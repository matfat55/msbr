use image::RgbImage;
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Grayscale conversion
// ---------------------------------------------------------------------------

pub fn grayscale(img: &RgbImage) -> Vec<f32> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let raw = img.as_raw();
    let mut out = vec![0.0f32; w * h];
    for i in 0..w * h {
        let idx = i * 3;
        out[i] =
            0.299 * raw[idx] as f32 + 0.587 * raw[idx + 1] as f32 + 0.114 * raw[idx + 2] as f32;
    }
    out
}

// ---------------------------------------------------------------------------
// Gaussian blur
// ---------------------------------------------------------------------------

fn gaussian_kernel(sigma: f32) -> Vec<f32> {
    let radius = (sigma * 3.0).ceil() as i32;
    let size = (2 * radius + 1) as usize;
    let mut kernel = vec![0.0f32; size];
    let denom = 2.0 * sigma * sigma;
    let mut sum = 0.0f32;
    for i in 0..size {
        let x = (i as i32 - radius) as f32;
        let val = (-x * x / denom).exp();
        kernel[i] = val;
        sum += val;
    }
    for v in &mut kernel {
        *v /= sum;
    }
    kernel
}

pub fn gaussian_blur(data: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    if sigma < 0.5 {
        return data.to_vec();
    }
    let kernel = gaussian_kernel(sigma);
    let radius = (kernel.len() / 2) as i32;

    // Horizontal pass
    let mut tmp = vec![0.0f32; w * h];
    tmp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let mut sum = 0.0f32;
            for k in 0..kernel.len() {
                let sx = (x as i32 + k as i32 - radius).clamp(0, w as i32 - 1) as usize;
                sum += data[y * w + sx] * kernel[k];
            }
            row[x] = sum;
        }
    });

    // Vertical pass
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let mut sum = 0.0f32;
            for k in 0..kernel.len() {
                let sy = (y as i32 + k as i32 - radius).clamp(0, h as i32 - 1) as usize;
                sum += tmp[sy * w + x] * kernel[k];
            }
            row[x] = sum;
        }
    });

    out
}

// ---------------------------------------------------------------------------
// Gradient field (Sobel on blurred grayscale)
// ---------------------------------------------------------------------------

pub struct GradientField {
    /// Gradient angle at each pixel (radians)
    pub angles: Vec<f32>,
    /// Gradient magnitude at each pixel (0..~max)
    pub magnitudes: Vec<f32>,
    pub width: u32,
    pub height: u32,
}

impl GradientField {
    pub fn from_image(img: &RgbImage, blur_sigma: f32) -> Self {
        let w = img.width() as usize;
        let h = img.height() as usize;
        let gray = grayscale(img);
        let blurred = gaussian_blur(&gray, w, h, blur_sigma);

        let mut angles = vec![0.0f32; w * h];
        let mut magnitudes = vec![0.0f32; w * h];

        // Sobel on interior pixels; boundary stays zero
        for y in 1..(h - 1) {
            for x in 1..(w - 1) {
                let gx = sobel_x(&blurred, w, x, y);
                let gy = sobel_y(&blurred, w, x, y);
                let mag = (gx * gx + gy * gy).sqrt();
                // Stroke angle = perpendicular to gradient direction
                let angle = gy.atan2(gx) + std::f32::consts::FRAC_PI_2;
                let idx = y * w + x;
                angles[idx] = angle;
                magnitudes[idx] = mag;
            }
        }

        GradientField {
            angles,
            magnitudes,
            width: w as u32,
            height: h as u32,
        }
    }

    #[inline]
    pub fn angle_at(&self, x: u32, y: u32) -> f32 {
        self.angles[(y * self.width + x) as usize]
    }

    #[inline]
    pub fn magnitude_at(&self, x: u32, y: u32) -> f32 {
        self.magnitudes[(y * self.width + x) as usize]
    }
}

#[inline]
fn sobel_x(data: &[f32], w: usize, x: usize, y: usize) -> f32 {
    let g =
        |dx: i32, dy: i32| -> f32 { data[(y as i32 + dy) as usize * w + (x as i32 + dx) as usize] };
    -g(-1, -1) + g(1, -1) - 2.0 * g(-1, 0) + 2.0 * g(1, 0) - g(-1, 1) + g(1, 1)
}

#[inline]
fn sobel_y(data: &[f32], w: usize, x: usize, y: usize) -> f32 {
    let g =
        |dx: i32, dy: i32| -> f32 { data[(y as i32 + dy) as usize * w + (x as i32 + dx) as usize] };
    -g(-1, -1) - 2.0 * g(0, -1) - g(1, -1) + g(-1, 1) + 2.0 * g(0, 1) + g(1, 1)
}

// ---------------------------------------------------------------------------
// Canny edge detection
// ---------------------------------------------------------------------------

pub struct EdgeMap {
    /// Binary edge mask (true = edge pixel)
    pub edges: Vec<bool>,
    /// Edge positions cached for fast sampling
    pub edge_positions: Vec<(u32, u32)>,
    pub width: u32,
    pub height: u32,
}

impl EdgeMap {
    pub fn from_gradient(gradient: &GradientField, low_thresh: f32, high_thresh: f32) -> Self {
        let w = gradient.width as usize;
        let h = gradient.height as usize;

        // Non-maximum suppression
        let mut nms = vec![0.0f32; w * h];
        for y in 1..(h - 1) {
            for x in 1..(w - 1) {
                let idx = y * w + x;
                let mag = gradient.magnitudes[idx];
                if mag < low_thresh {
                    continue;
                }

                // Quantize gradient direction to one of four neighbor pairs
                let angle = gradient.angles[idx] - std::f32::consts::FRAC_PI_2; // back to gradient dir
                let a = angle.rem_euclid(std::f32::consts::PI);

                let (n1, n2) =
                    if a < std::f32::consts::FRAC_PI_8 || a >= 7.0 * std::f32::consts::FRAC_PI_8 {
                        // ~horizontal gradient -> compare left/right
                        (
                            gradient.magnitudes[y * w + x - 1],
                            gradient.magnitudes[y * w + x + 1],
                        )
                    } else if a < 3.0 * std::f32::consts::FRAC_PI_8 {
                        // ~45 deg
                        (
                            gradient.magnitudes[(y - 1) * w + x + 1],
                            gradient.magnitudes[(y + 1) * w + x - 1],
                        )
                    } else if a < 5.0 * std::f32::consts::FRAC_PI_8 {
                        // ~vertical
                        (
                            gradient.magnitudes[(y - 1) * w + x],
                            gradient.magnitudes[(y + 1) * w + x],
                        )
                    } else {
                        // ~135 deg
                        (
                            gradient.magnitudes[(y - 1) * w + x - 1],
                            gradient.magnitudes[(y + 1) * w + x + 1],
                        )
                    };

                if mag >= n1 && mag >= n2 {
                    nms[idx] = mag;
                }
            }
        }

        // Double threshold + hysteresis via flood fill
        let mut edges = vec![false; w * h];
        let mut strong = Vec::new();

        for y in 1..(h - 1) {
            for x in 1..(w - 1) {
                let idx = y * w + x;
                if nms[idx] >= high_thresh {
                    edges[idx] = true;
                    strong.push((x, y));
                }
            }
        }

        // Flood fill from strong edges to connected weak edges
        while let Some((x, y)) = strong.pop() {
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                        continue;
                    }
                    let nidx = ny as usize * w + nx as usize;
                    if !edges[nidx] && nms[nidx] >= low_thresh {
                        edges[nidx] = true;
                        strong.push((nx as usize, ny as usize));
                    }
                }
            }
        }

        // Cache edge positions
        let mut edge_positions = Vec::new();
        for y in 0..h {
            for x in 0..w {
                if edges[y * w + x] {
                    edge_positions.push((x as u32, y as u32));
                }
            }
        }

        EdgeMap {
            edges,
            edge_positions,
            width: w as u32,
            height: h as u32,
        }
    }

    #[inline]
    pub fn is_edge(&self, x: u32, y: u32) -> bool {
        self.edges[(y * self.width + x) as usize]
    }
}

// ---------------------------------------------------------------------------
// Complexity map (local standard deviation of luminance)
// ---------------------------------------------------------------------------

pub struct ComplexityMap {
    /// Normalized complexity per pixel [0.0, 1.0]
    pub data: Vec<f32>,
    pub width: u32,
    pub height: u32,
}

impl ComplexityMap {
    pub fn compute(img: &RgbImage, window_radius: usize) -> Self {
        let w = img.width() as usize;
        let h = img.height() as usize;
        let gray = grayscale(img);

        let mut data = vec![0.0f32; w * h];

        // Compute local variance in parallel
        data.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for x in 0..w {
                let x0 = x.saturating_sub(window_radius);
                let x1 = (x + window_radius).min(w - 1);
                let y0 = y.saturating_sub(window_radius);
                let y1 = (y + window_radius).min(h - 1);

                let mut sum = 0.0f32;
                let mut sum_sq = 0.0f32;
                let mut count = 0.0f32;

                for wy in y0..=y1 {
                    for wx in x0..=x1 {
                        let v = gray[wy * w + wx];
                        sum += v;
                        sum_sq += v * v;
                        count += 1.0;
                    }
                }

                let mean = sum / count;
                let variance = (sum_sq / count - mean * mean).max(0.0);
                row[x] = variance.sqrt();
            }
        });

        // Normalize to [0, 1]
        let max_val = data.iter().cloned().fold(1.0f32, f32::max);
        let inv_max = 1.0 / max_val;
        for v in &mut data {
            *v *= inv_max;
        }

        ComplexityMap {
            data,
            width: w as u32,
            height: h as u32,
        }
    }

    #[inline]
    pub fn at(&self, x: u32, y: u32) -> f32 {
        self.data[(y * self.width + x) as usize]
    }
}
// ---------------------------------------------------------------------------
// Importance map (combines error, edge, complexity for stroke placement)
// ---------------------------------------------------------------------------

pub struct ImportanceMap {
    pub width: u32,
    pub height: u32,
    prob: Vec<f32>,
    alias: Vec<usize>,

    scratch: Vec<f32>,
}

impl ImportanceMap {
    /// Allocate an importance map for the given dimensions.
    /// Call `rebuild` to populate it before use.
    pub fn new(width: u32, height: u32) -> Self {
        let n = (width * height) as usize;
        Self {
            width,
            height,
            prob: vec![0.0f32; n],
            alias: vec![0usize; n],
            scratch: vec![0.0f32; n],
        }
    }

    /// Recompute the alias table in-place from the given inputs
    /// All input slices must have length == width * height.
    pub fn rebuild(
        &mut self,
        error: &[f32],
        edge: &[f32],
        complexity: &[f32],
        suppression: &[f32],
        error_weight: f32,
        edge_weight: f32,
        complexity_weight: f32,
    ) {
        let n = self.prob.len();

        // Compute the max error value in a single pass so we can normalize
        // inline, eliminating the need for ErrorMap::normalized() and its
        // full-image Vec allocation entirely.
        let max_err = error.iter().cloned().fold(0.001f32, f32::max);
        let inv_max_err = 1.0 / max_err;

        // Fill scratch with un-scaled weights, normalizing error on the fly.
        let mut sum = 0.0f32;
        for i in 0..n {
            // Suppression multiplies the whole weight -- a fully-dense pixel gets
            // weight 0.0001 (the floor) so it can still be sampled in extremis
            // but is strongly deprioritized.
            let base = error_weight * (error[i] * inv_max_err)
                + edge_weight * edge[i]
                + complexity_weight * complexity[i]
                + 0.0001;
            let w = base * suppression[i].max(0.0001);
            self.scratch[i] = w;
            sum += w;
        }

        // Scale so that the mean weight == 1.0
        let scale = n as f32 / sum;
        for v in &mut self.scratch {
            *v *= scale;
        }

        // Build the alias table directly into self.prob / self.alias using
        // two stack Vecs.
        let mut small: Vec<usize> = Vec::with_capacity(n / 2);
        let mut large: Vec<usize> = Vec::with_capacity(n / 2);

        for (i, &v) in self.scratch.iter().enumerate() {
            if v < 1.0 {
                small.push(i);
            } else {
                large.push(i);
            }
        }

        while let (Some(s), Some(l)) = (small.pop(), large.pop()) {
            self.prob[s] = self.scratch[s];
            self.alias[s] = l;

            self.scratch[l] = self.scratch[l] + self.scratch[s] - 1.0;

            if self.scratch[l] < 1.0 {
                small.push(l);
            } else {
                large.push(l);
            }
        }

        for i in large {
            self.prob[i] = 1.0;
        }
        for i in small {
            self.prob[i] = 1.0;
        }
    }

    /// Sample a pixel position weighted by importance.
    pub fn sample<R: rand::Rng>(&self, rng: &mut R) -> (u32, u32) {
        let n = self.prob.len();
        let i = rng.gen_range(0..n);

        let idx = if rng.gen::<f32>() < self.prob[i] {
            i
        } else {
            self.alias[i]
        };

        let w = self.width as usize;
        let x = (idx % w) as u32;
        let y = (idx / w) as u32;

        // Keep samples off the 1px border because Sobel/Canny/gradient access
        // rely on interior neighborhoods.
        (x.clamp(1, self.width - 2), y.clamp(1, self.height - 2))
    }
}

// ---------------------------------------------------------------------------
// Error map (per-pixel squared error between canvas and target)
// ---------------------------------------------------------------------------

pub struct ErrorMap {
    pub data: Vec<f32>,
    pub width: u32,
    pub height: u32,
}

impl ErrorMap {
    pub fn compute(canvas_data: &[u8], target_raw: &[u8], width: u32, height: u32) -> Self {
        let n = (width * height) as usize;
        let mut data = vec![0.0f32; n];

        for i in 0..n {
            let ci = i * 4;
            let ti = i * 3;
            let dr = canvas_data[ci] as f32 - target_raw[ti] as f32;
            let dg = canvas_data[ci + 1] as f32 - target_raw[ti + 1] as f32;
            let db = canvas_data[ci + 2] as f32 - target_raw[ti + 2] as f32;
            // Store squared error -- sqrt is not needed because the error map
            // is only ever used for relative importance weighting, and sqrt
            // is monotonic so it cannot change which pixels are more important.
            data[i] = dr * dr + dg * dg + db * db;
        }

        ErrorMap {
            data,
            width,
            height,
        }
    }

    pub fn update_region(
        &mut self,
        canvas_data: &[u8],
        target_raw: &[u8],
        x0: u32,
        y0: u32,
        x1: u32,
        y1: u32,
    ) {
        let w = self.width;
        for y in y0..=y1.min(self.height - 1) {
            for x in x0..=x1.min(w - 1) {
                let i = (y * w + x) as usize;
                let ci = i * 4;
                let ti = i * 3;
                let dr = canvas_data[ci] as f32 - target_raw[ti] as f32;
                let dg = canvas_data[ci + 1] as f32 - target_raw[ti + 1] as f32;
                let db = canvas_data[ci + 2] as f32 - target_raw[ti + 2] as f32;
                self.data[i] = dr * dr + dg * dg + db * db;
            }
        }
    }
}
