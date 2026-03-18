use crate::analysis::{EdgeMap, ErrorMap, GradientField, ImportanceMap};
use crate::stroke::{BrushKind, BrushStroke};
use rand::rngs::StdRng;
use rand::Rng;
use rayon::prelude::*;
use std::cell::RefCell;
use tiny_skia::Pixmap;

// ---------------------------------------------------------------------------
// Pass configuration
// ---------------------------------------------------------------------------
pub struct StrokeDensity {
    pub data: Vec<u16>,
    pub width: u32,
    pub height: u32,
}
impl StrokeDensity {
    /// Reset density to zero. Called between passes so each pass starts fresh.
    pub fn reset(&mut self) {
        self.data.fill(0);
    }

    /// Return a normalized suppression weight per pixel: 1.0 where density
    /// is zero, falling toward 0.0 as density approaches `cap`.
    pub fn suppression_weights(&self, cap: u16) -> Vec<f32> {
        let cap_f = cap as f32;
        self.data
            .iter()
            .map(|&d| {
                if d >= cap {
                    0.0
                } else {
                    1.0 - (d as f32 / cap_f)
                }
            })
            .collect()
    }
}
#[derive(Clone, Debug)]
pub struct PassConfig {
    pub name: &'static str,
    pub stroke_count: usize,
    pub batch_size: usize,
    pub refine_iterations: usize,
    pub min_length_frac: f32, // fraction of min_dim
    pub max_length_frac: f32,
    pub min_width_frac: f32,
    pub max_width_frac: f32,
    pub error_weight: f32,
    pub edge_weight: f32,
    pub complexity_weight: f32,
    pub alpha_range: (u8, u8),
    /// How much angle jitter to add on top of gradient direction
    pub angle_jitter: f32,
}

impl PassConfig {
    pub fn coarse(total_strokes: usize, image_pixels: u32) -> Self {
        // Cap: a coarse stroke covers ~1-2% of canvas area on average.
        let fraction_count = (total_strokes as f32 * 0.10) as usize;
        let area_cap = ((image_pixels as f32 / 500.0) as usize).max(50).min(800);
        PassConfig {
            name: "Coarse",
            stroke_count: fraction_count.min(area_cap),
            batch_size: 64,
            refine_iterations: 3,
            min_length_frac: 0.06,
            max_length_frac: 0.18,
            min_width_frac: 0.015,
            max_width_frac: 0.06,
            error_weight: 0.6,
            edge_weight: 0.0,
            complexity_weight: 0.4,
            alpha_range: (160, 230),
            angle_jitter: 0.4,
        }
    }

    pub fn medium(total_strokes: usize, image_pixels: u32) -> Self {
        // Cap: medium strokes cover ~0.2% of canvas each.
        // Beyond ~3000 medium strokes the mid-frequency detail is filled.
        let fraction_count = (total_strokes as f32 * 0.35) as usize;
        let area_cap = ((image_pixels as f32 / 120.0) as usize).max(200).min(6000);
        PassConfig {
            name: "Medium",
            stroke_count: fraction_count.min(area_cap),
            batch_size: 48,
            refine_iterations: 4,
            min_length_frac: 0.025,
            max_length_frac: 0.09,
            min_width_frac: 0.006,
            max_width_frac: 0.03,
            error_weight: 0.5,
            edge_weight: 0.2,
            complexity_weight: 0.3,
            alpha_range: (140, 210),
            angle_jitter: 0.3,
        }
    }

    pub fn fine(stroke_count: usize) -> Self {
        PassConfig {
            name: "Fine",
            stroke_count,
            batch_size: 32,
            refine_iterations: 5,
            min_length_frac: 0.008,
            max_length_frac: 0.04,
            min_width_frac: 0.002,
            max_width_frac: 0.012,
            error_weight: 0.3,
            edge_weight: 0.5,
            complexity_weight: 0.2,
            alpha_range: (100, 200),
            angle_jitter: 0.2,
        }
    }

    pub fn topup(stroke_count: usize) -> Self {
        PassConfig {
            name: "Top-up",
            stroke_count,
            batch_size: 32,
            refine_iterations: 3,
            min_length_frac: 0.008,
            max_length_frac: 0.04,
            min_width_frac: 0.002,
            max_width_frac: 0.012,
            error_weight: 0.35,
            edge_weight: 0.45,
            complexity_weight: 0.2,
            alpha_range: (100, 190),
            angle_jitter: 0.25,
        }
    }
}

/// Walk along the local direction field and return the final heading.
/// This approximates "flow-following" orientation so longer strokes can bend
/// toward structure instead of using only the center pixel's angle.
fn follow_gradient(gradient: &GradientField, x: f32, y: f32, length: f32, step: f32) -> f32 {
    let mut px = x;
    let mut py = y;

    let mut angle = gradient.angle_at(x as u32, y as u32);

    let steps = (length / step) as usize;

    for _ in 0..steps {
        let gx = px + step * angle.cos();
        let gy = py + step * angle.sin();

        if gx < 1.0
            || gy < 1.0
            || gx >= (gradient.width - 2) as f32
            || gy >= (gradient.height - 2) as f32
        {
            break;
        }

        px = gx;
        py = gy;

        angle = gradient.angle_at(px as u32, py as u32);
    }

    angle
}
// ---------------------------------------------------------------------------
// Batch stroke generation
// ---------------------------------------------------------------------------

pub fn generate_batch(
    rng: &mut StdRng,
    config: &PassConfig,
    importance: &ImportanceMap,
    gradient: &GradientField,
    edge_map: &EdgeMap,
    target_raw: &[u8],
    width: u32,
    height: u32,
) -> Vec<BrushStroke> {
    let min_dim = width.min(height) as f32;
    let min_len = min_dim * config.min_length_frac;
    let max_len = min_dim * config.max_length_frac;
    let min_wid = min_dim * config.min_width_frac;
    let max_wid = min_dim * config.max_width_frac;

    // Phase 1 -- sequential: sample positions.
    struct Candidate {
        x: u32,
        y: u32,
        angle: f32,
        length: f32,
        width_f: f32,
        curve: f32,
        taper_start: f32,
        taper_end: f32,
        alpha: u8,
        kind: BrushKind,
    }

    let mut candidates: Vec<Candidate> = Vec::with_capacity(config.batch_size);

    for _ in 0..config.batch_size {
        let (x, y) = importance.sample(rng);

        let base_angle = follow_gradient(gradient, x as f32, y as f32, max_len, 2.0);
        let jitter = rng.gen_range(-config.angle_jitter..config.angle_jitter);
        let angle = base_angle + jitter;

        let kind = if edge_map.is_edge(x, y) {
            match rng.gen_range(0u8..8) {
                0..=1 => BrushKind::Round,
                2..=3 => BrushKind::Flat,
                _ => BrushKind::Rigger,
            }
        } else {
            match rng.gen_range(0u8..8) {
                0..=3 => BrushKind::Round,
                4..=5 => BrushKind::Flat,
                _ => BrushKind::Rigger,
            }
        };

        let edge_factor = if edge_map.is_edge(x, y) { 0.4 } else { 1.0 };
        let (length, width_f, curve_scale) = match kind {
            BrushKind::Round => (
                rng.gen_range(min_len..max_len) * edge_factor,
                rng.gen_range(min_wid..max_wid),
                0.2,
            ),
            BrushKind::Flat => (
                rng.gen_range(min_len * 0.5..max_len * 0.7) * edge_factor,
                rng.gen_range(min_wid * 2.0..max_wid * 3.0),
                0.05,
            ),
            BrushKind::Rigger => (
                rng.gen_range(max_len * 0.4..max_len * 1.2) * edge_factor,
                rng.gen_range(min_wid * 0.3..min_wid * 1.2),
                0.3,
            ),
        };

        let alpha = rng.gen_range(config.alpha_range.0..=config.alpha_range.1);
        let curve_sign = if rng.gen_bool(0.5) { 1.0f32 } else { -1.0 };
        let curve = curve_sign * length * curve_scale * rng.gen_range(0.0f32..1.0);

        let (taper_start, taper_end) = match rng.gen_range(0u8..4) {
            0 => (rng.gen_range(0.1f32..0.5), 1.0),
            1 => (1.0, rng.gen_range(0.1f32..0.5)),
            2 => (rng.gen_range(0.15f32..0.5), rng.gen_range(0.1f32..0.4)),
            _ => (1.0, 1.0),
        };

        candidates.push(Candidate {
            x,
            y,
            angle,
            length,
            width_f,
            curve,
            taper_start,
            taper_end,
            alpha,
            kind,
        });
    }

    // Phase 2 -- parallel: color sample + BrushStroke construction
    candidates
        .into_par_iter()
        .map(|c| {
            let (r, g, b) = if matches!(c.kind, BrushKind::Rigger) && edge_map.is_edge(c.x, c.y) {
                let mut r_sum = 0u32;
                let mut g_sum = 0u32;
                let mut b_sum = 0u32;
                let mut count = 0u32;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let nx = c.x as i32 + dx;
                        let ny = c.y as i32 + dy;
                        if nx >= 0 && ny >= 0 && nx < width as i32 && ny < height as i32 {
                            let i = (ny as u32 * width + nx as u32) as usize * 3;
                            r_sum += target_raw[i] as u32;
                            g_sum += target_raw[i + 1] as u32;
                            b_sum += target_raw[i + 2] as u32;
                            count += 1;
                        }
                    }
                }
                (
                    (r_sum / count) as u8,
                    (g_sum / count) as u8,
                    (b_sum / count) as u8,
                )
            } else {
                let i = (c.y * width + c.x) as usize * 3;
                (target_raw[i], target_raw[i + 1], target_raw[i + 2])
            };

            BrushStroke::new(
                c.x as f32,
                c.y as f32,
                c.angle,
                c.length,
                c.width_f,
                c.curve,
                c.taper_start,
                c.taper_end,
                r,
                g,
                b,
                c.alpha,
                c.kind,
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Stroke evaluation (tile-based MSE improvement)
// ---------------------------------------------------------------------------

thread_local! {
    static TILE_BUF: RefCell<Vec<u8>> = RefCell::new(Vec::new());
}

pub fn measure_improvement(
    canvas_data: &[u8],
    stroke: &BrushStroke,
    target_raw: &[u8],
    width: u32,
    height: u32,
) -> f64 {
    let (x0, y0, x1, y1) = stroke.bbox(width, height);
    let tile_w = (x1 - x0 + 1) as usize;
    let tile_h = (y1 - y0 + 1) as usize;
    if tile_w == 0 || tile_h == 0 {
        return 0.0;
    }

    // Fast color reject: if the stroke color is already close to canvas at
    // center, it almost certainly won't improve anything.
    let cx = stroke.x.clamp(0.0, (width - 1) as f32) as usize;
    let cy = stroke.y.clamp(0.0, (height - 1) as f32) as usize;
    let ci = (cy * width as usize + cx) * 4;
    let dr = stroke.r as i32 - canvas_data[ci] as i32;
    let dg = stroke.g as i32 - canvas_data[ci + 1] as i32;
    let db = stroke.b as i32 - canvas_data[ci + 2] as i32;
    if dr * dr + dg * dg + db * db < 64 {
        return 0.0;
    }

    TILE_BUF.with(|cell| {
        let mut buf = cell.borrow_mut();

        let needed = tile_w * tile_h * 4;
        if buf.len() < needed {
            buf.resize(needed, 0);
        }

        // --- Copy canvas region into tile buffer ---
        // Use running pointer arithmetic instead of index recomputation:
        // si/ti advance by fixed strides each row, inner loop increments by 4.
        let tile_w4 = tile_w * 4;
        let canvas_stride = width as usize * 4;

        let mut si = (y0 as usize * width as usize + x0 as usize) * 4;

        for ty in 0..tile_h {
            let di = ty * tile_w4;
            buf[di..di + tile_w4].copy_from_slice(&canvas_data[si..si + tile_w4]);
            si += canvas_stride;
        }

        // --- Render the stroke onto the tile buffer via PixmapMut::from_bytes ---
        // This wraps our pre-allocated Vec<u8> directly -- zero allocation.
        let mut tile_mut =
            tiny_skia::PixmapMut::from_bytes(&mut buf[..needed], tile_w as u32, tile_h as u32)
                .expect("tile dimensions valid");

        let mut os = stroke.clone();
        os.x -= x0 as f32;
        os.y -= y0 as f32;
        os.render_to_mut(&mut tile_mut);

        // --- Score: sum(old_err² - new_err²) with pointer arithmetic ---
        // si: canvas RGBA pointer, restarted at bbox origin
        // ti: target RGB pointer, restarted at bbox origin
        // Both advance by fixed row strides; inner loop uses +4 / +3.
        let canvas_stride4 = width as usize * 4;
        let target_stride3 = width as usize * 3;

        let mut si = (y0 as usize * width as usize + x0 as usize) * 4;
        let mut ti = (y0 as usize * width as usize + x0 as usize) * 3;

        let mut delta: i64 = 0;

        for ty in 0..tile_h {
            let mut si_row = si;
            let mut ti_row = ti;
            let mut di = ty * tile_w4; // tile RGBA pointer

            for _ in 0..tile_w {
                let tr = target_raw[ti_row] as i32;
                let tg = target_raw[ti_row + 1] as i32;
                let tb = target_raw[ti_row + 2] as i32;

                let or_ = canvas_data[si_row] as i32;
                let og = canvas_data[si_row + 1] as i32;
                let ob = canvas_data[si_row + 2] as i32;

                let nr = buf[di] as i32;
                let ng = buf[di + 1] as i32;
                let nb = buf[di + 2] as i32;

                delta += ((or_ - tr) * (or_ - tr) + (og - tg) * (og - tg) + (ob - tb) * (ob - tb))
                    as i64;
                delta -=
                    ((nr - tr) * (nr - tr) + (ng - tg) * (ng - tg) + (nb - tb) * (nb - tb)) as i64;

                si_row += 4;
                ti_row += 3;
                di += 4;
            }

            si += canvas_stride4;
            ti += target_stride3;
        }

        delta as f64 / (tile_w * tile_h) as f64
    })
}
// ---------------------------------------------------------------------------
// Batch optimization via local coordinate search
// ---------------------------------------------------------------------------

/// Refine a batch of strokes by perturbing each parameter and keeping improvements.

pub fn refine_batch(
    batch: &mut Vec<BrushStroke>,
    canvas_data: &[u8],
    target_raw: &[u8],
    width: u32,
    height: u32,
    iterations: usize,
    gradient: &GradientField,
) {
    if iterations == 0 || batch.is_empty() {
        return;
    }

    // Compute base scores once -- carried forward so each iteration starts
    // from the previous best without re-evaluating.
    let mut scores: Vec<f64> = batch
        .par_iter()
        .map(|s| measure_improvement(canvas_data, s, target_raw, width, height))
        .collect();

    for _iter in 0..iterations {
        let (refined, new_scores): (Vec<BrushStroke>, Vec<f64>) = batch
            .par_iter()
            .zip(scores.par_iter())
            .map(|(stroke, &base_score)| {
                let mut best = stroke.clone();
                let mut best_score = base_score;

                let pos_delta = stroke.length * 0.15;
                let ang_delta = 0.15f32;
                let len_delta = stroke.length * 0.12;
                let wid_delta = stroke.width * 0.15;

                // Position perturbations
                for &(dx, dy) in &[
                    (pos_delta, 0.0f32),
                    (-pos_delta, 0.0),
                    (0.0, pos_delta),
                    (0.0, -pos_delta),
                ] {
                    let mut s = stroke.clone();
                    s.x = (s.x + dx).clamp(1.0, (width - 2) as f32);
                    s.y = (s.y + dy).clamp(1.0, (height - 2) as f32);
                    let px = s.x as u32;
                    let py = s.y as u32;
                    let tidx = ((py * width + px) as usize) * 3;
                    if tidx + 2 < target_raw.len() {
                        s.r = target_raw[tidx];
                        s.g = target_raw[tidx + 1];
                        s.b = target_raw[tidx + 2];
                    }
                    let score = measure_improvement(canvas_data, &s, target_raw, width, height);
                    if score > best_score {
                        best_score = score;
                        best = s;
                    }
                }

                // Angle perturbations -- sync trig after each mutation
                for &da in &[ang_delta, -ang_delta] {
                    let mut s = best.clone();
                    s.angle += da;
                    s.sync_trig(); // keeps cos_angle/sin_angle in sync
                    let score = measure_improvement(canvas_data, &s, target_raw, width, height);
                    if score > best_score {
                        best_score = score;
                        best = s;
                    }
                }

                // Snap to gradient direction
                {
                    let px = best.x as u32;
                    let py = best.y as u32;
                    if px < width && py < height {
                        let mut s = best.clone();
                        s.angle = gradient.angle_at(px, py);
                        s.sync_trig();
                        let score = measure_improvement(canvas_data, &s, target_raw, width, height);
                        if score > best_score {
                            best_score = score;
                            best = s;
                        }
                    }
                }

                // Length perturbations -- no trig change, no sync needed
                for &dl in &[len_delta, -len_delta] {
                    let mut s = best.clone();
                    s.length = (s.length + dl).max(2.0);
                    let score = measure_improvement(canvas_data, &s, target_raw, width, height);
                    if score > best_score {
                        best_score = score;
                        best = s;
                    }
                }

                // Width perturbations -- no trig change, no sync needed
                for &dw in &[wid_delta, -wid_delta] {
                    let mut s = best.clone();
                    s.width = (s.width + dw).max(1.0);
                    let score = measure_improvement(canvas_data, &s, target_raw, width, height);
                    if score > best_score {
                        best_score = score;
                        best = s;
                    }
                }

                (best, best_score)
            })
            .unzip();

        *batch = refined;
        scores = new_scores;
    }
}
/// Commit a batch of strokes to the canvas. Only apply strokes that improve the result.
/// Returns the number of strokes accepted.
pub fn commit_batch(
    batch: &[BrushStroke],
    pixmap: &mut Pixmap,
    target_raw: &[u8],
    error_map: &mut ErrorMap,
    width: u32,
    height: u32,
    density: &mut StrokeDensity,
) -> usize {
    let mut accepted = 0;

    // Score each stroke against current canvas state, then apply in score order
    let canvas_data = pixmap.data();
    let mut scored: Vec<(f64, usize)> = batch
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let imp = measure_improvement(canvas_data, s, target_raw, width, height);
            (imp, i)
        })
        .collect();

    // Sort by improvement descending: apply best strokes first.
    // `total_cmp` avoids panic on NaN ordering edge cases.
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));

    for &(imp, idx) in &scored {
        if imp <= 0.0 {
            break;
        }
        let stroke = &batch[idx];

        // Re-check against current canvas state because earlier accepted strokes
        // in this same batch may have invalidated the original score.
        let current_imp = measure_improvement(pixmap.data(), stroke, target_raw, width, height);
        if current_imp <= 0.0 {
            continue;
        }

        let (x0, y0, x1, y1) = stroke.bbox(width, height);
        stroke.render(pixmap);

        let cx = stroke.x as i32;
        let cy = stroke.y as i32;
        let radius = ((stroke.length / 2.0 + stroke.width) as i32).max(4);
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let nx = cx + dx;
                let ny = cy + dy;

                if nx >= 0 && ny >= 0 && nx < width as i32 && ny < height as i32 {
                    let i = (ny as u32 * density.width + nx as u32) as usize;
                    density.data[i] = density.data[i].saturating_add(1);
                }
            }
        }

        error_map.update_region(pixmap.data(), target_raw, x0, y0, x1, y1);
        accepted += 1;
    }

    accepted
}
