use image::Rgb;
use png::text_metadata::TEXtChunk;
use png::{BitDepth, ColorType, Encoder};
use std::io::BufWriter;
use tiny_skia::Pixmap;

pub struct Canvas {
    pub pixmap: Pixmap,
    pub width: u32,
    pub height: u32,
    display_cache: Vec<u32>,
    display_width: u32,
    display_height: u32,
}

/// All render parameters and results to be embedded in PNG metadata.
pub struct RenderMetadata<'a> {
    pub input_path: &'a str,
    pub orig_width: u32,
    pub orig_height: u32,
    pub canvas_width: u32,
    pub canvas_height: u32,
    pub strokes_requested: usize,
    pub strokes_accepted: usize,
    pub strokes_attempted: usize,
    pub blur_sigma: f32,
    pub canny_low: f32,
    pub canny_high: f32,
    pub seed: u64,
    pub elapsed_secs: f64,
    pub coarse_accepted: usize,
    pub coarse_attempted: usize,
    pub medium_accepted: usize,
    pub medium_attempted: usize,
    pub fine_accepted: usize,
    pub fine_attempted: usize,
    pub topup_accepted: usize,
    pub topup_attempted: usize,
}

impl Canvas {
    pub fn new(width: u32, height: u32, bg: Rgb<u8>) -> Self {
        let mut pixmap = Pixmap::new(width, height).expect("Failed to create pixmap");
        let [r, g, b] = bg.0;
        pixmap.fill(tiny_skia::Color::from_rgba8(r, g, b, 255));
        let display_cache = vec![0u32; (width * height) as usize];
        Canvas {
            pixmap,
            width,
            height,
            display_cache,
            display_width: width,
            display_height: height,
        }
    }

    pub fn save(&self, path: &str) {
        self.pixmap.save_png(path).expect("Failed to save PNG");
    }

    /// Save the final output PNG with render metadata embedded as tEXt chunks.
    pub fn save_with_metadata(&self, path: &str, meta: &RenderMetadata) {
        let file = std::fs::File::create(path).expect("Failed to create output file");
        let w = BufWriter::new(file);
        let mut encoder = Encoder::new(w, self.width, self.height);
        encoder.set_color(ColorType::Rgba);
        encoder.set_depth(BitDepth::Eight);

        // Build all metadata entries as (keyword, text) pairs.
        let accept_rate = if meta.strokes_attempted > 0 {
            meta.strokes_accepted as f64 / meta.strokes_attempted as f64 * 100.0
        } else {
            0.0
        };
        let coarse_rate = if meta.coarse_attempted > 0 {
            meta.coarse_accepted as f64 / meta.coarse_attempted as f64 * 100.0
        } else {
            0.0
        };
        let medium_rate = if meta.medium_attempted > 0 {
            meta.medium_accepted as f64 / meta.medium_attempted as f64 * 100.0
        } else {
            0.0
        };
        let fine_rate = if meta.fine_attempted > 0 {
            meta.fine_accepted as f64 / meta.fine_attempted as f64 * 100.0
        } else {
            0.0
        };

        let entries = vec![
            ("Software", "msbr".into()),
            ("Source-File", meta.input_path.into()),
            (
                "Source-Dimensions",
                format!("{}x{}", meta.orig_width, meta.orig_height),
            ),
            (
                "Canvas-Dimensions",
                format!("{}x{}", meta.canvas_width, meta.canvas_height),
            ),
            ("Strokes-Requested", meta.strokes_requested.to_string()),
            ("Strokes-Accepted", meta.strokes_accepted.to_string()),
            ("Strokes-Attempted", meta.strokes_attempted.to_string()),
            ("Accept-Rate", format!("{:.1}%", accept_rate)),
            ("Blur-Sigma", format!("{}", meta.blur_sigma)),
            ("Canny-Low", format!("{}", meta.canny_low)),
            ("Canny-High", format!("{}", meta.canny_high)),
            ("Seed", meta.seed.to_string()),
            ("Render-Time", format!("{:.1}s", meta.elapsed_secs)),
            ("Coarse-Accepted", meta.coarse_accepted.to_string()),
            ("Coarse-Attempted", meta.coarse_attempted.to_string()),
            ("Coarse-Rate", format!("{:.1}%", coarse_rate)),
            ("Medium-Accepted", meta.medium_accepted.to_string()),
            ("Medium-Attempted", meta.medium_attempted.to_string()),
            ("Medium-Rate", format!("{:.1}%", medium_rate)),
            ("Fine-Accepted", meta.fine_accepted.to_string()),
            ("Fine-Attempted", meta.fine_attempted.to_string()),
            ("Fine-Rate", format!("{:.1}%", fine_rate)),
            ("Topup-Accepted", meta.topup_accepted.to_string()),
            ("Topup-Attempted", meta.topup_attempted.to_string()),
            (
                "Comment",
                format!(
                    "msbr | {}x{} | {} accepted/{} requested | \
                 blur={} canny={}/{} seed={} | {:.1}s",
                    meta.canvas_width,
                    meta.canvas_height,
                    meta.strokes_accepted,
                    meta.strokes_requested,
                    meta.blur_sigma,
                    meta.canny_low,
                    meta.canny_high,
                    meta.seed,
                    meta.elapsed_secs,
                ),
            ),
        ];

        let mut writer = encoder.write_header().expect("Failed to write PNG header");

        for (key, val) in &entries {
            writer
                .write_text_chunk(&TEXtChunk {
                    keyword: key.to_string(),
                    text: val.clone(),
                })
                .expect("Failed to write PNG text chunk");
        }

        writer
            .write_image_data(self.pixmap.data())
            .expect("Failed to write PNG image data");
    }

    pub fn update_display_cache(&mut self, out_width: u32, out_height: u32) {
        let data = self.pixmap.data();
        self.display_width = out_width;
        self.display_height = out_height;

        if out_width == self.width && out_height == self.height {
            self.display_cache
                .resize((self.width * self.height) as usize, 0);
            for (i, chunk) in data.chunks_exact(4).enumerate() {
                self.display_cache[i] =
                    ((chunk[0] as u32) << 16) | ((chunk[1] as u32) << 8) | (chunk[2] as u32);
            }
            return;
        }

        let dst_len = (out_width * out_height) as usize;
        self.display_cache.resize(dst_len, 0);

        for out_y in 0..out_height {
            let src_y0 = (out_y * self.height) / out_height;
            let src_y1 = (((out_y + 1) * self.height) / out_height).max(src_y0 + 1);

            for out_x in 0..out_width {
                let src_x0 = (out_x * self.width) / out_width;
                let src_x1 = (((out_x + 1) * self.width) / out_width).max(src_x0 + 1);

                let mut r_sum = 0u32;
                let mut g_sum = 0u32;
                let mut b_sum = 0u32;
                let mut count = 0u32;

                for src_y in src_y0..src_y1.min(self.height) {
                    let row_start = (src_y * self.width) as usize;
                    for src_x in src_x0..src_x1.min(self.width) {
                        let i = (row_start + src_x as usize) * 4;
                        r_sum += data[i] as u32;
                        g_sum += data[i + 1] as u32;
                        b_sum += data[i + 2] as u32;
                        count += 1;
                    }
                }

                let i = (out_y * out_width + out_x) as usize;
                self.display_cache[i] =
                    ((r_sum / count) << 16) | ((g_sum / count) << 8) | (b_sum / count);
            }
        }
    }

    pub fn display_buffer(&self) -> &[u32] {
        &self.display_cache
    }
}
