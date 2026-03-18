use tiny_skia::{BlendMode, Color, FillRule, Paint, PathBuilder, Pixmap, Transform};

#[derive(Clone, Debug)]
pub enum BrushKind {
    Round,
    Flat,
    Rigger,
}

#[derive(Clone, Debug)]
pub struct BrushStroke {
    pub x: f32,
    pub y: f32,
    pub angle: f32,
    pub cos_angle: f32,
    pub sin_angle: f32,
    pub length: f32,
    pub width: f32,
    pub curve: f32,
    pub taper_start: f32,
    pub taper_end: f32,
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub alpha: u8,
    pub kind: BrushKind,
}

impl BrushStroke {
    /// Construct a stroke, computing and caching cos/sin of the angle once.
    pub fn new(
        x: f32,
        y: f32,
        angle: f32,
        length: f32,
        width: f32,
        curve: f32,
        taper_start: f32,
        taper_end: f32,
        r: u8,
        g: u8,
        b: u8,
        alpha: u8,
        kind: BrushKind,
    ) -> Self {
        let (sin_angle, cos_angle) = angle.sin_cos();
        BrushStroke {
            x,
            y,
            angle,
            cos_angle,
            sin_angle,
            length,
            width,
            curve,
            taper_start,
            taper_end,
            r,
            g,
            b,
            alpha,
            kind,
        }
    }

    #[inline]
    pub fn sync_trig(&mut self) {
        (self.sin_angle, self.cos_angle) = self.angle.sin_cos();
    }

    /// Bounding box as (x0, y0, x1, y1), clamped to canvas dimensions.
    pub fn bbox(&self, canvas_w: u32, canvas_h: u32) -> (u32, u32, u32, u32) {
        let half = self.length / 2.0 + self.width + self.curve.abs();
        let x0 = ((self.x - half) as i32).max(0) as u32;
        let y0 = ((self.y - half) as i32).max(0) as u32;
        let x1 = ((self.x + half) as u32).min(canvas_w - 1);
        let y1 = ((self.y + half) as u32).min(canvas_h - 1);
        (x0, y0, x1, y1)
    }

    /// Build the stroke outline path. Uses cached cos/sin -- no trig calls.
    fn build_path(&self) -> Option<tiny_skia::Path> {
        let cos_a = self.cos_angle;
        let sin_a = self.sin_angle;
        let perp_x = -sin_a;
        let perp_y = cos_a;

        let half = self.length / 2.0;
        let x0 = self.x - cos_a * half;
        let y0 = self.y - sin_a * half;
        let x1 = self.x + cos_a * half;
        let y1 = self.y + sin_a * half;
        let cx = self.x + perp_x * self.curve;
        let cy = self.y + perp_y * self.curve;

        // Fixed tessellation keeps path construction predictable in the hot loop.
        const N: usize = 10;
        let mut pb = PathBuilder::new();
        let mut first = true;

        // Forward pass: left edge
        for i in 0..=N {
            let t = i as f32 / N as f32;
            let bx = bezier(x0, cx, x1, t);
            let by = bezier(y0, cy, y1, t);
            let taper = taper_at(t, self.taper_start, self.taper_end);
            let half_w = self.width * taper * 0.5;
            let (tx, ty) = bezier_tangent(x0, cx, x1, y0, cy, y1, t);
            let lx = bx - ty * half_w;
            let ly = by + tx * half_w;
            if first {
                pb.move_to(lx, ly);
                first = false;
            } else {
                pb.line_to(lx, ly);
            }
        }
        // Backward pass: right edge
        for i in (0..=N).rev() {
            let t = i as f32 / N as f32;
            let bx = bezier(x0, cx, x1, t);
            let by = bezier(y0, cy, y1, t);
            let taper = taper_at(t, self.taper_start, self.taper_end);
            let half_w = self.width * taper * 0.5;
            let (tx, ty) = bezier_tangent(x0, cx, x1, y0, cy, y1, t);
            let rx = bx + ty * half_w;
            let ry = by - tx * half_w;
            pb.line_to(rx, ry);
        }
        pb.close();
        pb.finish()
    }

    /// Render onto an owned Pixmap (used when committing accepted strokes).
    pub fn render(&self, pixmap: &mut Pixmap) {
        if let Some(path) = self.build_path() {
            let mut paint = Paint::default();
            paint.set_color(Color::from_rgba8(self.r, self.g, self.b, self.alpha));
            paint.anti_alias = true;
            paint.blend_mode = BlendMode::SourceOver;
            pixmap.fill_path(
                &path,
                &paint,
                FillRule::Winding,
                Transform::identity(),
                None,
            );
        }
    }

    /// Render onto a borrowed PixmapMut (used by measure_improvement via the
    /// thread-local tile buffer -- zero allocation).
    pub fn render_to_mut(&self, pixmap: &mut tiny_skia::PixmapMut) {
        if let Some(path) = self.build_path() {
            let mut paint = Paint::default();
            paint.set_color(Color::from_rgba8(self.r, self.g, self.b, self.alpha));
            paint.anti_alias = true;
            paint.blend_mode = BlendMode::SourceOver;
            pixmap.fill_path(
                &path,
                &paint,
                FillRule::Winding,
                Transform::identity(),
                None,
            );
        }
    }
}

#[inline(always)]
fn bezier(p0: f32, pc: f32, p1: f32, t: f32) -> f32 {
    let it = 1.0 - t;
    it * it * p0 + 2.0 * it * t * pc + t * t * p1
}

#[inline(always)]
fn bezier_tangent(x0: f32, cx: f32, x1: f32, y0: f32, cy: f32, y1: f32, t: f32) -> (f32, f32) {
    let it = 1.0 - t;
    let dx = 2.0 * (it * (cx - x0) + t * (x1 - cx));
    let dy = 2.0 * (it * (cy - y0) + t * (y1 - cy));
    let len = (dx * dx + dy * dy).sqrt().max(1e-6);
    (dx / len, dy / len)
}

#[inline(always)]
fn taper_at(t: f32, start: f32, end: f32) -> f32 {
    if t < 0.5 {
        let s = t * 2.0;
        start + (1.0 - start) * smooth(s)
    } else {
        let s = (t - 0.5) * 2.0;
        1.0 - (1.0 - end) * smooth(s)
    }
}

#[inline(always)]
fn smooth(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}
