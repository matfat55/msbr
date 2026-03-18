mod analysis;
mod canvas;
mod pipeline;
mod stroke;

use analysis::{ComplexityMap, EdgeMap, ErrorMap, GradientField, ImportanceMap};
use canvas::Canvas;
use pipeline::{commit_batch, generate_batch, refine_batch, PassConfig};

use clap::{Parser, ValueEnum};
use image::{imageops, open, Rgb};
use minifb::{Key, Scale, ScaleMode, Window, WindowOptions};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Copy, Clone, Debug, ValueEnum)]
enum FrameSchedule {
    Uniform,
    Frontloaded,
}

#[derive(Parser, Debug)]
#[command(name = "msbr")]
#[command(about = "Stroke-based rendering with multi-pass optimization pipeline")]
struct Args {
    #[arg(short, long)]
    input: String,

    #[arg(short, long, default_value = "output.png")]
    output: String,

    #[arg(short, long, default_value_t = 5000)]
    strokes: usize,

    /// Save a frame every N accepted strokes (0 = disabled)
    #[arg(short, long, default_value_t = 0)]
    frame_every: usize,

    #[arg(long, default_value = "frames")]
    frame_dir: String,

    /// Frame spacing mode: `uniform` (every N accepted strokes) or
    /// `frontloaded` (denser at the beginning, sparser later).
    #[arg(long, value_enum, default_value_t = FrameSchedule::Uniform)]
    frame_schedule: FrameSchedule,

    /// Curvature for `--frame-schedule frontloaded`.
    /// Values >1.0 bias harder toward early frames.
    #[arg(long, default_value_t = 2.0)]
    frame_curve: f32,

    #[arg(long, default_value_t = 67)]
    seed: u64,

    /// Refresh the live window every N accepted strokes
    #[arg(long, default_value_t = 20)]
    display_every: usize,

    #[arg(long, default_value_t = false)]
    no_preview: bool,

    /// Gaussian blur sigma for gradient computation (higher = smoother strokes)
    #[arg(long, default_value_t = 1.5)]
    blur_sigma: f32,

    /// Canny low threshold (fraction of max gradient magnitude)
    #[arg(long, default_value_t = 0.05)]
    canny_low: f32,

    /// Canny high threshold (fraction of max gradient magnitude)
    #[arg(long, default_value_t = 0.15)]
    canny_high: f32,

    /// Resize input so its long edge is at most this many pixels before
    /// rendering. 0 = no resize.
    #[arg(long, default_value_t = 0)]
    max_size: u32,
}

fn main() {
    let args = Args::parse();
    let images_dir = PathBuf::from("images");
    std::fs::create_dir_all(&images_dir).expect("Failed to create images directory");
    let output_file = Path::new(&args.output)
        .file_name()
        .unwrap_or_else(|| std::ffi::OsStr::new("output.png"));
    let output_path = images_dir.join(output_file);
    let output_path_str = output_path.to_string_lossy().to_string();
    let frame_dir = images_dir.join(&args.frame_dir);
    let frame_dir_str = frame_dir.to_string_lossy().to_string();

    // If frame saving is enabled, clear the frames directory at start so
    // timelapse frames are fresh for each run.
    if args.frame_every > 0 {
        if frame_dir.exists() {
            // Remove the directory and its contents. Ignore errors here to avoid
            // aborting the run just because of a stale file lock on some systems.
            if let Err(e) = std::fs::remove_dir_all(&frame_dir) {
                eprintln!(
                    "Warning: failed to clear frame dir {}: {}",
                    frame_dir_str, e
                );
            } else {
                println!("Cleared existing frames directory: {}", frame_dir_str);
            }
        }
        std::fs::create_dir_all(&frame_dir).expect("Failed to create frame directory");
    }

    let target_dyn = open(&args.input).expect("Could not open input image");
    let target_img = target_dyn.to_rgb8();
    let (orig_width, orig_height) = target_img.dimensions();
    println!("Loaded image: {}x{}", orig_width, orig_height);

    let target_img = if args.max_size > 0 {
        let long_edge = orig_width.max(orig_height);
        if long_edge > args.max_size {
            let scale = args.max_size as f32 / long_edge as f32;
            let new_w = ((orig_width as f32 * scale).round() as u32).max(1);
            let new_h = ((orig_height as f32 * scale).round() as u32).max(1);
            println!(
                "Resizing to {}x{} (--max-size {})",
                new_w, new_h, args.max_size
            );
            imageops::resize(&target_img, new_w, new_h, imageops::FilterType::Lanczos3)
        } else {
            println!(
                "Image already within --max-size {} -- no resize needed.",
                args.max_size
            );
            target_img
        }
    } else {
        target_img
    };

    let (width, height) = target_img.dimensions();

    let bg = average_color(&target_img);
    println!("Background color: ({}, {}, {})", bg[0], bg[1], bg[2]);

    // Flatten target into contiguous RGB for fast access
    let target_raw: Vec<u8> = target_img.as_raw().clone();

    let mut canvas = Canvas::new(width, height, bg);

    // Warn if stroke count is likely too low to cover the canvas.
    // Rule of thumb: you need at least 1 fine stroke per ~300 pixels.
    let image_pixels = width * height;
    let recommended_min = (image_pixels / 300) as usize;
    if args.strokes < recommended_min {
        println!(
            "Warning: --strokes {} may be too low for a {}x{} image. \
                 Recommended minimum: {} (use --strokes {} or higher for full coverage).",
            args.strokes, width, height, recommended_min, recommended_min
        );
    }
    // --- Analysis phase ---
    println!("Analyzing image...");

    let t0 = std::time::Instant::now();
    let gradient = GradientField::from_image(&target_img, args.blur_sigma);
    println!(
        "  Gradient field: {:.1}ms",
        t0.elapsed().as_secs_f64() * 1000.0
    );

    // Compute Canny thresholds from gradient magnitude statistics
    let max_mag = gradient.magnitudes.iter().cloned().fold(0.0f32, f32::max);
    let canny_low = max_mag * args.canny_low;
    let canny_high = max_mag * args.canny_high;

    let t0 = std::time::Instant::now();
    let edge_map = EdgeMap::from_gradient(&gradient, canny_low, canny_high);
    println!(
        "  Canny edges: {} edge pixels ({:.1}ms)",
        edge_map.edge_positions.len(),
        t0.elapsed().as_secs_f64() * 1000.0
    );

    let t0 = std::time::Instant::now();
    let complexity = ComplexityMap::compute(&target_img, 4);
    println!(
        "  Complexity map: {:.1}ms",
        t0.elapsed().as_secs_f64() * 1000.0
    );

    // Edge strength as a float map for importance computation
    let edge_float: Vec<f32> = edge_map
        .edges
        .iter()
        .map(|&e| if e { 1.0 } else { 0.0 })
        .collect();

    // Live preview window
    let max_display = 1200u32;
    let (dw, dh) = if width > max_display {
        let s = max_display as f32 / width as f32;
        ((width as f32 * s) as usize, (height as f32 * s) as usize)
    } else {
        (width as usize, height as usize)
    };

    let mut window: Option<Window> = None;
    if !args.no_preview {
        match Window::new(
            &format!("msbr | {} strokes | ESC/Q to stop", args.strokes),
            dw,
            dh,
            WindowOptions {
                resize: false,
                scale: Scale::X1,
                scale_mode: ScaleMode::Center,
                ..WindowOptions::default()
            },
        ) {
            Ok(w) => {
                window = Some(w);
                println!("Live preview ({}x{}). ESC/Q to stop early.", dw, dh);
            }
            Err(e) => eprintln!("Preview unavailable: {}. Running headless.", e),
        }
    }

    let mut rng = StdRng::seed_from_u64(args.seed);
    let mut total_accepted = 0usize;
    let mut total_attempted = 0usize;
    let mut coarse_accepted = 0usize;
    let mut coarse_attempted = 0usize;
    let mut medium_accepted = 0usize;
    let mut medium_attempted = 0usize;
    let mut fine_accepted = 0usize;
    let mut fine_attempted = 0usize;
    let mut topup_accepted = 0usize;
    let mut topup_attempted = 0usize;
    let mut frame_count = 0u32;
    let frame_targets = build_frame_targets(
        args.strokes,
        args.frame_every,
        args.frame_schedule,
        args.frame_curve,
    );
    let mut next_frame_idx = 0usize;
    if args.frame_every > 0 {
        println!(
            "Frame capture: {:?}, {} planned frames",
            args.frame_schedule,
            frame_targets.len()
        );
    }
    let print_every = (args.strokes / 50).max(1);
    let mut last_print_at = 0usize;
    let mut last_print_time = std::time::Instant::now();

    // Initial error map
    let mut error_map = ErrorMap::compute(canvas.pixmap.data(), &target_raw, width, height);

    let mut density = pipeline::StrokeDensity {
        data: vec![0; (width * height) as usize],
        width,
        height,
    };

    // Initial display
    if let Some(ref mut win) = window {
        canvas.update_display_cache(dw as u32, dh as u32);
        let _ = win.update_with_buffer(canvas.display_buffer(), dw, dh);
    }

    let total_start = std::time::Instant::now();
    let density_cap = (((width * height) as f32 / 400.0).sqrt().round() as u16).max(8);

    // Shared stop flag
    let stop = AtomicBool::new(false);

    macro_rules! run_pass {
        ($pass:expr, $allow_early_exit:expr) => {{
            let pass: &PassConfig = $pass;
            println!(
                "\n=== {} pass: {} strokes (batches of {}, {} refine iters) ===",
                pass.name, pass.stroke_count, pass.batch_size, pass.refine_iterations
            );

            let pass_start = std::time::Instant::now();
            let mut pass_accepted = 0usize;
            let mut pass_attempted = 0usize;
            let mut pass_batches = 0usize;
            let early_exit_window = 8usize;

            let early_exit_min_rate: f32 = match pass.name {
                "Coarse" => 0.08,
                "Medium" => 0.08,
                "Fine" => 0.02,
                _ => 0.02, // top-up
            };
            let mut recent_attempted = 0usize;
            let mut recent_accepted = 0usize;

            density.reset();
            let suppression = density.suppression_weights(density_cap);
            let mut importance = ImportanceMap::new(width, height);
            importance.rebuild(
                &error_map.data,
                &edge_float,
                &complexity.data,
                &suppression,
                pass.error_weight,
                pass.edge_weight,
                pass.complexity_weight,
            );

            while pass_accepted < pass.stroke_count && !stop.load(Ordering::Relaxed) {
                // User stop
                if let Some(ref mut win) = window {
                    if !win.is_open() || win.is_key_down(Key::Escape) || win.is_key_down(Key::Q) {
                        println!("\nStopped early by user at {} strokes.", total_accepted);
                        stop.store(true, Ordering::Relaxed);
                        break;
                    }
                }

                let mut batch = generate_batch(
                    &mut rng,
                    pass,
                    &importance,
                    &gradient,
                    &edge_map,
                    &target_raw,
                    width,
                    height,
                );
                let batch_attempted = batch.len();
                total_attempted += batch_attempted;
                pass_attempted += batch_attempted;
                recent_attempted += batch_attempted;

                refine_batch(
                    &mut batch,
                    canvas.pixmap.data(),
                    &target_raw,
                    width,
                    height,
                    pass.refine_iterations,
                    &gradient,
                );

                let accepted = commit_batch(
                    &batch,
                    &mut canvas.pixmap,
                    &target_raw,
                    &mut error_map,
                    width,
                    height,
                    &mut density,
                );
                pass_accepted += accepted;
                total_accepted += accepted;
                pass_batches += 1;
                recent_accepted += accepted;

                // Progress print
                if total_accepted >= last_print_at + print_every {
                    let now = std::time::Instant::now();
                    let secs = now.duration_since(last_print_time).as_secs_f64();
                    let rate = (total_accepted - last_print_at) as f64 / secs.max(0.001);
                    let elapsed = now.duration_since(total_start).as_secs_f64();
                    let pct = total_accepted as f32 / args.strokes as f32 * 100.0;
                    let accept_pct = pass_accepted as f32 / pass_attempted as f32 * 100.0;
                    println!(
                        "  [{:5.1}%] {:5} strokes  {:6.1}/s  elapsed {:.1}s  \
                             pass accept {:.1}%  [{}]",
                        pct, total_accepted, rate, elapsed, accept_pct, pass.name,
                    );
                    last_print_at = total_accepted;
                    last_print_time = now;
                }

                // Early exit
                if $allow_early_exit && pass_batches % early_exit_window == 0 {
                    let recent_rate = recent_accepted as f32 / recent_attempted as f32;
                    if recent_rate < early_exit_min_rate {
                        println!(
                            "  Early exit: {:.1}% accept rate over last {} batches \
                                 ({} accepted, {} short of configured target)",
                            recent_rate * 100.0,
                            early_exit_window,
                            pass_accepted,
                            pass.stroke_count.saturating_sub(pass_accepted),
                        );
                        break;
                    }
                    recent_attempted = 0;
                    recent_accepted = 0;
                }

                // Display
                if let Some(ref mut win) = window {
                    if total_accepted % args.display_every == 0 || accepted > 0 {
                        canvas.update_display_cache(dw as u32, dh as u32);
                        if win
                            .update_with_buffer(canvas.display_buffer(), dw, dh)
                            .is_err()
                        {
                            stop.store(true, Ordering::Relaxed);
                            break;
                        }
                        win.set_title(&format!(
                            "msbr | {}/{} [{}] | ESC/Q to stop",
                            total_accepted, args.strokes, pass.name,
                        ));
                    }
                }

                // Frame save
                if next_frame_idx < frame_targets.len()
                    && total_accepted >= frame_targets[next_frame_idx]
                {
                    canvas.save(&format!("{}/frame_{:05}.png", frame_dir_str, frame_count));
                    frame_count += 1;
                    // Advance through all thresholds already crossed by this batch.
                    // A single batch can accept multiple strokes, so this preserves
                    // monotonic frame numbering without duplicating saves.
                    while next_frame_idx < frame_targets.len()
                        && frame_targets[next_frame_idx] <= total_accepted
                    {
                        next_frame_idx += 1;
                    }
                }

                // Rebuild importance
                if pass_batches % 8 == 0 {
                    let suppression = density.suppression_weights(density_cap);
                    importance.rebuild(
                        &error_map.data,
                        &edge_float,
                        &complexity.data,
                        &suppression,
                        pass.error_weight,
                        pass.edge_weight,
                        pass.complexity_weight,
                    );
                }
            }
            // Accumulate per-pass stats for metadata
            match pass.name {
                "Coarse" => {
                    coarse_accepted += pass_accepted;
                    coarse_attempted += pass_attempted;
                }
                "Medium" => {
                    medium_accepted += pass_accepted;
                    medium_attempted += pass_attempted;
                }
                "Fine" => {
                    fine_accepted += pass_accepted;
                    fine_attempted += pass_attempted;
                }
                _ => {
                    topup_accepted += pass_accepted;
                    topup_attempted += pass_attempted;
                }
            }
            let elapsed = pass_start.elapsed().as_secs_f64();
            println!(
                "  {} pass done: {}/{} strokes accepted in {:.1}s ({:.0}/s, {:.1}% accept)",
                pass.name,
                pass_accepted,
                pass_attempted,
                elapsed,
                pass_accepted as f64 / elapsed.max(0.001),
                pass_accepted as f32 / pass_attempted as f32 * 100.0,
            );
        }};
    }

    run_pass!(&PassConfig::coarse(args.strokes, image_pixels), true);

    if !stop.load(Ordering::Relaxed) {
        run_pass!(&PassConfig::medium(args.strokes, image_pixels), true);
    }

    if !stop.load(Ordering::Relaxed) {
        let fine_target = args.strokes.saturating_sub(total_accepted);
        run_pass!(&PassConfig::fine(fine_target), true);
    }

    let mut topup_stalls = 0u32;
    while !stop.load(Ordering::Relaxed) {
        let shortfall = args.strokes.saturating_sub(total_accepted);
        if shortfall == 0 {
            break;
        }
        if topup_stalls >= 2 {
            println!(
                "\n  Note: canvas fully converged after top-up. \
                     {} of {} requested strokes placed \
                     ({} could not improve the image further).",
                total_accepted, args.strokes, shortfall
            );
            break;
        }
        println!(
            "\n  {} strokes short of target -- running top-up pass.",
            shortfall
        );
        let before = total_accepted;
        run_pass!(&PassConfig::topup(shortfall), true);
        // If a top-up delivered fewer than 1% of the shortfall, count it as a stall.
        let delivered = total_accepted - before;
        if delivered < (shortfall / 100).max(1) {
            topup_stalls += 1;
        } else {
            topup_stalls = 0;
        }
    }

    let still_short = args.strokes.saturating_sub(total_accepted);
    if still_short > 0 && !stop.load(Ordering::Relaxed) {
        println!(
            "\n  Note: canvas fully converged. {} of {} requested strokes placed \
                 ({} could not improve the image further).",
            total_accepted, args.strokes, still_short
        );
    }

    // Save final image immediately after generation (no need to close the preview).
    use canvas::RenderMetadata;
    let total_elapsed = total_start.elapsed().as_secs_f64();
    canvas.save_with_metadata(
        &output_path_str,
        &RenderMetadata {
            input_path: &args.input,
            orig_width,
            orig_height,
            canvas_width: width,
            canvas_height: height,
            strokes_requested: args.strokes,
            strokes_accepted: total_accepted,
            strokes_attempted: total_attempted,
            blur_sigma: args.blur_sigma,
            canny_low: args.canny_low,
            canny_high: args.canny_high,
            seed: args.seed,
            elapsed_secs: total_elapsed,
            coarse_accepted,
            coarse_attempted,
            medium_accepted,
            medium_attempted,
            fine_accepted,
            fine_attempted,
            topup_accepted,
            topup_attempted,
        },
    );

    if args.frame_every > 0 && frame_count > 0 {
        println!("\nTo create a timelapse video:");
        println!(
            "  ffmpeg -framerate 30 -i {}/frame_%05d.png \
             -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" \
             -c:v libx264 -pix_fmt yuv420p {}.mp4",
            frame_dir_str, output_path_str
        );
    }

    // Keep window open for user to inspect, but final image has already been saved.
    if let Some(ref mut win) = window {
        canvas.update_display_cache(dw as u32, dh as u32);
        let _ = win.update_with_buffer(canvas.display_buffer(), dw, dh);
        win.set_title(&format!(
            "msbr | DONE: {} strokes | close to exit",
            total_accepted
        ));
        println!("Preview shown. You can close the preview window or press ESC/Q to exit.");
        while win.is_open() && !win.is_key_down(Key::Escape) && !win.is_key_down(Key::Q) {
            let _ = win.update_with_buffer(canvas.display_buffer(), dw, dh);
        }
    }
}

fn average_color(img: &image::RgbImage) -> Rgb<u8> {
    let (w, h) = img.dimensions();
    let n = (w * h) as u64;
    let (mut r, mut g, mut b) = (0u64, 0u64, 0u64);
    for px in img.pixels() {
        r += px[0] as u64;
        g += px[1] as u64;
        b += px[2] as u64;
    }
    Rgb([(r / n) as u8, (g / n) as u8, (b / n) as u8])
}

fn build_frame_targets(
    strokes: usize,
    frame_every: usize,
    schedule: FrameSchedule,
    frame_curve: f32,
) -> Vec<usize> {
    if strokes == 0 || frame_every == 0 {
        return Vec::new();
    }

    // Keep the same rough frame budget as uniform spacing.
    let frame_count = strokes / frame_every;
    if frame_count == 0 {
        return Vec::new();
    }

    if matches!(schedule, FrameSchedule::Uniform) {
        return (1..=frame_count).map(|i| i * frame_every).collect();
    }

    let curve = frame_curve.max(1.0) as f64;
    let mut out = Vec::with_capacity(frame_count);
    let mut prev = 0usize;

    for i in 1..=frame_count {
        let progress = i as f64 / frame_count as f64;
        let raw = (strokes as f64 * progress.powf(curve)).round() as usize;
        let min_next = prev + 1;
        let remaining = frame_count - i;
        let max_next = strokes.saturating_sub(remaining);
        let target = raw.clamp(min_next, max_next);
        out.push(target);
        prev = target;
    }

    out
}
