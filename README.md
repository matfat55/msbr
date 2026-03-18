# MSBR (Stroke-Based Renderer)

MSBR turns a photo into a painting made of procedural brush strokes.

It uses a multi-pass pipeline (coarse -> medium -> fine -> top-up), with
importance-driven sampling, per-stroke local refinement, and convergence-aware
early exits.

## Roadmap
1. Artistic Styles
2. Batch
2. GUI

## Highlights

- Adaptive multi-pass with per-pass stroke budgets
- Importance-map sampling
- Batch generation + parallel refinement
- Deterministic with `--seed`
- Live preview and timelapse export
- Output metadata

## Build

```bash
cargo build --release
```

## Quick Start

```bash
./target/release/msbr --input images/photo.jpg
```

Output is written under `images/`:

- `--output painting.png` -> `images/painting.png`

## Common Commands

Basic:

```bash
./target/release/msbr --input images/photo.jpg --output painting.png
```

Headless

```bash
./target/release/msbr --input images/photo.jpg --no-preview
```

Timelapse frames:

```bash
./target/release/msbr \
  --input images/photo.jpg \
  --strokes 15000 \
  --frame-every 50 \
  --frame-schedule frontloaded \
  --frame-curve 2.5 \
  --frame-dir frames
```

## CLI Options

| Flag | Default | Description |
|---|---|---|
| `--input`, `-i` | required | Input image (PNG or JPEG) |
| `--output`, `-o` | `output.png` | Final output filename (saved under `images/`) |
| `--strokes`, `-s` | `5000` | Target stroke count |
| `--max-size` | `0` | Resize input so long edge <= this value (0 = no resize) |
| `--blur-sigma` | `1.5` | Gaussian blur sigma before gradient computation |
| `--canny-low` | `0.05` | Low Canny threshold as fraction of max gradient magnitude |
| `--canny-high` | `0.15` | High Canny threshold as fraction of max gradient magnitude |
| `--display-every` | `20` | Live preview refresh interval (accepted strokes) |
| `--frame-every` | `0` | Save timelapse frame every N accepted strokes (0 = off) |
| `--frame-schedule` | `uniform` | `uniform` or `frontloaded` |
| `--frame-curve` | `2.0` | Frontload strength (>1.0 saves more early frames) |
| `--frame-dir` | `frames` | Frame subdirectory under `images/` |
| `--seed` | `67` | RNG seed |
| `--no-preview` | `false` | Disable preview window |

## How It Works

### 1. Analysis Stage

- Gradient field from Sobel on Gaussian-blurred grayscale
- Canny edge map (non-max suppression + hysteresis)
- Complexity map from local luminance variance
- Error map between target and current canvas
- Importance map built with Walker alias sampling

Sampling weight combines:

- Error (where canvas mismatches target)
- Edge strength (contours)
- Local complexity (detail regions)
- Density suppression (avoid overpainting the same area)

### 2. Multi-Pass Rendering

| Pass | Typical role |
|---|---|
| Coarse | Large strokes for broad color blocks |
| Medium | Mid structure and color detail |
| Fine | Detail |
| Top-up | Repeated shortfall until target or convergence |

Each pass can exit early when recent acceptance drops below a threshold.

### 3. Batch Workflow

1. Generate a candidate batch from the importance map
2. Refine each stroke locally (position, angle, length, width)
3. Commit accepted strokes in descending improvement order
4. Update the error map only in affected regions


## Image Look Tuning

- Smoother, painting like look:
  `--blur-sigma 2.2 --canny-low 0.06 --canny-high 0.18`
  Higher blur favors broad stroke flow.

- Sharper edges and emphasis on outlines:
  `--blur-sigma 1.1 --canny-low 0.03 --canny-high 0.09`
  Detects more edges due to lower thresholds

- Soft, impressionistic rendering:
  `--blur-sigma 2.5 --canny-low 0.08 --canny-high 0.22 --strokes 12000`

- Detail-heavy rendering:
  `--blur-sigma 1.3 --canny-low 0.03 --canny-high 0.10 --strokes 30000`
  Drives more refinement into small structures.

Practical rules of thumb:

- `--blur-sigma`: lower: more texture/noise response, higher: smoother brush flow.
- `--canny-low` / `--canny-high`: lower values lead to stronger outline-following behavior.
- `--strokes`: self explanatory, biggest impact on quality
- `--max-size`: large speed impact, lower if laggy. 

## Timelapse Video

Convert saved frames into MP4:

```bash
ffmpeg -framerate 30 \
  -i images/frames/frame_%05d.png \
  -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
  -c:v libx264 \
  -pix_fmt yuv420p \
  timelapse.mp4
```

`pad=ceil(iw/2)*2:ceil(ih/2)*2` ensures even dimensions for `yuv420p`.

## Project Layout

```text
src/
  main.rs       # CLI + render orchestration
  analysis.rs   # gradient, edges, complexity, importance, error map
  stroke.rs     # stroke geometry + tiny-skia rendering
  pipeline.rs   # generation, refinement, commit path
  canvas.rs     # canvas state, preview buffer, PNG metadata export
```

## Output Metadata

Final PNGs embed render metadata as `tEXt` chunks, including:

- Input and canvas dimensions
- Requested/accepted/attempted strokes
- Pass-level acceptance stats
- Core parameters (`blur`, `canny`, `seed`)
- Render time
