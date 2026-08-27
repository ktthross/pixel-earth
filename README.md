# pixel-earth

Select the Earth out of an image and crop it to a transparent PNG.

```bash
uv sync --extra dev
uv run pixel-earth                              # interactive UI, :7860
uv run pixel-earth-fetch --last 1               # mirror a day of DSCOVR/EPIC
uv run pixel-earth-batch data/epic -r           # batch a directory
uv run pytest                                   # hermetic; -m network for live
```

## Piece 1 — pixel thresholding (done)

`luminance → gaussian blur → threshold → largest blob → fill holes → erode/dilate → bbox crop`

Pure numpy + scipy, in [segment.py](src/pixel_earth/segment.py). The UI in
[app.py](src/pixel_earth/app.py) is a thin shell over it.

Otsu picks the threshold automatically. The gaussian pre-blur is the one
convolution that earns its place here: it kills sensor noise and JPEG ringing
that would otherwise speckle the mask.

Read the **mask overlay** pane, not the cutout — the overlay dims the
background and outlines the mask, so a mask that eats the limb or clips the
night side is obvious. `fill ratio` is the numeric version of the same check: a
full disc fills π/4 ≈ 0.785 of its bounding box.

### What piece 1 handles

| case | handled by |
|---|---|
| stars, moon, caption text | keep largest blob |
| dark ocean, night-side interior | fill holes |
| sensor noise, JPEG ringing | pre-blur, then erode 1–2px |
| background not pure black | Otsu, or manual threshold |

The pre-blur costs about a pixel of radius per side (it softens the limb, which
moves the Otsu cutoff). `--edge-adjust 1` buys it back if you care.

### What it does not

A terminator that runs out to the limb is an open notch, not an enclosed hole,
so hole-filling cannot recover it and the crop clips the night side. That is a
property of thresholding, not a tuning problem — see
`test_terminator_reaching_the_limb_defeats_thresholding`. Fixing it needs a
shape prior; piece 2 at least *detects* it and flags the image for review.

## Piece 2 — batch a directory (done)

```bash
uv run pixel-earth-batch <dir> [-r] [--threshold N] [--blur SIGMA]
                               [--edge-adjust PX] [--pad PX] [--min-area FRAC]
                               [--no-fill-holes] [--all-blobs] [--no-overlays]
                               [--force] [--dry-run] [-o OUT]
```

Writes into `outputs/<run_id>/` at the repo root (gitignored):

```text
outputs/<run_id>/
    manifest.json     settings, per-image stats, which images need review
    cutouts/…​.png      transparent PNGs, input tree mirrored
    overlays/…​.png     QC views
outputs/latest -> <run_id>
```

**`run_id` is a hash of the source directory plus the settings**, not a
timestamp. So:

- rerunning the same command reuses the folder and **skips finished images** —
  an interrupted batch resumes; `--force` redoes them;
- changing any setting writes a **new** folder, so two settings sit side by side
  instead of clobbering each other.

Recursion mirrors the input tree, so `a.png` and `nested/a.png` cannot collide.
EXIF rotation is applied on load. A corrupt file is recorded as `failed` and the
batch carries on.

### Reviewing a batch

`needs_review` in the manifest, and the closing CLI summary, flag every image
whose mask is not a plausible full disc. Two independent signals, because either
alone gets fooled:

| signal | catches | blind to |
|---|---|---|
| `fill_ratio` vs π/4 | mask leaked into background | clipped crescent — a 35% terminator moves it only 4% |
| `aspect_ratio` vs 1.0 | clipped disc, terminator | mask that grew symmetrically |

`min_area` (default 0.1% of frame) is the third guard: Otsu always finds
*something*, so on an empty frame it latches onto a hot pixel. Without the floor
that becomes a 3×3 "cutout"; with it, an honest `empty`.

Real check — the Meteosat full disc in [data/development/](data/development/)
comes out at `fill_ratio 0.7864` (π/4 = 0.78540) and `aspect 1.0035`, outline
sitting on the limb.

## Piece 3 — mirror DSCOVR/EPIC (done)

```bash
uv run pixel-earth-fetch --from 2024-01-01 --to 2024-12-31 --spread 3 --dry-run
uv run pixel-earth-fetch --from 2024-01-01 --to 2024-12-31 --spread 3
uv run pixel-earth-batch data/epic -r
```

DSCOVR sits at the Earth–Sun L1 point and photographs the fully lit disc all
day, so **one UTC day of frames is one full rotation** — 9 to 22 frames
depending on era, sweeping a complete 360° of longitude.

### The politeness problem

`epic.gsfc.nasa.gov` publishes **no rate limit, no API key, and a `robots.txt`
of bare `Disallow:` with no `Crawl-delay`.** Nothing upstream will ever throttle
us, so every restraint is self-imposed and untestable against the server —
which is why [tests/test_epic.py](tests/test_epic.py) spends most of its length
on exactly these guarantees:

| mechanism | default | note |
|---|---|---|
| serialised requests | `--delay 0.5` | injectable clock, so tests never sleep |
| skip existing files | on | **zero** requests — verified by a tripwire client |
| conditional GET | `--revalidate` | `If-None-Match` → `304`, no body |
| resume | automatic | `.part` + `Range:` → `206` |
| atomic writes | always | verify length, then `os.replace` |
| retry | 5 attempts, 1→16 s | honours `Retry-After`; 404 is never retried |
| byte ceiling | `--max-bytes 10GiB` | refuses up front; overshoot bounded to one file |

`--dry-run` HEADs each missing file for an exact total and writes nothing:

```text
collection natural, format png, ceiling 10.0 GiB

  2024-01-01  13 frames, 13 missing   40.0 MiB    324deg swept
  2024-07-01  21 frames, 21 missing   60.9 MiB    343deg swept
  2024-12-31  12 frames, 12 missing   37.9 MiB    323deg swept

  46 frames, 0 already present, 138.8 MiB to download
  mirror 0 B -> 138.8 MiB  (1.4% of ceiling)
```

`deg swept` sums the gaps *between* frames, so a complete day reads
`360 × (N−1) / N` — 343 for 21 frames, not 360. Well under that means the day
has a hole in its rotation.

### Mirror, not output

Downloads reproduce the archive path verbatim, kept separate from `outputs/`:

```text
data/epic/natural/2024/06/01/png/epic_1b_20240601004554.png
data/epic/natural/2024/06/01/metadata.json     ← the API response, saved
```

The mirror is the expensive-to-acquire resource; run outputs are disposable.
Re-segmenting with new settings never touches the network. This needed **no
change to `batch.py` or `segment.py`** — `pixel-earth-batch data/epic -r` just
works, and mirrors the tree into `cutouts/`.

### Gotchas the archive will not tell you about

- **Missing dates return `[]`, not 404.** Always intersect with
  `/api/<collection>/available` (3606 dates, 2015-06-13 onward, 46 KB).
- **The archive has real holes**: 230 days (2019-06-27 → 2020-02-12, DSCOVR
  safe hold), 86 days in 2025. `--spread` snaps to the nearest available date
  and prints how far it moved, so a hole is visible instead of silent.
- **The archive directory comes from the image *name*, not the `date` field.**
  On 2024-06-01 they disagree (00:45:54 vs 00:41:06); only the name matches the
  URL.
- **Metadata is `Cache-Control: no-cache, private`** with no ETag, so we cache
  it ourselves — forever for past days, one hour for today.

### Measured on 46 frames across 3 days of 2024

| | value |
|---|---|
| `fill_ratio` | 0.7805 – 0.7861 (π/4 = 0.78540) |
| `aspect_ratio` | 0.9964 – 1.0082 |
| `needs_review` | **0 of 46** |
| disc width | 1638 – 1718 px (varies with L1 distance) |
| download | 138.8 MiB, 96 requests, 2m35s |
| segmentation | 24 s for 46 × 2048² frames |

Piece 1 needed no changes for real EPIC data. Full disc, always lit, black
space — the easy case, exactly as designed for.

## Piece 4 — cloud-free rotation (done)

```bash
uv run pixel-earth-turntable data/epic --frames 72
```

A full 360° rotation, cloud-free, built entirely from frames already mirrored
locally — no reprojection round-trip through one persistent global raster.
For each of `--frames` evenly-spaced output longitudes,
[catalog.py](src/pixel_earth/catalog.py) picks the mirrored frames whose own
sub-satellite point could plausibly see that viewpoint (metadata only, no
image decoded yet — one UTC day sweeps the whole 360°, but a given longitude
is only ever lit the same way on *some* days, since sub-satellite latitude
tracks the season), then [mosaic.py](src/pixel_earth/mosaic.py) reprojects
each candidate frame's own pixels straight into the output viewpoint's
orthographic geometry (same projection as [segment.py](src/pixel_earth/segment.py)'s
disc, via [geometry.py](src/pixel_earth/geometry.py)) and keeps, pixel by
pixel, the single least-cloudy candidate's whole `(R, G, B)` — never a
per-channel synthesis across frames, which is what darkened and discoloured
an earlier, unshipped attempt at this badly enough that it needed a
brightness-gain-matching patch afterwards.

Cloudiness is scored by brightness × whiteness² (bright + colourless = cloud,
[cloud_score.py](src/pixel_earth/cloud_score.py)) — visible light only, so it
cannot tell cloud from snow or ice. NASA also publishes a dedicated
per-pixel cloud-fraction product; `scripts/spike_cloudfraction.py` checked
whether the *quicklook* version of it (the only part this client can reach
without a separate Earthdata login) was a usable drop-in, and found it isn't:
it's a rendered figure — title, coastline overlay, a 4-class legend — at a
different resolution, not a bare disc with a continuous per-pixel value. The
classification underneath it visibly tracks real cloud structure, so it's a
plausible future upgrade; wiring it in would mean detecting and cropping the
disc out of that chrome first, which is real scope beyond "swap the score
function," so it's parked as a documented no-go for now
(`cloud_score.cloudfraction_score`).

### Measured on the full 1014-frame, 63-day mirror

| | value |
|---|---|
| `mean_suspect_fraction` (no trustworthy pixel found) | 0.49% |
| worst single frame | 0.91% |
| candidates considered per output frame | 46–48 (of `--max-candidates 48`) |
| render time | ~4s/frame at `--radius 360`, ~24s/frame at `--radius 800` |

`suspect_fraction` per frame is in `manifest.json` — deliberately not hidden
or covered up, since coverage is naturally thinner at longitudes only ever
photographed far from the equinoxes.

### What it does not do

Thin or broken cloud that isn't bright-and-white (haze, low stratus — the
Pacific is the visible example) can score as merely "less clear" rather than
"cloud," so the least-cloudy real candidate sometimes still shows soft, muted
texture there instead of a crisp cutout. Blending across more near-tied
candidates (`--blend-k`) doesn't fix this — it's a scoring blind spot, not a
selection one — so it's the same class of known limitation as snow/ice.

## Next pieces

5. **Click to select** — `gr.Image.select` gives the click point for free; use
   it to seed a flood fill, or to pick among candidate blobs.
6. **GrabCut** — for when the Earth is not a disc (a map, a globe on a desk).
7. **SAM** — click-to-mask, best quality, ~350MB of torch.

**Hough circle**, previously piece 3, is deferred indefinitely. It was the fix
for a clipped terminator, and DSCOVR never produces one.
