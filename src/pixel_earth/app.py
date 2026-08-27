"""Gradio shell around :mod:`pixel_earth.segment`.

Piece 1: upload an image, tune the threshold pipeline, see the mask overlay
next to the transparent cutout. The overlay is the point -- it tells you when
the mask is wrong before you trust the crop.
"""

from __future__ import annotations

import numpy as np
import gradio as gr

from pixel_earth.segment import cutout, overlay, segment

_EMPTY_REPORT = "No object found. Lower the threshold, or turn off *auto threshold*."


def run(
    image: np.ndarray | None,
    auto_threshold: bool,
    threshold: float,
    blur_sigma: float,
    edge_adjust: float,
    fill_holes: bool,
    keep_largest: bool,
    pad: float,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """Segment ``image`` and return (overlay, cutout, markdown report)."""
    if image is None:
        return None, None, "Upload an image."

    result = segment(
        image,
        threshold=None if auto_threshold else int(threshold),
        blur_sigma=float(blur_sigma),
        edge_adjust=int(edge_adjust),
        fill_holes=fill_holes,
        keep_largest=keep_largest,
        pad=int(pad),
    )

    if result.is_empty:
        return None, None, _EMPTY_REPORT

    left, top, right, bottom = result.bbox
    width, height = right - left, bottom - top
    verdict = "looks like a full disc" if result.looks_like_disc() else "not a full disc"

    report = "\n".join(
        [
            f"**threshold** {result.threshold}"
            + ("  (Otsu)" if auto_threshold else "  (manual)"),
            f"**bbox** {left},{top} → {right},{bottom}  ({width}×{height})",
            f"**coverage** {result.coverage:.1%} of frame",
            f"**fill ratio** {result.fill_ratio:.3f} — {verdict}",
        ]
    )
    return overlay(image, result), cutout(image, result), report


def build() -> gr.Blocks:
    with gr.Blocks(title="pixel-earth") as demo:
        gr.Markdown("## pixel-earth — threshold the Earth out of the background")

        with gr.Row():
            with gr.Column(scale=1):
                source = gr.Image(label="source", type="numpy", image_mode="RGB")
                auto_threshold = gr.Checkbox(value=True, label="auto threshold (Otsu)")
                threshold = gr.Slider(0, 255, value=40, step=1, label="threshold")
                blur_sigma = gr.Slider(
                    0, 8, value=1.0, step=0.1, label="pre-blur sigma (px)"
                )
                edge_adjust = gr.Slider(
                    -10, 10, value=0, step=1, label="edge adjust (px: − erode, + dilate)"
                )
                pad = gr.Slider(0, 100, value=0, step=1, label="crop padding (px)")
                fill_holes = gr.Checkbox(value=True, label="fill interior holes")
                keep_largest = gr.Checkbox(value=True, label="keep largest blob only")

            with gr.Column(scale=2):
                with gr.Row():
                    overlay_out = gr.Image(label="mask overlay", type="numpy")
                    cutout_out = gr.Image(
                        label="cutout (RGBA)",
                        type="numpy",
                        image_mode="RGBA",
                        format="png",
                    )
                report = gr.Markdown()

        inputs = [
            source,
            auto_threshold,
            threshold,
            blur_sigma,
            edge_adjust,
            fill_holes,
            keep_largest,
            pad,
        ]
        outputs = [overlay_out, cutout_out, report]

        # Re-run on any change; every op is O(pixels) so this stays interactive.
        source.change(run, inputs, outputs)
        for control in inputs[1:]:
            control.change(run, inputs, outputs)

    return demo


def main() -> None:
    build().launch()


if __name__ == "__main__":
    main()
