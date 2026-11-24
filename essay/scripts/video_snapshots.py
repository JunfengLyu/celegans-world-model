"""Extract and arrange five frames from the held-out posture-rollout MP4."""

import argparse
from pathlib import Path

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ESSAY = Path(__file__).resolve().parents[1]
VIDEO = ESSAY / "demos" / "hybrid-rollout.mp4"
OUTPUT = ESSAY / "figures" / "video-snapshots.pdf"
TIMES = (0.5, 1.0, 2.0, 3.0, 4.0)
HISTORY = 12
FPS = 16
FRAME_IDS = tuple(HISTORY - 1 + round(time * FPS) for time in TIMES)

# Interior of the three bottom-row body axes in hybrid_demo.py, before resizing.
REFERENCE_WIDTH, REFERENCE_HEIGHT = 1120, 616
BODY_AXES = ((164, 353, 393, 579), (496, 353, 725, 579), (828, 353, 1057, 579))


def read_frames(video):
    frames = []
    reader = imageio_ffmpeg.read_frames(str(video), pix_fmt="rgb24")
    try:
        metadata = next(reader)
        width, height = metadata["size"]
        if metadata["fps"] != FPS:
            raise ValueError(f"Expected {FPS} Hz, got {metadata['fps']} Hz.")
        for frame, raw in enumerate(reader):
            if frame in FRAME_IDS:
                frames.append(np.frombuffer(raw, np.uint8).reshape(height, width, 3))
            if frame == FRAME_IDS[-1]:
                break
    finally:
        reader.close()
    if len(frames) != len(FRAME_IDS):
        raise ValueError("The supplied video does not reach the last requested frame.")
    return frames


def crop_body_panels(frames):
    height, width = frames[0].shape[:2]
    boxes = [(round(x0 * width / REFERENCE_WIDTH), round(y0 * height / REFERENCE_HEIGHT),
              round(x1 * width / REFERENCE_WIDTH), round(y1 * height / REFERENCE_HEIGHT))
             for x0, y0, x1, y1 in BODY_AXES]
    panels = [[frame[y0:y1, x0:x1] for x0, y0, x1, y1 in boxes] for frame in frames]
    common_height, common_width = np.min([panel.shape[:2] for group in panels for panel in group], axis=0)
    # Crop every panel by the same union bounding box so that posture and center stay comparable.
    stack = np.stack([[panel[:common_height, :common_width] for panel in group] for group in panels])
    foreground = stack.min(axis=-1) < 172
    if np.any(foreground.sum(axis=(-2, -1)) < 50):
        raise ValueError("Every extracted cell must contain a visible body contour.")
    y, x = np.nonzero(foreground.any(axis=(0, 1)))
    pad = max(6, round(width / REFERENCE_WIDTH * 9))
    x0, x1 = max(0, x.min() - pad), min(common_width, x.max() + pad + 1)
    y0, y1 = max(0, y.min() - pad), min(common_height, y.max() + pad + 1)
    return stack[:, :, y0:y1, x0:x1]


def draw_strip(video, output):
    crops = crop_body_panels(read_frames(video))
    plt.rcParams.update({"font.family": "STIXGeneral", "font.size": 10, "pdf.fonttype": 42})
    figure, axes = plt.subplots(3, len(TIMES), figsize=(6.85, 1.78))
    figure.subplots_adjust(left=.170, right=.987, bottom=.025, top=.835, wspace=.075, hspace=.045)
    for column, (letter, time, frames) in enumerate(zip("ABCDE", TIMES, crops)):
        for axis, frame in zip(axes[:, column], frames):
            axis.imshow(frame, interpolation="none", aspect="equal")
            axis.set_axis_off()
        box = axes[0, column].get_position()
        figure.text(box.x0, .965, letter, va="top", ha="left", fontweight="normal")
        figure.text((box.x0 + box.x1) / 2, .965, f"{time:g} s", va="top", ha="center")
    for axis, label in zip(axes[:, 0], ("Observed", "GRU", "Phase-embedded\nGRU")):
        box = axis.get_position()
        figure.text(.150, (box.y0 + box.y1) / 2, label, ha="right", va="center", linespacing=0.95)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, metadata={"Title": "Posture-rollout video frames"})
    plt.close(figure)
    print(f"{output} | video frame indices {', '.join(map(str, FRAME_IDS))}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=VIDEO)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    draw_strip(args.video, args.output)


if __name__ == "__main__":
    main()
