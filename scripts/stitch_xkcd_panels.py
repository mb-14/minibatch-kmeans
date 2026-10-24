"""Crop whitespace and stack chart.xkcd panel PNGs into the README figures."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "reports" / "figures" / "xkcd"


def crop(image: Image.Image, pad: int = 12) -> Image.Image:
    background = Image.new(image.mode, image.size, "white")
    diff = ImageChops.difference(image, background)
    bbox = diff.getbbox()
    if bbox is None:
        return image
    left, top, right, bottom = bbox
    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(image.width, right + pad)
    bottom = min(image.height, bottom + pad)
    return image.crop((left, top, right, bottom))


def stitch(names: list[str], dest: str) -> None:
    panels = [crop(Image.open(SRC / name).convert("RGB")) for name in names]
    width = max(panel.width for panel in panels)
    gap = 16
    height = sum(panel.height for panel in panels) + gap * (len(panels) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for panel in panels:
        canvas.paste(panel, ((width - panel.width) // 2, y))
        y += panel.height + gap
    canvas.save(SRC / dest, optimize=True)


def main() -> None:
    for name in [
        "panel_quality_ari.png",
        "panel_quality_nmi.png",
        "panel_wall_vs_n.png",
        "panel_mem_bars.png",
    ]:
        image = crop(Image.open(SRC / name).convert("RGB"))
        image.save(SRC / name.removeprefix("panel_"), optimize=True)
        (SRC / name).unlink()
    for panel in SRC.glob("panel_*.png"):
        panel.unlink()
    manifest = SRC / "manifest.json"
    if manifest.exists():
        manifest.unlink()


if __name__ == "__main__":
    main()
