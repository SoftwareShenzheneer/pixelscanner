import subprocess
import sys
from dataclasses import dataclass

import numpy as np
import cv2

@dataclass(frozen=True)
class ColorRange:
    h_min: int
    h_max: int
    s_min: int
    s_max: int
    v_min: int
    v_max: int

@dataclass
class Result:
    hit: bool
    area_ratio: float
    pixel_count: int

IRON_ORE = ColorRange(
    h_min=18, h_max=30,     # 24 ± 6
    s_min=120, s_max=255,   # strong yellow only
    v_min=120, v_max=255    # bright, avoid dark noise
)

def hue_mask(h: np.ndarray, h_min: int, h_max: int) -> np.ndarray:
    if h_min <= h_max:
        return (h >= h_min) & (h <= h_max)
    else:
        # wrap-around (e.g. red)
        return (h >= h_min) | (h <= h_max)

def select_region():
    try:
        out = subprocess.check_output(["slurp"], text=True).strip()
    except subprocess.CalledProcessError:
        return None

    pos, size = out.split()
    x, y = map(int, pos.split(","))
    w, h = map(int, size.split("x"))

    return x, y, w, h

def capture_region(x, y, w, h) -> np.ndarray:
    # Grim captures a region and outputs to stdout as PNG
    cmd = [
        "grim",
        "-g", f"{x},{y} {w}x{h}",
        "-"
    ]
    png_bytes = subprocess.check_output(cmd)
    arr = np.frombuffer(png_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # returns BGR
    return img[..., ::-1]  # convert BGR -> RGB

@dataclass
class ColorPresenceScanner:
    color_range: ColorRange
    min_pixels: int = 50
    kernel_size: int = 3

    def scan(self, rgb: np.ndarray) -> Result:
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Expected RGB image (H, W, 3)")

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        h_mask = hue_mask(h, self.color_range.h_min, self.color_range.h_max)

        mask = (
            h_mask &
            (s >= self.color_range.s_min) & (s <= self.color_range.s_max) &
            (v >= self.color_range.v_min) & (v <= self.color_range.v_max)
        )

        # Convert to uint8 for OpenCV ops
        mask = mask.astype(np.uint8) * 255

        # Morphological cleanup (noise suppression)
        if self.kernel_size > 1:
            kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        pixel_count = cv2.countNonZero(mask)
        area_ratio = pixel_count / mask.size

        return Result(
            hit=pixel_count >= self.min_pixels,
            area_ratio=area_ratio,
            pixel_count=pixel_count
        )

def main():
    region = select_region()
    if region is None:
        print("Region selection cancelled.")
        sys.exit(1)

    x, y, w, h = region
    print(f"Locked region: {x},{y} {w}x{h}")

    # Placeholder: replace with your actual capture method
    rgb = capture_region(x, y, w, h)

    scanner = ColorPresenceScanner(IRON_ORE, min_pixels=100)
    result = scanner.scan(rgb)
    if result.hit:
        print(
            f"HIT: {result.pixel_count} px "
            f"({result.area_ratio:.2%} of region)"
        )
    else:
        print(
            f"MISS: {result.pixel_count} px "
            f"({result.area_ratio:.2%} of region)"
    )
    print(result)

if __name__ == "__main__":
    main()

