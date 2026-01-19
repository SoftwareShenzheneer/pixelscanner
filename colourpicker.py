#!/usr/bin/env python3

import subprocess
import tempfile
import os

import numpy as np
import cv2


def select_region():
    out = subprocess.check_output(["slurp"], text=True).strip()
    pos, size = out.split()
    x, y = map(int, pos.split(","))
    w, h = map(int, size.split("x"))
    return x, y, w, h


def capture_region(x, y, w, h) -> np.ndarray:
    geometry = f"{x},{y} {w}x{h}"

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name

    try:
        subprocess.check_call(["grim", "-g", geometry, path])

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Failed to read screenshot")

        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    finally:
        os.unlink(path)


def pick_color(rgb: np.ndarray):
    # average pixels (handles antialiasing / subpixel junk)
    r, g, b = rgb.reshape(-1, 3).mean(axis=0)
    r, g, b = int(r), int(g), int(b)

    hsv = cv2.cvtColor(
        np.array([[[r, g, b]]], dtype=np.uint8),
        cv2.COLOR_RGB2HSV,
    )[0, 0]

    h, s, v = (int(x) for x in hsv)
    hex_color = f"#{r:02X}{g:02X}{b:02X}"

    return (r, g, b), (h, s, v), hex_color


def main():
    print("Click to pick a color…")
    x, y, w, h = select_region()

    # force tiny region for pipette behavior
    w = max(1, min(w, 3))
    h = max(1, min(h, 3))

    rgb = capture_region(x, y, w, h)
    rgb_val, hsv_val, hex_val = pick_color(rgb)

    print("\nPicked color:")
    print(f"RGB: {rgb_val}")
    print(f"HSV: {hsv_val}")
    print(f"HEX: {hex_val}")


if __name__ == "__main__":
    main()

