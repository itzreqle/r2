"""Render R2's app icon (glowing orb on black squircle) to a 1024 px PNG."""

import sys

import matplotlib.pyplot as plt
import numpy as np

N = 1024
out = sys.argv[1] if len(sys.argv) > 1 else "icon_1024.png"
y, x = np.mgrid[0:N, 0:N].astype(np.float32) / N - 0.5

# macOS-style squircle mask with transparent margin
m = 0.40
squircle = (np.abs(x / m) ** 5 + np.abs(y / m) ** 5) <= 1.0

img = np.zeros((N, N, 3), np.float32)
r = np.sqrt(x**2 + y**2)
R = 0.27

# halo
for col, k, a in [((0.1, 0.75, 1.0), 0.20, 0.55), ((0.65, 0.3, 1.0), 0.13, 0.5)]:
    img += a * np.array(col) * np.exp(-((r / k) ** 2))[..., None] * (r > R * 0.7)[..., None]

# blobs inside sphere
blobs = [((-0.07, -0.05), 0.15, (0.1, 0.8, 1.0)), ((0.08, 0.02), 0.14, (1.0, 0.2, 0.6)),
         ((0.0, 0.09), 0.13, (0.55, 0.3, 1.0)), ((-0.02, -0.11), 0.10, (0.1, 1.0, 0.8))]
inside = np.zeros_like(img)
for (cx, cy), s, col in blobs:
    inside += np.array(col) * np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / s**2))[..., None]
sphere = (r <= R)[..., None]
body = 0.04 + inside * 0.9
img = np.where(sphere, body, img)

# specular highlight + rim
hl = np.exp(-(((x + 0.08) / 0.12) ** 2 + ((y + 0.15) / 0.06) ** 2)) * (r <= R)
img += 0.35 * hl[..., None]
rim = np.exp(-(((r - R) / 0.004) ** 2))
img += 0.25 * rim[..., None]

img = np.clip(img, 0, 1)
alpha = squircle.astype(np.float32)
rgba = np.dstack([img * alpha[..., None], alpha])
plt.imsave(out, rgba)
print("icon:", out)
