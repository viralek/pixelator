import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import cKDTree
import os
from datetime import datetime

# ----------- CONFIGURATION ----------- #

CAMERA_ID = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

PIXEL_SIZE_DEFAULT = 10
PIXEL_SIZE_MIN = 2
PIXEL_SIZE_MAX = 50

COLOR_LEVELS_DEFAULT = 4
COLOR_LEVELS_MIN = 2
COLOR_LEVELS_MAX = 64

INFO_BAR_HEIGHT = 30

FILTER_COUNT = 10

# Background settings
USE_CUSTOM_BACKGROUND = True
BGCOLOR = (151, 169, 189)  # #97a9bd in BGR

# Palette settings
USE_PALETTE = True
PALETTE_COLORS = ["#02a8fe", "#f4441d", "#008c23", "#3c2ebf", "#f0332f", "#bb4cff", "#6c4317", "#ff92eb", "#ffffff", "#000000"]

# ----------- END CONFIGURATION ----------- #

# Convert hex palette to RGB
palette_rgb = np.array([tuple(int(color[i:i+2], 16) for i in (1, 3, 5)) for color in PALETTE_COLORS])
palette_tree = cKDTree(palette_rgb)

# Create image export folder
os.makedirs("imgexp", exist_ok=True)

# Inicializácia kamery
cap = cv2.VideoCapture(CAMERA_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# Mediapipe pre segmentáciu osôb
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmenter = mp_selfie_segmentation.SelfieSegmentation()

# Premenné
pixel_size = PIXEL_SIZE_DEFAULT
color_levels = COLOR_LEVELS_DEFAULT
color_filter = 1

# Kvantizácia farieb na 8-bit efekt
def quantize_color(img, levels):
    img = np.floor(img / (256 / levels)) * (256 / levels)
    return img.astype(np.uint8)

# Fast palette mapping using KD-tree
def apply_palette_fast(img, tree, palette):
    flat_img = img.reshape(-1, 3)
    _, idx = tree.query(flat_img)
    return palette[idx].reshape(img.shape).astype(np.uint8)

# Aplikácia farebných filtrov
def apply_color_filter(img, filter_id):
    filters = {
        1: lambda x: x,
        2: lambda x: x[..., [1, 2, 0]],
        3: lambda x: x[..., [2, 0, 1]],
        4: lambda x: 255 - x,
        5: lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2HSV),
        6: lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2LAB),
        7: lambda x: cv2.applyColorMap(x, cv2.COLORMAP_OCEAN),
        8: lambda x: cv2.applyColorMap(x, cv2.COLORMAP_HOT),
        9: lambda x: cv2.applyColorMap(x, cv2.COLORMAP_COOL),
        10: lambda x: cv2.applyColorMap(x, cv2.COLORMAP_JET),
    }
    return filters.get(filter_id, lambda x: x)(img)

cv2.namedWindow('8-bit Pixelated People', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('8-bit Pixelated People', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = segmenter.process(img_rgb)

    mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255

    small_frame = cv2.resize(frame, (FRAME_WIDTH // pixel_size, FRAME_HEIGHT // pixel_size), interpolation=cv2.INTER_LINEAR)
    pixelated_frame = cv2.resize(small_frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_NEAREST)

    pixelated_frame = quantize_color(pixelated_frame, color_levels)
    pixelated_frame = apply_color_filter(pixelated_frame, color_filter)

    if USE_PALETTE:
        pixelated_frame = apply_palette_fast(pixelated_frame, palette_tree, palette_rgb)

    bg_frame = np.full(frame.shape, BGCOLOR, dtype=np.uint8) if USE_CUSTOM_BACKGROUND else frame

    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    final_frame = np.where(mask_3ch == 255, pixelated_frame, bg_frame)

    info_bar = np.zeros((INFO_BAR_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    info_text = f'Pixel size [W/S]: {pixel_size} | Color levels [E/D]: {color_levels} | Color filter [R/F]: {color_filter} | Palette: {"ON" if USE_PALETTE else "OFF"} | BG: {"Custom" if USE_CUSTOM_BACKGROUND else "Natural"} | Press Q to quit'
    cv2.putText(info_bar, info_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    final_frame = np.vstack([final_frame, info_bar])

    cv2.imshow('8-bit Pixelated People', final_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('a'):
        filename = datetime.now().strftime('imgexp/capture_%Y%m%d_%H%M%S.png')
        cv2.imwrite(filename, final_frame)
    elif key == ord('w'):
        pixel_size = min(pixel_size + 1, PIXEL_SIZE_MAX)
    elif key == ord('s'):
        pixel_size = max(pixel_size - 1, PIXEL_SIZE_MIN)
    elif key == ord('e'):
        color_levels = min(color_levels + 1, COLOR_LEVELS_MAX)
    elif key == ord('d'):
        color_levels = max(color_levels - 1, COLOR_LEVELS_MIN)
    elif key == ord('r'):
        color_filter = (color_filter % FILTER_COUNT) + 1
    elif key == ord('f'):
        color_filter = FILTER_COUNT if color_filter == 1 else color_filter - 1

cap.release()
cv2.destroyAllWindows()
