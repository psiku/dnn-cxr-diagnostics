import cv2
import numpy as np
from src.constants import DISPLAY_SIZE


def generate_heatmap_overlay(img_pil, cam_map, target_size=DISPLAY_SIZE):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    img_resized = np.array(img_pil.resize(target_size))
    heatmap_resized = cv2.resize(heatmap, target_size)

    overlay = cv2.addWeighted(img_resized, 0.6, heatmap_resized, 0.4, 0)
    return img_resized, overlay