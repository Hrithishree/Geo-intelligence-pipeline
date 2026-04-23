import os
import cv2
import argparse
import numpy as np
import torch

from scripts.config import CONFIG
from scripts.predictor import load_model, predict
from scripts.postprocess import clean_mask, get_polygons
from scripts.visualize import draw_overlay


# =========================
# MAIN PIPELINE
# =========================
def main(args):

    # -------------------------
    # LOAD CONFIG PATHS
    # -------------------------
    model_path = CONFIG["model"]
    input_path = CONFIG["inference_input"]
    output_path = CONFIG["outputs"]

    os.makedirs(output_path + "/masks", exist_ok=True)
    os.makedirs(output_path + "/overlays", exist_ok=True)
    os.makedirs(output_path + "/polygons", exist_ok=True)

    print("Loading model...")
    model = load_model(model_path)

    # -------------------------
    # INPUT FILES
    # -------------------------
    files = sorted(os.listdir(input_path))

    if len(files) == 0:
        print("No input images found ❌")
        return

    print(f"Found {len(files)} images for inference")

    # -------------------------
    # INFERENCE LOOP
    # -------------------------
    for fname in files:

        img_path = os.path.join(input_path, fname)

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # -------------------------
        # PREDICTION
        # -------------------------
        mask = predict(model, img_rgb, args)

        mask = clean_mask(mask)

        # resize mask to original image size (SAFE FIX)
        h, w = img_rgb.shape[:2]
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

        polygons = get_polygons(mask)

        # -------------------------
        # VISUALIZATION
        # -------------------------
        overlay = draw_overlay(img_rgb, mask, polygons)

        # -------------------------
        # SAVE OUTPUTS
        # -------------------------
        name = os.path.splitext(fname)[0]

        cv2.imwrite(f"{output_path}/masks/{name}.png", mask * 255)
        cv2.imwrite(
            f"{output_path}/overlays/{name}.png",
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        )

        # save polygons (simple format)
        np.save(f"{output_path}/polygons/{name}.npy", polygons)

    print("Inference completed successfully ✅")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # only parameters that should remain flexible
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.35)

    args = parser.parse_args()

    main(args)