import os
import cv2
import json
import numpy as np
from shapely.geometry import Polygon, mapping

MASK_DIR = "/content/drive/MyDrive/geo-intelligence-pipeline/01_Building_Extraction/outputs/masks"
OUT_FILE = "/content/drive/MyDrive/geo-intelligence-pipeline/01_Building_Extraction/outputs/buildings.geojson"

features = []
def mask_to_polygons(mask):

    mask = (mask > 127).astype(np.uint8)

    # IMPORTANT: clean noise
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polys = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < 200:   # increase threshold (removes noise blobs)
            continue

        epsilon = 0.005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) < 3:
            continue

        coords = [(int(p[0][0]), int(p[0][1])) for p in approx]

        try:
            poly = Polygon(coords)

            # IMPORTANT FIX: buffer(0) fixes invalid polygons
            poly = poly.buffer(0)

            if poly.is_valid and poly.area > 50:
                polys.append(poly)

        except:
            continue

    return polys
files = sorted(os.listdir(MASK_DIR))

total_polys = 0

for f in files:
    path = os.path.join(MASK_DIR, f)

    mask = cv2.imread(path, 0)
    if mask is None:
        continue

    polys = mask_to_polygons(mask)

    total_polys += len(polys)

    for poly in polys:
        features.append({
            "type": "Feature",
            "geometry": mapping(poly),
            "properties": {
                "image": f,
                "area": float(poly.area)
            }
        })

geojson = {
    "type": "FeatureCollection",
    "features": features
}

with open(OUT_FILE, "w") as f:
    json.dump(geojson, f)

print("GeoJSON created ✅")
print("Total polygons:", total_polys)
