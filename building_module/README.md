# 🏗️ Building Extraction Module

This module performs **automatic building footprint extraction** from satellite imagery using a deep learning segmentation model.

---

## 🚀 Features
- Semantic segmentation using DeepLabV3+
- Binary mask generation
- Polygon extraction using contours
- GeoJSON export for GIS applications

---

## 📂 Project Structure


building_module/
├── inference/
│ └── run_inference.py
├── scripts/
│ └── export_geojson.py
├── demo/
│ ├── sample1.png
│ ├── sample2.png
│ ├── sample3.png
│ └── buildings.geojson
├── README.md
├── requirements.txt


---

## ▶️ How to Run

```bash
python inference/run_inference.py \
  --input dataset/images \
  --output outputs \
  --model model.pth

Then generate GeoJSON:

python scripts/export_geojson.py
📊 Sample Results
Overlay Output

Extracted GeoJSON

See: demo/buildings.geojson

🧠 Approach
Input satellite image
Deep learning segmentation
Binary mask generation
Contour detection
Polygon conversion
GeoJSON export
⚠️ Notes
Only sample outputs are included (full dataset not uploaded)
Model weights are excluded due to size constraints
🔥 Future Work
Multi-class segmentation (roads, water, roofs)
Real-time pipeline integration
Edge deployment (Jetson / Raspberry Pi)
