# Building Extraction Module

## Overview
Deep learning pipeline for building segmentation from satellite imagery.

## Model
DeepLabV3+

## Pipeline
Image → Mask → Contour → Polygon → GeoJSON

## Run
python run_inference.py --input sample/ --output outputs/ --model model.pth

## Output
- Binary masks
- Overlay images
- GeoJSON polygons
