# Road Extraction Pipeline

## Overview
This project performs:
- Road segmentation using DeepLabV3+
- Skeletonization
- Graph extraction (nodes + edges)

## How to Run
Run:
python inference/run_full_pipeline.py

## Outputs
- masks/
- skeletons/
- graphs/
- json/

## Note
Model is pre-trained and loaded from checkpoint.