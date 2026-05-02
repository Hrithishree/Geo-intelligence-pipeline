
import sys
import os

ROOT = "/content/drive/MyDrive/Geo-Intelligence-Pipeline-Outputs/02_Road_Extraction"
sys.path.append(ROOT)

import cv2
import torch
import json
import numpy as np
from skimage.morphology import skeletonize

from scripts.model import load_model
from scripts.utils import preprocess, postprocess
from graph_pipeline.node_detection import detect_nodes
from graph_pipeline.edge_tracing import trace_edges
sys.path.append(ROOT)
IMAGE_DIR = f"{ROOT}/dataset/clean_road/images"
MODEL_PATH = f"{ROOT}/models/road_segmentation_v1/best.pth"

OUT_DIR = f"{ROOT}/outputs"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = load_model(MODEL_PATH, DEVICE)

images = [f for f in os.listdir(IMAGE_DIR)][:5]

for img_name in images:
    path = os.path.join(IMAGE_DIR, img_name)

    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    inp, H, W = preprocess(img_rgb)

    tensor = torch.tensor(inp).permute(2,0,1).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        pred = model(tensor)
        pred = torch.sigmoid(pred)[0,0].cpu().numpy()

    mask = postprocess(pred, H, W)

    skel = skeletonize(mask > 0).astype(np.uint8)

    endpoints, junctions = detect_nodes(skel)
    edges = trace_edges(skel, endpoints, junctions)

    print(f"{img_name} -> Nodes: {len(endpoints)+len(junctions)}, Edges: {len(edges)}")

print("DONE")
