
import numpy as np

def get_neighbors(skel, y, x):
    pts = []
    for dy in [-1,0,1]:
        for dx in [-1,0,1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx = y+dy, x+dx
            if 0 <= ny < skel.shape[0] and 0 <= nx < skel.shape[1]:
                if skel[ny, nx]:
                    pts.append((ny, nx))
    return pts

def cluster(points, d):
    out = []
    for p in points:
        if all(np.hypot(p[0]-q[0], p[1]-q[1]) > d for q in out):
            out.append(p)
    return out

def detect_nodes(skel):
    endpoints = []
    junctions = []

    for y in range(skel.shape[0]):
        for x in range(skel.shape[1]):
            if skel[y, x]:
                n = len(get_neighbors(skel, y, x))
                if n == 1:
                    endpoints.append((y, x))
                elif n >= 3:
                    junctions.append((y, x))

    endpoints = cluster(endpoints, 10)
    junctions = cluster(junctions, 15)

    return endpoints, junctions
