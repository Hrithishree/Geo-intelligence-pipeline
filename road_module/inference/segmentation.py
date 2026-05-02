import torch

def run_segmentation(model, tensor):
    with torch.no_grad():
        pred = model(tensor)
        pred = torch.sigmoid(pred)[0,0].cpu().numpy()
    return pred