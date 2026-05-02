
DEVICE = "cuda"

NUM_CLASSES = 1

IMAGE_SIZE = 512
BATCH_SIZE = 4
NUM_WORKERS = 2

ENCODER_NAME = "resnet34"
ENCODER_WEIGHTS = "imagenet"

LR = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 20

CE_WEIGHT = 0.0
DICE_WEIGHT = 1.0

MODEL_SAVE_PATH = "road_best_model.pth"
EXPERIMENT_NAME = "road_unetpp_resnet34"

IMAGE_DIR = "/content/drive/MyDrive/Geo-Intelligence-Pipeline-Outputs/02_Road_Extraction/dataset/clean_road/images"
MASK_DIR = "/content/drive/MyDrive/Geo-Intelligence-Pipeline-Outputs/02_Road_Extraction/dataset/clean_road/masks"
