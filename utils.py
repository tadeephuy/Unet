import torchvision
import numpy as np

def transform_preprocess(height, width):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((height, width)),
        torchvision.transforms.ToTensor()
    ])
    return transform

def weighted_dice(outputs, targets, weight):
    return 0

def get_iou(preds, labels):
    smooth = 1e-6
    # preds (m, 1, h, w)
    preds = preds.squeeze(1) # (m, h, w)
    union = (preds | labels).float().sum((1, 2)) # (m)
    intersection = (preds & labels).float().sum((1, 2)) # (m)

    iou = (intersection + smooth) / (union + smooth) # (m)
    return iou

def segmentation_correct(preds, labels, threshold=0.85):
    iou = get_iou(preds, labels) # (m)
    binarize = np.vectorize(lambda x: 1 if x >= threshold else 0)
    n_corrects = binarize(iou.cpu()).sum()
    return n_corrects