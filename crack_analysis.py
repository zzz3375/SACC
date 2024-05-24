import os
import sys
# sys.path.append("../")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def edge_supression(mask, e=8):
    h,w=mask.shape
    out = np.zeros_like(mask)
    out[e:h-e,e:w-e] = mask[e:h-e,e:w-e]
    return out

def sam_quantify_crack(img):
    model_type = "vit_h"
    device = "cuda"
    
    sam = sam_model_registry[model_type](checkpoint=str(sam_checkpoint))
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(img)
    filtered_masks = filter(lambda  mask: mask["area"]>70_000, masks)
    
    crack = np.ones_like(masks[0]["segmentation"], dtype=bool)
    for mask in filtered_masks:
        crack[mask["segmentation"]]=False
    crack = edge_supression(crack)
    
    return crack.astype(int)*255

if __name__=="__main__":
    data_path = Path("./data/concrete_crack");data_path.mkdir(parents=1,exist_ok=1)
    out_path = Path("./out/crack_mask");out_path.mkdir(parents=1,exist_ok=1)
    sam_checkpoint = Path("./sam_vit_h_4b8939.pth")    
    for im_name in os.listdir(data_path):
        
        img = cv2.imread(str(data_path/im_name))
        crack = sam_quantify_crack(img)
        cv2.imwrite(str(out_path/im_name), crack)   
        # del mask_generator
        # del masks
        # torch.cuda.empty_cache()
    
    