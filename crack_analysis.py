import os
import sys
# sys.path.append("../")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


data_path = Path("./data/concrete_crack");data_path.mkdir(parents=1,exist_ok=1)
out_path = Path("./out/crack_mask");out_path.mkdir(parents=1,exist_ok=1)
sam_checkpoint = Path("./sam_vit_h_4b8939.pth")

def edge_supression(mask, e=8):
    h,w=mask.shape
    out = np.zeros_like(mask)
    out[e:h-e,e:w-e] = mask[e:h-e,e:w-e]
    return out


if __name__=="__main__":
    
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=str(sam_checkpoint))
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    def area_criteria(mask):
        return mask["area"]>50_000
    for im_name in os.listdir(data_path):
        
        img = cv2.imread(str(data_path/im_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(img)
        # sorted_mask = sorted(masks,key=(lambda x: x['area']),reverse=True)
        filtered_masks = filter(area_criteria, masks)
        crack = np.zeros_like(masks[0]["segmentation"], dtype=bool)
        
        for mask in filtered_masks:
            crack = np.bitwise_or( crack, mask["segmentation"]) 
        crack = np.bitwise_not(crack).astype(int)*255
        crack = edge_supression(crack)
        cv2.imwrite(str(out_path/im_name), crack)   
        # del mask_generator
        # del masks
        # torch.cuda.empty_cache()
    
    