#%%
import os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
# import torch
import cv2
import numpy as np
from pathlib import Path
# import matplotlib.pyplot as plt

#%%
def edge_supression(mask):
    h,w=mask.shape
    out = np.zeros_like(mask)
    dh,dw = np.round(np.array(mask.shape)/110 ).astype(int) 
    out[dh:h-dh,dw:w-dw] = mask[dh:h-dh,dw:w-dw]
    return out

def sam_seg_crack_area(img):
    model_type = "vit_h"
    device = "cuda"
    
    img_area = np.prod(img.shape[:2])
    sam = sam_model_registry[model_type](checkpoint=str(sam_checkpoint))
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(img)
    
    filtered_masks = filter(lambda  mask: (mask["area"]/ img_area) > 0.15, masks)
    
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
    
    