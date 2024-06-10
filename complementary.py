#%%
import os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
# import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import os
import subprocess
#%%
def edge_supression(mask):
    h,w=mask.shape
    out = np.zeros_like(mask)
    dh,dw = np.round(np.array(mask.shape)/110 ).astype(int) 
    out[dh:h-dh,dw:w-dw] = mask[dh:h-dh,dw:w-dw]
    return out

def sam_seg_crack_comp(source):
    model_type = "vit_h"
    device = "cuda"
    
    # img = cv2.imread(source)
    img = np.asarray(Image.open(source))
    img_area = np.prod(img.shape[:2])
    sam = sam_model_registry[model_type](checkpoint="sam_vit_h_4b8939.pth")
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(img)
    
    filtered_masks = filter(lambda  mask:  (mask["area"]/ img_area) > 0.15, masks) # 
    
    crack = np.ones_like(masks[0]["segmentation"], dtype=bool)
    for file in os.listdir(r"tmp"):
        if "mask_" in file: subprocess.run(f"del tmp\\{file}",shell=1)
    for i, mask in enumerate(filtered_masks):
        crack[mask["segmentation"]]=False
        cv2.imwrite(f"tmp\mask_{i}.jpg",mask["segmentation"].astype(int)*255)

    crack = edge_supression(crack).astype(int)*255
    cv2.imwrite(r"tmp\sam_comp_result.jpg",crack)
    
    return crack
#%%
if __name__=="__main__":
    source = r"data\crack_dataset_cleaned\混凝土桥梁裂缝optic_disc_seg\JPEGImages\N0019.jpg"
    crack = sam_seg_crack_comp(source)
    
    
# %%
