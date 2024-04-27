import os
import sys
# sys.path.append("../")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
import cv2
import numpy as np
from pathlib import Path


data_path = Path("./concrete_crack");data_path.mkdir(parents=1,exist_ok=1)
out_path = Path("./crack_mask");out_path.mkdir(parents=1,exist_ok=1)
sam_checkpoint = Path("./sam_vit_h_4b8939.pth")

if __name__=="__main__":
    torch.cuda.empty_cache()  
    
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=str(sam_checkpoint))
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)
    def area_criteria(mask):
        return mask["area"]>50_000

    torch.cuda.empty_cache() 
    # os.mkdir("../crack_mask")
    for im_name in os.listdir(data_path):
        mask_generator = SamAutomaticMaskGenerator(sam)
        img = cv2.imread(str(data_path/im_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(img)
        # sorted_mask = sorted(masks,key=(lambda x: x['area']),reverse=True)
        filtered_masks = filter(area_criteria, masks)
        crack = np.zeros_like(masks[0]["segmentation"], dtype=bool)
        
        for mask in filtered_masks:
            crack = np.bitwise_or( crack, mask["segmentation"]) 
        crack = np.bitwise_not(crack).astype(int)*255
        cv2.imwrite(str(out_path/im_name), crack)   
        del mask_generator
        torch.cuda.empty_cache()     
    