# %%
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from yolo_suggestion import sam_seg_crack_by_prompt
# %%
data_dir = Path(r"data\crack_dataset_cleaned\混凝土桥梁裂缝optic_disc_seg\JPEGImages")
methods = ["GAN", "YOLOv8-Seg", "2nd Round", "Final Desicion (Proposed)"]
demo_files = ["H0003.jpg", "H0007.jpg", "H0009.jpg", "H0020.jpg", "P0018.jpg"]
i=0
for file_name in demo_files:
    sam_seg_crack_by_prompt(data_dir/file_name)
    j=0
    for method in methods:
        if method == "GAN":
            file_dir = Path(r"gan-result")/(file_name[:-4]+"_fake"+".png")
        elif method == "YOLOv8-Seg":
            file_dir = Path(r"tmp\yolo_raw_result.jpg")
        elif method == "2nd Round":
            file_dir = Path(r"tmp\SAM_prompting.jpg")
        elif method == "Final Desicion (Proposed)":
            file_dir = Path(r"tmp\accepted_result.jpg")
        
        j+=1
        
    i+=1