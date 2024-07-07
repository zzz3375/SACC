# %%
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from yolo_suggestion import sam_seg_crack_by_prompt
import numpy as np
plt.rcParams.update({'font.size': 6})
# %%
data_dir = Path(r"data\crack_dataset_cleaned\混凝土桥梁裂缝optic_disc_seg\JPEGImages")
methods = ["Original Image", "GAN", "YOLOv8-Seg", "2nd Round", "Proposed", "Manual Annotation"]
demo_files = ["H0003.jpg", "H0007.jpg", "H0009.jpg", "H0020.jpg", "P0018.jpg"]
plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=0.1, hspace=0.1)
i=0
for file_name in demo_files:
    sam_seg_crack_by_prompt(data_dir/file_name)
    j=0
    for method in methods:
        if method == "GAN":
            file_dir = Path(r"gan-result")/(file_name[:-4]+"_fake"+".png")
        elif method == "Original Image":
            file_dir = Path(r"tmp\original_image.jpg")
        elif method == "YOLOv8-Seg":
            file_dir = Path(r"tmp\yolo_raw_result.jpg")
        elif method == "2nd Round":
            file_dir = Path(r"tmp\SAM_prompting.jpg")
        elif method == "Proposed":
            file_dir = Path(r"tmp\accepted_result.jpg")
        elif method == "Manual Annotation":
            file_dir = Path(r"data\crack_dataset_cleaned\混凝土桥梁裂缝optic_disc_seg\Annotations") / (file_name[:-3]+"png")
        
        img = np.asarray(Image.open(file_dir))
        plt.subplot(len(demo_files), len(methods), len(methods)*i+j+1)
        plt.axis("off")
        plt.imshow(img, "gray")
        if i == 0: plt.title(method)
        j+=1
        
        
    i+=1
plt.savefig("qualitative.svg", dpi=2000)
# %%
