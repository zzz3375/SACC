#%%
from yolo_suggestion import sam_seg_crack_by_prompt
from pathlib import Path
import numpy as np
import os
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt
# %% Initialize overall metrics table
eval_metrics = "Precision	Recall	IoU	PA".split("\t") # ['Precision', 'Recall', 'IoU', 'PA']
statistic_metrics = "Mean STD".split(" ")
metric_table_head_up = ["Size"]
metric_table_head_down = [" "]

for item in eval_metrics:   
    for s in statistic_metrics:
        metric_table_head_up += [item]
        metric_table_head_down += [s]    
    pass

metric_table_head = pd.MultiIndex.from_arrays(np.array([metric_table_head_up,
                                                        metric_table_head_down]))
pp=[]
rr=[]
# %%
data_dirs = [Path(r"data\crack_dataset_cleaned\混凝土桥梁裂缝optic_disc_seg")]
for data_dir in data_dirs:
    img_names = os.listdir(data_dir/"JPEGImages")
    

    for img_name in img_names:
        source = str(data_dir/"JPEGImages"/img_name)
        print(source)
        ann_file = str(data_dir/"Annotations"/(img_name[:-3]+"png"))
        proposed_mask = sam_seg_crack_by_prompt(source, debug = 0)
        ground_truth = np.asarray(Image.open(ann_file))
        # compare !
        p = (ground_truth/255*proposed_mask/255).sum()/(proposed_mask.sum()/255)
        r = (ground_truth/255*proposed_mask/255).sum()/(ground_truth.sum()/255)
        pp.append(p)
        rr.append(r)
        pass

plt.scatter(rr,pp)
plt.savefig(r"tmp\pr_curve.jpg")