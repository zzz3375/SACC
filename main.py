#%%
from yolo_suggestion import sam_seg_crack_by_prompt
from pathlib import Path
import numpy as np
import os
import pandas as pd
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
metric_table_head = pd.MultiIndex.from_arrays(np.array([metric_table_head_up,metric_table_head_down]))

# %%
data_dirs = [Path(r"data\crack_dataset_cleaned\混凝土桥梁裂缝optic_disc_seg")]
for data_dir in data_dirs:
    img_names = os.listdir(data_dir/"JPEGImages")
    

    for img_name in img_names:
        source = str(data_dir/"JPEGImages"/img_name)
        ann_file = str(data_dir/"Annotations"/(img_name[:-3]+"png"))

        # compare !

        pass