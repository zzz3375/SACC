#%%
from yolo_suggestion import sam_seg_crack_by_prompt
from pathlib import Path
import numpy as np
import os
import pandas as pd
from PIL import Image
import cv2
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from skimage.morphology import medial_axis, skeletonize
from scipy.ndimage import distance_transform_edt
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

pp_yolo = []
rr_yolo = []
crack_width = []
# %%
data_dirs = [Path(r"data\crack_dataset_cleaned\混凝土桥梁裂缝optic_disc_seg")]
for data_dir in data_dirs:
    img_names = os.listdir(data_dir/"JPEGImages")
    

    for img_name in img_names:
        source = str(data_dir/"JPEGImages"/img_name)
        print(source)
        ann_file = str(data_dir/"Annotations"/(img_name[:-3]+"png"))
        proposed_mask = sam_seg_crack_by_prompt(source, debug = 0, sampling_points=40)
        ground_truth = np.asarray(Image.open(ann_file))
        yolo_mask = cv2.imread(r"tmp\yolo_raw_result.jpg", cv2.IMREAD_GRAYSCALE)
        # ske, dst = medial_axis(ground_truth, return_distance=1 )
        ske = skeletonize(ground_truth)
        dst = distance_transform_edt(ground_truth)
        dst[ske==0]=0
        # compare !
        if proposed_mask.sum()>0 and ground_truth.sum()>0 and yolo_mask.sum()>0:
            
            p = (ground_truth/255*proposed_mask/255).sum()/(proposed_mask.sum()/255)
            r = (ground_truth/255*proposed_mask/255).sum()/(ground_truth.sum()/255)
            pp.append(p)
            rr.append(r)

            p_yolo = (ground_truth/255*yolo_mask/255).sum()/(yolo_mask.sum()/255)
            r_yolo = (ground_truth/255*yolo_mask/255).sum()/(ground_truth.sum()/255)
            
            pp_yolo.append(p_yolo)
            rr_yolo.append(r_yolo)
            crack_width.append(np.median(dst[ dst > 0 ]))

        pass

np.save(r"tmp\pp",pp)
np.save(r"tmp\rr",rr)

np.save(r"tmp\pp_yolo",pp_yolo)
np.save(r"tmp\rr_yolo",rr_yolo)

np.save(r"tmp\width", crack_width)

# plt.stem(crack_width, rr-rr_yolo)
# plt.show()