# %%
import supervision as sv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import cv2
# %%
raw_data_infos = [
    {
    "path":"data/crack dataset (bridge, pavement, rail)/混凝土桥梁裂缝optic_disc_seg",
    "annotated":True,
    "type":"Segmentation",
    "transpose":True
    },
    {
    "path":"data/crack dataset (bridge, pavement, rail)/混凝土桥裂缝Original_Crack_DataSet_1024_1024",
    "annotated":False
    },
    {
    "path":"data/crack dataset (bridge, pavement, rail)/震后路面裂缝",
    "annotated":False
    },
    {
    "path":"data/crack dataset (bridge, pavement, rail)/自建的五类沥青路面虚拟裂缝数据集（UE5+无人机）",
    "annotated":True,
    "type":"Detection"
    },
    {
    "path":"data/crack dataset (bridge, pavement, rail)/CFD通用数据集",
    "annotated":True,
    "type":"Segmentation"
    },
    {
    "path":"data/crack dataset (bridge, pavement, rail)/roboflow",
    "annotated":True,
    "type":"Segmentation"
    }
]
# %%
if __name__ == '__main__':

    for raw_data_info in raw_data_infos:
        if raw_data_info["annotated"] and raw_data_info["type"] == "Segmentation":
            raw_data_path = Path(raw_data_info["path"])
            if (raw_data_path/"JPEGImages").exists():
                image_files = os.listdir(raw_data_path/"JPEGImages")
                random_id = np.random.randint(0,len(image_files))
                image_name = image_files[random_id]
                image_file = raw_data_path/"JPEGImages"/image_name
                annotation_file = raw_data_path/"Annotations"/(image_name[:-3]+"png")

                img = np.array(Image.open(image_file).convert('L'))
                ann = np.array(Image.open(annotation_file).convert('L'))
                ann = (ann/np.max(ann)).astype(int)*255
                if "transpose" in raw_data_info.keys():
                    # ann = cv2.rotate(ann,cv2.ROTATE_90_COUNTERCLOCKWISE)
                    ann = ann.T
                out = Image.fromarray(np.concatenate([img,ann],axis=1))
                out.show()

            pass
        else: continue
    pass

