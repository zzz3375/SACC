# %%
import supervision as sv
from pathlib import Path
import matplotlib.pyplot as plt
# %%
if __name__ == '__main__':
    raw_data_infos = [
        {
        "path":"data/crack dataset (bridge, pavement, rail)/混凝土桥梁裂缝optic_disc_seg",
        "annotated":True,
        "type":"Segmentation"
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
    for raw_data_info in raw_data_infos:
        if raw_data_info["annotated"] and raw_data_info["type"] == "Segmentation":
            raw_data_path = Path(raw_data_info["path"])
            pass
        else: continue
    pass

