# %%
import supervision as sv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import cv2
import shutil
import xmltodict
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
    out_dir = Path("./data/crack_dataset_cleaned")
    out_dir.mkdir(parents=1,exist_ok=1)
    for raw_data_info in raw_data_infos:
        
        if raw_data_info["annotated"] and raw_data_info["type"] == "Segmentation":
            raw_data_path = Path(raw_data_info["path"])
            if (raw_data_path/"JPEGImages").exists():
                
                image_files = os.listdir(raw_data_path/"JPEGImages")
                img_out_dir = raw_data_path.parent.parent/"crack_dataset_cleaned"/raw_data_path.name/"JPEGImages"
                if not img_out_dir.exists():
                    shutil.copytree(raw_data_path/"JPEGImages", img_out_dir)
                for id in range(len(image_files)):
                
                    image_name = image_files[id]
                    image_file = raw_data_path/"JPEGImages"/image_name
                    annotation_file = raw_data_path/"Annotations"/(image_name[:-3]+"png")
                    ann_out_dir = raw_data_path.parent.parent/"crack_dataset_cleaned"/raw_data_path.name/"Annotations"/(image_name[:-3]+"png")
                    
                    if not ann_out_dir.exists():
                        
                        img = np.array(Image.open(image_file).convert('L'))
                        ann = np.array(Image.open(annotation_file).convert('L'))
                        ann = (ann/np.max(ann)).astype(img.dtype)*255
                        if "transpose" in raw_data_info.keys():
                            ann = ann.T
                        
                        ann = Image.fromarray(ann)
                        
                        ann_out_dir.parent.mkdir(parents=1,exist_ok=1)
                        ann.save(ann_out_dir)


                    pass
                pass
            pass

        elif raw_data_info["annotated"] and raw_data_info["type"] == "Detection":
            raw_data_path = Path(raw_data_info["path"])
            if (raw_data_path/"JPEGImages").exists():
                image_files = os.listdir(raw_data_path/"JPEGImages")
                img_out_dir = raw_data_path.parent.parent/"crack_dataset_cleaned"/raw_data_path.name/"JPEGImages"
                if not img_out_dir.exists():
                    shutil.copytree(raw_data_path/"JPEGImages", img_out_dir)
                for id in range(len(image_files)):

                    id = np.random.randint(0,len(image_files)) #!!!!!!!!!!!!!!!!!!!!!!!!!   

                    image_name = image_files[id]
                    image_file = raw_data_path/"JPEGImages"/image_name
                    annotation_file = raw_data_path/"Annotations"/(image_name[:-3]+"xml")
                    
                    with open(annotation_file,"r") as f:
                        ann_xml = f.read()   
                        ann_dict = xmltodict.parse(ann_xml)['annotation']['object']['bndbox']
                        for key in ann_dict.keys():
                            ann_dict[key]=eval(ann_dict[key])
                        globals().update(ann_dict)

                    
                    pass       

                pass

            pass

        pass

            

        
        


    

