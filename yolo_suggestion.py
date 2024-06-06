#%%
# import subprocess
# import shutil
from ultralytics import YOLO
import cv2
# %%
model = YOLO(r"C:\Users\13694\yolov8-adaptated\runs\segment\train-achieved\weights\best.pt")
source = r"data\concrete_crack\0.jpg"
results = model(source)
mask_raw = results[0].masks[0].cpu().data.numpy().transpose(1,2,0).astype(int)*255 
cv2.imwrite("result.jpg",mask_raw)

cmd = f"yolo segment predict model={model} source={source}"
#%%
# shutil.rmtree(r"runs\segment\predict")
# subprocess.run(cmd,shell=1)
# %%
