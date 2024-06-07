#%%
# import subprocess
# import shutil
import numpy as np
from ultralytics import YOLO
import cv2
from skimage.morphology import skeletonize
# %%
def yolo_predict(source = r"data\concrete_crack\0.jpg", model_path = r"C:\Users\13694\yolov8-adaptated\runs\segment\train-achieved\weights\best.pt"):
    model = YOLO(model_path)
    results = model(source)
    mask_raw = results[0].masks[0].cpu().data.numpy().transpose(1,2,0).squeeze(-1).astype(int)*255 
    cv2.imwrite(r"tmp\yolo_raw_result.jpg",mask_raw)
    return mask_raw

# %%
# mask 2 skeleton points
mask_raw = yolo_predict(r"data\concrete_crack\0.jpg")
skeleton_bool = skeletonize(mask_raw)
skeleton_img = skeleton_bool.astype(int)*255
cv2.imwrite(r"tmp\skeleton.jpg",skeleton_img)
h,w = skeleton_bool.shape[:2]
skeleton_index = np.where(skeleton_bool.ravel())[0]
y = skeleton_index//w
x = skeleton_index%w
points = np.array([x,y]).T

# %%
