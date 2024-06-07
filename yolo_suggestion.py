#%%
# import subprocess
# import shutil
from ultralytics import YOLO
import cv2
from skimage.morphology import skeletonize
# %%
def yolo_predict(source = r"data\concrete_crack\0.jpg", model_path = r"C:\Users\13694\yolov8-adaptated\runs\segment\train-achieved\weights\best.pt"):
    model = YOLO(model_path)
    results = model(source)
    mask_raw = results[0].masks[0].cpu().data.numpy().transpose(1,2,0).astype(int)*255 
    cv2.imwrite(r"tmp\yolo_raw_result.jpg",mask_raw)
    return mask_raw

# cmd = f"yolo segment predict model={model} source={source}"
#%%
# shutil.rmtree(r"runs\segment\predict")
# subprocess.run(cmd,shell=1)
# %%
mask_raw = yolo_predict(r"data\concrete_crack\0.jpg")
skeleton = skeletonize(mask_raw).squeeze(-1).astype(int)*255
cv2.imwrite(r"tmp\skeleton.jpg",skeleton)
# %%
