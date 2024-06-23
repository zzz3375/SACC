#%%
import numpy as np
import cupy as cp
import cv2
from pathlib import Path
from skimage.morphology import skeletonize
# %%
mask_source = Path(r"tmp\yolo_raw_result.jpg")
mask = cv2.imread(str(mask_source), cv2.IMREAD_GRAYSCALE)
skeleton = skeletonize(mask).view(np.uint8)*255
cv2.imwrite(r"tmp_width\skeleton.jpg", skeleton)
# %%
