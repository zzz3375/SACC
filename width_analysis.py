#%%
import numpy as np
import cupy as cp
from PIL import Image
import cv2
from pathlib import Path
from skimage.morphology import medial_axis, skeletonize
# %%
mask_source = Path(r"tmp\accepted_result.jpg")
mask = np.asarray(Image.open(mask_source))
skeleton= skeletonize(mask, method="lee")
skeleton = skeleton.view(np.uint8)*255
cv2.imwrite(r"tmp_width\skeleton.jpg", skeleton)
# %%
