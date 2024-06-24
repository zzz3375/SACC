#%%
import numpy as np
import cupy as cp
from PIL import Image
import cv2
from pathlib import Path
from skimage.morphology import medial_axis, skeletonize
# %%
mask_source = Path(r"data\crack_dataset_cleaned\混凝土桥梁裂缝optic_disc_seg\Annotations\H0021.png")
mask = np.asarray(Image.open(mask_source))
# skeleton= skeletonize(mask, method="lee")
skeleton, dist = medial_axis(mask,return_distance=1)
skeleton = skeleton.view(np.uint8)*255
cv2.imwrite(r"tmp_width\skeleton.jpg", skeleton)
# %%
