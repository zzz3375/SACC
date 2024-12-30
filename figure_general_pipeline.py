# %%
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2
from yolo_suggestion import sam_seg_crack_by_prompt, mask2points

from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
# %%
source = r"data\crack_dataset_cleaned\混凝土桥梁裂缝optic_disc_seg\JPEGImages\H0021.jpg"
# h1h2w1w2 = [470, 580, 400, 510]
h1h2w1w2 = [0,256,768,1024]
h1, h2, w1, w2 = h1h2w1w2
# sam_seg_crack_by_prompt(source)


#original image
src = r"tmp\original_image.jpg"
img = cv2.imread(src)
cv2.imwrite(r"result_demonstration\figures\original_image.jpg",img[h1:h2, w1:w2])


# distance heatmap
distance = np.load(r"tmp\distance-inplace.npy")
distance_heatmap = (distance/distance.max()*255).astype(np.uint8)
distance_heatmap = cv2.applyColorMap(distance_heatmap, cv2.COLORMAP_JET)
cv2.imwrite(r"result_demonstration\figures\heatmap.jpg",distance_heatmap[h1:h2, w1:w2])

#yolov8-seg
src = r"tmp\original_image.jpg"
img = cv2.imread(src)
original_image = cv2.imread(src)
yolo_mask = cv2.imread("tmp\yolo_raw_result.jpg", cv2.IMREAD_GRAYSCALE)
out = yolo_mask[:,:,None].repeat(3,-1)*np.array([0,0,1])*0.5 + img*0.5
out = out.astype(np.uint8)
cv2.imwrite(r"result_demonstration\figures\yolo.jpg", out[h1:h2, w1:w2])

#axis and edge
ret, yolo_mask = cv2.threshold(yolo_mask,127,255,cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(yolo_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
edge = cv2.drawContours(np.zeros_like(yolo_mask), contours, -1, 255, 1)

ske = cv2.imread(r"tmp\skeleton.jpg", cv2.IMREAD_GRAYSCALE)
ske = ske[:,:,None].repeat(3,-1)
ske[edge>127]=[0,255,0]
points = mask2points(yolo_mask, sampling_points=60)
for p in points:
    ske = cv2.circle(ske,p,3,(0,0,255),2)
cv2.imwrite(r"result_demonstration\figures\prompting_points.jpg", ske[h1:h2, w1:w2])



# h1, h2, w1, w2 = h1h2w1w2
# ret, sam_raw = cv2.threshold(np.asarray(Image.open(r"tmp\SAM_prompting.jpg")),127,255,cv2.THRESH_BINARY)
# con, hie = cv2.findContours(sam_raw,cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)



# SAM_result
src = r"tmp\SAM_prompting.jpg"
img = cv2.imread(src)
cv2.imwrite(r"result_demonstration\figures\SAM_prompting.jpg",img[h1:h2, w1:w2])

# contour
src = r"tmp\SAM_prompting.jpg"
img_color = cv2.imread(src, cv2.IMREAD_COLOR)
img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
ret, img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
img = cv2.drawContours(np.zeros_like(img_color), contours, -1, (0,255,0), 2)
cv2.imwrite(r"result_demonstration\figures\contours_SAM.jpg", img[h1:h2, w1:w2])
# %% final output
src = r"tmp\accepted_result.jpg"
accept_result = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
out = accept_result[:,:,None].repeat(3,-1)*np.array([0,0,1])*0.5 + original_image*0.5
cv2.imwrite(r"result_demonstration\figures\accepted_result.jpg",out[h1:h2, w1:w2])