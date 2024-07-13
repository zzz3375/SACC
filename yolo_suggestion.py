#%%
# import subprocess
# import shutil
import numpy as np

# image processing
from ultralytics import YOLO
import cv2
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from segment_anything import sam_model_registry, SamPredictor

# visulization
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics.utils.ops import scale_masks
from pathlib import Path
from PIL import Image


#%%
def yolo_predict(source, debug=True, yolo_model_path = "best.pt"):
    model_path = yolo_model_path
    source_image = cv2.imread(source)
    # h,w=source_image.shape[:2]
    model = YOLO(model_path)
    results = model.predict(source)
    # mask_raw = results[0].masks[0].cpu().data.numpy().transpose(1,2,0).squeeze(-1).astype(int)*255
    original_shape = source_image.shape[:2]
    conf = results[0].boxes.conf.cpu().detach().numpy()
    if results[0].masks is not None:
        mask_raw_stack = scale_masks(results[0].masks.data[None,:],original_shape).cpu().numpy()
    else: mask_raw_stack = np.zeros([1,1]+list(source_image.shape[:2]), dtype = np.uint8)
    mask_raw = mask_raw_stack.squeeze(axis=0)[conf>0.5].sum(axis=0)  
    mask_raw = ( mask_raw >0 ).astype(np.uint8) *255
    # if debug: 
    cv2.imwrite(r"tmp\yolo_raw_result.jpg",mask_raw)
    return mask_raw, conf

def mask2points(mask_raw, debug=True, sampling_points = 12) -> np.ndarray:
    skeleton_bool = skeletonize(mask_raw)
    dist = distance_transform_edt(mask_raw)
    if debug: np.save(r"tmp\distance-inplace", dist)
    dist[skeleton_bool==0]=0
    skeleton_img = skeleton_bool.astype(np.uint8)*255
    if debug: 
        cv2.imwrite(r"tmp\skeleton.jpg",skeleton_img)
        np.save(r"tmp\skeleton_length", skeleton_bool.sum())
    h,w = skeleton_bool.shape[:2]
    skeleton_index = np.where(skeleton_bool.ravel())[0]
    y = skeleton_index//w
    x = skeleton_index%w
    points = np.array([x,y]).T

    # sample
    n = points.shape[0]
    step = n//sampling_points
    # step = 80
    # points = points[int(step/2):-1-int(step/2):step]
    points = points[::step] if step > 0 else points
    return points

def sam_prompt(source, points, debug=True):
    source_image = cv2.imread(source)
    # sample

    input_label = np.ones(points.shape[:1])
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(source_image)
    masks, scores, logits = predictor.predict(
        point_coords=points,
        point_labels=input_label,
        multimask_output=False,
    )
    if debug:
        cv2.imwrite(r"tmp\SAM_prompting.jpg",masks[scores.argmax()].astype(int)*255)
        cv2.imwrite(r"tmp\original_image.jpg",source_image)
        print(scores[0])
    # plt.figure(figsize=(10,10))
    # plt.imshow(source_image)
    # show_points(points, input_label, plt.gca())
    # plt.axis('on')
    # plt.savefig(r"tmp\points_prompting.jpg")
    
    for p in points: 
        # p = np.round(p)
        source_image = cv2.circle(source_image,p,4,(0,0,255),4)
    if debug: cv2.imwrite(r"tmp\points_prompting.jpg",source_image)
    return masks[0].astype(np.uint8)*255, scores[0]

def morno_coorrection(img, debug=1):
    m = np.ceil(max(img.shape[:2])*0.015).astype(int)
    n = np.ceil(max(img.shape[:2])*0.007).astype(int)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (m,m))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (n,n))
    # out = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel_open)
    out = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel_close)
    if debug: cv2.imwrite(r"tmp\SAM_prompting_morno.jpg", out)        
    return out

def contour_aera(con,mask):
    tmp = np.zeros_like(mask)
    mask_i = cv2.fillPoly(tmp,con,255)
    aera = np.sum(mask_i>0)
    # print(aera)
    return aera

def contour_optimization(mask,yolo_conf, debug=1):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    out = cv2.drawContours(np.zeros_like(mask), contours, -1, 255, 1)
    if debug: cv2.imwrite(r"tmp\contours.jpg",out)
    contours_sorted = sorted(contours, key = lambda con: contour_aera(con, mask=mask), reverse=1)
    contours_accepted = contours_sorted[:np.sum(yolo_conf>0.6)]
    out = cv2.drawContours(np.zeros_like(mask), contours_accepted, -1, 255, 1)
    if debug: cv2.imwrite(r"tmp\contours_filtered.jpg", out)
    out = np.zeros_like(mask)
    for con in contours_accepted:
        out = cv2.fillPoly(out,[con.squeeze(axis=1)],255)
    if debug: cv2.imwrite(r"tmp\SAM_contour_filter.jpg", out)
    return out

def sam_seg_crack_by_prompt(source, debug=1, sampling_points = 12):
    Path("tmp").mkdir(parents=1,exist_ok=1)
    mask_yolo, yolo_conf = yolo_predict(source,debug)
    points = mask2points(mask_yolo,debug, sampling_points=sampling_points)

    mask_sam, sam_scores = sam_prompt(source,points,debug)
    mask_sam = contour_optimization(mask_sam,yolo_conf,debug)
    mask_sam = morno_coorrection(mask_sam,debug)

    # contradictary analysis: 2 model should have coherent prediction
    conflict = (mask_sam - mask_sam*mask_yolo).astype(bool).sum()/ (mask_yolo).astype(bool).sum() if (mask_yolo).astype(bool).sum() > 0 else 100
    if  conflict > 2:
        mask_accepted = mask_yolo # prevent the condition that YOLO raw result is not reliable, making some sampling point out of the crack
        if debug: print("accept yolo")
    else: 
        mask_accepted = mask_sam
        if debug: print("accept SAM")
    if debug: 
        cv2.imwrite(r"tmp\accepted_result.jpg",mask_accepted)
    return mask_accepted


# %%
if __name__ == '__main__':
    source = r"data\crack_dataset_cleaned\混凝土桥梁裂缝optic_disc_seg\JPEGImages\H0021.jpg"
    sam_seg_crack_by_prompt(source)
    distance = np.load(r"tmp\distance-inplace.npy")
    sns.heatmap(distance)
    plt.show()

    pass

# %%
