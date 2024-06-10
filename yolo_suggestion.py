#%%
# import subprocess
# import shutil
import numpy as np
from ultralytics import YOLO
import cv2
from skimage.morphology import skeletonize
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
from ultralytics.utils.ops import scale_masks

#%%
def yolo_predict(source):
    model_path = "best.pt"
    source_image = cv2.imread(source)
    # h,w=source_image.shape[:2]
    model = YOLO(model_path)
    results = model.predict(source)
    # mask_raw = results[0].masks[0].cpu().data.numpy().transpose(1,2,0).squeeze(-1).astype(int)*255
    original_shape = source_image.shape[:2]
    conf = results[0].boxes.conf.cpu().detach().numpy()
    mask_raw_stack = scale_masks(results[0].masks.data[None,:],original_shape)
    mask_raw = mask_raw_stack[conf>0.5].sum(axis=1).squeeze().cpu().numpy()   
    mask_raw = ( mask_raw >0 ).astype(int) *255
    cv2.imwrite(r"tmp\yolo_raw_result.jpg",mask_raw)
    return mask_raw


# mask 2 skeleton points
# mask_raw = yolo_predict(source, model_path)
def mask2points(mask_raw) -> np.ndarray:
    skeleton_bool = skeletonize(mask_raw,method="lee")
    skeleton_img = skeleton_bool.astype(int)*255
    cv2.imwrite(r"tmp\skeleton.jpg",skeleton_img)
    h,w = skeleton_bool.shape[:2]
    skeleton_index = np.where(skeleton_bool.ravel())[0]
    y = skeleton_index//w
    x = skeleton_index%w
    points = np.array([x,y]).T

    # sample
    n = points.shape[0]
    step = n//6
    points = points[::step]

    return points


# show_points
# def show_points(coords, labels, ax, marker_size=375):
#     pos_points = coords[labels==1]
#     neg_points = coords[labels==0]
#     ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
#     ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

# source  and points 2 prompted mask
# points = mask2points(mask_raw)

def sam_prompt(source, points):
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

    cv2.imwrite(r"tmp\SAM_prompting.jpg",masks[0].astype(int)*255)
    cv2.imwrite(r"tmp\original_image.jpg",source_image)
    # print(scores)
    # plt.figure(figsize=(10,10))
    # plt.imshow(source_image)
    # show_points(points, input_label, plt.gca())
    # plt.axis('on')
    # plt.savefig(r"tmp\points_prompting.jpg")
    
    for p in points: 
        # p = np.round(p)
        source_image = cv2.circle(source_image,p,10,(0,0,255),4)
    cv2.imwrite("tmp\points_prompting.jpg",source_image)
    return masks[0].astype(int)*255, scores[0]

def sam_seg_crack_by_prompt(source):
    mask_raw = yolo_predict(source)
    points = mask2points(mask_raw)
    mask, sam_scores = sam_prompt(source,points)
    return mask, sam_scores
# %%
if __name__ == '__main__':
    source = r"data\crack_dataset_cleaned\混凝土桥梁裂缝optic_disc_seg\JPEGImages\P0213.jpg"
    mask,_ = sam_seg_crack_by_prompt(source)
    pass
