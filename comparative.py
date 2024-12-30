from yolo_suggestion import *
import shutil
import pandas as pd


img_dir = r"data\crack_dataset_cleaned\混凝土桥梁裂缝optic_disc_seg\JPEGImages"
msk_dir = r"data\crack_dataset_cleaned\混凝土桥梁裂缝optic_disc_seg\Annotations"

def predict_proposed(img_dir, msk_dir, out_dir = "./full_results"):
    img_dir = Path(img_dir)
    msk_dir = Path(msk_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=1, parents=1)
    for img_file in img_dir.iterdir():
        sam_seg_crack_by_prompt(img_file)
        shutil.copyfile(r"tmp\yolo_raw_result.png", out_dir/(img_file.name[:-4]+"_yolo"+".png"))
        shutil.copyfile(r"tmp\SAM_prompting.png", out_dir/(img_file.name[:-4]+"_sam"+".png"))
        shutil.copyfile(r"tmp\accepted_result.png", out_dir/(img_file.name[:-4]+"_proposed"+".png"))
        pass

    pass

def compute_metrics(predict_msk_path: Path, gt_msk_path: Path) -> float:
    # Load the predicted mask and ground truth mask
    predict_msk = np.array(Image.open(predict_msk_path).convert('L'))
    gt_msk = np.array(Image.open(gt_msk_path).convert('L'))
    
    # Ensure the masks are binary
    predict_msk = (predict_msk > 127).astype(np.uint8)
    gt_msk = (gt_msk > 127).astype(np.uint8)
    
    # Compute intersection and union
    intersection = np.logical_and(predict_msk, gt_msk).sum()
    union = np.logical_or(predict_msk, gt_msk).sum()
    
    # Compute IoU
    iou = intersection / union if union != 0 else 0.0
    
    return iou

def compute_iou_avg(predict_dir: Path, gt_dir: Path, type = "CycleGAN"):
    name_surfix = {"CycleGAN":"_fake_A"}
    pass


if __name__ == '__main__':
    
    predict_msk_path = Path(r"C:\Users\13694\pytorch-CycleGAN-and-pix2pix\results\cyclegan_crack\test_latest\images\H0017_fake_A.png")
    gt_msk_path = Path(str(predict_msk_path).replace("fake", "real"))

    iou = compute_iou(predict_msk_path, gt_msk_path)
    print(f"IoU: {iou:.4f}")
    pass
