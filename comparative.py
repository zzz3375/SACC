from yolo_suggestion import *
import shutil


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

if __name__ == '__main__':
    predict_proposed(img_dir, msk_dir) 
    pass
