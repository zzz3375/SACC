#%%
from yolo_suggestion import *
from scipy.optimize import curve_fit
import shutil
# source = r"data\crack_dataset_cleaned\混凝土桥梁裂缝optic_disc_seg\JPEGImages\H0021.jpg" #good result
# source = r"data\crack_dataset_cleaned\混凝土桥梁裂缝optic_disc_seg\JPEGImages\N0042.jpg" #multi-crack
#%%
def sample_study(out_dir = Path("result-sample-study\\ss-650"), max_sample = 650):
    source = Path(r"data\crack_dataset_cleaned\混凝土桥梁裂缝optic_disc_seg\JPEGImages\P0018.jpg")
    gt_path = source.parent.parent/"Annotations"/(source.name[:-3]+"png")
    ske_len = np.load(r"result-sample-study\skeleton_length.npy")
    test_sample = np.arange(1,max_sample, 6)
    rr = []
    pp = []
    rr_sam=[]
    pp_sam=[]
    for sampling_points in test_sample:
        mask = sam_seg_crack_by_prompt(source, debug=1, sampling_points=sampling_points).astype(bool)
        mask_sam = np.asarray(Image.open(r"tmp\SAM_prompting.jpg")).astype(bool)
        gt = np.asarray(Image.open(gt_path)).astype(bool)
        r = np.bitwise_and(gt, mask).sum()/gt.sum()
        p = np.bitwise_and(gt, mask).sum()/mask.sum()
        r_sam = np.bitwise_and(gt, mask_sam).sum()/gt.sum()
        p_sam = np.bitwise_and(gt, mask_sam).sum()/mask_sam.sum()
        rr.append(r)
        pp.append(p)
        rr_sam.append(r_sam)
        pp_sam.append(p_sam)
    # np.save(r"tmp\pp-ss", pp)
    # np.save(r"tmp\rr-ss", rr)
    np.save(out_dir/"pp-ss", pp_sam)
    np.save(out_dir/"rr-ss", rr_sam)
    np.save(out_dir/"test_sample", test_sample)
    # np.save(r"tmp\samples-ss", test_sample)
    # shutil.copy(r"tmp\pp_sam-ss.npy", r"result-sample-study")
    # shutil.copy(r"tmp\rr_sam-ss.npy", r"result-sample-study")

def analysis_sample_study_250():
    # sample_study()
    p = np.load(r"result-sample-study\pp-ss.npy")
    r = np.load(r"result-sample-study\rr-ss.npy")
    p_sam = np.load(r"result-sample-study\pp_sam-ss.npy")
    r_sam = np.load(r"result-sample-study\rr_sam-ss.npy")
    # samples = np.arange()
    plt.subplot(121)
    
    samples = np.load(r"result-sample-study\samples-ss.npy")
    analysis_end = samples.max()+1

    plt.scatter(samples, p, label = "Proposed" )
    popt, pcov = curve_fit(curve4, samples[samples<analysis_end], p[samples<analysis_end])
    xx = np.linspace(samples[0], analysis_end, 101)
    yy = curve4(xx, *popt)
    plt.plot(xx, yy, color = "red")

    plt.scatter(samples, p_sam, label = "Original SAM (2023)")
    popt, pcov = curve_fit(curve4, samples[samples<analysis_end], p_sam[samples<analysis_end])
    xx = np.linspace(samples[0], analysis_end, 101)
    yy = curve4(xx, *popt)
    plt.plot(xx, yy, color = "green")
    
    plt.xlabel("Samples per Image")
    plt.title("Precision")
    plt.legend()

    plt.subplot(122)
    plt.scatter(samples, r, label = "Proposed" )
    plt.scatter(samples, r_sam, label = "Original SAM (2023)" )
    
    # popt, pcov = curve_fit(curve4, samples[1:], r[1:])
    # xx = np.linspace(samples, analysis_end, 101)
    # yy = curve4(xx, *popt)    
    # plt.plot(xx, yy, color = "red")

    # popt, pcov = curve_fit(curve4, samples[1:], r_sam[1:])
    # xx = np.linspace(samples, analysis_end, 101)
    # yy = curve4(xx, *popt)  

    plt.plot(xx, yy, color = "green")
    plt.ylim((0.8,0.95))
    plt.xlabel("Samples per Image")
    plt.title("Recall Rate")
    plt.legend()

    plt.show()

def curve4(x, a, b, c, d, e):
    return a*x**4 + b*x**3 +c*x**2+d*x +e
def hook(x, a,b,c):
    return a/x+b*np.log(x)+c


if __name__ == '__main__':
    # sample_study()
    # analysis_sample_study_250()
    # p = np.load(r"result-sample-study\ss-650\pp_sam-ss.npy")
    # samples = np.load(r"result-sample-study\ss-650\test_sample.npy")
    # plt.scatter(samples, p)
    # plt.xlabel("Samples per Image")
    # plt.title("Precision")
    # plt.show()

    # source = Path(r"data\crack_dataset_cleaned\混凝土桥梁裂缝optic_disc_seg\JPEGImages\P0018.jpg")
    # source = Path(r"data\crack_dataset_cleaned\混凝土桥梁裂缝optic_disc_seg\JPEGImages\H0021.jpg")
    # sam_seg_crack_by_prompt(source, sampling_points=20).astype(bool)
    # mask = np.asarray(Image.open(r"tmp\SAM_prompting_morno.jpg")).astype(bool)
    # gt = np.asarray(Image.open(source.parent.parent/"Annotations"/(source.name[:-3]+"png"))).astype(bool)
    # out = np.zeros(list(gt.shape)+[3], dtype = np.uint8)
    # out[gt] = [255, 255, 255]
    
    # t = np.bitwise_and(mask, gt).astype(bool)
    # out[t] = [0,255,0]

    # f = (mask.astype(int)-t.astype(int)).astype(bool)
    # out[f] = [0,0,255]

    # cv2.imwrite(r"tmp\sample-study-2.jpg", out)
    analysis_sample_study_250()

    
    pass
