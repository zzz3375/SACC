# %%
import matplotlib.pyplot as plt 
# import matplotlib as mpl
# mpl.use('webAgg')
import numpy as np
from pathlib import Path
# %%
tmpdir = Path(r"archived_result\tmp-yolo-best")
wid = np.load(tmpdir/"width.npy")
recall = np.load(tmpdir/"rr.npy")
recall_yolo = np.load(tmpdir/"rr_yolo.npy")
wid[wid>30] = wid[wid>30]/wid.max()*10
precision = np.load(tmpdir/"pp.npy")
precision_yolo = np.load(tmpdir/"pp_yolo.npy")
print(np.sum((recall-recall_yolo)>0)/len(wid))
print(np.sum((precision-precision_yolo)>0)/len(wid))
print(recall.mean(),recall_yolo.mean())
print(recall[wid > 10].mean(),recall_yolo[wid > 10].mean())
print(precision.mean(),precision_yolo.mean())
print(precision[wid > 10].mean(),precision_yolo[wid > 10].mean())
#%%
def show_recall_improvements():
    plt.clf()
    plt.stem(wid,recall - recall_yolo)

    plt.xlabel("Crack Width in Pixel")
    # plt.title("recall_proposed - recall_yolo")
    plt.ylabel("Improvements on Recall Rate")
    # plt.show()

    width = np.load("training_width.npy")
    xmin= np.min(width)
    xmax = np.max(width)
    y = -0.3
    color = "black"
    plt.plot([xmin, xmax], [y, y], color)
    plt.plot([xmax,xmax], [-0.25,-0.35], color)
    plt.plot([xmin, xmin], [-0.25, -0.35], color)
    plt.show()

# show_recall_improvements()