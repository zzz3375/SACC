# %%
import matplotlib.pyplot as plt 
# import matplotlib as mpl
# mpl.use('webAgg')
import numpy as np
from pathlib import Path
import seaborn as sns
import pandas as pd
# %%
tmpdir = Path(r"archived_result\tmp-yolo-best")
wid = np.load(tmpdir/"width.npy")
recall = np.load(tmpdir/"rr.npy")
recall_yolo = np.load(tmpdir/"rr_yolo.npy")
wid[wid>30] = 0
# wid = wid[wid != wid.max()]
# recall = recall[wid != wid.max()]
# recall_yolo = recall_yolo[wid != wid.max()]
precision = np.load(tmpdir/"pp.npy")
precision_yolo = np.load(tmpdir/"pp_yolo.npy")


ax = plt.subplot(221)
def contrast_in_bars(tmpdir, ax):
    wid = np.load(tmpdir/"width.npy")
    recall = np.load(tmpdir/"rr.npy")
    recall_yolo = np.load(tmpdir/"rr_yolo.npy")
    wid[wid>30] = 0
    # wid = wid[wid != wid.max()]
    # recall = recall[wid != wid.max()]
    # recall_yolo = recall_yolo[wid != wid.max()]
    precision = np.load(tmpdir/"pp.npy")
    precision_yolo = np.load(tmpdir/"pp_yolo.npy")
    # precision = precision[wid != wid.max()]
    # precision_yolo = precision_yolo[wid != wid.max()]
    # wid = wid[wid != wid.max()]
    # print(np.sum((recall-recall_yolo)>0)/len(wid))
    # print(np.sum((precision-precision_yolo)>0)/len(wid))
    # print(recall.mean(),recall_yolo.mean())
    # print(recall[wid > 10].mean(),recall_yolo[wid > 10].mean())
    # print(precision.mean(),precision_yolo.mean())
    # print(precision[wid > 10].mean(),precision_yolo[wid > 10].mean())
    data = np.array([recall.mean(), recall_yolo.mean(), 
                    recall[wid > 10].mean(), recall_yolo[wid > 10].mean(), 
                    precision.mean(), precision_yolo.mean(), 
                    precision[wid > 10].mean(), precision_yolo[wid > 10].mean()])
    x = ["Recall (All)"]*2 + ["Recall (Width>10)"]*2 + ["Precision (All)"]*2 + ["Precision (Width>10)"]*2
    # for i in "ABCD": x+=[i]*2
    id = ["Proposed", "YOLOv8-Seg"]*4
    df = pd.DataFrame(data=np.array([x, id]).T, columns=["x", "Method"])
    df["data"]=data
    sns.barplot(df,x="x", y="data", hue="Method", width=0.6)
    ax.grid(1,axis="y")
    ax.set_axisbelow(1)
    plt.ylim((df["data"].min() - 0.2, df["data"].max()+0.05))
    plt.xlabel(" ")
    plt.ylabel(" ")
    # plt.show()
    
# contrast_in_bars(tmpdir, ax)
#%%
def show_recall_improvements():
    plt.clf()
    plt.stem(wid,recall - recall_yolo)

    plt.xlabel("Crack Width in Pixel")
    # plt.title("recall_proposed - recall_yolo")
    plt.ylabel("Improvements on Recall Rate")
    # plt.show()

    width = np.load("training_width.npy") # training width
    xmin= np.min(width)
    xmax = np.max(width)
    y = -0.3
    color = "black"
    plt.plot([xmin, xmax], [y, y], color)
    plt.plot([xmax,xmax], [-0.25,-0.35], color)
    plt.plot([xmin, xmin], [-0.25, -0.35], color)
    plt.show()

# show_recall_improvements()
# %%
# sns.set_style('darkgrid')
# %%
dir_names = [f"tmp-{i}-{j}" for i in ["se","yolo"] for j in ["best", "latest"]]
# %%
i=0
for me in ["se","yolo"]:
    for epcho in ["best", "latest"]:
        tmpdir = Path(f"archived_result\\tmp-{me}-{epcho}")
        ax = plt.subplot(221+i)
        contrast_in_bars(tmpdir, ax)
        i+=1
plt.show()