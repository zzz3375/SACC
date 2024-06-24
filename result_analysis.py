# %%
import matplotlib.pyplot as plt 
import matplotlib as mpl
mpl.use('webAgg')
import numpy as np
# %%
wid = np.load(r"tmp\width.npy")
recall = np.load(r"tmp\rr.npy")
recall_yolo = np.load(r"tmp\rr_yolo.npy")
wid[wid>30] = wid[wid>30]/wid.max()*10
# %%
# plt.clf()
# plt.scatter(wid,recall)
# plt.scatter(wid,recall_yolo, marker="v")
# plt.show()
# %%
precision = np.load(r"tmp\pp.npy")
precision_yolo = np.load(r"tmp\pp_yolo.npy")

# # %%
# plt.clf()
# plt.scatter(wid,precision)
# plt.scatter(wid,precision_yolo, marker="v")
# plt.show()
# %%
plt.clf()
plt.stem(wid,recall - recall_yolo)
print(np.sum((recall-recall_yolo)>0)/len(wid))
print(recall.mean(),recall_yolo.mean())
print(recall[wid > 10].mean(),recall_yolo[wid > 10].mean())
print(precision.mean(),precision_yolo.mean())
print(precision[wid > 10].mean(),precision_yolo[wid > 10].mean())
plt.xlabel("crack width (pixel)")
plt.title("recall_proposed - recall_yolo")
plt.legend()
plt.show()