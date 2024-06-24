# %%
import matplotlib.pyplot as plt 
import numpy as np
# %%
wid = np.load(r"tmp\width.npy")
recall = np.load(r"tmp\rr.npy")
recall_yolo = np.load(r"tmp\rr_yolo.npy")
wid[wid>1000] = wid[wid>1000]/wid.max()*100
# %%
plt.clf()
plt.scatter(wid,recall)
plt.scatter(wid,recall_yolo, marker="v")
plt.show()
# %%
precision = np.load(r"tmp\pp.npy")
precision_yolo = np.load(r"tmp\pp_yolo.npy")

# %%
plt.clf()
plt.scatter(wid,precision)
plt.scatter(wid,precision_yolo, marker="v")
plt.show()
# %%
