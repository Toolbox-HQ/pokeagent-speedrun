import os
import sys
import time

if os.environ.get("OPENBLAS_NUM_THREADS") is None:
    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = "96"
    os.execve(sys.executable, [sys.executable] + sys.argv, env)

import numpy as np
import torch
from sklearn.cluster import DBSCAN

arr: np.ndarray = torch.load("./tmp/embs.pt", weights_only=False)
arr = np.ascontiguousarray(arr, dtype=np.float64)
print(arr.shape)

t = time.time()
cluster_labels = DBSCAN(
    eps=0.005,
    min_samples=5,
    n_jobs=192,
    algorithm="ball_tree",
).fit_predict(arr)
print(f"time: {time.time() - t}")