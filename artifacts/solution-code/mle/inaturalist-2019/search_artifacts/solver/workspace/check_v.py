import numpy as np
V = np.load("visual_sim.npy")
print("Min:", V.min())
print("Max:", V.max())
print("Mean:", V.mean())
print("Median:", np.median(V))
