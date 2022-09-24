import numpy as np
import matplotlib.pyplot as plt

running_loss = np.load("run_log.npy")
progress = np.load("progress.npy")

plt.figure(figsize=(12, 6))
plt.plot(progress[:-1])
plt.xlabel("Batch number")
plt.ylabel("Loss")
plt.show()