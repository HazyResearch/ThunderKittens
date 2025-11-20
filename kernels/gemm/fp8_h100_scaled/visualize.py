import numpy as np
import matplotlib.pyplot as plt

M, N = 4096, 4096  # Update if you are only sampling a subset.
cpu_ref = np.loadtxt("h_C_ref.csv", delimiter=',', dtype=np.float32).reshape(M, N)
gpu_result = np.loadtxt("h_C.csv", delimiter=',', dtype=np.float32).reshape(M, N)

difference = np.abs(cpu_ref - gpu_result)
log_diff = np.log(difference + 1e-8)

# plt.figure()
# plt.imshow(log_diff, aspect='auto')
# plt.title("Log Difference")
# plt.colorbar()
# plt.savefig("difference.png")

# plt.figure()
# plt.hist(difference.flatten(), bins=100)
# plt.title("Difference Histogram")
# plt.savefig("difference_histogram.png")

max_error_positions = np.loadtxt("max_error_positions.txt", delimiter=',', dtype=np.int32)

plt.figure()
plt.scatter(max_error_positions[:, 1], max_error_positions[:, 0], c=difference[max_error_positions[:, 0], max_error_positions[:, 1]], cmap='viridis')
plt.colorbar()
plt.title("Max Error Positions")
plt.savefig("max_error_positions.png")
