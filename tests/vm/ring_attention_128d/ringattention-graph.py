import matplotlib.pyplot as plt
import numpy as np

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Extract data
N_labels = [f'$2^{{{i}}}$' for i in range(15, 21)]
jax_values = [1.17, 4.6703, 17.2353, 48.2669, 100.0651, 135.7256]
xdit_values = [319.4930, 387.9966, 419.6126, 435.8104, 429.8288, 430.6051]
seq_values = [387, 615, 780, 810, 832, 827]
overlap_values = [709.8999, 808.4670, 865.3955, 843, 852, 833]

# Convert N to a readable format (K for thousands, M for millions)
x_labels = []
for n in N_labels:
    x_labels.append(n)

# Create the bar chart
x = np.arange(len(N_labels))
width = 0.2

# Plot bars
ax.bar(x - width*3/2, jax_values,     width, label='RingAttention (Jax)', color='#F29441')
ax.bar(x - width/2,   xdit_values,    width, label='xDiT', color='#9467BD')
ax.bar(x + width/2,   seq_values,     width, label='ThunderRing (Sequential)', color='#4EAE4E')
ax.bar(x + width*3/2, overlap_values, width, label='ThunderRing (Overlapped)', color='#67BBBD')

# Add labels and title
ax.set_xlabel('Sequence Length (N)', fontsize=12)
ax.set_ylabel('TFLOPs/GPU', fontsize=12)
ax.set_title('Ring Attention Performance (4xB200)', fontsize=14)

# Set x-axis ticks and labels
ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.tick_params(axis='both', which='major', labelsize=13)

# Add legend
ax.legend()

# Add grid for better readability
# ax.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('ringattention.png', dpi=300)
