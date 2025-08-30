import matplotlib.pyplot as plt
import numpy as np

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Extract data
N_labels = [f'$2^{{{i}}}$' for i in range(14, 21)]
comm_0_values =  [185.590861063713,  402.42394045062053, 684.0080230832069, 831.3073710520496, 855.9992523261375, 886.4029767166567, 889.7014210006454]
comm_2_values =  [268.6913106700555, 526.5637359594801,  808.270115030147,  872.4621283424646, 874.2722886322125, 892.6658221442152, 886.5143609934917]
comm_4_values =  [418.2157088369102, 730.76127976704,    804.7538130661654, 868.1141226293634, 869.9436662306031, 890.3894067038113, 884.3053061304734]
comm_8_values =  [551.6706408990542, 733.3903737311397,  801.4786905239077, 866.96001507527,   857.9247458159687, 875.035884075361,  872.5478959392061]
comm_16_values = [547.9503437215694, 735.7355139931854,  808.505628043081,  828.7697422433661, 827.046795296101,  841.3080830630765, 839.3379082628672]

# Convert N to a readable format (K for thousands, M for millions)
x_labels = []
for n in N_labels:
    x_labels.append(n)

# Create the bar chart
x = np.arange(len(N_labels)) * 1.2
width = 0.2

# Plot bars
ax.bar(x - width*2, comm_0_values,  width, label='0 Comm. SMs',  color='#F29441')
ax.bar(x - width,   comm_2_values,  width, label='2 Comm. SMs',  color='#9467BD')
ax.bar(x,           comm_4_values,  width, label='4 Comm. SMs',  color='#4EAE4E')
ax.bar(x + width,   comm_8_values,  width, label='8 Comm. SMs',  color='#67BBBD')
ax.bar(x + width*2, comm_16_values, width, label='16 Comm. SMs', color='#E67C7C')

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
plt.savefig('ringattention-ablation.png', dpi=300)
