
import matplotlib.pyplot as plt


# Data for different models and their corresponding accuracies
n_params_baseline = [3.02, 11.15, 43.7]
n_params_custom = [3.02, 11.15, 43.7]
n_params_ablation = [3.02, 3.02, 3.02]

accuracies_baseline = [0.47, 0.7, 0.81]
accuracies_custom = [0.405, 0.681, 0.803]
accuracies_ablation = [0.337, 0.352, 0.491]

# Create a figure and axis object
fig, ax = plt.subplots()

# Modify lines with linestyle parameter
ax.plot(n_params_baseline, accuracies_baseline, marker='o', ls='--', label='Baseline YOLOv8')
ax.plot(n_params_custom, accuracies_custom, marker='^', ls='--', label='Proposed Solution')
ax.plot(n_params_ablation, accuracies_ablation, marker='x', ls='', label='Ablation Study')

# Annotate each point with a label
labels = ['n', 's', 'l']
labels_ablation = ['gray', 'SAM', 'Harris']
for i, txt in enumerate(labels):
    if txt == 'n':
        ax.annotate(txt, (n_params_baseline[i], accuracies_baseline[i]), textcoords="offset points", xytext=(-2,-15), ha='center')
    else:
        ax.annotate(txt, (n_params_baseline[i], accuracies_baseline[i]), textcoords="offset points", xytext=(0,10), ha='center')
# for i, txt in enumerate(labels):
#     ax.annotate(txt, (n_params_custom[i], accuracies_custom[i]), textcoords="offset points", xytext=(0,10), ha='center')
for i, txt in enumerate(labels_ablation):
    if txt == 'Harris':
        ax.annotate(txt, (n_params_ablation[i], accuracies_ablation[i]), textcoords="offset points", xytext=(-15,16), ha='left')
    else:
        ax.annotate(txt, (n_params_ablation[i], accuracies_ablation[i]), textcoords="offset points", xytext=(10,-5), ha='left')


# Enable grid
ax.grid(ls='--')

# Set the chart title and axis labels
#ax.set_title("Improvement of TransMix on ViT-based Models")
ax.set_xlabel("Parameters (M)")
ax.set_ylabel("T-LESS $mAP_{50-95}$")
ax.set_ylim(top=0.9)

# Add a legend
ax.legend()

# Show the plot
plt.show()
