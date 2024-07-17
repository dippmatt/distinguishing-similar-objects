from ultralytics import YOLO
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

this_file = Path(__file__).resolve()    

eval_models = [
    "dsimo/runs/detect/3-ch-default-baseline__2__-50k-yolo8n-v8.2.45/weights/best.pt",
    #"dsimo/runs/detect/ablation_gray_only_yolo8n/weights/best.pt",
    "dsimo/runs/detect/ablation_harris_yolo8n/weights/best.pt",
    #"dsimo/runs/detect/ablation_sam_yolo8n/weights/best.pt",
]

baseline_missclass = []
baseline_missclass_labels = []
harris_missclass = []
harris_missclass_labels = []

j=0
for model_path in eval_models:
    j+=1
    print("Model path", model_path)
    model_path = Path("/home/matthias/Documents/", model_path)
    save_dir = model_path.parent.parent / "confusion_matrix.txt"

    # load the confution matrix
    loaded_array = np.loadtxt(save_dir, delimiter=',')
    # convert to int
    confusion_matrix = loaded_array.astype(int)

    # sum up the whole matrix:
    # confusion_matrix_sum = np.sum(confusion_matrix)
    # confusion_matrix_sum = np.sum(confusion_matrix[:30, :], axis=1)
    # print(f"Sum of the confusion matrix: {confusion_matrix_sum}")

    # remove the background class
    confusion_matrix = confusion_matrix[:-1, :-1]
    num_classes = 30

    # Initialize arrays to store FN and FP for each class
    FN = np.zeros((num_classes, num_classes))
    FP = np.zeros((num_classes, num_classes))

    # Calculate FN and FP for each class
    for class_index in range(num_classes):
        for i in range(num_classes):
            if i != class_index:
                FN[class_index, i] = confusion_matrix[class_index, i]
                FP[class_index, i] = confusion_matrix[i, class_index]

    # Print the results for all classes
    for class_index in range(num_classes):
        print(f"Class {class_index}:")
        for i in range(num_classes):
            if i != class_index:
                print(f"  vs Class {i}: FN = {FN[class_index, i]}, FP = {FP[class_index, i]}")

    # Find the class with the largest sum of FP and FN for each class
    largest_misclassifications = np.zeros(num_classes, dtype=int)
    largest_misclass_values = np.zeros(num_classes, dtype=int)

    for class_index in range(num_classes):
        max_misclass_value = -1
        misclass_class = -1
        for i in range(num_classes):
            if i != class_index:
                misclass_value = FP[class_index, i] + FN[class_index, i]
                if misclass_value > max_misclass_value:
                    max_misclass_value = misclass_value
                    misclass_class = i
        largest_misclassifications[class_index] = misclass_class
        largest_misclass_values[class_index] = max_misclass_value

    # Print the results for all classes
    total_instances = np.sum(confusion_matrix[:30, :], axis=1)

    for class_index in range(num_classes):
        misclass_class = largest_misclassifications[class_index]
        misclass_value = largest_misclass_values[class_index]
        misclass_rate = misclass_value / total_instances[class_index]
        print(f"Class {class_index} is most frequently misclassified as Class {misclass_class} with FP + FN = {misclass_value} with a missclassification rate of {misclass_rate:.2f}")
        if j == 1:
            baseline_missclass.append(misclass_rate)
            baseline_missclass_labels.append(misclass_class)
        elif j == 2:
            harris_missclass.append(misclass_rate)
            harris_missclass_labels.append(misclass_class)

print("Baseline missclass", len(baseline_missclass))
print("Harris missclass", len(harris_missclass))

print("Baseline missclass labels", len(baseline_missclass_labels))
print("Harris missclass labels", len(harris_missclass_labels))

for i, cls in enumerate(baseline_missclass_labels):
    print("Baseline missclass", baseline_missclass_labels[i])
    print("Harris missclass", harris_missclass_labels[i])
    print()

# Calculate average misclassification rate for Model 1
baseline_model_rate = np.mean(baseline_missclass)

# Calculate average misclassification rate for Model 2
harris_model_rate = np.mean(harris_missclass)

# Print the results
print(f"Average Misclassification Rate for Baseline: {baseline_model_rate:.4f}")
print(f"Average Misclassification Rate for Harris: {harris_model_rate:.4f}")

class_indices = np.arange(num_classes)  # Indices for the classes

# Width of the bars
bar_width = 0.35

# Plotting the bars
fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figsize as needed
bars_model1 = ax.bar(class_indices - bar_width/2, baseline_missclass, bar_width, label='RGB Baseline')
bars_model2 = ax.bar(class_indices + bar_width/2, harris_missclass, bar_width, label='Grayscale + Harris-Corner Channel')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# Adding labels, title, and legend
#ax.set_xlabel('Classes & Most misclassified class pair')
ax.set_xlabel('T-LESS Classes', fontsize=14)
ax.set_ylabel('Misclassification Rate', fontsize=14)
#ax.set_title('Misclassification Rates Comparison between Model 1 and Model 2')
ax.set_xticks(class_indices)
#ax.set_xticklabels([f'Class {i}' for i in range(num_classes)])
ax.set_xticklabels([f'Class {class_index + 1}: {baseline_missclass_labels[class_index] + 1}\nClass {class_index + 1}: {harris_missclass_labels[class_index] + 1}' for class_index in range(num_classes)], rotation='vertical')
# ax.set_xticklabels([f'Class {class_index + 1}' for class_index in range(num_classes)], rotation='vertical')

# add misclassification rate formula
formula_text = ("Missclassification Rate" "\n" r"$R_{ij}=\frac{FP_{ij}+FN_{ij}}{N_i}$")
plt.text(5.5, max(baseline_missclass) * 1.05 - 0.07, formula_text,
         horizontalalignment='center', fontsize=14, bbox=dict(facecolor='white', alpha=0.5, pad=10))
ax.legend()

# Adding value labels on top of each bar
# def autolabel(bars):
#     for bar in bars:
#         height = bar.get_height()
#         ax.annotate(f'{height:.2f}',
#                     xy=(bar.get_x() + bar.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom')

# autolabel(bars_model1)
# autolabel(bars_model2)

plt.tight_layout()
plt.show()

    # plt.figure(figsize=(8, 6))
    # sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")

    # # Adding labels
    # plt.title("Confusion Matrix")
    # plt.xlabel("Predicted Label")
    # plt.ylabel("True Label")
    # # there are 30 classes
    # classes = [str(i) for i in range(31)]
    # # plt.xticks(np.arange(3) + 0.5, classes)
    # # plt.yticks(np.arange(3) + 0.5, classes)

    # # Display the plot
    # plt.show()
    # import sys;sys.exit(0)


    # # Plotting the misclassification rates
    # classes = [f'{i}' for i in range(30)]

    # plt.figure(figsize=(10, 6))
    # plt.bar(classes, misclassification_rates, color='skyblue')
    # plt.xlabel('Class')
    # plt.ylabel('Misclassification Rate')
    # plt.title('Misclassification Rates for Each Class')
    # plt.ylim(0, max(misclassification_rates) * 1.1)  # Set y-axis limit to a bit higher than the max value    plt.grid(axis='y')
    # # Adding the data labels on the bars
    # for i, rate in enumerate(misclassification_rates):
    #     plt.text(i, rate + 0.02, f'{rate:.2f}', ha='center')
    # plt.show()
    # if j == 2:
    #     break