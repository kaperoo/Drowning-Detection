# Purpose: take labels_norm.pt and create a barchart of classes
import torch
import matplotlib.pyplot as plt

labels = torch.load("labels_norm.pt")
# print(labels.shape)
# print(labels)

graph_labels = ["drown", "swim", "misc", "idle"]
graph_values = [0, 0, 0, 0]

for label in labels:
    graph_values[label] += 1

plt.bar(graph_labels, graph_values)
plt.title("Distribution of frames by class")
# show the real values on the graph
for i, v in enumerate(graph_values):
    plt.text(i - 0.1, v + 0.1, str(v))

plt.show()
