
import matplotlib.pyplot as plt
import numpy as np

train_o = [18,27,10]
train_u = [13,24,11]
test_o = [4,3,2]
test_u = [6,3,4]

# sum values by index
train = [train_o[i] + train_u[i] for i in range(len(train_o))]
test = [test_o[i] + test_u[i] for i in range(len(test_o))]

labels = ["drown", "swim", "idle"]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(8,6))
rects1 = ax.bar(x - width/2, train, width, label='Training')
rects2 = ax.bar(x + width/2, test, width, label='Testing')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Number of videos')
ax.set_title('Training data')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()

plt.show()
