import os

folder = '../dataset/train/tr_overhead'

#print all folders in the directory
for directory in os.listdir(folder):
    for file in os.listdir(os.path.join(folder, directory)):
        print(file)