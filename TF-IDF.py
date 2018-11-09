from os import listdir
from os.path import isfile, join
import os
import sys
import pandas as pd


def tf_idf(root, output_file):
    ground_truth = pd.DataFrame(columns = ['file_name', 'author'])
    author_folders = [f for f in listdir(root) if not isfile(join(root, f))]
    files = []
    for folder in author_folders:
        print("folder = {}".format(folder))
        new_folder = root +'/'+folder
        folder_files = [f for f in listdir(new_folder) if isfile(join(new_folder, f))]
        files.extend(folder_files)
        for f in folder_files:
            ground_truth.loc[len(ground_truth)] = [f, folder]

    print("ground_truth = {}".format(ground_truth))
    print('files = {}'.format(files))
    ground_truth.to_csv("ground_truth.csv")


tf_idf(sys.argv[1], sys.argv[2])