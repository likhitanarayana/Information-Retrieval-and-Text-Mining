from os import listdir
from os.path import isfile, join
import os
import sys
import pandas as pd
import string
import numpy as np


def individual_tf_idf(document_name):
    """
    Given the entire tf-idf csv file, only output the rows which contain words in the document asked for
    :param document_name: Name of document which you want sparse vector representation of
    :return:
    """
    # calculate tf-idf * tf for each word and turn them to an list for each document given and use numpy's dot product
    raise NotImplementedError



def tf_idf(root, output_file):
    ground_truth = pd.DataFrame(columns = ['file_name', 'author'])
    tf = pd.DataFrame(columns = ['word'])
    author_folders = [f for f in listdir(root) if not isfile(join(root, f))]
    files = []
    counter = 0
    for folder in author_folders:
        print("folder = {}".format(folder))
        new_folder = root +'/'+folder
        folder_files = [f for f in listdir(new_folder) if isfile(join(new_folder, f))]
        files.extend(folder_files)
        for f in folder_files:
            counter += 1
            print("file = {}".format(f))
            ground_truth.loc[len(ground_truth)] = [f, folder]
            f_words = []
            with open(new_folder+"/"+f) as open_f:
                for line in open_f:
                    f_words.extend([word for word in line.strip(string.punctuation).split()])

            # print("f words = {}".format(f_words))
            tf[f] = 0
            for word in f_words:
                row_index = tf.loc[tf['word']==word].index
                if len(row_index) != 0: # word already exists in dataframe
                    tf.ix[row_index, f] += 1
                else: # word does not exist yet in dataframe
                    empty_list = [0] * len(tf.columns)
                    tf.loc[len(tf)] = empty_list
                    tf['word'][len(tf)-1] = word
                    tf[f][len(tf)-1] = 1
            tf[f+'_tf'] = tf[f]*1.0 / len(f_words)
            tf = tf.drop(f, 1)
            print("tf")
            print(tf)
            if counter >= 2:
                break


    tf['idf'] = np.log(len(files)*1.0/tf.eq(0).sum(axis=1))
    for file in files:
        file_str = file+'_tf'
        new_file_col = file+'_tf-idf'
        df_col = tf[[file_str]]
        print("df col = ")
        print(df_col)
        print("tf.idf ")
        print(tf[['idf']])
        print("mult")
        print(tf[[file_str]]*tf.idf)
        print("idf type = {}".format(type(tf[['idf']])))
        print("df type = {}".format(type(tf[[file_str]])))
        tf[new_file_col] = tf[[file_str]]*tf[['idf']]

    print("tf dataframe")
    print(tf)
    ground_truth.to_csv("ground_truth.csv")
    tf.to_csv("tf-idf.csv")


tf_idf(sys.argv[1], sys.argv[2])