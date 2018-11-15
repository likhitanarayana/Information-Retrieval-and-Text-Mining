from os import listdir
from os.path import isfile, join
import os
import sys
import pandas as pd
import string
import numpy as np
import pickle


class Document:
    def __init__(self):
        self.length = -1
        self.words = dict()

    def __repr__(self):
        return 'length = {}\n words = {}'.format(self.length, self.words)


class Word_Information:
    def __init__(self, tf, doc_frequency):
        self.term_frequency = tf # number of times words shows up in this document
        self.document_frequency = doc_frequency # number of documents the term shows up

    def __repr__(self):
        return 'term frequency = {} and doc frequency = {}'.format(self.term_frequency, self.document_frequency)



def tf_idf(root, stop_list_file=None):
    # stop words list
    if stop_list_file is not None:
        stop_list = []
        with open(stop_list_file) as fp:
            for count, line in enumerate(fp):
                #print("stop_word = {}".format(line))
                #print("isalpha = {}".format(line.isalpha))
                line = line.strip('\n')
                if len(line) != 0:
                    stop_list.append(line)

    if stop_list is not None:
        print("stop word = {}".format(stop_list))

    doc_list = dict()

    ground_truth = pd.DataFrame(columns = ['file_name', 'author'])
    author_folders = [f for f in listdir(root) if not isfile(join(root, f))]
    files = []
    counter = 0
    for folder in author_folders:
        print("folder = {}".format(folder))
        print("folder # = {}".format(counter))
        new_folder = root +'/'+folder
        folder_files = [f for f in listdir(new_folder) if isfile(join(new_folder, f))]
        files.extend(folder_files)
        for f in folder_files:
            #counter += 1
            print("file = {}".format(f))
            ground_truth.loc[len(ground_truth)] = [f, folder]

            curr_file = open(new_folder+"/"+f, 'r')
            text = curr_file.read()
            #print("text = {}".format(text))
            f_words = [word.strip(string.punctuation) for word in text.split()]
            f_words = [x.lower() for x in f_words]
            f_words = [word.replace(",", "") for word in f_words]
            f_words = [word.replace(".", "") for word in f_words]
            f_words = [word.replace("-", "") for word in f_words]

            doc_unique_words = list()
            #print("f-words = {}".format(f_words))
            doc_info = Document()
            length = 0
            for word in f_words:
                if word not in stop_list:
                    #print("word = {}".format(word))
                    length += 1
                    word_information = doc_info.words.get(word, None)
                    #print("word information = {}".format(word_information))
                    if word_information is None:
                        # does not in document dictionary yet
                        doc_info.words[word] = Word_Information(1, 1)
                    else:
                        # update word term frequency
                        word_information.term_frequency += 1
                        doc_info.words[word] = word_information # term frequency
                    doc_unique_words.append(word)
            doc_info.length = length
            print("##doc_info = {}".format(doc_info))

            # go through and update other documents frequencies

            doc_unique_words = set(doc_unique_words)

            #print("doc unique words = {}".format(doc_unique_words))
            for word in doc_unique_words:
                #print("word = {}".format(word))
                # loop through all documents
                num_found_in_other_documents = 0
                #print("doc list = {}".format(doc_list))
                for doc in doc_list:
                    #print("doc = {}".format(doc))
                    #print("document = {}".format(doc_list[doc]))
                    doc_words = doc_list[doc]
                    if word in doc_words.words:
                        #print("original doc frequency = {}".format(doc_words.words[word].document_frequency))
                        doc_words.words[word].document_frequency += 1
                        #print("after doc frequency = {}".format(doc_words.words[word].document_frequency))
                        num_found_in_other_documents += 1
                #update document frequency for word in current document
                #print("obj = {}".format(doc_info.words))
                (doc_info.words[word]).document_frequency += num_found_in_other_documents

            doc_list[f] = doc_info

            if counter <= 2:
                break

    with open('objs.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(doc_list, f)


    ground_truth.to_csv("ground_truth.csv")

    return doc_list


def test_reading_in_pickle(random):
    with open('objs.pkl', 'rb') as f:
        doc_vector = pickle.load(f)
        #print("doc vector = {}".format(doc_vector))

    for key in doc_vector:
        print("key = {}".format(key))
        print("value = {}".format(doc_vector[key]))


#tf_idf(sys.argv[1], sys.argv[2])
test_reading_in_pickle(sys.argv[1])
