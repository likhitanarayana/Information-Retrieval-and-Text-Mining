import pandas as pd
import numpy as np
import sys
from vector import cosineSimilarity
from vector import okapi
from vector import arrayWords
from vector import weightWords
from vector import avdl
from textVectorizer import *

def getKey(item):
   return item[0]

def computeKNN(index, D, k, flag, ArrayW, avd, WeightWords):
   KNearest = []
   dPoint = D[index]
   for key in D:
      if key != index:
         if flag == "cosine":
            sim = cosineSimilarity(dPoint, D[key], len(D), ArrayW, WeightWords[index], WeightWords[key])
         else: 
            sim = okapi(dPoint, D[key], len(D), avd)
         if len(KNearest) < k:
            KNearest.append((sim, key))
         else:
            KNearest = sorted(KNearest, key=getKey)
            for i,j in enumerate(KNearest):
               if j[0] < sim:
                  KNearest[i] = (sim, key)
                  break
   KNearest = sorted(KNearest, key=getKey)
   strin = ""
   for ind, row in enumerate(KNearest):
      if ind == len(KNearest) - 1:
         strin += str(row[1]) + str(row[0])
      else:
         strin +=  str(row[1]) + " " + str(row[0]) +  "," 
      
   print("{},{}".format(index, strin))

def KNN(vectorFile, k, flag):
   D = test_reading_in_pickle()
   k = int(k)
   ArrayW = arrayWords(D)
   avd = avdl(D)
   WeightWords = weightWords(D, ArrayW)
   for key in D:
       #print("key = {}".format(key))
       #print("value = {}".format(D[key].words))
       #print(len(D))
       computeKNN(key, D, k, flag, ArrayW, avd, WeightWords)

def test_reading_in_pickle():
    with open('objs.pkl', 'rb') as f:
        doc_vector = pickle.load(f)
        #print("doc vector = {}".format(doc_vector))

    return doc_vector
  
KNN(sys.argv[1], sys.argv[2], sys.argv[3])
