import pandas as pd
import numpy as np
import sys
from vector import cosineSimilarity
from vector import convertToMatrix

def getKey(item):
   return item[0]

def computeKNN(D, index, k, flag):
   KNearest = []
   dPoint = D[index]
   for ind, row in enumerate(D):
      if ind != index:
         sim = cosineSimilarity(dPoint, row)
         if len(KNearest) < k:
            KNearest.append((sim, row[0]))
         else:
            KNearest = sorted(KNearest, key=getKey)
            for i,j in enumerate(KNearest):
               if j[0] < sim:
                  KNearest[i] = (sim, row[0])
                  break
   print("Document : {} KNearest : {}".format(dPoint[0], KNearest))


def KNN(vectorFile, k, flag):
   D = []
   k = int(k)
   fp = open(vectorFile, "r")
   D = convertToMatrix(fp)
   for index, row in enumerate(D):
      #print(row)
      computeKNN(D, index, k, flag)

  
KNN(sys.argv[1], sys.argv[2], sys.argv[3])
