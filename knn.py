import pandas as pd
import numpy as np
import sys
from vector import cosineSimilarity
from vector import convertToMatrix

def computeKNN(D, index, k, flag):
   KNearest = []
   dPoint = D[index]
   for ind, row in enumerate(D):
      if ind != index:
         sim = cosineSimilarity(dPoint, row)
         if len(KNearest) < k:
            KNearest.append((sim, D[index][0]))
         else:
            KNearest = sorted(KNearest)
            for i,j in enumerate(KNearest):
               if j[0] < sim:
                  KNearest[i] = (sim, D[index][0])
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
