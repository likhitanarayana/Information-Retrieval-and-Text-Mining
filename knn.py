import pandas as pd
import numpy as np
import sys
from vector import cosineSimilarity
from vector import convertToMatrix

def computeKNN():

def KNN(vectorFile, k, flag):
   D = []
   fp = open(vectorFile, "r")
   D = convertToMatrix(fp)
   for index, row in enumerate(D):
      print(row)
      computeKNN(index, k, flag)

  
KNN(sys.argv[1], sys.argv[2], sys.argv[3])
