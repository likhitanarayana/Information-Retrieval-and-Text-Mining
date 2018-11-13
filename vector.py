#import pandas as pd
import numpy as np
import csv

def cosineSimilarity(vector1, vector2):
   weightsOfVector1 = 0.0
   weightsOfVector2 = 0.0
   weightMultiplication = 0.0
   startingIndex = 1
   for x in range(1, len(vector1)):
      weightsOfVector1 += np.square(vector1[x][1])
      for y in range(startingIndex, len(vector2)):
         if vector1[x][0] == vector2[y][0]:
            weightMultiplication += vector1[x][1] * vector2[y][1]
            startingIndex = y + 1
            break
   for y in range(1, len(vector2)):
      weightsOfVector2 += np.square(vector2[y][1])
   return weightMultiplication / (np.sqrt(weightsOfVector1*weightsOfVector2))

def convertToMatrix(fp):
   D = []
   lines = fp.readlines()
   #takes out the id and the word
   D.append((lines[0].strip().split(","))[2:])
   #index4 = 0
   for line in lines[1:]:
      row = (line.strip().split(','))[2:]
      for index, elem in enumerate(row):
         row[index] = float(elem)         
      #index4 += 1
      #print("{} , {}".format(index4, row))
      D.append(row)
   invertedD = []
   indexIDF = 0
   for index, elem in enumerate(D[0]):
      if elem == "idf":
         indexIDF = index
         break

   for index, elem in enumerate(D[0]):
      if index > indexIDF:
         newRow = []
         newRow.append(elem)
         for index3, row in enumerate(D[1:]):
            for index2, elem2 in enumerate(row):
               if index2 == index and elem2 != 0.0:
                  newRow.append((index3, elem2))
         invertedD.append(newRow)
         
   return invertedD
