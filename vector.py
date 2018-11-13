#import pandas as pd
import numpy as np
import csv

def cosineSimilarity(vector1, vector2):
   weightsOfVector1 = 0.0
   weightsOfVector2 = 0.0
   weightMultiplication = 0.0
   for index, element in enumerate(vector1[1:]):
      elementOfVector2 = vector2[index + 1]
      weightsOfVector1 += np.square(element)
      weightsOfVector2 += np.square(elementOfVector2)
      weightMultiplication += element * elementofVector2
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
   for index, elem in enumerate(D[0]):
      newRow = []
      newRow.append(elem)
      for row in D[1:]:
         for index2, elem2 in enumerate(row):
            if index2 == index:
               newRow.append(elem2)
      invertedD.append(newRow)
         
   return invertedD
