#import pandas as pd
import numpy as np
import csv

def cosineSimilarity(vector1, vector2, lenD, ArrayWords, weightsOfVector1, weightsOfVector2):
   weightMultiplication = 0.0
   for row in vector2.words:
      if row in vector1.words:
         dfi = vector1.words[row].document_frequency
         tfi = vector1.words[row].term_frequency
         tfq = vector2.words[row].term_frequency
         ArrW = ArrayWords[row]
         weightMultiplication += (np.log2(lenD/dfi)*(tfi/ArrW)) * (np.log2(lenD/dfi)*(tfq/ArrW))
   return weightMultiplication / (weightsOfVector1*weightsOfVector2)

def cosineSimilarityKMeans(vector1, vector2, lenD, ArrayWords):
   weightsOfVector1 = 0.0
   weightsOfVector2 = 0.0
   weightMultiplication = 0.0
   for row in vector1.words:
      weightsOfVector1 += vector1.words[row].term_frequency
   for row in vector2.words:
      tfq = vector2.words[row].term_frequency
      weightsOfVector2 += tfq
      if row in vector1.words:
         dfi = vector1.words[row].document_frequency
         tfi = vector1.words[row].term_frequency
         ArrW = ArrayWords[row]
         weightMultiplication += (np.log2(lenD/dfi)*(tfi/ArrW)) * (np.log2(lenD/dfi)*(tfq/ArrW))
   return weightMultiplication / (weightsOfVector1*weightsOfVector2)

def okapi(vector1, vector2, lenD, avdl):
   summ = 0.0
   for row in vector1.words:
      if row in vector2.words:
         dfi = vector1.words[row].document_frequency
         fij = vector1.words[row].term_frequency
         fiq = vector2.words[row].term_frequency
         k1 = 1.5
         k2 = 100
         summ += (np.log(lenD - dfi + .5) * ((k1 + 1) * fij / (k1 * (1 - .75 + .75*vector1.length/avdl) + fij)) * (((k2 + 1) * fiq)/(k2 + fiq)))
   return summ
         
def arrayWords(D):
   ArrayWords = {}
   for row in D:
      dRow = D[row]
      for word in dRow.words:
         df = dRow.words[word].term_frequency
         if word in ArrayWords:
            if df > ArrayWords[word]:
               ArrayWords[word] = df
         else:
            ArrayWords[word] = df
   return ArrayWords

def weightWords(D, ArrayWords):
   weightWords = {}
   lenD = len(D)
   for row in D:
      dRow = D[row]
      weightsOfVector1 = 0.0
      for word in dRow.words:
         weightsOfVector1 += np.log2(lenD/dRow.words[word].document_frequency)*((dRow.words[word].term_frequency)/ArrayWords[word])
      weightWords[row] = (np.sqrt(np.square(weightsOfVector1)))
   return weightWords

def avdl(D):
   avg = 0.0
   for row in D:
      avg += D[row].length
   return avg/len(D)

