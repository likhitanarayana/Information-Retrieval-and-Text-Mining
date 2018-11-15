import sys
import operator

def groundTruthSplit(fp):
   Ground = {}
   lines = fp.readlines()
   for line in lines[1:]:
      newline = line.strip().split(",")
      Ground[newline[1]] = newline[2]
   return Ground

def knnOutcomeSplit(fp):
   KNN = {}
   lines = fp.readlines()
   for line in lines:
      newline = line.strip().split(",")
      KNN[newline[0]] = newline[1:]
   return KNN

def knnEvaluation(fileKnn, fileGround):
   KNN = {}
   confusionMatrix= []
   Ground = {}
   totalCorr = 0.0
   totalIncorr = 0.0
   fp1 = open(fileKnn, "r")
   fp2 = open(fileGround, "r")
   KNN = knnOutcomeSplit(fp1)
   Ground = groundTruthSplit(fp2)
   Authors = {}
   for row in Ground:
      Authors[Ground[row]] = [0, 0, 0, 0]
   for ind in KNN:
      Classes = {}
      KNearest = KNN[ind]
      for row in KNearest:
         author = Ground[row]
         if author in Classes:
            Classes[author] = Classes[author] + 1
         else:
            Classes[author] = 1
      maxy = 0
      classy = ""
      for row in Classes:
         if Classes[row] > maxy:
            maxy = Classes[row]
            classy = row
      if Ground[ind] == classy:
         totalCorr += 1
         Authors[classy][0] = Authors[classy][0] + 1
         for row in Authors:
            if row != classy:
               Authors[row][3] = Authors[row][3] + 1
      else:
         totalIncorr +=1
         Authors[classy][1] = Authors[classy][1] + 1
         Authors[Ground[ind]][2] = Authors[Ground[ind]][2] + 1
         for row in Authors:
            if row != classy and row != Ground[ind]:
               Authors[row][3] = Authors[row][3] + 1
   for row in Authors:
      Recall = (float(Authors[row][0]))/(Authors[row][0] + Authors[row][2])
      Precision = (float(Authors[row][0]))/(Authors[row][0] + Authors[row][1])
      F_measure = ((2 * Precision  * Recall)/(Precision + Recall))
      print("Author : {}, Hits : {}, Strikes : {}, Misses : {}".format(row, Authors[row][0], Authors[row][1], Authors[row][2]))
      print("   Precision : {}, Recall : {}, F-measure : {}".format(Precision, Recall, F_measure))

   print("Correctly Predicted : {}, Incorrectly Predicted : {}, Overall Accuracy : {}".format(totalCorr, totalIncorr, totalCorr/(totalIncorr + totalCorr)))

      
   

knnEvaluation(sys.argv[1], sys.argv[2])
