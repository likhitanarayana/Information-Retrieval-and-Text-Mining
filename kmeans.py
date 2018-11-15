import math
import pickle
import numpy as np
import sys
from textVectorizer import *
import random


def diskKMeans(D, k):
    # print("len of D orig {}".format(len(D)))
    m = random_initial_centroids(D, k)
    # print("init centroids = {}".format(m))
    s = []
    num = []
    cl = []
    old_m = []
    stopCondition = False
    while stopCondition is False:
        for j in range(0, k):
            # print("j = {}".format(j))
            s.append([])  # s.append(list(np.zeros(len(D)))) # s[] - family of vectors of size dim(D)
            num.append(0)  # num[] - number of points in each cluster
            cl.append([])  # cl[] - actual clusters
        # print("init s = {}".format(s))
        for x in D:
            print("x = {}".format(x))
            cluster = assign_cluster(x, m)
            # print("cluster = {}".format(cluster))
            cl[cluster].append(x)
            s[cluster].append(x)
            # print("s = {}".format(s))
            num[cluster] += 1
        old_m = m
        for j in range(0, k):
            # print("\n\nj = {}".format(j))
            # print("num = {}".format(num))
            # print("s = {}".format(s[j]))
            # print("len of s = {}".format(len(s)))
            # print("old m = {}".format(m))
            if len(s[j]) != 0:
                m[j] = [float(sum(col)) / len(col) for col in zip(*s[j])]
            else:
                m[j] = old_m[j]
            # print("m[j] = {}".format(m[j]))
        # print("new centroids = {}".format(m))
        stopCondition = is_stopping_condition(m, old_m)
        # print("stopCondition = {}".format(stopCondition))

    # print("\n\n")
    return cl, m


def mean(list):
    return sum(list) / len(list)


def is_stopping_condition(m, old_m):
    # print("m = {}".format(m))
    # print("old_m = {}".format(old_m))
    for i in range(0, len(m)):
        for j in range(0, len(m[i])):
            if m[i][j] - old_m[i][j] > 0.3:
                return False
    return True


def assign_cluster(x, clusters):
    # print("\nin assign_cluster")
    shortest_distance = float('inf')
    cluster = -1
    for i in range(0, len(clusters)):
        # print("i = {}".format(i))
        distance_to_cluster = 0
        for attribute in range(0, len(x)):
            distance_to_cluster += np.square(float(x[attribute]) - float(clusters[i][attribute]))
        distance = np.sqrt(distance_to_cluster)
        # print("distance = {}".format(distance))
        if distance < shortest_distance:
            shortest_distance = distance
            cluster = i
            # print("cluster = {}".format(i))

    return cluster


def random_initial_centroids(D, k):
    centers = dict()
    #for i in range(0, k):
    #    rand = randint(0, len(D) - 1)
    #    centers.append(D[rand])
    keys = random.sample(list(D), k)
    for key in keys:
        centers[key] = D[key]
        D.pop(key)
    print("centers = {}".format(centers))
    return centers


def calc_stats(cluster, center):
    max_dist = 0
    min_dist = float('inf')
    average = 0
    distances = []
    for i in cluster:
        if len(i) != 0:
            dist = 0
            for attribute in range(0, len(i)):
                dist += np.square(float(i[attribute]) - float(center[attribute]))
            distance = np.sqrt(dist)
            # do i square root distance for sse?
            distances.append(distance)
            average += distance
            if distance > max_dist:
                max_dist = distance
            if distance < min_dist:
                min_dist = distance
        average = average / len(cluster)

    # calculate SSE
    sse = 0
    for i in distances:
        sse += np.square(i - average)

    return max_dist, min_dist, average, sse


def runkMeans(fileName, k):
    D = []
    #fp = open(fileName, "r")
    #D = convertToMatrix(fp)
    with open(fileName, 'rb') as f:
        doc_vector = pickle.load(f)
    clusters, ms = diskKMeans(doc_vector, int(k))
    for i in range(0, len(clusters)):
        if len(clusters[i]) != 0:
            max, min, average, sse = calc_stats(clusters[i], ms[i])
        else:
            max = min = average = sse = 0
        print("Cluster {}: ".format(i))
        print("\tCentroid = {}".format(ms[i]))
        print("\tMax Distance to Center = {}".format(max))
        print("\tMin Distance to Center = {}".format(min))
        print("\tAverage Distance to Center = {}".format(min))
        print("\tCluster SSE = {}".format(sse))
        print("\tAll points in this Cluster = {}".format(len(clusters[i])))
        print("\tAll rows in this cluster: ")
        for j in clusters[i]:
            print("\t\t{}".format(j))

runkMeans(sys.argv[1], sys.argv[2])