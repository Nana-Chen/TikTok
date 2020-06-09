#!use/bin/env python
# -*- coding:utf-8 _*-
"""
@author : chenmeiyi
@file : FCM.py
@time : 2020/06/09
@desc : 
"""
import copy
import math
import random
import time

global MAX  # 用于初始化隶属度矩阵U
MAX = 10000.0

global Epsilon  # 结束条件
Epsilon = 0.0000001


def print_matrix(list):
    for i in range(0, len(list)):
        print(list[i])


def initialize_U(data, cluster_number):
    global MAX
    U = []
    for i in range(0, len(data)):
        current = []
        rand_sum = 0.0
        for j in range(0, cluster_number):
            dummy = random.randint(1, int(MAX))
            current.append(dummy)
            rand_sum += dummy
        for j in range(0, cluster_number):
            current[j] = current[j] / rand_sum
        U.append(current)
    return U


def distance(point, center):
    if len(point) != len(center):
        return -1
    dummy = 0.0
    for i in range(0, len(point)):
        dummy += abs(point[i] - center[i]) ** 2
    return math.sqrt(dummy)


def end_conditon(U, U_old):
    global Epsilon
    for i in range(0, len(U)):
        for j in range(0, len(U[0])):
            if abs(U[i][j] - U_old[i][j]) > Epsilon:
                return False
    return True


def normalise_U(U):
    for i in range(0, len(U)):
        maximum = max(U[i])
        for j in range(0, len(U[0])):
            if U[i][j] != maximum:
                U[i][j] = 0
            else:
                U[i][j] = 1
    return U


def fuzzy(data, cluster_number, m):
    U = initialize_U(data, cluster_number)
    while (True):
        U_old = copy.deepcopy(U)
        C = []
        for j in range(0, cluster_number):
            current_cluster_center = []
            for i in range(0, len(data[0])):
                dummy_sum_num = 0.0
                dummy_sum_dum = 0.0
                for k in range(0, len(data)):
                    dummy_sum_num += (U[k][j] ** m) * data[k][i]
                    dummy_sum_dum += (U[k][j] ** m)
                current_cluster_center.append(dummy_sum_num / dummy_sum_dum)
            C.append(current_cluster_center)

        distance_matrix = []
        for i in range(0, len(data)):
            current = []
            for j in range(0, cluster_number):
                current.append(distance(data[i], C[j]))
            distance_matrix.append(current)

        for j in range(0, cluster_number):
            for i in range(0, len(data)):
                dummy = 0.0
                for k in range(0, cluster_number):
                    dummy += (distance_matrix[i][j] / distance_matrix[i][k]) ** (2 / (m - 1))
                U[i][j] = 1 / dummy

        if end_conditon(U, U_old):
            print("end")
            break

    U = normalise_U(U)
    return U


if __name__ == '__main__':
    data = pd.read_csv('/Users/chenmeiyi/Desktop/数据/housing.csv', header=0, delim_whitespace=True)
    data = array.data
    start = time.time()
    final_location = fuzzy(data, 5, 2)

    final_location = de_randomise_data(final_location, order)

    print(checker_iris(final_location))
    print("用时：{0}".format(time.time() - start))