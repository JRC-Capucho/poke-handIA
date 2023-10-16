import random as rd

import numpy as np


def perceptron_model(nx, ny, v, datas):
    # layer input
    x = np.zeros(nx + 1, float)
    # layer output
    y = np.zeros(ny, float)

    # Step 1 assing data in layer input
    for i in range(nx):
        x[i] = datas[i]
    x[nx] = 1

    # Step 3 propagar dados para a camada de saida
    y = 0
    for i in range(nx + 1):
        y += x[i] * v[i]

    # Step 4 aplica funcao de ativacao
    if y >= 0:
        classes = +1
    else:
        classes = -1

    return classes


def perceptron_test(nx, ny, v, datas, classes):
    # layer input
    x = np.zeros(nx + 1, float)
    # layer output
    y = np.zeros(ny, float)

    ac = 0

    # Step 2 select a data for training
    for row in range(len(datas)):
        # Step 1 assing data in layer input
        for i in range(nx):
            x[i] = datas[row][i]
        x[nx] = 1

        # Step 2 assing data in layer output
        se = classes[row]

        # Step 3 propagar dados para a camada de saida
        y = 0
        for i in range(nx + 1):
            y += x[i] * v[i]

        # Step 4 aplica funcao de ativacao
        if y >= 0:
            sr = +1
        else:
            sr = -1

        # Step 5 verify answer
        if se == sr:
            ac += 1

    return 100.0 * ac / len(datas)


def perceptron_training(nx, ny, alpha, time, datas, classes):
    # layer input
    x = np.zeros(nx + 1, float)
    # layer output
    y = np.zeros(ny, float)

    # layer input and output
    v = np.zeros((nx + 1, ny), float)

    # Step 1: generate 'weights' random
    for i in range(nx + 1):
        for j in range(ny):
            v[i][j] = rd.uniform(-1, 1)

    # Step 2 select a data for training
    for t in range(time):
        row = rd.randrange(len(datas))

        # Step 3 assing data in layer input
        for i in range(nx):
            x[i] = datas[row][i]
        x[nx] = 1

        # Step 4 assing data in layer output
        se = classes[row]

        # Step 5 propagar dados para a camada de saida
        y = 0
        for i in range(nx + 1):
            y += x[i] * v[i]

        # Step 6
        if y >= 0:
            sr = +1
        else:
            sr = -1

        # Step 6 update weights if wrong
        if se != sr:
            for i in range(nx + 1):
                v[i] += alpha * (se - sr) * x[i]

    return v


def read_file(file):
    f = open(file, "r")

    datas_str = []

    for row in f:
        content = row.strip("\n").split(",")
        datas_str.append(content)
    f.close()

    data = []
    classes = []

    for reg in datas_str:
        row = []
        for i in range(len(reg) - 1):
            row.append(int(reg[i]))
        data.append(row)
        if reg[len(reg) - 1] == "1":
            classes.append(1)
        else:
            classes.append(-1)

    return data, classes


# Leitura de dados treino
datas, classes = read_file("poker-hand-training-true.data")

# Leitura de test
datas_t, classes_t = read_file("poker-hand-testing.data")

# Attributes for perceptron
nx = len(datas[0])
ny = 1
alpha = 0.4
# time = 50
time = 25010

nxt = len(datas_t[0])

# Training
v = perceptron_training(nx, ny, alpha, time, datas, classes)

# vt = perceptron_training(nxt,ny,alpha, time ,datas_t,classes_t)

print("Final Weights")
print(v)

# Test
prec = perceptron_test(nx, ny, v, datas, classes)
# prect = perceptron_test(nxt, ny, vt, datas_t, classes_t)
print("Precisao")
print(prec)
