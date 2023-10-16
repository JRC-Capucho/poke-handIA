import numpy as np
import random as rd
import math as ma


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


def mlp_test(nx, ny, nz, v, w, datas, classes):
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


def mlp_treino(nx, nz, ny, alpha, time, datas, classes):
    # layer input
    x = np.zeros(nx + 1, float)
    # layer hide
    z = np.zeros(nz + 1, float)
    # layer output
    y = np.zeros(ny, float)

    # layer input and hide
    v = np.zeros((nx + 1, nz), float)
    # layer hide and output
    w = np.zeros((nz + 1, ny), float)

    # Calc error in layer output
    dy = np.zeros(ny, float)
    # Calc error in layer hide
    dz = np.zeros(nz, float)

    # Step 1: generate 'weights' random
    for i in range(nx + 1):
        for j in range(nz):
            v[i][j] = rd.uniform(-1, 1)

    for i in range(nz + 1):
        for j in range(ny):
            w[i][j] = rd.uniform(-1, 1)

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
        for j in range(nz):
            z[j] = 0
            for i in range(nx + 1):
                z[j] += x[i] * v[i][j]
        z[nz] = 1

        for j in range(nz + 1):
            z[j] = -1 + 2 / (1 + ma.exp(-z[j]))

        y = 0
        for i in range(nz + 1):
            y += z[i] * w[i]

        y = -1 + 2 / (1 + ma.exp(-y))

        # Step 6
        if y >= 0:
            sr = +1
        else:
            sr = -1

        # Step 7 update weights if wrong
        if se != sr:
            # Step 8 erro da camada de saida
            dy = (se - sr) * (1 - y) * (1 + y) / 2

            for i in range(nz):
                # Propragar o erro para a camada oculta
                dz[i] = dy * w[i]
                # erro da camada oculta
                dz[i] = dz[i] * (1 - z[i]) * (1 + z[i]) / 2
            # atualizar pesos
            for i in range(nz + 1):
                w[i] += alpha * dy * z[i]

            for i in range(nx + 1):
                for j in range(nz):
                    v[i][j] += alpha * dz[j] * x[i]
            # Complete another lecture
    return v, w


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

# Attributes for perceptron
nx = len(datas[0])
nz = 7
ny = 1
alpha = 0.4
time = 800
# time = 25010

# Training
v, w = mlp_treino(nx, nz, ny, alpha, time, datas, classes)

print("Final Weights")
print(v)

print()
print(w)

# # Test
# prec = mlp_test(nx,nz, ny, v,w, datas ,classes)
# print("Precisao")
# print(prec)
