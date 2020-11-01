import random
from math import exp
import numpy

weights_l2_l3 = []
weights_l1_l2 = []

training_data = []
for i in range(8):
    temp_list = []
    for j in range(8):
        if i == j:
            temp_list.append(1)
        else:
            temp_list.append(0)
    training_data.append(temp_list)

for x in range(8):
    temporary_list = []
    for y in range(3):
        temporary_list.append((random.uniform(0, 0.01)))
    weights_l1_l2.append(temporary_list)

for x in range(3):
    temporary_list = []
    for y in range(8):
        temporary_list.append((random.uniform(0, 0.01)))
    weights_l2_l3.append(temporary_list)

bias_l1 = [random.uniform(0, 0.01) for x in range(3)]
bias_l2 = [random.uniform(0, 0.01) for x in range(8)]

learning_rate = 0.9
weight_decay = 0.0002


def sigmoid(x):
    return 1 / (1 + exp(-x))

def forward_propagate(input, w1, w2, b1, b2):
    activation_hidden = []
    for x in range(3):
        temp_value = b1[x]
        for y in range(8):
            temp_value += w1[y][x] * input[y]
        activation_hidden.append(sigmoid(temp_value))

    activation_output = []
    for x in range(8):
        temp_value = b2[x]
        for y in range(3):
            temp_value += w2[y][x] * activation_hidden[y]
        activation_output.append(sigmoid(temp_value))

    return activation_output, activation_hidden


def backpropagation(input, w1, w2, b1, b2):
    act_o, act_h = forward_propagate(input, w1, w2, b1, b2)
    delta_output = []
    for x in range(8):
        delta_output.append((act_o[x] - input[x]) * act_o[x] * (1 - act_o[x]))

    delta_hidden = []
    for x in range(3):
        temp_value = 0
        for y in range(8):
            temp_value += w2[x][y] * delta_output[y]
        temp_value = temp_value * act_h[x] * (1 - act_h[x])
        delta_hidden.append(temp_value)

    par_der_w2 = []
    for x in range(3):
        temp_list = []
        for y in range(8):
            temp_list.append(delta_output[y]*act_h[x])
        par_der_w2.append(temp_list)

    # print(par_der_w2)

    par_der_w1 = []
    for x in range(8):
        temp_list = []
        for y in range(3):
            temp_list.append(input[x] * delta_hidden[y])
        par_der_w1.append(temp_list)

    par_der_b_l2 = delta_output
    par_der_b_l1 = delta_hidden

    return par_der_w2, par_der_b_l2, par_der_w1, par_der_b_l1

def update_weights(input, w1, w2, b1, b2):
    der_w2 = []
    for x in range(3):
        temp_list = []
        for y in range(8):
            temp_list.append(0)
        der_w2.append(temp_list)

    der_w1 = []
    for x in range(8):
        temp_list = []
        for y in range(3):
            temp_list.append(0)
        der_w1.append(temp_list)

    der_b2 = []
    for x in range(8):
        der_b2.append(0)

    der_b1 = []
    for x in range(3):
        der_b1.append(0)

    for row in input:
        par_der_w2, par_der_b2, par_der_w1, par_der_b1 = backpropagation(row, w1, w2, b1, b2)
        for x in range(3):
            der_b1[x] += par_der_b1[x]

        for x in range(8):
            der_b2[x] += par_der_b2[x]

        for x in range(8):
            for y in range(3):
                der_w1[x][y] += par_der_w1[x][y]

        for x in range(3):
            for y in range(8):
                der_w2[x][y] += par_der_w2[x][y]

    for x in range(8):
        for y in range(3):
            w1[x][y] = w1[x][y] - learning_rate * ((1/len(input)) * der_w1[x][y] + weight_decay * w1[x][y])
    for x in range(3):
        for y in range(8):
            w2[x][y] = w2[x][y] - learning_rate * ((1/len(input)) * der_w2[x][y] + weight_decay * w2[x][y])

    for x in range(3):
        b1[x] = b1[x] - learning_rate * ((1/len(input)) * der_b1[x])

    for y in range(8):
        b2[y] = b2[y] - learning_rate * ((1/len(input)) * der_b2[y])

    return w1, w2, b1, b2

def gradient_descent(input, w1, w2, b1, b2):
    count = 0
    while count < 7000:
        count += 1
        w1, w2, b1, b2 = update_weights(input, w1, w2, b1, b2)
        for row in input:
            act_o, act_h = forward_propagate(row, w1, w2, b1, b2)
    for i in input:
        o, h = forward_propagate(i, w1, w2, b1, b2)
        output = [0 for i in range(8)]
        output[o.index(max(o))] = 1
        print("input", i, "  output", o, " output2",  output)


gradient_descent(training_data, weights_l1_l2, weights_l2_l3, bias_l1, bias_l2)

