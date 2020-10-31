import random
from math import exp

weights_l2_l3 = []
weights_l1_l2 = []

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

# print(weights_l1_l2)

bias_l1 = [random.uniform(0, 0.01)]
bias_l2 = [random.uniform(0, 0.01)]

learning_rate = 0.9
weight_decay = 0.01


def sigmoid(x):
    return 1 / (1 + exp(-x))


def forward_propagate(input, weight_l1_l2, weight_l2_l3, b1, b2):
    activation_hidden = []
    for i in range(3):
        temporary_activation = b1[0]
        for j in range(8):
            temporary_activation += weight_l1_l2[j][i] * input[j]
        activation_hidden.append(sigmoid(temporary_activation))

    activation_output = []
    for i in range(8):
        temporary_activation = b2[0]
        for j in range(3):
            temporary_activation += weight_l2_l3[j][i] * activation_hidden[j]
        activation_output.append(sigmoid(temporary_activation))

    return activation_hidden, activation_output

# forward_propagate([1, 0, 0, 0, 0, 0, 0, 0], weights_l1_l2, weights_l2_l3, bias_l1, bias_l2)


def backpropagation(input, weight_l1_l2, weight_l2_l3, b1, b2):
    activation_hidden, activation_output = forward_propagate(input, weight_l1_l2, weight_l2_l3, b1, b2)
    delta_output = []
    for x in range(8):
        delta_output.append((activation_output[x] - input[x])*activation_output[x]*(1-activation_output[x]))

    delta_hidden = []
    for x in range(3):
        temporary_error = 0
        for y in range(8):
            temporary_error += weight_l2_l3[x][y] * delta_output[y]
        temporary_error = temporary_error * (activation_hidden[x] * (1-activation_hidden[x]))
        delta_hidden.append(temporary_error)

    # print(weight_l2_l3)
    #
    # print(activation_hidden)
    # print(activation_output)
    # print(delta_output)
    # print(delta_hidden)
    partial_derivative_b_output = delta_output
    partial_derivative_b_hidden = delta_hidden
    #
    partial_derivative_weight_output = []
    for x in range(3):
        temporary_list = []
        for y in range(8):
            temporary_list.append(delta_output[y]*activation_hidden[x])
        partial_derivative_weight_output.append(temporary_list)
    # print(partial_derivative_weight_output)
    partial_derivative_weight_hidden = []
    for x in range(8):
        temporary_list = []
        for y in range(3):
            temporary_list.append(input[x]*delta_hidden[y])
        partial_derivative_weight_hidden.append(temporary_list)

    # print(partial_derivative_weight_hidden)
    return partial_derivative_b_output, partial_derivative_b_hidden, partial_derivative_weight_output, partial_derivative_weight_hidden

# backpropagation([1,0,0,0,0,0,0,0], weights_l1_l2, weights_l2_l3, bias_l1, bias_l2)


