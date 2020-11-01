import numpy
import random
from math import exp

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


def forward_propagate(input, weight_l1_l2, weight_l2_l3, b1, b2):
    activation_hidden = [sigmoid(x + y) for x, y in zip(b1, numpy.dot(input, weight_l1_l2).tolist())]
    activation_output = [sigmoid(x + y) for x, y in zip(numpy.dot(activation_hidden, weight_l2_l3), b2)]

    return activation_hidden, activation_output


def backpropagation(input, weight_l1_l2, weight_l2_l3, b1, b2):
    activation_hidden, activation_output = forward_propagate(input, weight_l1_l2, weight_l2_l3, b1, b2)
    delta_output = [(a - y) * a * (1 - a) for y, a in zip(input, activation_output)]
    delta_hidden = numpy.dot(numpy.array(weight_l2_l3), numpy.array(delta_output).transpose())
    delta_hidden = [a * (1 - a) * x for x, a in zip(delta_hidden, activation_hidden)]

    par_der_w_l2_l3 = numpy.dot(numpy.array(delta_output).reshape(8, 1), numpy.array(activation_hidden).reshape(1, 3))
    par_der_b_l2 = delta_output
    par_der_w_l1_l2 = numpy.dot(numpy.array(input).reshape(8, 1), numpy.array(delta_hidden).reshape(1, 3))
    par_der_b_l1 = delta_hidden

    print(par_der_w_l2_l3.transpose())

    return par_der_w_l2_l3.transpose(), par_der_b_l2, par_der_w_l1_l2, par_der_b_l1


# backpropagation([1,0,0,0,0,0,0,0], weights_l1_l2, weights_l2_l3, bias_l1, bias_l2)

def update_weights(input, weight_l1_l2, weight_l2_l3, b1, b2):
    der_w_l2_l3 = numpy.array([0 for i in range(24)]).reshape(3, 8)
    der_b_l2 = numpy.array([0 for i in range(8)]).reshape(1, 8)
    der_w_l1_l2 = numpy.array([0 for i in range(24)]).reshape(8, 3)
    der_b_l1 = numpy.array([0 for i in range(3)]).reshape(1, 3)

    for row in input:
        par_der_w_l2_l3, par_der_b_l2, par_der_w_l1_l2, par_der_b_l1 = \
            backpropagation(row, weight_l1_l2, weight_l2_l3, b1, b2)
        der_w_l2_l3 = der_w_l2_l3 + numpy.array(par_der_w_l2_l3)
        der_b_l2 = der_b_l2 + numpy.array(par_der_b_l2)
        der_w_l1_l2 = der_w_l1_l2 + numpy.array(par_der_w_l1_l2)
        der_b_l1 = der_b_l1 + numpy.array(par_der_b_l1)

    weight_l1_l2 = weight_l1_l2 - learning_rate * \
                   ((1 / len(input)) * der_w_l1_l2 + weight_decay * numpy.array(weight_l1_l2))
    weight_l2_l3 = weight_l2_l3 - learning_rate * ((1/ len(input))
                                                   * der_w_l2_l3 + weight_decay * numpy.array(weight_l2_l3))
    b1 = numpy.array(b1) - learning_rate * ((1/len(input)) * numpy.array(der_b_l1[0]).transpose())
    b2 = numpy.array(b2) - learning_rate * ((1/len(input)) * numpy.array(der_b_l2[0]).transpose())

    return weight_l1_l2, weight_l2_l3, b1, b2


# def gradient_descent(input, weight_l1_l2, weight_l2_l3, b1, b2):
#     count = 0
#     current_count = 0
#     while (count < 7000) :
#         count += 1
#         current_count = 0
#         weight_l1_l2, weight_l2_l3, b1, b2 = update_weights(input, weight_l1_l2, weight_l2_l3, b1, b2)
#         for row in input:
#             activation_hidden, activation_output = forward_propagate(row, weight_l1_l2, weight_l2_l3, b1, b2)
#             # todo: add currentcount
#     for i in input:
#         h, o = forward_propagate(i, weight_l1_l2, weight_l2_l3, b1, b2)
#         output = [0 for i in range(8)]
#         output[o.index(max(o))] = 1
#         print("input", i, "  output", o, " output2",  output)


# gradient_descent(training_data, weights_l1_l2, weights_l2_l3, bias_l1, bias_l2)
