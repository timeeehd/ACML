# Authors: Omendra Manhar i6131589, Jacob Salam i6184256
import numpy
import random
import math
import pprint
import copy

##
## BACKPROPAGATION ALGORITHM IMPLEMENTATION
##

pp = pprint.PrettyPrinter(indent=4)


def sigmoid_function(x):
    return 1 / (1 + math.exp(-x))


# Formulate inputs
inputs = []

for x in range(8):
    temp = []
    for y in range(8):
        if x == y:
            temp.append(1)
        else:
            temp.append(0)
    inputs.append(temp)
print(inputs)
# The learning rate and decay parameter can be changed here.
learning_rate = 0.9
lambda_value = 0.0002
weights_l1_l2 = []
weights_l2_l3 = []
b_l1 = []
b_l2 = []
b_inp = [1 for i in range(8)]
b_hidden = [1 for i in range(3)]

for x in range(3):
    b_l1.append(random.uniform(0, 0.0001))

for x in range(8):
    b_l2.append(random.uniform(0, 0.0001))

for x in range(8):
    temp_1 = []
    for y in range(3):
        temp_1.append(random.uniform(0, 0.0001))
    weights_l1_l2.append(temp_1)

for y in range(3):
    temp_1 = []
    for x in range(8):
        temp_1.append(random.uniform(0, 0.0001))
    weights_l2_l3.append(temp_1)


def feedforward(inp, W1, W2, b1, b2):
    # Feed Forward
    # hidden layer
    if (type(W1) != type([1, 2])):
        W1 = W1.tolist()

    if (type(W2) != type([1, 2])):
        W2 = W2.tolist()

    if (type(inp) != type([1, 2])):
        inp = inp.tolist()

    if (type(b2) != type([1, 2])):
        b2 = b2.tolist()

    if (type(b1) != type([1, 2])):
        b1 = b1.tolist()

    h = [sigmoid_function(x + y) for x, y in zip(b1, numpy.dot(inp, W1).tolist())]

    # Output layer
    o = [sigmoid_function(x + y) for x, y in zip(numpy.dot(h, W2), b2)]

    return o, h


def backpropagation(input, W1, W2, b1, b2):
    o, h = feedforward(input, W1, W2, b1, b2)
    delta = [-1 * (y - a) * a * (1 - a) for y, a in zip(input, o)]
    delta_hidden = numpy.dot(numpy.array(W2), numpy.array(delta).transpose())
    delta_hidden = [a * (1 - a) * x for x, a in zip(delta_hidden, h)]
    # print(W2)
    # print(h)
    # print(o)
    # print(delta)
    # print(delta_hidden)
    # print(delta)
    # print(numpy.array(delta).reshape(8, 1))
    # print(h)
    # print(numpy.array(h).reshape(1, 3))
    pd_W = numpy.dot(numpy.array(delta).reshape(8, 1), numpy.array(h).reshape(1, 3))
    # print(pd_W.transpose())
    pd_B = delta
    pd_W_H = numpy.dot(numpy.array(input).reshape(8, 1), numpy.array(delta_hidden).reshape(1, 3))
    pd_B_H = delta_hidden

    # print(pd_W_H)

    return pd_W.transpose(), pd_B, pd_W_H, pd_B_H

# backpropagation([1,0,0,0,0,0,0,0], weights_l1_l2, weights_l2_l3, b_l1, b_l2)

def update_parameters(input, W1, W2, b1, b2):
    d_W = numpy.array([0 for i in range(24)]).reshape(3, 8)
    d_B = numpy.array([0 for i in range(8)]).reshape(1, 8)
    d_W_H = numpy.array([0 for i in range(24)]).reshape(8, 3)
    d_B_H = numpy.array([0 for i in range(3)]).reshape(1, 3)



    for i in input:
        pd_W, pd_B, pd_W_H, pd_B_H = backpropagation(i, W1, W2, b1, b2)
        # print(pd_W)
        d_W = d_W + numpy.array(pd_W)
        d_B = d_B + numpy.array(pd_B)
        d_W_H = d_W_H + numpy.array(pd_W_H)
        d_B_H = d_B_H + numpy.array(pd_B_H)
    # print(f'dw {d_W} ')
    W1 = W1 - learning_rate * ((1 / len(input)) * d_W_H + lambda_value * numpy.array(W1))
    W2 = W2 - learning_rate * ((1 / len(input)) * d_W + lambda_value * numpy.array(W2))
    b1 = numpy.array(b1) - learning_rate * ((1 / len(input)) * numpy.array(d_B_H[0]).transpose())
    b2 = numpy.array(b2) - learning_rate * ((1 / len(input)) * numpy.array(d_B[0]).transpose())

    # print(f'b2 {b2}')
    # print(f'b1 {b1}')
    # print(f'w1 {W1}')
    # print(f'w2 {W2}')

    return W1, W2, b1, b2

update_parameters([[1,0,0,0,0,0,0,0]], weights_l1_l2, weights_l2_l3, b_l1, b_l2)

#
#
def gradient_descent(input, W1, W2, b1, b2):
    count = 0
    correct_count = 0
    # training
    # while (correct_count != len(input)):
    while(count < 10000):
        count = count + 1
        correct_count = 0
        # The number of inputs to be learned on can be changed here by adapting input[:8] for example:
        # first 3 and last 2: input[:3]+ input[6:8]
        W1, W2, b1, b2 = update_parameters(input[0:8], W1, W2, b1, b2)
        for i in input:
            o, h = feedforward(i, W1, W2, b1, b2)
            if i.index(max(i)) == o.index(max(o)):
                correct_count = correct_count + 1
        if (count % 10000 == 0):
            print(count)
            for i in input:
                o, h = feedforward(i, W1, W2, b1, b2)
                print("Expected - ", i.index(max(i)), " Actual - ", o.index(max(o)))
    print("Convergence reached after", count, "iterations for a Learning rate of", learning_rate, "and a Lambda of",
          lambda_value)
    for i in input:
        o, h = feedforward(i, W1, W2, b1, b2)
        output = [0 for i in range(8)]
        output[o.index(max(o))] = 1
        print("input", i, "  output", o, " output2",  output)


gradient_descent(inputs, weights_l1_l2, weights_l2_l3, b_l1, b_l2)
