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
    activation_hidden = []
    for i in range(3):
        temporary_activation = b1[i]
        for j in range(8):
            temporary_activation += weight_l1_l2[j][i] * input[j]
        activation_hidden.append(sigmoid(temporary_activation))

    activation_output = []
    for i in range(8):
        temporary_activation = b2[i]
        for j in range(3):
            temporary_activation += weight_l2_l3[j][i] * activation_hidden[j]
        activation_output.append(sigmoid(temporary_activation))

    return activation_hidden, activation_output


# forward_propagate([1, 0, 0, 0, 0, 0, 0, 0], weights_l1_l2, weights_l2_l3, bias_l1, bias_l2)


def backpropagation(input, weight_l1_l2, weight_l2_l3, b1, b2):
    activation_hidden, activation_output = forward_propagate(input, weight_l1_l2, weight_l2_l3, b1, b2)
    delta_output = []
    for x in range(8):
        delta_output.append((activation_output[x] - input[x]) * activation_output[x] * (1 - activation_output[x]))

    delta_hidden = []
    for x in range(3):
        temporary_error = 0
        for y in range(8):
            temporary_error += weight_l2_l3[x][y] * delta_output[y]
        temporary_error = temporary_error * (activation_hidden[x] * (1 - activation_hidden[x]))
        delta_hidden.append(temporary_error)

    # print(weight_l2_l3)
    #
    # print(activation_hidden)
    # print(activation_output)
    # print(delta_output)
    # print(delta_hidden)
    part_der_b_l2 = delta_output
    part_der_b_l1 = delta_hidden
    # print(part_der_b_l1)
    #
    par_der_weight_l2_l3 = []
    for x in range(3):
        temporary_list = []
        for y in range(8):
            temporary_list.append(delta_output[y] * activation_hidden[x])
        par_der_weight_l2_l3.append(temporary_list)
    # print(partial_derivative_weight_output)
    par_der_weight_l1_l2 = []
    for x in range(8):
        temporary_list = []
        for y in range(3):
            temporary_list.append(input[x] * delta_hidden[y])
        par_der_weight_l1_l2.append(temporary_list)

    # print(partial_derivative_weight_hidden)
    return part_der_b_l2, part_der_b_l1, par_der_weight_l2_l3, par_der_weight_l1_l2


def update_weights(input, weight_l1_l2, weight_l2_l3, b1, b2):
    der_b_l2 = []
    for i in range(8):
        der_b_l2.append(0)

    der_b_l1 = []
    for i in range(3):
        der_b_l1.append(0)

    der_weight_l2_l3 = []
    for i in range(3):
        temp_list = []
        for j in range(8):
            temp_list.append(0)
        der_weight_l2_l3.append(temp_list)

    der_weight_l1_l2 = []
    for i in range(8):
        temp_list = []
        for j in range(3):
            temp_list.append(0)
        der_weight_l1_l2.append(temp_list)


    for row in input:
        par_der_b_l2, par_der_b_l1, par_der_weight_l2_l3, par_der_weight_l1_l2 = \
            backpropagation(row, weight_l1_l2, weight_l2_l3, b1, b2)

        for i in range(8):
            der_b_l2[i] += par_der_b_l2[i]
        for i in range(3):
            der_b_l1[i] += par_der_b_l1[i]

        for i in range(3):
            for j in range(8):
                der_weight_l2_l3[i][j] += par_der_weight_l2_l3[i][j]

        for i in range(8):
            for j in range(3):
                der_weight_l1_l2[i][j] += par_der_weight_l1_l2[i][j]

    for i in range(8):
        b2[i] = b2[i] - learning_rate * ((1 / len(input)) * der_b_l2[i])

    for i in range(3):
        b1[i] = b1[i] - learning_rate * ((1 / len(input)) * der_b_l1[i])

    for i in range(3):
        for j in range(8):
            weight_l1_l2[j][i] = weight_l1_l2[j][i] - learning_rate * \
                                 ((1 / len(input)) * der_weight_l1_l2[j][i]
                                  + weight_decay * weight_l1_l2[j][i])

    for i in range(8):
        for j in range(3):
            weight_l2_l3[j][i] = weight_l2_l3[j][i] - learning_rate * \
                                 ((1 / len(input)) * der_weight_l2_l3[j][i] +
                                  weight_decay * weight_l2_l3[j][i])

        # print(b1)
        # print(b2)
        # print(weight_l1_l2)
        # print(weight_l2_l3)

        return b1, b2, weight_l1_l2, weight_l2_l3


# update_weights([[1,0,0,0,0,0,0,0]], weights_l1_l2, weights_l2_l3, bias_l1, bias_l2)

def gradient_descent(input, weight_l1_l2, weight_l2_l3, b1, b2):
    count = 0
    current_count = 0
    while (count < 7000):
        count += 1
        current_count = 0
        b1, b2, weight_l1_l2, weight_l2_l3 = update_weights(input, weight_l1_l2,
                                                            weight_l2_l3, b1, b2)
        for row in input:
            # print(row)
            activation_hidden, activation_output = forward_propagate(row, weight_l1_l2, weight_l2_l3,
                                                                     b1, b2)
            # print(activation_output)
            if activation_output.index(max(activation_output)) == row.index(max(row)):
                current_count += 1

            if count % 10000 == 0:
                activation_hidden, activation_output = forward_propagate(row, weight_l1_l2, weight_l2_l3,
                                                                         b1, b2)
                print(f'Expected: {row.index(max(row))}. Actual: {activation_output.index(max(activation_output))}')

    print(f'Convergence reached in {count} iterations with learning rate {learning_rate} and'
      f'weight decay {weight_decay}')

    for row in input:
        activation_hidden, activation_output = forward_propagate(row, weight_l1_l2, weight_l2_l3,
                                                           b1, b2)
        print(row)
        print(activation_output)


gradient_descent(training_data, weights_l1_l2, weights_l2_l3, bias_l1, bias_l2)
