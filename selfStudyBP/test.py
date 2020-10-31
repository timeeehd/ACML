# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from math import exp
from random import seed
from random import random
import random
from random import randrange


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def initialize_network(nr_input_nodes, nr_hidden_nodes, nr_output_nodes):
    network = list()
    hidden_layer = [
        # {'weights': [0.5 for i in range(nr_input_nodes + 1)],
        {'weights': [(random.uniform(0, 1)) / 100 for i in range(nr_input_nodes + 1)],
         'gradient': [0 for i in range(nr_input_nodes + 1)],
         'partial_der': [0 for i in range(nr_input_nodes + 1)]}
        for j in range(nr_hidden_nodes)]
    network.append(hidden_layer)
    output_layer = [
        # {'weights': [0.5 for i in range(nr_hidden_nodes + 1)],
        {'weights': [(random.uniform(0, 1)) / 100 for i in range(nr_hidden_nodes + 1)],
         'gradient': [0 for i in range(nr_hidden_nodes + 1)],
         'partial_der': [0 for i in range(nr_hidden_nodes + 1)]}
        for j in range(nr_output_nodes)]
    network.append(output_layer)
    return network


def weighted_sum(inputs, weights):
    # start with the bias node, which is at the end of the list
    activation = weights[-1]
    # print(f'inputs ${inputs}')
    # print(f'weights ${weights}')
    for i in range(len(weights) - 1):
        activation += inputs[i] * weights[i]
    return activation


def sigmoid(activation):
    return 1 / (1 + exp(-activation))


def forward_propagate(network, input):
    inputs = input
    outputs = []
    for layer in network:
        # the output of one layer, is the input of the next layer
        outputs = []
        for node in layer:
            ws = weighted_sum(inputs, node['weights'])
            output = sigmoid(ws)
            node['output'] = output
            outputs.append(output)
        inputs = outputs
    # it returns the output of the last layer (confusing with name)
    return outputs


def gradient(output):
    return output * (1 - output)


def backpropagate_error(network, expected_value):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if (i == len(network) - 1):
            for j in range(len(layer)):
                node = layer[j]
                errors.append(node['output'] - expected_value[j])
        else:
            for j in range(len(layer)):
                error = 0.0
                for node in network[i + 1]:
                    error += (node['weights'][j] * node['delta'])
                errors.append(error)
        for j in range(len(layer)):
            node = layer[j]
            node['delta'] = errors[j] * gradient(node['output'])


def update_gradient(network, row):
    # print(f'row ${row}')
    for i in range(len(network)):
        inputs = row[:-1][0]
        # print(inputs)
        if i != 0:
            inputs = [node['output'] for node in network[i - 1]]
        for node in network[i]:
            for j in range(len(inputs)):
                node['gradient'][j] += node['delta'] * inputs[j]
            node['gradient'][-1] += node['delta']


# def partial_gradient(network, row):
#     for i in range(len(network)):
#         inputs = row[:-1]
#         if i != 0:
#             inputs = [neuron['output'] for neuron in network[i - 1]]
#         for neuron in network[i]:
#             for j in range(len(inputs)):
#                 neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
#             neuron['weights'][-1] += l_rate * neuron['delta']


def partial_derivative(network, training_data, weight_decay):
    for layer in network:
        for j in range(len(layer)):
            node = layer[j]
            # print(f'node ${node}')
            for i in range(len(node['weights'])):
                if j != len(layer):
                    node['partial_der'][i] = (node['gradient'][i] + weight_decay * node['weights'][i]) / len(
                        training_data)
                else:
                    node['partial_der'][i] = (node['gradient'][i]) / len(training_data)


def update_weight(network, learning_rate):
    for layer in network:
        for j in range(len(layer)):
            node = layer[j]
            for i in range(len(node['weights'])):
                node['weights'][i] -= learning_rate * node['partial_der'][i]


def train_network(network, training_data, l_rate, weight_decay, training_reps):
    # gradient needs to be reset
    for rep in range(training_reps):
        sum_error = 0
        for row in training_data:
            # print(row)
            input = row[1]
            # print(f'input ${input}')
            outputs = forward_propagate(network, input)
            expected_result = row[-1]
            # print(f'expected_result ${expected_result}')
            sum_error += sum([(expected_result[i] - outputs[i]) ** 2 for i in range(len(expected_result))])
            backpropagate_error(network, expected_result)
            update_gradient(network, row)
        # print(f'error: {sum_error}')
        partial_derivative(network, training_data, weight_decay)
        update_weight(network, l_rate)
        reset_gradient(network)


def reset_gradient(network):
    for layer in network:
        for neuron in layer:
            for i in range(len(neuron['gradient']) - 1):
                neuron['gradient'][i] = 0


def othermain():
    network = initialize_network(2, 1, 2)
    for layer in network:
        print(layer)

    output = forward_propagate(network, [1, 0])
    print(output)
    # train_network(network, [[1, 0], [1, 0]], 1, 1)
    # row = [1, 0]
    # output = forward_propagate(network, row)
    # print(output)
    # network = [
    #     [{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
    #     [{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095]},
    #      {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763]}]]
    # expected = [0, 1]
    # backpropagate_error(network, expected)
    train_network(network, [[[1, 0], [1, 0]]], 1, 1, 3)
    # for layer in network:
    #     print(layer)
    # print(randrange(0, 10)/1000)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # seed(1)
    nn = initialize_network(8, 3, 8)
    for layer in nn:
        print(layer)
    # reps = 1
    # for k in range(15):
    #     reps = reps * 2
    #     print()
    #     print(f'training reps {reps}')
    #     nn = initialize_network(4, 3, 4)
    #
    #     training_data = [[[1, 0, 0, 0], [1, 0, 0, 0]],
    #                      [[0, 1, 0, 0], [0, 1, 0, 0]],
    #                      [[0, 0, 1, 0], [0, 0, 1, 0]],
    #                      [[0, 0, 0, 1], [0, 0, 0, 1]]]
    #
    #     train_network(nn, training_data, 0.9, 0.0005, reps)
    #     for set in training_data:
    #         print(f'in = {set[0]} out {forward_propagate(nn, set[0])}')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
