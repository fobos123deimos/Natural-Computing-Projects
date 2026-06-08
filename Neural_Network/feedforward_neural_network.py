"""
Feedforward Neural Network

This module contains a simple feedforward neural network structure.

The network receives:
- A list of weight matrices.
- A list describing the number of neurons in each layer.

The current transfer function is the hyperbolic tangent function.
"""

import numpy as np


class FeedForwardNeuralNetwork(object):
    """
    Simple feedforward neural network.

    This class represents a neural network that performs only the forward
    propagation step. It does not include training or backpropagation.
    """

    def __init__(self, weight_matrices, layer_sizes):
        self.layer_sizes = layer_sizes
        self.input_count = layer_sizes[0]
        self.output_count = layer_sizes[-1]
        self.weight_matrices = weight_matrices

    def initialize_weight_matrices(self):
        """
        Initializes all network weight matrices with zeros.
        """

        weight_matrices = []

        for layer_index in range(len(self.layer_sizes) - 1):
            weight_matrix = np.zeros(
                (
                    self.layer_sizes[layer_index + 1],
                    self.layer_sizes[layer_index],
                )
            )

            weight_matrices.append(weight_matrix)

        self.weight_matrices = weight_matrices

    def compute_layer_activation(self, input_vector, weight_matrix):
        """
        Computes the activation values for one layer.
        """

        return np.array(
            [
                np.dot(input_vector, weight_row)
                for weight_row in weight_matrix
            ]
        )

    def apply_transfer_function(self, activation_values):
        """
        Applies the hyperbolic tangent transfer function.
        """

        return np.array(
            [
                np.tanh(activation_value)
                for activation_value in list(activation_values)
            ]
        )

    def feed_forward(self, input_vector):
        """
        Performs the feedforward process.
        """

        layer_output = np.array(input_vector)

        for layer_index in range(len(self.weight_matrices)):
            activation_values = self.compute_layer_activation(
                layer_output,
                self.weight_matrices[layer_index],
            )

            layer_output = self.apply_transfer_function(activation_values)

        return layer_output

    def set_weight_matrices(self, new_weights, layer_index=-1):
        """
        Updates the network weight matrices.

        If layer_index is negative, the full list of weight matrices is replaced.
        Otherwise, only the selected layer matrix is replaced.
        """

        if layer_index < 0:
            self.weight_matrices = new_weights
        else:
            self.weight_matrices[layer_index] = new_weights