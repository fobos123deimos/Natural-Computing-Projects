"""
Neural Network Implementation

This module contains two educational neural network implementations:
1. SingleLayerPerceptron
2. MultiLayerPerceptron

The original algorithmic structure was preserved, but class names, method names,
variable names, comments, and graph labels were rewritten in English.
"""

import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal
from math import exp
from random import choice


class SingleLayerPerceptron(object):
    """
    Single-layer perceptron implementation.

    This class represents a simple perceptron with:
    - A fixed number of inputs.
    - A fixed number of outputs.
    - A learning rate.
    - A weight matrix initialized with zeros.

    The bias is handled by adding an extra input with value 1.
    """

    def __init__(self, input_count, output_count, learning_rate):
        """
        Initializes the single-layer perceptron.

        Parameters
        ----------
        input_count : int
            Number of input features.
        output_count : int
            Number of output neurons.
        learning_rate : float
            Learning rate used during the weight update process.
        """

        self.input_count = input_count
        self.output_count = output_count
        self.learning_rate = learning_rate

        # Weight matrix. The extra column is used for the bias term.
        self.weights = np.zeros((self.output_count, self.input_count + 1))

    def compute_activation(self, weights, input_vector):
        """
        Computes the activation value before thresholding.

        This method calculates the dot product between the weight vector
        and the input vector.
        """

        return np.dot(weights, input_vector)

    def apply_transfer_function(self, activation_value):
        """
        Applies a binary step transfer function.

        Returns
        -------
        float
            0.0 if the activation value is less than or equal to 0.
            1.0 otherwise.
        """

        if activation_value <= 0.0:
            return 0.0

        return 1.0

    def train(self, epoch_batches, target_table):
        """
        Trains the single-layer perceptron.

        Sequential learning is used. The bias value is fixed as 1.
        The weight vector is initialized with zeros, and the error
        correction method is used to update the weights.

        Parameters
        ----------
        epoch_batches : list
            List of epochs. Each epoch contains training cycles.
        target_table : dict
            Lookup table containing the expected output for each input.
        """

        # Stores the average error measured during training.
        mean_errors = []

        for epoch in epoch_batches:
            for training_cycle in epoch:
                input_vectors_with_bias = []
                accumulated_error = Decimal("0.0")

                # Adds the bias value 1 to each input vector.
                for input_vector in training_cycle:
                    input_vectors_with_bias.append(np.array([1] + input_vector))

                sample_index = 0

                for input_vector_with_bias in input_vectors_with_bias:
                    weight_correction = np.zeros(
                        (self.output_count, self.input_count + 1)
                    )

                    predicted_output = np.array(
                        [0 for _ in range(self.output_count)]
                    )

                    # Forward process: input to output.
                    for output_index in range(self.output_count):
                        activation_value = self.compute_activation(
                            self.weights[output_index],
                            input_vector_with_bias,
                        )

                        predicted_output[output_index] = self.apply_transfer_function(
                            activation_value
                        )

                    expected_output = np.array(
                        target_table[str(training_cycle[sample_index])]
                    )

                    # Error vector.
                    error_vector = expected_output - predicted_output

                    # Calculates the correction factor for each output neuron.
                    for output_index in range(self.output_count):
                        weight_correction[output_index] += (
                            input_vector_with_bias
                            * self.learning_rate
                            * error_vector[output_index]
                        )

                    # Updates the weights.
                    self.weights += weight_correction

                    # Accumulates the average absolute error.
                    accumulated_error += (
                        Decimal(str(sum(abs(error_vector))))
                        / Decimal(str(self.output_count))
                    )

                    sample_index += 1

                mean_error = accumulated_error / Decimal(str(self.output_count))
                mean_errors.append(float(mean_error))

        # Learning graph.
        plt.plot(list(range(1, len(mean_errors) + 1)), mean_errors, linewidth=2)
        plt.title("Learning Graph", fontsize=24)
        plt.xlabel("Iterations", fontsize=14)
        plt.ylabel("Average Error / Cycle", fontsize=14)
        plt.tick_params(axis="both", which="major", labelsize=5)
        plt.show()

    def predict(self, input_vector):
        """
        Uses the trained perceptron to classify a new input.

        Parameters
        ----------
        input_vector : list
            Input vector without the bias value.

        Returns
        -------
        list
            Output generated by the perceptron.
        """

        # Adds the bias value to the input.
        input_vector_with_bias = np.array([1] + input_vector)

        predicted_output = []

        for output_index in range(self.output_count):
            activation_value = self.compute_activation(
                self.weights[output_index],
                input_vector_with_bias,
            )

            predicted_output.append(
                self.apply_transfer_function(activation_value)
            )

        return predicted_output


# Layer component indexes used by the multilayer perceptron.
WEIGHTS = 0
ACTIVATIONS = 1
TRANSFORMED_OUTPUTS = 2
LOCAL_GRADIENTS = 3
WEIGHT_CORRECTIONS = 4


class MultiLayerPerceptron(object):
    """
    Multilayer perceptron implementation.

    This class creates a neural network with:
    - One input layer.
    - A configurable number of hidden layers.
    - One output layer.
    - Randomly initialized weights.
    - Local gradients for backpropagation.
    - Momentum-based weight correction.

    The configuration list controls the architecture and training behavior.
    """

    def __init__(self, input_count, output_count, config, target_table):
        """
        Initializes the multilayer perceptron.

        Parameters
        ----------
        input_count : int
            Number of input features.
        output_count : int
            Number of output neurons.
        config : list
            Configuration list used to define the network.

            Expected structure:
            config[0] -> hidden layer count
            config[1] -> learning rate
            config[2] -> momentum factor
            config[3] -> transfer function identifier
            config[4] -> random weight range limit
            config[5] -> decimal places used for output rounding
            config[6] -> neuron count per hidden layer
        target_table : dict
            Lookup table containing the expected output for each input.
        """

        network_layers = []

        hidden_layer_count = config[0]
        hidden_neuron_count = config[6]
        random_weight_limit = config[4]

        # Creates the input layer, hidden layers, and output layer.
        for layer_index in range(hidden_layer_count + 2):

            # Input layer.
            if layer_index == 0:
                input_layer = np.array([0.0 for _ in range(input_count)])
                network_layers.append(input_layer)

            # Hidden layers.
            elif 0 < layer_index <= hidden_layer_count:

                # First hidden layer.
                if layer_index == 1:
                    weights = np.zeros((hidden_neuron_count, input_count + 1))
                    weight_corrections = np.zeros(
                        (hidden_neuron_count, input_count + 1)
                    )

                    activations = np.array(
                        [0.0 for _ in range(hidden_neuron_count)]
                    )
                    transformed_outputs = np.array(
                        [0.0 for _ in range(hidden_neuron_count)]
                    )
                    local_gradients = np.array(
                        [0.0 for _ in range(hidden_neuron_count)]
                    )

                    # Random weight initialization.
                    for neuron_index in range(hidden_neuron_count):
                        weights[neuron_index] += np.array(
                            [
                                choice(
                                    [
                                        float(value)
                                        for value in range(
                                            -random_weight_limit,
                                            random_weight_limit,
                                        )
                                    ]
                                )
                                for _ in range(input_count + 1)
                            ]
                        )

                    network_layers.append(
                        [
                            weights,
                            activations,
                            transformed_outputs,
                            local_gradients,
                            weight_corrections,
                        ]
                    )

                # Hidden layers after the first one.
                else:
                    weights = np.zeros(
                        (hidden_neuron_count, hidden_neuron_count + 1)
                    )
                    weight_corrections = np.zeros(
                        (hidden_neuron_count, hidden_neuron_count + 1)
                    )

                    activations = np.array(
                        [0.0 for _ in range(hidden_neuron_count)]
                    )
                    transformed_outputs = np.array(
                        [0.0 for _ in range(hidden_neuron_count)]
                    )
                    local_gradients = np.array(
                        [0.0 for _ in range(hidden_neuron_count)]
                    )

                    # Random weight initialization.
                    for neuron_index in range(hidden_neuron_count):
                        weights[neuron_index] += np.array(
                            [
                                choice(
                                    [
                                        float(value)
                                        for value in range(
                                            -random_weight_limit,
                                            random_weight_limit,
                                        )
                                    ]
                                )
                                for _ in range(hidden_neuron_count + 1)
                            ]
                        )

                    network_layers.append(
                        [
                            weights,
                            activations,
                            transformed_outputs,
                            local_gradients,
                            weight_corrections,
                        ]
                    )

            # Output layer.
            else:
                weights = np.zeros((output_count, hidden_neuron_count + 1))
                weight_corrections = np.zeros(
                    (output_count, hidden_neuron_count + 1)
                )

                activations = np.array([0.0 for _ in range(output_count)])
                transformed_outputs = np.array(
                    [0.0 for _ in range(output_count)]
                )
                local_gradients = np.array([0.0 for _ in range(output_count)])

                # Random weight initialization.
                for neuron_index in range(output_count):
                    weights[neuron_index] += np.array(
                        [
                            choice(
                                [
                                    float(value)
                                    for value in range(
                                        -random_weight_limit,
                                        random_weight_limit,
                                    )
                                ]
                            )
                            for _ in range(hidden_neuron_count + 1)
                        ]
                    )

                network_layers.append(
                    [
                        weights,
                        activations,
                        transformed_outputs,
                        local_gradients,
                        weight_corrections,
                    ]
                )

        self.layers = network_layers
        self.config = config
        self.input_count = input_count
        self.output_count = output_count
        self.target_table = target_table

    def compute_layer_activation(self, weights, input_vector):
        """
        Computes the activation values for a layer.

        Parameters
        ----------
        weights : array
            Weight matrix of the current layer.
        input_vector : array
            Input vector, including the bias value.

        Returns
        -------
        array
            Activation values for all neurons in the current layer.
        """

        return np.array(
            [
                np.dot(weights[row_index], input_vector)
                for row_index in range(len(weights))
            ]
        )

    def apply_transfer_function(self, should_apply, activation_values):
        """
        Applies the transfer function.

        In the current implementation, config[3] is used as the transfer
        function identifier. When config[3] == 1, a sigmoid-like function
        is applied.

        Notes
        -----
        The original mathematical expression was preserved intentionally.
        A future improvement should review the sigmoid expression because
        the current implementation may not behave like the standard sigmoid.
        """

        transfer_function_id = self.config[3]

        if should_apply:
            if transfer_function_id == 1:
                transformed_values = []

                for activation_value in activation_values:
                    if activation_value < 0:
                        transformed_values.append(
                            1 - (1 / 1 + exp(activation_value))
                        )
                    else:
                        transformed_values.append(
                            1 / 1 + exp(-activation_value)
                        )

                return np.array(transformed_values)

        else:
            return transfer_function_id

    def compute_transfer_derivative(self, transfer_function_id, activation_values):
        """
        Computes the derivative of the transfer function.

        Parameters
        ----------
        transfer_function_id : int
            Transfer function identifier.
        activation_values : array
            Activation values.

        Returns
        -------
        array
            Derivative values for each activation.
        """

        if transfer_function_id == 1:
            derivative_values = []

            for activation_value in activation_values:
                if activation_value < 0:
                    derivative_values.append(
                        exp(activation_value) / pow((1 + exp(activation_value)), 2)
                    )
                else:
                    derivative_values.append(
                        exp(-activation_value) / pow((1 + exp(-activation_value)), 2)
                    )

            return np.array(derivative_values)

    def compute_local_gradient(self, current_layer_index, error_vector):
        """
        Computes the local gradient for a given layer.

        Parameters
        ----------
        current_layer_index : int
            Current layer index.
        error_vector : array
            Error vector from the output layer.
        """

        if current_layer_index != 0:
            hidden_layer_count = self.config[0]
            hidden_neuron_count = self.config[6]

            # Output layer gradient.
            if current_layer_index == hidden_layer_count + 1:
                self.layers[current_layer_index][LOCAL_GRADIENTS] = (
                    -error_vector
                    * self.compute_transfer_derivative(
                        self.apply_transfer_function(False, 0.0),
                        self.layers[current_layer_index][ACTIVATIONS],
                    )
                )

            # Hidden layer gradient.
            else:
                gradient_factor = np.array(
                    [
                        np.dot(
                            self.layers[current_layer_index + 1][LOCAL_GRADIENTS],
                            np.array(
                                list(
                                    self.layers[current_layer_index + 1][WEIGHTS][
                                        0:len(
                                            self.layers[current_layer_index + 1][
                                                TRANSFORMED_OUTPUTS
                                            ]
                                        ),
                                        column_index:column_index + 1,
                                    ].T[0]
                                )
                            ),
                        )
                        for column_index in range(1, hidden_neuron_count + 1)
                    ]
                )

                self.layers[current_layer_index][LOCAL_GRADIENTS] = (
                    self.compute_transfer_derivative(
                        self.apply_transfer_function(False, 0.0),
                        self.layers[current_layer_index][ACTIVATIONS],
                    )
                    * gradient_factor
                )

    def forward_propagate(self, input_vector):
        """
        Performs the forward propagation process for one input.

        Parameters
        ----------
        input_vector : list
            Input vector.

        Returns
        -------
        array
            Error vector between the expected output and the network output.
        """

        hidden_layer_count = self.config[0]

        # Stores the input vector in the input layer.
        self.layers[0] = np.array(input_vector)

        # Processes the first hidden layer.
        self.layers[1][ACTIVATIONS] = self.compute_layer_activation(
            self.layers[1][WEIGHTS],
            np.array([1.0] + list(self.layers[0])),
        )

        self.layers[1][TRANSFORMED_OUTPUTS] = self.apply_transfer_function(
            True,
            self.layers[1][ACTIVATIONS],
        )

        # Processes the remaining hidden layers.
        for layer_index in range(2, hidden_layer_count + 1):
            self.layers[layer_index][ACTIVATIONS] = self.compute_layer_activation(
                self.layers[layer_index][WEIGHTS],
                np.array(
                    [1.0]
                    + list(self.layers[layer_index - 1][TRANSFORMED_OUTPUTS])
                ),
            )

            self.layers[layer_index][TRANSFORMED_OUTPUTS] = (
                self.apply_transfer_function(
                    True,
                    self.layers[layer_index][ACTIVATIONS],
                )
            )

        # Processes the output layer.
        output_layer_index = hidden_layer_count + 1

        self.layers[output_layer_index][ACTIVATIONS] = self.compute_layer_activation(
            self.layers[output_layer_index][WEIGHTS],
            np.array(
                [1.0]
                + list(self.layers[hidden_layer_count][TRANSFORMED_OUTPUTS])
            ),
        )

        self.layers[output_layer_index][TRANSFORMED_OUTPUTS] = [
            output_value
            for output_value in self.apply_transfer_function(
                True,
                self.layers[output_layer_index][ACTIVATIONS],
            )
        ]

        expected_output = np.array(self.target_table[str(input_vector)])

        error_vector = (
            expected_output
            - self.layers[output_layer_index][TRANSFORMED_OUTPUTS]
        )

        return error_vector

    def predict(self, input_vectors):
        """
        Uses the trained multilayer perceptron to classify inputs.

        Parameters
        ----------
        input_vectors : list
            List of input vectors.

        Returns
        -------
        list
            List of network outputs.
        """

        hidden_layer_count = self.config[0]
        decimal_places = self.config[5]

        predicted_outputs = []

        for input_vector in input_vectors:
            self.layers[0] = np.array(input_vector)

            # Processes the first hidden layer.
            self.layers[1][ACTIVATIONS] = self.compute_layer_activation(
                self.layers[1][WEIGHTS],
                np.array([1.0] + list(self.layers[0])),
            )

            self.layers[1][TRANSFORMED_OUTPUTS] = self.apply_transfer_function(
                True,
                self.layers[1][ACTIVATIONS],
            )

            # Processes the remaining hidden layers.
            for layer_index in range(2, hidden_layer_count + 1):
                self.layers[layer_index][ACTIVATIONS] = self.compute_layer_activation(
                    self.layers[layer_index][WEIGHTS],
                    np.array(
                        [1.0]
                        + list(self.layers[layer_index - 1][TRANSFORMED_OUTPUTS])
                    ),
                )

                self.layers[layer_index][TRANSFORMED_OUTPUTS] = (
                    self.apply_transfer_function(
                        True,
                        self.layers[layer_index][ACTIVATIONS],
                    )
                )

            # Processes the output layer.
            output_layer_index = hidden_layer_count + 1

            self.layers[output_layer_index][ACTIVATIONS] = (
                self.compute_layer_activation(
                    self.layers[output_layer_index][WEIGHTS],
                    np.array(
                        [1.0]
                        + list(
                            self.layers[hidden_layer_count][TRANSFORMED_OUTPUTS]
                        )
                    ),
                )
            )

            self.layers[output_layer_index][TRANSFORMED_OUTPUTS] = [
                round(output_value, decimal_places)
                for output_value in self.apply_transfer_function(
                    True,
                    self.layers[output_layer_index][ACTIVATIONS],
                )
            ]

            predicted_outputs.append(
                self.layers[output_layer_index][TRANSFORMED_OUTPUTS]
            )

        return predicted_outputs

    def backpropagate(self, error_vector):
        """
        Performs the backpropagation process.

        Parameters
        ----------
        error_vector : array
            Output error vector.

        Returns
        -------
        array
            The same error vector received as input.
        """

        hidden_layer_count = self.config[0]
        learning_rate = self.config[1]
        momentum_factor = self.config[2]

        # Iterates backward through the network layers.
        for reverse_index in range(hidden_layer_count + 1):
            current_layer_index = hidden_layer_count + (1 - reverse_index)

            # Computes the local gradient.
            self.compute_local_gradient(current_layer_index, error_vector)

            # Weight correction for the first hidden layer.
            if current_layer_index == 1:
                for neuron_index in range(
                    len(self.layers[current_layer_index][ACTIVATIONS])
                ):
                    self.layers[current_layer_index][WEIGHT_CORRECTIONS][
                        neuron_index
                    ] = (
                        [
                            self.layers[current_layer_index][LOCAL_GRADIENTS][
                                neuron_index
                            ]
                            * learning_rate
                        ]
                        + [
                            self.layers[current_layer_index][LOCAL_GRADIENTS][
                                neuron_index
                            ]
                            * input_value
                            * learning_rate
                            for input_value in self.layers[current_layer_index - 1]
                        ]
                        + self.layers[current_layer_index][WEIGHT_CORRECTIONS][
                            neuron_index
                        ]
                        * momentum_factor
                    )

            # Weight correction for the other layers.
            else:
                for neuron_index in range(
                    len(self.layers[current_layer_index][ACTIVATIONS])
                ):
                    self.layers[current_layer_index][WEIGHT_CORRECTIONS][
                        neuron_index
                    ] = (
                        [
                            self.layers[current_layer_index][LOCAL_GRADIENTS][
                                neuron_index
                            ]
                            * learning_rate
                        ]
                        + [
                            self.layers[current_layer_index][LOCAL_GRADIENTS][
                                neuron_index
                            ]
                            * output_value
                            * learning_rate
                            for output_value in self.layers[current_layer_index - 1][
                                TRANSFORMED_OUTPUTS
                            ]
                        ]
                        + self.layers[current_layer_index][WEIGHT_CORRECTIONS][
                            neuron_index
                        ]
                        * momentum_factor
                    )

            # Applies the weight corrections.
            for neuron_index in range(
                len(self.layers[current_layer_index][WEIGHTS])
            ):
                self.layers[current_layer_index][WEIGHTS][neuron_index] += (
                    self.layers[current_layer_index][WEIGHT_CORRECTIONS][
                        neuron_index
                    ]
                )

        return error_vector

    def train_and_test(self, epoch_batches, test_groups, test_target_table):
        """
        Trains the multilayer perceptron and evaluates it on test groups.

        Parameters
        ----------
        epoch_batches : list
            List of training epochs.
        test_groups : list
            List of test groups.
        test_target_table : dict
            Test lookup table containing the expected outputs.
        """

        mean_errors = []
        accuracy_percentages = []

        test_group_index = 0

        for epoch in epoch_batches:
            epoch_error = 0.0

            # Training process for each input in the epoch.
            for input_vector in epoch:
                epoch_error += sum(
                    0.5
                    * ((self.backpropagate(self.forward_propagate(input_vector))) ** 2)
                ) / len(epoch)

            mean_errors.append(epoch_error)

            # Test process.
            if test_group_index <= len(test_groups) - 1:
                correct_prediction_count = 0.0

                predicted_outputs = self.predict(test_groups[test_group_index])

                for sample_index in range(len(predicted_outputs)):
                    if all(
                        predicted_outputs[sample_index][0]
                        == test_target_table[
                            str(test_groups[test_group_index][sample_index])
                        ]
                    ):
                        correct_prediction_count += 1.0

                accuracy_percentages.append(
                    (correct_prediction_count / len(test_groups[test_group_index]))
                    * 100
                )

            test_group_index += 1

        # Learning graph.
        plt.plot(list(range(1, len(epoch_batches) + 1)), mean_errors, linewidth=2)
        plt.title("Learning Graph", fontsize=24)
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Average Error / Epoch", fontsize=14)
        plt.tick_params(axis="both", which="major", labelsize=5)
        plt.show()

        # Test graph.
        plt.plot(
            list(range(1, len(accuracy_percentages) + 1)),
            accuracy_percentages,
            linewidth=2,
        )
        plt.title("Test Graph", fontsize=24)
        plt.xlabel("Test Group", fontsize=14)
        plt.ylabel("Accuracy Percentage (%)", fontsize=14)
        plt.tick_params(axis="both", which="major", labelsize=5)
        plt.show()