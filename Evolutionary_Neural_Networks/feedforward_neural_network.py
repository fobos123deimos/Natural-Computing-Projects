import numpy as np


# Sigmoidal neural network.
# In the future, this can be replaced by a linear or hyperbolic tangent-based network.
# Important: normalize the input data before passing it to the network.


class FeedForwardNeuralNetwork(object):
    """
    Simple feed-forward neural network.

    This class stores a list of weight matrices and applies a feed-forward
    process using the hyperbolic tangent as the transfer function.
    """

    def __init__(self, weight_matrices, layer_structure):
        """
        Initialize the neural network.

        Parameters
        ----------
        weight_matrices : list
            List containing the weight matrices of the network.

        layer_structure : list or tuple
            List containing the number of neurons in each layer.
            Example: [2, 4, 1] means:
            - 2 input neurons
            - 4 hidden neurons
            - 1 output neuron
        """

        self.layer_structure = layer_structure

        # Number of input neurons
        self.input_size = layer_structure[0]

        # Number of output neurons
        self.output_size = layer_structure[len(layer_structure) - 1]

        # List of weight matrices
        self.weight_matrices = weight_matrices

    @classmethod
    def create_random_network(
        cls,
        layer_structure,
        minimum_weight=-1.0,
        maximum_weight=1.0,
        random_seed=None
    ):
        """
        Create a feed-forward neural network with random weights.

        Parameters
        ----------
        layer_structure : list or tuple
            Neural network architecture.
            Example: (2, 3, 1)

        minimum_weight : float
            Minimum random weight value.

        maximum_weight : float
            Maximum random weight value.

        random_seed : int, optional
            Random seed used to make the result reproducible.

        Returns
        -------
        FeedForwardNeuralNetwork
            Neural network initialized with random weight matrices.
        """

        random_generator = np.random.default_rng(random_seed)

        weight_matrices = []

        # Create one random weight matrix for each pair of consecutive layers
        for i in range(len(layer_structure) - 1):
            matrix = random_generator.uniform(
                minimum_weight,
                maximum_weight,
                size=(
                    layer_structure[i + 1],
                    layer_structure[i]
                )
            )

            weight_matrices.append(matrix)

        return cls(
            weight_matrices,
            layer_structure
        )

    def initialize_network(self):
        """
        Initialize all weight matrices with zeros.

        For each pair of consecutive layers, a weight matrix is created.
        The matrix shape is:

        number of neurons in next layer x number of neurons in current layer
        """

        weight_matrices = []

        for i in range(len(self.layer_structure) - 1):
            matrix = np.zeros(
                (
                    self.layer_structure[i + 1],
                    self.layer_structure[i]
                )
            )

            weight_matrices.append(matrix)

        self.weight_matrices = weight_matrices

    def activation_function(self, inputs, weight_matrix):
        """
        Calculate the weighted input for a layer.

        Parameters
        ----------
        inputs : array
            Input values coming from the previous layer.

        weight_matrix : array
            Weight matrix of the current layer.

        Returns
        -------
        array
            Weighted sum for each neuron in the current layer.
        """

        return np.array(
            [
                np.dot(inputs, row)
                for row in weight_matrix
            ]
        )

    def transfer_function(self, inputs):
        """
        Apply the transfer function to the weighted inputs.

        The original sigmoid function was replaced by tanh.

        Sigmoid option:
            1 / (1 + exp(-x))

        Current function:
            tanh(x)
        """

        return np.array(
            [
                np.tanh(value)
                for value in list(inputs)
            ]
        )

    def feed_forward(self, inputs):
        """
        Execute the feed-forward process.

        Parameters
        ----------
        inputs : array-like
            Input values of the neural network.

        Returns
        -------
        array
            Final output of the neural network.
        """

        result = np.array(inputs)

        # Pass the input through each layer of the network
        for i in range(len(self.weight_matrices)):
            weighted_input = self.activation_function(
                result,
                self.weight_matrices[i]
            )

            result = self.transfer_function(weighted_input)

        return result

    def set_weight_matrices(self, modification, index):
        """
        Modify the network weight matrices.

        Parameters
        ----------
        modification : array or list
            New weight matrix or new list of weight matrices.

        index : int
            If index < 0, replace the full list of weight matrices.
            Otherwise, replace only the matrix at the given index.
        """

        if index < 0:
            self.weight_matrices = modification
        else:
            self.weight_matrices[index] = modification