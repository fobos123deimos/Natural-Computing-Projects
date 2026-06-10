from random import randint

import numpy as np

from Algoritmos_Geneticos.genetic_algorithm import Gene, Chromosome, GeneticMachine
from Algoritmos_Geneticos.feedforward_neural_network import FeedForwardNeuralNetwork


def decode_chromosome(chromosome):
    """
    Decode a neural chromosome into weight matrices and bias vectors.

    The chromosome stores connections as dictionaries. Each dictionary
    represents either:

    1. A connection between neurons, stored in a weight matrix.
    2. A bias contribution, stored in a bias vector.

    Returns
    -------
    tuple
        A tuple containing:
        - weight_matrices: list of numpy arrays
        - bias_vectors: list of numpy arrays
    """

    weight_matrices = []
    bias_vectors = []

    layer_structure = chromosome.get_gene().get_network_structure()

    # Create empty weight matrices and bias vectors for each layer transition
    for i in range(len(layer_structure) - 1):
        weight_matrix = np.zeros(
            (
                layer_structure[i + 1],
                layer_structure[i]
            )
        )

        bias_vector = np.array(
            [
                0.0
                for _ in range(layer_structure[i + 1])
            ]
        )

        weight_matrices.append(weight_matrix)
        bias_vectors.append(bias_vector)

    # Decode each connection stored in the chromosome
    for connection in chromosome.get_string():

        # Connection with 4 fields represents a weight between neurons
        if len(connection) == 4:
            layer_index = connection["matrix"]
            target_neuron = connection["target_neuron"]
            source_neuron = connection["source_neuron"]
            weight = connection["weight"]

            weight_matrices[layer_index][target_neuron][source_neuron] += weight

        # Otherwise, the connection represents a bias value
        else:
            layer_index = connection["bias_vector"]
            neuron_index = connection["neuron_index"]
            weight = connection["weight"]

            bias_vectors[layer_index][neuron_index] += weight

    return weight_matrices, bias_vectors


# Neural network and genetic algorithm parameters
bias = 1
population_size = 10
layer_structure = (2, 3, 1)
mutation_rate = 10
selection_probability = 100
elitism = 10


# Create the feed-forward neural network
neural_network = FeedForwardNeuralNetwork(
    bias,
    [],
    [],
    layer_structure
)


# Create the genetic machine using a neural gene
genetic_machine = GeneticMachine(
    Gene.create_neural_gene(neural_network.layer_structure),
    max_size=-1,
    population=[],
    size=population_size,
    selection_probability=selection_probability,
    elitism=elitism
)


# Initialize the neural network weights and biases
neural_network.initialize_network()


# Fill the genetic machine with neural chromosomes
for _ in range(genetic_machine.get_size()):
    chromosome = Chromosome.create_neural_chromosome(
        mutation_rate,
        [],
        neural_network.layer_structure
    )

    filled_chromosome = Chromosome.fill_random_neural_chromosome(
        chromosome
    )

    genetic_machine.add(filled_chromosome)


# Assign a random score to each chromosome.
# This is only for testing the selection process.
for i in range(len(genetic_machine.get_population())):
    genetic_machine.get_population()[i].score = (
        randint(0, 100000) / 100000
    )


# Print the initial population
for chromosome in genetic_machine.get_population():
    print(chromosome)


# Apply selection and generate the next population
genetic_machine.selection()


print("\n\n")


# Print the new population after selection
for chromosome in genetic_machine.get_population():
    print(chromosome)


# Optional decoding test:
#
# print(genetic_machine.get_population()[1].get_string())
#
# weight_matrices, bias_vectors = decode_chromosome(
#     genetic_machine.get_population()[1]
# )
#
# print("\n\n")
# print(weight_matrices)
# print("\n\n")
# print(bias_vectors)


# Study notes:
#
# - Each chromosome should classify dataset samples and receive a score
#   according to its classification error.
#
# - The dataset may need to be presented with repetition and random order
#   so that chromosomes are evaluated more fairly.
#
# - The selection method may need to be reviewed.
#
# - The encoding strategy may also need to be reviewed.
#   One alternative would be to treat the population as a population of
#   neurons or network structures instead of only a population of weights.