from random import choice

import matplotlib.pyplot as plt
import numpy as np

from Evolutionary_Neural_Networks.genetic_algorithm import Gene, Chromosome, GeneticMachine
from Evolutionary_Neural_Networks.feedforward_neural_network import FeedForwardNeuralNetwork


# There are no negative weights in the current gene configuration.
# This may limit the network's ability to represent some decision boundaries,
# depending on the transfer function and the dataset.


def decode_chromosome(chromosome, neural_network):
    """
    Decode a chromosome into a list of neural network weight matrices.

    Each value stored in the chromosome is copied into the corresponding
    position of the feed-forward neural network weight matrices.

    Parameters
    ----------
    chromosome : Chromosome
        Chromosome containing the encoded network weights.

    neural_network : FeedForwardNeuralNetwork
        Neural network whose layer structure defines the matrix shapes.

    Returns
    -------
    list
        List of decoded weight matrices.
    """

    weight_matrices = []

    chromosome_values = list(chromosome.get_string())

    # Create empty weight matrices according to the neural network architecture
    for i in range(len(neural_network.layer_structure) - 1):
        matrix = np.zeros(
            (
                neural_network.layer_structure[i + 1],
                neural_network.layer_structure[i]
            )
        )

        weight_matrices.append(matrix)

    # Fill each matrix with values from the chromosome
    for i in range(len(weight_matrices)):
        for j in range(len(weight_matrices[i])):
            for k in range(len(weight_matrices[i][j])):
                weight_matrices[i][j][k] = chromosome_values[0]
                del chromosome_values[0]

    return weight_matrices


# Dataset variables
input_samples = []
expected_outputs = []
target_table = {}

# Class-separated data
class_one_samples = []
class_zero_samples = []

# Training subsets
class_one_training_samples = []
class_zero_training_samples = []

# Test subsets
class_one_test_samples = []
class_zero_test_samples = []

# Normalization auxiliary lists
first_feature_values = []
second_feature_values = []

first_feature_median = 0
second_feature_median = 0

first_feature_std = 0
second_feature_std = 0


# Genetic algorithm parameters
population_size = 100
layer_structure = (2, 3, 1)

gene = Gene(
    (0, 1),
    "F",
    4
)

mutation_rate = 5
truncation_rate = 90
chromosome_size = 0

# Experiment parameters
repetitions = 3
test_size = 30


# Metrics collected during evolution
training_median_scores = []
test_median_scores = []

best_generation_scores = []
worst_generation_scores = []

class_one_scores = []
class_zero_scores = []


# Calculate the number of weights required by the neural network
for i in range(len(layer_structure) - 1):
    chromosome_size += layer_structure[i] * layer_structure[i + 1]


# Load and preprocess the dataset
with open("binary_admission_dataset.txt", "r") as file:
    for line in file:
        if line != "\n":
            values = line.strip().split(",")

            values = [
                float(value)
                for value in values
            ]

            # First two columns are input features
            input_samples.append(values[0:2])

            # Convert labels to the neural network output format:
            # Class 1 becomes [1.0]
            # Class 0 becomes [-1.0]
            if values[2] == 1.0:
                expected_outputs.append([1.0])
            else:
                expected_outputs.append([-1.0])


# Normalize the input data using median and standard deviation
for sample in input_samples:
    first_feature_values.append(sample[0])
    second_feature_values.append(sample[1])

first_feature_median = np.median(first_feature_values)
second_feature_median = np.median(second_feature_values)

first_feature_std = np.std(first_feature_values)
second_feature_std = np.std(second_feature_values)

for i in range(len(input_samples)):
    normalized_first_feature = (
        input_samples[i][0] - first_feature_median
    ) / first_feature_std

    normalized_second_feature = (
        input_samples[i][1] - second_feature_median
    ) / second_feature_std

    input_samples[i] = [
        normalized_first_feature,
        normalized_second_feature
    ]


# Create the target table.
# The sample converted to string is used as the dictionary key.
for i in range(len(input_samples)):
    target_table[str(input_samples[i])] = expected_outputs[i]


# Create and initialize the feed-forward neural network
neural_network = FeedForwardNeuralNetwork(
    [],
    layer_structure
)

neural_network.initialize_network()


# Create the genetic machine
genetic_machine = GeneticMachine(
    gene,
    -1,
    [],
    population_size,
    truncation_rate
)


# Fill the initial population with random chromosomes
for _ in range(genetic_machine.get_size()):
    chromosome = Chromosome(
        gene,
        chromosome_size,
        [],
        0,
        mutation_rate
    )

    filled_chromosome = Chromosome.fill_random_chromosome(chromosome)

    genetic_machine.add(filled_chromosome)


# Separate the dataset by class
for sample in input_samples:
    if target_table[str(sample)] == [1.0]:
        class_one_samples.append(sample)
    else:
        class_zero_samples.append(sample)


# Split each class into training and testing subsets
class_one_training_samples = [
    class_one_samples[i]
    for i in range(int(len(class_one_samples) / 2))
]

class_zero_training_samples = [
    class_zero_samples[i]
    for i in range(int(len(class_zero_samples) / 2))
]

class_one_test_samples = [
    class_one_samples[i]
    for i in range(
        int(len(class_one_samples) / 2),
        len(class_one_samples)
    )
]

class_zero_test_samples = [
    class_zero_samples[i]
    for i in range(
        int(len(class_zero_samples) / 2),
        len(class_zero_samples)
    )
]


# Neuroevolution training loop
for _ in range(repetitions):
    for _ in range(len(input_samples)):

        training_scores = []
        testing_scores = []

        class_one_training_scores = []
        class_zero_training_scores = []

        # Randomly select samples for training and testing
        class_one_training_batch = [
            choice(class_one_training_samples)
            for _ in range(int(test_size / 2))
        ]

        class_zero_training_batch = [
            choice(class_zero_training_samples)
            for _ in range(int(test_size / 2))
        ]

        class_one_testing_batch = [
            choice(class_one_test_samples)
            for _ in range(int(test_size / 2))
        ]

        class_zero_testing_batch = [
            choice(class_zero_test_samples)
            for _ in range(int(test_size / 2))
        ]

        # Training evaluation
        for chromosome in genetic_machine.get_population():
            class_one_hits = 0
            class_zero_hits = 0

            for i in range(int(test_size / 2)):
                weight_matrices = decode_chromosome(
                    chromosome,
                    neural_network
                )

                neural_network.set_weight_matrices(
                    weight_matrices,
                    -1
                )

                class_one_output = neural_network.feed_forward(
                    class_one_training_batch[i]
                )

                class_zero_output = neural_network.feed_forward(
                    class_zero_training_batch[i]
                )

                # Class 1 is expected to produce a non-negative output
                if class_one_output[0] >= 0.0:
                    class_one_hits += 1

                # Class 0 is expected to produce a non-positive output
                if class_zero_output[0] <= 0.0:
                    class_zero_hits += 1

            score = (class_one_hits + class_zero_hits) / test_size

            training_scores.append(score)
            class_one_training_scores.append(
                2 * class_one_hits / test_size
            )
            class_zero_training_scores.append(
                2 * class_zero_hits / test_size
            )

            chromosome.set_score(score)

        # Testing evaluation
        for chromosome in genetic_machine.get_population():
            class_one_hits = 0
            class_zero_hits = 0

            for i in range(int(test_size / 2)):
                weight_matrices = decode_chromosome(
                    chromosome,
                    neural_network
                )

                neural_network.set_weight_matrices(
                    weight_matrices,
                    -1
                )

                class_one_output = neural_network.feed_forward(
                    class_one_testing_batch[i]
                )

                class_zero_output = neural_network.feed_forward(
                    class_zero_testing_batch[i]
                )

                if class_one_output[0] >= 0.0:
                    class_one_hits += 1

                if class_zero_output[0] <= 0.0:
                    class_zero_hits += 1

            testing_scores.append(
                (class_one_hits + class_zero_hits) / test_size
            )

        # Store generation metrics
        training_median_scores.append(
            np.median(training_scores)
        )

        class_one_scores.append(
            np.median(class_one_training_scores)
        )

        class_zero_scores.append(
            np.median(class_zero_training_scores)
        )

        test_median_scores.append(
            np.median(testing_scores)
        )

        best_generation_scores.append(
            max(genetic_machine.get_population()).get_score()
        )

        worst_generation_scores.append(
            min(genetic_machine.get_population()).get_score()
        )

        # Apply selection and generate the next population
        genetic_machine.selection()


# X-axis used in all plots
generations = list(
    range(
        1,
        len(input_samples) * repetitions + 1
    )
)


# Plot the median training score
plt.plot(
    generations,
    training_median_scores,
    linewidth=2
)

plt.title(
    "Median Training Score",
    fontsize=24
)

plt.xlabel(
    "Generation",
    fontsize=14
)

plt.ylabel(
    "Median Score per Generation",
    fontsize=14
)

plt.tick_params(
    axis="both",
    which="major",
    labelsize=5
)

plt.axis(
    [
        1,
        len(input_samples) * repetitions + 1,
        0,
        1
    ]
)

plt.show()


# Plot class 1 training score
plt.plot(
    generations,
    class_one_scores,
    linewidth=2
)

plt.title(
    "Class 1 Training Score",
    fontsize=24
)

plt.xlabel(
    "Generation",
    fontsize=14
)

plt.ylabel(
    "Class 1 Score per Generation",
    fontsize=14
)

plt.tick_params(
    axis="both",
    which="major",
    labelsize=5
)

plt.axis(
    [
        1,
        len(input_samples) * repetitions + 1,
        0,
        1
    ]
)

plt.show()


# Plot class 0 training score
plt.plot(
    generations,
    class_zero_scores,
    linewidth=2
)

plt.title(
    "Class 0 Training Score",
    fontsize=24
)

plt.xlabel(
    "Generation",
    fontsize=14
)

plt.ylabel(
    "Class 0 Score per Generation",
    fontsize=14
)

plt.tick_params(
    axis="both",
    which="major",
    labelsize=5
)

plt.axis(
    [
        1,
        len(input_samples) * repetitions + 1,
        0,
        1
    ]
)

plt.show()


# Plot the best and worst score of each generation
plt.plot(
    generations,
    best_generation_scores,
    linewidth=2,
    label="Best"
)

plt.plot(
    generations,
    worst_generation_scores,
    linewidth=2,
    color="red",
    label="Worst"
)

plt.title(
    "Best and Worst Scores per Generation",
    fontsize=24
)

plt.xlabel(
    "Generation",
    fontsize=14
)

plt.ylabel(
    "Score per Generation",
    fontsize=14
)

plt.tick_params(
    axis="both",
    which="major",
    labelsize=5
)

plt.axis(
    [
        1,
        len(input_samples) * repetitions + 1,
        0,
        1
    ]
)

plt.legend()

plt.show()


# Plot test score
plt.plot(
    generations,
    test_median_scores,
    linewidth=2
)

plt.title(
    "Test Score",
    fontsize=24
)

plt.xlabel(
    "Generation",
    fontsize=14
)

plt.ylabel(
    "Score per Generation",
    fontsize=14
)

plt.tick_params(
    axis="both",
    which="major",
    labelsize=5
)

plt.axis(
    [
        1,
        len(input_samples) * repetitions + 1,
        0,
        1
    ]
)

plt.show()