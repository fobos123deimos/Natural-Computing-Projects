from random import choice

import matplotlib.pyplot as plt
import numpy as np

from Algoritmos_Geneticos.genetic_algorithm import Gene, Chromosome, GeneticMachine
from Algoritmos_Geneticos.feedforward_neural_network import FeedForwardNeuralNetwork


# The current gene configuration does not use negative weights.
# This may be a limitation depending on the dataset and the decision boundary
# the neural network needs to learn.


def decode_chromosome(chromosome, neural_network):
    """
    Decode a chromosome into the neural network weight matrices.

    The chromosome stores all network weights as a single sequence.
    This function reshapes that sequence into the matrix format expected
    by the feed-forward neural network.

    Parameters
    ----------
    chromosome : Chromosome
        Chromosome containing the encoded weights.

    neural_network : FeedForwardNeuralNetwork
        Neural network whose layer structure defines the matrix shapes.

    Returns
    -------
    list
        List of decoded weight matrices.
    """

    weight_matrices = []

    chromosome_values = list(chromosome.get_string())

    # Create one weight matrix for each pair of consecutive layers
    for i in range(len(neural_network.layer_structure) - 1):
        matrix = np.zeros(
            (
                neural_network.layer_structure[i + 1],
                neural_network.layer_structure[i]
            )
        )

        weight_matrices.append(matrix)

    # Fill the matrices using the chromosome values
    for i in range(len(weight_matrices)):
        for j in range(len(weight_matrices[i])):
            for k in range(len(weight_matrices[i][j])):
                weight_matrices[i][j][k] = chromosome_values[0]
                del chromosome_values[0]

    return weight_matrices


# Dataset variables
input_samples = []
expected_outputs = []
test_scores = []
target_table = {}

# Class-separated samples
class_negative_samples = []
class_positive_samples = []

# Feature values used for normalization
first_feature_values = []
second_feature_values = []
third_feature_values = []

# Feature medians
first_feature_median = 0
second_feature_median = 0
third_feature_median = 0

# Feature standard deviations
first_feature_std = 0
second_feature_std = 0
third_feature_std = 0


# Genetic algorithm parameters
population_size = 100
layer_structure = (3, 4, 1)

gene = Gene(
    (0, 1),
    "F",
    4
)

mutation_rate = 2.25
truncation_rate = 85
chromosome_size = 0

# Experiment parameters
repetitions = 1
test_size = 40


# Metrics collected during evolution
median_training_scores = []
best_generation_scores = []
worst_generation_scores = []

class_negative_scores = []
class_positive_scores = []


# Calculate how many weights are needed by the neural network
for i in range(len(layer_structure) - 1):
    chromosome_size += layer_structure[i] * layer_structure[i + 1]


# Load and preprocess the dataset
with open("three_feature_binary_dataset.txt", "r") as file:
    for line in file:
        # Ignore empty lines
        if line != "\n":
            values = line.strip().split(",")

            # Convert all values to float
            values = [
                float(value)
                for value in values
            ]

            # The first three columns are input features
            input_samples.append(values[0:3])

            # Convert labels to the neural network output format:
            # Original class 1 becomes [-1.0]
            # Other classes become [1.0]
            if values[3] == 1:
                expected_outputs.append([-1.0])
            else:
                expected_outputs.append([1.0])


# Normalize the input data using median and standard deviation
for sample in input_samples:
    first_feature_values.append(sample[0])
    second_feature_values.append(sample[1])
    third_feature_values.append(sample[2])

first_feature_median = np.median(first_feature_values)
second_feature_median = np.median(second_feature_values)
third_feature_median = np.median(third_feature_values)

first_feature_std = np.std(first_feature_values)
second_feature_std = np.std(second_feature_values)
third_feature_std = np.std(third_feature_values)

for i in range(len(input_samples)):
    normalized_first_feature = (
        input_samples[i][0] - first_feature_median
    ) / first_feature_std

    normalized_second_feature = (
        input_samples[i][1] - second_feature_median
    ) / second_feature_std

    normalized_third_feature = (
        input_samples[i][2] - third_feature_median
    ) / third_feature_std

    input_samples[i] = [
        normalized_first_feature,
        normalized_second_feature,
        normalized_third_feature
    ]


# Create the learning table.
# Each input sample is converted to string and used as a dictionary key.
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
    if target_table[str(sample)] == [-1.0]:
        class_negative_samples.append(sample)
    else:
        class_positive_samples.append(sample)


# The original train/test split was disabled.
# The model currently samples from the full class-separated dataset.

# class_negative_test_samples = [
#     class_negative_samples[i]
#     for i in range(int(len(class_negative_samples) / 2))
# ]
#
# class_positive_test_samples = [
#     class_positive_samples[i]
#     for i in range(int(len(class_positive_samples) / 2))
# ]
#
# class_negative_samples = [
#     class_negative_samples[i]
#     for i in range(
#         int(len(class_negative_samples) / 2),
#         len(class_negative_samples)
#     )
# ]
#
# class_positive_samples = [
#     class_positive_samples[i]
#     for i in range(
#         int(len(class_positive_samples) / 2),
#         len(class_positive_samples)
#     )
# ]


# Neuroevolution training loop
for _ in range(repetitions):
    for _ in range(len(input_samples)):

        training_scores = []
        testing_scores = []

        negative_class_batch_scores = []
        positive_class_batch_scores = []

        # Randomly select samples from each class
        negative_class_batch = [
            choice(class_negative_samples)
            for _ in range(int(test_size / 2))
        ]

        positive_class_batch = [
            choice(class_positive_samples)
            for _ in range(int(test_size / 2))
        ]

        # Test batches were disabled in the original code.
        # negative_class_test_batch = [
        #     choice(class_negative_test_samples)
        #     for _ in range(int(test_size / 2))
        # ]
        #
        # positive_class_test_batch = [
        #     choice(class_positive_test_samples)
        #     for _ in range(int(test_size / 2))
        # ]

        # Training evaluation
        for chromosome in genetic_machine.get_population():
            negative_class_hits = 0
            positive_class_hits = 0

            for i in range(int(test_size / 2)):
                weight_matrices = decode_chromosome(
                    chromosome,
                    neural_network
                )

                neural_network.set_weight_matrices(
                    weight_matrices,
                    -1
                )

                negative_class_output = neural_network.feed_forward(
                    negative_class_batch[i]
                )

                positive_class_output = neural_network.feed_forward(
                    positive_class_batch[i]
                )

                # Negative class is expected to produce a non-positive output
                if negative_class_output[0] <= 0.0:
                    negative_class_hits += 1

                # Positive class is expected to produce a non-negative output
                if positive_class_output[0] >= 0.0:
                    positive_class_hits += 1

            score = (
                negative_class_hits + positive_class_hits
            ) / test_size

            training_scores.append(score)

            negative_class_batch_scores.append(
                2 * negative_class_hits / test_size
            )

            positive_class_batch_scores.append(
                2 * positive_class_hits / test_size
            )

            chromosome.set_score(score)

        # Testing evaluation was disabled in the original code.
        #
        # for chromosome in genetic_machine.get_population():
        #     negative_class_hits = 0
        #     positive_class_hits = 0
        #
        #     for i in range(int(test_size / 2)):
        #         weight_matrices = decode_chromosome(
        #             chromosome,
        #             neural_network
        #         )
        #
        #         neural_network.set_weight_matrices(
        #             weight_matrices,
        #             -1
        #         )
        #
        #         negative_class_output = neural_network.feed_forward(
        #             negative_class_test_batch[i]
        #         )
        #
        #         positive_class_output = neural_network.feed_forward(
        #             positive_class_test_batch[i]
        #         )
        #
        #         if negative_class_output[0] <= 0.0:
        #             negative_class_hits += 1
        #
        #         if positive_class_output[0] >= 0.0:
        #             positive_class_hits += 1
        #
        #     testing_scores.append(
        #         (negative_class_hits + positive_class_hits) / test_size
        #     )

        # Store generation metrics
        median_training_scores.append(
            np.median(training_scores)
        )

        negative_class_batch_scores_median = np.median(
            negative_class_batch_scores
        )

        positive_class_batch_scores_median = np.median(
            positive_class_batch_scores
        )

        class_negative_scores.append(
            negative_class_batch_scores_median
        )

        class_positive_scores.append(
            positive_class_batch_scores_median
        )

        # test_scores.append(np.median(testing_scores))

        best_generation_scores.append(
            max(genetic_machine.get_population()).get_score()
        )

        worst_generation_scores.append(
            min(genetic_machine.get_population()).get_score()
        )

        # Select individuals and generate the next population
        genetic_machine.selection()


# X-axis used in all plots
generations = list(
    range(
        1,
        len(input_samples) * repetitions + 1
    )
)


# Plot median training score
plt.plot(
    generations,
    median_training_scores,
    linewidth=2
)

plt.title(
    "Median Score",
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


# Plot negative class score
plt.plot(
    generations,
    class_negative_scores,
    linewidth=2
)

plt.title(
    "Negative Class Median Score",
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


# Plot positive class score
plt.plot(
    generations,
    class_positive_scores,
    linewidth=2
)

plt.title(
    "Positive Class Median Score",
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


# Plot best and worst score for each generation
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


# Test plot was disabled in the original code.
#
# plt.plot(
#     generations,
#     test_scores,
#     linewidth=2
# )
#
# plt.title(
#     "Test Score",
#     fontsize=24
# )
#
# plt.xlabel(
#     "Generation",
#     fontsize=14
# )
#
# plt.ylabel(
#     "Median Score per Generation",
#     fontsize=14
# )
#
# plt.tick_params(
#     axis="both",
#     which="major",
#     labelsize=5
# )
#
# plt.axis(
#     [
#         1,
#         len(input_samples) * repetitions + 1,
#         0,
#         1
#     ]
# )
#
# plt.show()