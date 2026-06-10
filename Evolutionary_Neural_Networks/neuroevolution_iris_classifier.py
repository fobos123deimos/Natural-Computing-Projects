from random import choice, randint, shuffle

import matplotlib.pyplot as plt
import numpy as np

from Evolutionary_Neural_Networks.genetic_algorithm import Gene, Chromosome, GeneticMachine
from Evolutionary_Neural_Networks.feedforward_neural_network import FeedForwardNeuralNetwork


def decode_chromosome(chromosome, neural_network):
    """
    Decode a chromosome into a list of neural network weight matrices.

    The chromosome stores all network weights in a single sequence.
    This function reshapes that sequence according to the neural network
    layer structure.
    """

    weight_matrices = []
    chromosome_values = list(chromosome.get_string())

    # Create empty matrices based on the neural network architecture
    for i in range(len(neural_network.layer_structure) - 1):
        matrix = np.zeros(
            (
                neural_network.layer_structure[i + 1],
                neural_network.layer_structure[i]
            )
        )

        weight_matrices.append(matrix)

    # Fill the matrices with the chromosome values
    for i in range(len(weight_matrices)):
        for j in range(len(weight_matrices[i])):
            for k in range(len(weight_matrices[i][j])):
                weight_matrices[i][j][k] = chromosome_values[0]
                del chromosome_values[0]

    return weight_matrices


# Training samples separated by Iris class
setosa_inputs = []
setosa_outputs = []

versicolor_inputs = []
versicolor_outputs = []

virginica_inputs = []
virginica_outputs = []

# Training lookup tables
setosa_training_table = {}
versicolor_training_table = {}
virginica_training_table = {}

# Test samples
test_inputs = []
test_outputs = []
test_table = {}


# Genetic algorithm and neural network parameters
population_size = 60
layer_structure = (4, 5, 2)

gene = Gene(
    (0, 1),
    "F",
    4
)

mutation_rate = 5
selection_probability = 10
chromosome_size = 0

repetitions = 1


# Calculate the number of weights required by the neural network
for i in range(len(layer_structure) - 1):
    chromosome_size += layer_structure[i] * layer_structure[i + 1]


# Load the training dataset
with open("iris_train_dataset.txt", "r") as file:
    for line in file:
        if line != "\n":
            values = line.strip().split(",")

            # Convert the four numerical Iris features to float
            for i in range(len(values) - 1):
                values[i] = float(values[i])

            input_features = values[0:4]
            species_name = values[4].strip().lower()

            # Avoid duplicated samples across the three training lists
            if (
                input_features in setosa_inputs
                or input_features in versicolor_inputs
                or input_features in virginica_inputs
            ):
                continue

            # Output encoding:
            # Iris-setosa     -> [1.0, 1.0]
            # Iris-versicolor -> [0.0, 0.0]
            # Iris-virginica  -> [1.0, 0.0]
            if "setosa" in species_name:
                setosa_inputs.append(input_features)
                setosa_outputs.append([1.0, 1.0])

            elif "versicolor" in species_name:
                versicolor_inputs.append(input_features)
                versicolor_outputs.append([0.0, 0.0])

            else:
                virginica_inputs.append(input_features)
                virginica_outputs.append([1.0, 0.0])


# Load the test dataset
with open("iris_test_dataset.txt", "r") as file:
    for line in file:
        if line != "\n":
            values = line.strip().split(",")

            # Convert the four numerical Iris features to float
            for i in range(len(values) - 1):
                values[i] = float(values[i])

            input_features = values[0:4]
            species_name = values[4].strip().lower()

            # Avoid duplicated test samples
            if input_features in test_inputs:
                continue

            test_inputs.append(input_features)

            # Same output encoding used in the training set
            if "setosa" in species_name:
                test_outputs.append([1.0, 1.0])

            elif "versicolor" in species_name:
                test_outputs.append([0.0, 0.0])

            else:
                test_outputs.append([1.0, 0.0])


# Create the training tables
for i in range(len(setosa_inputs)):
    setosa_training_table[str(setosa_inputs[i])] = setosa_outputs[i]

for i in range(len(versicolor_inputs)):
    versicolor_training_table[str(versicolor_inputs[i])] = versicolor_outputs[i]

for i in range(len(virginica_inputs)):
    virginica_training_table[str(virginica_inputs[i])] = virginica_outputs[i]


# Create the test table
for i in range(len(test_inputs)):
    test_table[str(test_inputs[i])] = test_outputs[i]


# Create and initialize the feed-forward neural network
neural_network = FeedForwardNeuralNetwork(
    [],
    layer_structure
)

neural_network.initialize_network()


# Create the genetic machine
genetic_machine = GeneticMachine(
    gene,
    max_size=-1,
    population=[],
    size=population_size,
    selection_probability=selection_probability
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


# Metrics collected during evolution
median_scores = []
best_scores = []
worst_scores = []
classification_hits_history = []


# Neuroevolution training loop
for _ in range(repetitions):
    shuffle(setosa_inputs)
    shuffle(versicolor_inputs)
    shuffle(virginica_inputs)
    shuffle(test_inputs)

    total_training_samples = (
        len(setosa_inputs)
        + len(versicolor_inputs)
        + len(virginica_inputs)
    )

    for _ in range(total_training_samples):
        generation_scores = []

        # Randomly select one sample from each class
        setosa_sample = choice(setosa_inputs)
        versicolor_sample = choice(versicolor_inputs)
        virginica_sample = choice(virginica_inputs)

        # Evaluate each chromosome
        for chromosome in genetic_machine.get_population():
            weight_matrices = decode_chromosome(
                chromosome,
                neural_network
            )

            neural_network.set_weight_matrices(
                weight_matrices,
                -1
            )

            setosa_output = neural_network.feed_forward(setosa_sample)
            versicolor_output = neural_network.feed_forward(versicolor_sample)
            virginica_output = neural_network.feed_forward(virginica_sample)

            # Mean squared error for each class sample
            setosa_error = np.sum(
                0.5
                * (
                    np.array(setosa_training_table[str(setosa_sample)])
                    - setosa_output
                ) ** 2
            )

            versicolor_error = np.sum(
                0.5
                * (
                    np.array(versicolor_training_table[str(versicolor_sample)])
                    - versicolor_output
                ) ** 2
            )

            virginica_error = np.sum(
                0.5
                * (
                    np.array(virginica_training_table[str(virginica_sample)])
                    - virginica_output
                ) ** 2
            )

            # Geometric mean of the three class errors
            error = (
                setosa_error
                * versicolor_error
                * virginica_error
            ) ** (1 / 3)

            # The genetic algorithm maximizes score.
            # Therefore, lower error becomes higher score by using -error.
            score = -error

            generation_scores.append(score)
            chromosome.set_score(score)

        median_scores.append(np.median(generation_scores))
        best_scores.append(
            max(genetic_machine.get_population()).get_score()
        )
        worst_scores.append(
            min(genetic_machine.get_population()).get_score()
        )

        # Generate the next population
        genetic_machine.selection()

        # Test one random instance against the new population
        hit_count = 0
        test_sample = choice(test_inputs)

        for chromosome in genetic_machine.get_population():
            weight_matrices = decode_chromosome(
                chromosome,
                neural_network
            )

            neural_network.set_weight_matrices(
                weight_matrices,
                -1
            )

            output = neural_network.feed_forward(test_sample)

            # The original code averages both output neurons and uses
            # that value as a probability for generating the final class.
            output_probability = (output[0] + output[1]) / 2

            first_random_value = randint(0, 100) / 100
            second_random_value = randint(0, 100) / 100

            first_class_value = (
                1.0
                if first_random_value <= output_probability
                else 0.0
            )

            second_class_value = (
                1.0
                if second_random_value <= output_probability
                else 0.0
            )

            predicted_class = np.array(
                [
                    first_class_value,
                    second_class_value
                ]
            )

            true_class = np.array(test_table[str(test_sample)])

            if (
                predicted_class[0] == true_class[0]
                and predicted_class[1] == true_class[1]
            ):
                hit_count += 1

        classification_hits_history.append(hit_count)


# X-axis used in all plots
generations = list(
    range(
        1,
        (
            len(setosa_inputs)
            + len(versicolor_inputs)
            + len(virginica_inputs)
        )
        * repetitions
        + 1
    )
)


# Plot median score
plt.plot(
    generations,
    median_scores,
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

plt.show()


# Plot best and worst scores
plt.plot(
    generations,
    best_scores,
    linewidth=2,
    label="Best"
)

plt.plot(
    generations,
    worst_scores,
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

plt.legend()

plt.show()


# Plot classification hits
plt.plot(
    generations,
    classification_hits_history,
    linewidth=2
)

plt.title(
    "Classification Hits",
    fontsize=24
)

plt.xlabel(
    "Generation",
    fontsize=14
)

plt.ylabel(
    "Hits per Generation",
    fontsize=14
)

plt.tick_params(
    axis="both",
    which="major",
    labelsize=5
)

plt.show()


# Manual test examples:
#
# Setosa:
# [5.1, 3.5, 1.4, 0.2]
#
# Versicolor:
# [6.1, 2.8, 4.0, 1.3]
#
# Virginica:
# [6.3, 3.4, 5.6, 2.4]