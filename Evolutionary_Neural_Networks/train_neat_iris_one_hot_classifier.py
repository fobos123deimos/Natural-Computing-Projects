from __future__ import print_function

import os

import neat
import numpy as np

import Evolutionary_Neural_Networks.neat_visualization as neat_visualization


# Dataset inputs and expected outputs
input_samples = []
expected_outputs = []

# Auxiliary lists used for feature normalization
sepal_length_values = []
sepal_width_values = []
petal_length_values = []
petal_width_values = []


def encode_iris_class(species_name):
    """
    Convert the Iris species name into a one-hot encoded output.

    Output encoding:
    - Iris-setosa     -> (1.0, 0.0, 0.0)
    - Iris-versicolor -> (0.0, 1.0, 0.0)
    - Iris-virginica  -> (0.0, 0.0, 1.0)
    """

    species_name = species_name.strip().lower()

    if "setosa" in species_name:
        return (1.0, 0.0, 0.0)

    if "versicolor" in species_name:
        return (0.0, 1.0, 0.0)

    if "virginica" in species_name:
        return (0.0, 0.0, 1.0)

    raise ValueError(f"Unknown Iris species: {species_name}")


def predict_class(network_output):
    """
    Convert the raw neural network output into a one-hot class prediction.

    The highest output value is selected as the predicted class.

    Example:
    [0.2, 0.8, 0.1] -> (0.0, 1.0, 0.0)
    """

    predicted_index = int(np.argmax(network_output))

    prediction = [0.0, 0.0, 0.0]
    prediction[predicted_index] = 1.0

    return tuple(prediction)


# Load and preprocess the Iris dataset
with open("iris_train_dataset.txt", "r") as file:
    for line in file:
        # Ignore empty lines
        if line != "\n":
            values = line.strip().split(",")

            # Convert the first four columns to float.
            # These columns are:
            # 0 - sepal length
            # 1 - sepal width
            # 2 - petal length
            # 3 - petal width
            for i in range(len(values) - 1):
                values[i] = float(values[i])

            input_features = tuple(values[0:4])
            species_name = values[4]

            input_samples.append(input_features)
            expected_outputs.append(
                encode_iris_class(species_name)
            )


# Normalize input data using median and standard deviation
for sample in input_samples:
    sepal_length_values.append(sample[0])
    sepal_width_values.append(sample[1])
    petal_length_values.append(sample[2])
    petal_width_values.append(sample[3])


sepal_length_median = np.median(sepal_length_values)
sepal_width_median = np.median(sepal_width_values)
petal_length_median = np.median(petal_length_values)
petal_width_median = np.median(petal_width_values)

sepal_length_std = np.std(sepal_length_values)
sepal_width_std = np.std(sepal_width_values)
petal_length_std = np.std(petal_length_values)
petal_width_std = np.std(petal_width_values)


for i in range(len(input_samples)):
    normalized_sepal_length = (
        input_samples[i][0] - sepal_length_median
    ) / sepal_length_std

    normalized_sepal_width = (
        input_samples[i][1] - sepal_width_median
    ) / sepal_width_std

    normalized_petal_length = (
        input_samples[i][2] - petal_length_median
    ) / petal_length_std

    normalized_petal_width = (
        input_samples[i][3] - petal_width_median
    ) / petal_width_std

    input_samples[i] = (
        normalized_sepal_length,
        normalized_sepal_width,
        normalized_petal_length,
        normalized_petal_width
    )


def evaluate_genomes(genomes, config):
    """
    Evaluate all genomes in the current NEAT generation.

    Each genome is converted into a feed-forward neural network.
    The network receives all Iris samples and receives one point
    for each correctly classified sample.
    """

    for genome_id, genome in genomes:
        correct_predictions = 0.0

        # Create a neural network from the genome
        network = neat.nn.FeedForwardNetwork.create(
            genome,
            config
        )

        # Evaluate the genome on all samples
        for input_sample, expected_output in zip(
            input_samples,
            expected_outputs
        ):
            network_output = network.activate(input_sample)

            predicted_output = predict_class(network_output)

            if predicted_output == expected_output:
                correct_predictions += 1

        # Fitness is the classification accuracy
        genome.fitness = correct_predictions / len(input_samples)


def run_neat(config_file):
    """
    Run the NEAT evolutionary process.

    Parameters
    ----------
    config_file : str
        Path to the NEAT configuration file.
    """

    # Load NEAT configuration
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )

    # Create the population, which is the main object of a NEAT run
    population = neat.Population(config)

    # Add a reporter to show progress in the terminal
    population.add_reporter(
        neat.StdOutReporter(True)
    )

    # Add a statistics reporter to collect evolution data
    statistics = neat.StatisticsReporter()
    population.add_reporter(statistics)

    # Run NEAT for up to 400 generations
    winner = population.run(
        evaluate_genomes,
        400
    )

    # Display the winning genome
    print("\nBest genome:\n{!s}".format(winner))

    # Show the output of the best genome against the training data
    print("\nOutput:")

    winner_network = neat.nn.FeedForwardNetwork.create(
        winner,
        config
    )

    for input_sample, expected_output in zip(
        input_samples,
        expected_outputs
    ):
        network_output = winner_network.activate(input_sample)
        predicted_output = predict_class(network_output)

        print(
            "input {!r}, expected output {!r}, got {!r}".format(
                input_sample,
                expected_output,
                predicted_output
            )
        )

    # Node names used in the network visualization
    node_names = {
        -1: "Sepal length",
        -2: "Sepal width",
        -3: "Petal length",
        -4: "Petal width",
        0: "Setosa",
        1: "Versicolor",
        2: "Virginica"
    }

    # Draw the winning neural network
    neat_visualization.draw_net(
        config,
        winner,
        view=True,
        node_names=node_names
    )

    # Plot fitness statistics
    neat_visualization.plot_stats(
        statistics,
        ylog=False,
        view=True
    )

    # Plot species evolution
    neat_visualization.plot_species(
        statistics,
        view=True
    )


if __name__ == "__main__":
    # Determine the path to the configuration file.
    # This allows the script to run correctly regardless of
    # the current working directory.

    local_directory = os.path.dirname(__file__)

    config_path = os.path.join(
        local_directory,
        "neat_iris_classifier_config"
    )

    run_neat(config_path)


# Study notes:
# - Understand how the __main__ block works.
# - Study the neat_visualization module.
# - Study how neat.nn creates neural networks from genomes.

