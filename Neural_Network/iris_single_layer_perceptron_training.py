"""
Iris Single-Layer Perceptron Training Script

This script trains and evaluates a single-layer perceptron on Iris datasets.

Expected dataset files:
- iris.txt       : training dataset
- bezdekIris.txt : evaluation dataset

The script:
1. Loads and preprocesses the Iris dataset.
2. Converts Iris species labels into numeric target vectors.
3. Builds randomized epoch batches.
4. Trains a single-layer perceptron.
5. Evaluates the trained model by species.

Target classes:
- Iris-setosa     -> [0.0, 0.0]
- Iris-versicolor -> [0.0, 1.0]
- Iris-virginica  -> [1.0, 1.0]
"""

from random import shuffle

from perceptron_implementation import SingleLayerPerceptron


# -------------------------------------------------------------------------
# Dataset paths
# -------------------------------------------------------------------------

TRAINING_DATASET_PATH = "iris.txt"
EVALUATION_DATASET_PATH = "bezdekIris.txt"


# -------------------------------------------------------------------------
# Training configuration
# -------------------------------------------------------------------------

NUMBER_OF_EPOCHS = 10
SAMPLES_PER_TRAINING_CYCLE = 7

INPUT_FEATURE_COUNT = 4
OUTPUT_NEURON_COUNT = 2
LEARNING_RATE = 1.0


# -------------------------------------------------------------------------
# Iris class mapping
# -------------------------------------------------------------------------

IRIS_CLASS_MAPPING = {
    "Iris-setosa": [0.0, 0.0],
    "Iris-versicolor": [0.0, 1.0],
    "Iris-virginica": [1.0, 1.0],
}


IRIS_CLASS_NAMES = {
    str([0.0, 0.0]): "Iris-setosa",
    str([0.0, 1.0]): "Iris-versicolor",
    str([1.0, 1.0]): "Iris-virginica",
}


def parse_iris_dataset_line(line):
    """
    Parses one line from an Iris dataset file.

    Parameters
    ----------
    line : str
        Raw line from the dataset file.

    Returns
    -------
    tuple or None
        A tuple containing:
        - input_features : list
        - target_output : list

        Returns None when the line is empty or malformed.
    """

    if line == "\n":
        return None

    values = line.strip().split(",")

    if len(values) != 5:
        return None

    input_features = [float(value) for value in values[0:4]]
    class_label = values[4]

    if class_label not in IRIS_CLASS_MAPPING:
        return None

    target_output = IRIS_CLASS_MAPPING[class_label]

    return input_features, target_output


def load_iris_dataset(dataset_path):
    """
    Loads and preprocesses an Iris dataset file.

    Duplicate feature vectors are ignored, preserving the behavior of the
    original script.

    Parameters
    ----------
    dataset_path : str
        Path to the Iris dataset file.

    Returns
    -------
    tuple
        A tuple containing:
        - feature_vectors : list
        - target_outputs : list
    """

    feature_vectors = []
    target_outputs = []

    with open(dataset_path, "r") as dataset_file:
        for line in dataset_file:
            parsed_line = parse_iris_dataset_line(line)

            if parsed_line is None:
                continue

            input_features, target_output = parsed_line

            if input_features not in feature_vectors:
                feature_vectors.append(input_features)
                target_outputs.append(target_output)

    return feature_vectors, target_outputs


def create_target_table(feature_vectors, target_outputs):
    """
    Creates a lookup table that maps each input vector to its expected output.
    """

    target_table = {}

    for sample_index in range(len(feature_vectors)):
        target_table[str(feature_vectors[sample_index])] = target_outputs[sample_index]

    return target_table


def create_training_cycle_groups(feature_vectors, samples_per_cycle):
    """
    Creates randomized training cycles.

    Each cycle contains up to samples_per_cycle input vectors.

    The original script manually selected seven random samples at a time.
    This function preserves that idea while avoiding repeated code and avoiding
    crashes when the dataset size is not perfectly divisible by the cycle size.
    """

    remaining_features = list(feature_vectors)
    shuffle(remaining_features)

    training_cycles = []

    while len(remaining_features) > 0:
        current_cycle = remaining_features[0:samples_per_cycle]
        del remaining_features[0:samples_per_cycle]

        training_cycles.append(current_cycle)

    return training_cycles


def create_epoch_batches(feature_vectors, number_of_epochs, samples_per_cycle):
    """
    Creates the full list of epoch batches.

    Each epoch contains randomized training cycles.
    """

    epoch_batches = []

    for _ in range(number_of_epochs):
        epoch_batches.append(
            create_training_cycle_groups(
                feature_vectors,
                samples_per_cycle,
            )
        )

    return epoch_batches


def train_perceptron(perceptron, epoch_batches, target_table):
    """
    Trains the perceptron using either the refactored or original method name.
    """

    try:
        perceptron.train(epoch_batches, target_table)
    except AttributeError:
        # Fallback for the original method name used by S_Perceptron.
        perceptron.learning(epoch_batches, target_table)


def predict_with_perceptron(perceptron, input_vector):
    """
    Predicts the class of one input vector using either the refactored or
    original method name.
    """

    try:
        return perceptron.predict(input_vector)
    except AttributeError:
        # Fallback for the original method name used by S_Perceptron.
        return perceptron.usar(input_vector)


def evaluate_per_species_accuracy(perceptron, feature_vectors, target_outputs):
    """
    Evaluates the perceptron accuracy separately for each Iris species.

    Parameters
    ----------
    perceptron : SingleLayerPerceptron
        Trained perceptron model.
    feature_vectors : list
        Input feature vectors.
    target_outputs : list
        Expected target vectors.

    Returns
    -------
    dict
        Accuracy percentage for each Iris species.
    """

    correct_predictions_by_class = {
        str([0.0, 0.0]): 0,
        str([0.0, 1.0]): 0,
        str([1.0, 1.0]): 0,
    }

    total_samples_by_class = {
        str([0.0, 0.0]): target_outputs.count([0.0, 0.0]),
        str([0.0, 1.0]): target_outputs.count([0.0, 1.0]),
        str([1.0, 1.0]): target_outputs.count([1.0, 1.0]),
    }

    target_table = create_target_table(feature_vectors, target_outputs)

    for input_vector in feature_vectors:
        predicted_output = predict_with_perceptron(perceptron, input_vector)
        expected_output = target_table[str(input_vector)]

        if predicted_output == expected_output:
            correct_predictions_by_class[str(expected_output)] += 1

    accuracy_by_class = {}

    for class_key, class_name in IRIS_CLASS_NAMES.items():
        total_samples = total_samples_by_class[class_key]

        if total_samples == 0:
            accuracy_by_class[class_name] = 0.0
        else:
            accuracy_by_class[class_name] = (
                correct_predictions_by_class[class_key] / total_samples
            ) * 100

    return accuracy_by_class


def print_accuracy_report(accuracy_by_class):
    """
    Prints the accuracy report by Iris species.
    """

    print("Accuracy by Iris species:")

    for class_name, accuracy in accuracy_by_class.items():
        print(f"{class_name}: {accuracy:.2f}%")


def main():
    """
    Main execution function.

    Loads training and evaluation data, trains the perceptron, evaluates it,
    and prints the species-level accuracy report.
    """

    training_features, training_targets = load_iris_dataset(TRAINING_DATASET_PATH)

    training_target_table = create_target_table(
        training_features,
        training_targets,
    )

    epoch_batches = create_epoch_batches(
        training_features,
        NUMBER_OF_EPOCHS,
        SAMPLES_PER_TRAINING_CYCLE,
    )

    iris_classifier = SingleLayerPerceptron(
        INPUT_FEATURE_COUNT,
        OUTPUT_NEURON_COUNT,
        LEARNING_RATE,
    )

    train_perceptron(
        iris_classifier,
        epoch_batches,
        training_target_table,
    )

    evaluation_features, evaluation_targets = load_iris_dataset(
        EVALUATION_DATASET_PATH
    )

    accuracy_by_class = evaluate_per_species_accuracy(
        iris_classifier,
        evaluation_features,
        evaluation_targets,
    )

    print_accuracy_report(accuracy_by_class)


if __name__ == "__main__":
    main()