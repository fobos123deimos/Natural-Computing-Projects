"""
Yeast Protein Localization MLP Training Script

This script trains and tests a multilayer perceptron on the Yeast dataset.

Expected dataset file:
- yeast.txt

The script:
1. Loads and preprocesses the Yeast dataset.
2. Splits the dataset into training and test subsets.
3. Encodes protein localization classes as numeric target vectors.
4. Builds shuffled epoch batches.
5. Trains a multilayer perceptron to classify protein localization sites.

Target classes:
- CYT -> cytosolic or cytoskeletal
- NUC -> nuclear
- MIT -> mitochondrial
- ME3 -> membrane protein, no N-terminal signal
- ME2 -> membrane protein, uncleaved signal
- ME1 -> membrane protein, cleaved signal
- EXC -> extracellular
- VAC -> vacuolar
- POX -> peroxisomal
- ERL -> endoplasmic reticulum lumen
"""

from random import shuffle

from perceptron_implementation import MultiLayerPerceptron


YEAST_DATASET_PATH = "yeast.txt"

TRAINING_SAMPLE_COUNT = 742
TEST_SAMPLE_COUNT = 742

NUMBER_OF_EPOCHS = 200

HIDDEN_LAYER_COUNT = 1
LEARNING_RATE = 0.985
MOMENTUM_CONSTANT = 0.08
TRANSFER_FUNCTION_VARIANT = 1
WEIGHT_RANGE_LIMIT = 5
OUTPUT_DECIMAL_PLACES = 3
HIDDEN_NEURON_COUNT = 8

INPUT_FEATURE_COUNT = 8
OUTPUT_NEURON_COUNT = 4


# Previous experimental configurations:
#
# 35 epochs:
# HIDDEN_LAYER_COUNT / LEARNING_RATE / MOMENTUM_CONSTANT /
# TRANSFER_FUNCTION_VARIANT / WEIGHT_RANGE_LIMIT /
# OUTPUT_DECIMAL_PLACES / HIDDEN_NEURON_COUNT
#
# 1 / 0.985 / 0.0 / 1 / 3 / 3 / 6
# 1 / 0.995 / 0.0 / 1 / 4 / 3 / 7


PROTEIN_LOCALIZATION_MAPPING = {
    "CYT": [0.0, 0.0, 0.0, 0.0],
    "MIT": [0.0, 0.0, 0.0, 1.0],
    "ME3": [0.0, 0.0, 1.0, 0.0],
    "ME2": [0.0, 0.0, 1.0, 1.0],
    "ME1": [0.0, 1.0, 0.0, 0.0],
    "EXC": [0.0, 1.0, 0.0, 1.0],
    "VAC": [0.0, 1.0, 1.0, 0.0],
    "POX": [0.0, 1.0, 1.0, 1.0],
    "ERL": [1.0, 0.0, 0.0, 0.0],
    "NUC": [1.0, 0.0, 0.0, 1.0],
}


def parse_yeast_dataset_line(line):
    """
    Parses one line from the Yeast dataset.

    Parameters
    ----------
    line : str
        Raw line from the dataset file.

    Returns
    -------
    list or None
        Parsed values containing:
        - 8 numeric input features.
        - 1 protein localization class label.

        Returns None when the line is empty.
    """

    if line == "\n":
        return None

    # The original dataset format has irregular spacing.
    # Splitting by two spaces preserves the behavior of the original code.
    parsed_values = line.replace("\n", "").split("  ")

    # Removes leading fields until only the 8 features and the class label remain.
    while len(parsed_values) > 9:
        del parsed_values[0]

    for value_index in range(len(parsed_values) - 1):
        parsed_values[value_index] = float(parsed_values[value_index])

    return parsed_values


def load_yeast_dataset(dataset_path):
    """
    Loads and preprocesses the Yeast dataset.

    Parameters
    ----------
    dataset_path : str
        Path to the Yeast dataset file.

    Returns
    -------
    list
        List of parsed dataset rows.
    """

    parsed_dataset_rows = []

    with open(dataset_path, "r") as dataset_file:
        for line in dataset_file:
            parsed_line = parse_yeast_dataset_line(line)

            if parsed_line is None:
                continue

            parsed_dataset_rows.append(parsed_line)

    return parsed_dataset_rows


def split_features_and_targets(dataset_rows):
    """
    Splits the dataset into training and test features/targets.

    The original script uses:
    - First 742 samples for training.
    - Remaining 742 samples for testing.

    Parameters
    ----------
    dataset_rows : list
        Parsed Yeast dataset rows.

    Returns
    -------
    tuple
        A tuple containing:
        - training_features
        - training_targets
        - test_features
        - test_targets
    """

    training_features = []
    training_targets = []

    test_features = []
    test_targets = []

    training_rows = dataset_rows[0:TRAINING_SAMPLE_COUNT]
    test_rows = dataset_rows[TRAINING_SAMPLE_COUNT:TRAINING_SAMPLE_COUNT + TEST_SAMPLE_COUNT]

    for row in training_rows:
        input_features = row[0:8]
        class_label = row[8]

        training_features.append(input_features)
        training_targets.append(PROTEIN_LOCALIZATION_MAPPING[class_label])

    for row in test_rows:
        input_features = row[0:8]
        class_label = row[8]

        test_features.append(input_features)
        test_targets.append(PROTEIN_LOCALIZATION_MAPPING[class_label])

    return training_features, training_targets, test_features, test_targets


def create_epoch_batches(feature_vectors, number_of_epochs):
    """
    Creates shuffled epoch batches.

    A copy of the shuffled feature list is stored for each epoch.
    This avoids the repeated-reference issue present in the original script.
    """

    epoch_batches = []
    shuffled_features = list(feature_vectors)

    for _ in range(number_of_epochs):
        shuffle(shuffled_features)
        epoch_batches.append(shuffled_features.copy())

    return epoch_batches


def create_target_table(feature_vectors, target_outputs):
    """
    Creates a lookup table that maps each input vector to its expected output.
    """

    target_table = {}

    for sample_index in range(len(feature_vectors)):
        target_table[str(feature_vectors[sample_index])] = target_outputs[sample_index]

    return target_table


def main():
    """
    Main execution function.

    Loads the dataset, prepares training and test structures, initializes the
    neural network, and starts the training/test process.
    """

    dataset_rows = load_yeast_dataset(YEAST_DATASET_PATH)

    (
        training_features,
        training_targets,
        test_features,
        test_targets,
    ) = split_features_and_targets(dataset_rows)

    training_epoch_batches = create_epoch_batches(
        training_features,
        NUMBER_OF_EPOCHS,
    )

    test_epoch_batches = create_epoch_batches(
        test_features,
        NUMBER_OF_EPOCHS,
    )

    training_target_table = create_target_table(
        training_features,
        training_targets,
    )

    test_target_table = create_target_table(
        test_features,
        test_targets,
    )

    network_parameters = (
        HIDDEN_LAYER_COUNT,
        LEARNING_RATE,
        MOMENTUM_CONSTANT,
        TRANSFER_FUNCTION_VARIANT,
        WEIGHT_RANGE_LIMIT,
        OUTPUT_DECIMAL_PLACES,
        HIDDEN_NEURON_COUNT,
    )

    yeast_localization_classifier = MultiLayerPerceptron(
        INPUT_FEATURE_COUNT,
        OUTPUT_NEURON_COUNT,
        network_parameters,
        training_target_table,
    )

    try:
        yeast_localization_classifier.train_and_test(
            training_epoch_batches,
            test_epoch_batches,
            test_target_table,
        )
    except AttributeError:
        # Fallback for the original method name used by M_Perceptron.
        yeast_localization_classifier.learning_teste(
            training_epoch_batches,
            test_epoch_batches,
            test_target_table,
        )


if __name__ == "__main__":
    main()