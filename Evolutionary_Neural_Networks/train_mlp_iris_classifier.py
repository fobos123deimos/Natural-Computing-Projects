from random import shuffle

from Neural_Network.perceptron_implementation import MultilayerPerceptron


# Dataset files
TRAIN_DATASET_FILE = "iris_train_dataset.txt"
TEST_DATASET_FILE = "iris_test_dataset.txt"


# Input samples and expected outputs for training
training_inputs = []
training_outputs = []

# Input samples and expected outputs for testing
test_inputs = []
test_outputs = []

# Epoch lists
training_epochs = []
test_epochs = []

# Lookup tables:
# key   -> input sample converted to string
# value -> expected output
training_table = {}
test_table = {}


# Training parameters
number_of_epochs = 200

hidden_layers = 1
learning_rate = 0.8
momentum_constant = 0.0
transfer_function_variant = 1
weight_interval = 3
decimal_approximation = 3
hidden_neuron_count = 6


# Transfer function variants:
# 1 - Sigmoid
# 2 - Hyperbolic tangent
# 3 - Gaussian


def encode_iris_class(class_name):
    """
    Convert the Iris class name into a numerical output.

    Output encoding:
    - Iris-setosa     -> [0.0, 0.0]
    - Iris-versicolor -> [0.0, 1.0]
    - Iris-virginica  -> [1.0, 1.0]

    The same encoding must be used for both training and testing.
    """

    class_name = class_name.strip().lower()

    if "setosa" in class_name:
        return [0.0, 0.0]

    if "versicolor" in class_name:
        return [0.0, 1.0]

    if "virginica" in class_name:
        return [1.0, 1.0]

    raise ValueError(f"Unknown Iris class: {class_name}")


def load_iris_dataset(filename):
    """
    Load an Iris dataset file.

    Each line is expected to have the following format:

    sepal_length,sepal_width,petal_length,petal_width,class_name

    Returns
    -------
    tuple
        A tuple containing:
        - input samples
        - expected outputs
    """

    input_samples = []
    expected_outputs = []

    with open(filename, "r") as file:
        for line in file:
            # Ignore empty lines
            if line != "\n":
                values = line.strip().split(",")

                # Convert the first four values to float.
                # These values are the Iris numerical features.
                for i in range(len(values) - 1):
                    values[i] = float(values[i])

                input_features = values[0:4]
                class_name = values[4]

                # Avoid duplicated input samples
                if input_features not in input_samples:
                    input_samples.append(input_features)
                    expected_outputs.append(
                        encode_iris_class(class_name)
                    )

    return input_samples, expected_outputs


# Load training dataset
training_inputs, training_outputs = load_iris_dataset(
    TRAIN_DATASET_FILE
)

# Load test dataset
test_inputs, test_outputs = load_iris_dataset(
    TEST_DATASET_FILE
)


# Auxiliary lists used to create shuffled epochs
training_auxiliary_inputs = list(training_inputs)
test_auxiliary_inputs = list(test_inputs)


# Create the training epoch list.
# Important: append a copy of the shuffled list, not the same list reference.
for _ in range(number_of_epochs):
    shuffle(training_auxiliary_inputs)
    training_epochs.append(
        list(training_auxiliary_inputs)
    )


# Create the test epoch list.
# Important: append a copy of the shuffled list, not the same list reference.
for _ in range(number_of_epochs):
    shuffle(test_auxiliary_inputs)
    test_epochs.append(
        list(test_auxiliary_inputs)
    )


# Create the training lookup table
for i in range(len(training_inputs)):
    training_table[str(training_inputs[i])] = training_outputs[i]


# Create the test lookup table
for i in range(len(test_inputs)):
    test_table[str(test_inputs[i])] = test_outputs[i]


# Group all network parameters into a single tuple
network_parameters = (
    hidden_layers,
    learning_rate,
    momentum_constant,
    transfer_function_variant,
    weight_interval,
    decimal_approximation,
    hidden_neuron_count
)


# Create the multilayer perceptron.
#
# Parameters:
# 4 -> number of input neurons
# 2 -> number of output neurons
# network_parameters -> training and architecture configuration
# training_table -> expected output table for training samples
iris_classifier = MultilayerPerceptron(
    4,
    2,
    network_parameters,
    training_table
)


# Train and test the network
iris_classifier.learning_test(
    training_epochs,
    test_epochs,
    test_table
)