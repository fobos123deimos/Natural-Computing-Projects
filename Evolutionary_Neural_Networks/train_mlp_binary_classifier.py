from Neural_Network.perceptron_implementation import MultilayerPerceptron
from random import shuffle


# Input samples and expected outputs
input_samples = []
expected_outputs = []

# Training and testing batches for each epoch
training_epochs = []
test_epochs = []

# Dictionaries that associate each sample with its expected output
training_table = {}
test_table = {}

# Number of training epochs
number_of_epochs = 200

# Neural network parameters
hidden_layers = 1
learning_rate = 0.935
momentum_constant = 0.0
transfer_function_variant = 3
weight_interval = 5

# Transfer function variants:
# 1 - Sigmoid
# 2 - Hyperbolic tangent
# 3 - Gaussian


# Read the dataset file
with open("binary_admission_dataset.txt", "r") as file:
    for line in file:
        # Ignore empty lines
        if line != "\n":
            # Remove line break and split values by comma
            values = line.strip().split(",")

            # Convert all values to float
            values = [float(value) for value in values]

            # The first two values are the input features
            input_samples.append(values[0:2])

            # Convert class labels:
            # Original class 0 becomes 1.0
            # Original class 1 becomes -1.0
            if values[2] == 0:
                expected_outputs.append(1.0)
            else:
                expected_outputs.append(-1.0)


# Auxiliary list with all input samples
auxiliary_samples = list(input_samples)


# Create shuffled training batches for each epoch
for _ in range(number_of_epochs):
    training_batch = auxiliary_samples[0:70]
    shuffle(training_batch)

    training_epochs.append(training_batch)


# Create shuffled testing batches for each epoch
for _ in range(number_of_epochs):
    test_batch = auxiliary_samples[70:100]
    shuffle(test_batch)

    test_epochs.append(test_batch)


# Create the training table using the first 70 samples
for i in range(70):
    training_table[str(input_samples[i])] = expected_outputs[i]


# Create the testing table using the last 30 samples
for i in range(30):
    sample_index = len(input_samples) - (i + 1)

    test_table[str(input_samples[sample_index])] = expected_outputs[sample_index]


# Group all network parameters in a single tuple
network_parameters = (
    hidden_layers,
    learning_rate,
    momentum_constant,
    transfer_function_variant,
    weight_interval
)


# Create the multilayer perceptron
classifier = MultilayerPerceptron(
    2,
    1,
    network_parameters,
    training_table
)


# Train and test the model
classifier.learning_test(
    training_epochs,
    test_epochs,
    test_table
)