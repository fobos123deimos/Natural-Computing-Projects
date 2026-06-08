"""
Adult Income MLP Training Script

This script trains and tests a multilayer perceptron on the Adult Income dataset.

Expected dataset files:
- adult_L.txt : training dataset
- adult_T.txt : test dataset

The script:
1. Loads and preprocesses the Adult Income dataset.
2. Converts categorical values into numeric values.
3. Builds shuffled epoch batches for training and testing.
4. Trains a multilayer perceptron to classify income.

Target classes:
- >50K  -> [0.0]
- <=50K -> [1.0]
"""

from random import shuffle

from perceptron_implementation import MultiLayerPerceptron


# -------------------------------------------------------------------------
# Dataset paths
# -------------------------------------------------------------------------

TRAINING_DATASET_PATH = "adult_L.txt"
TEST_DATASET_PATH = "adult_T.txt"


# -------------------------------------------------------------------------
# Training configuration
# -------------------------------------------------------------------------

NUMBER_OF_EPOCHS = 35

HIDDEN_LAYER_COUNT = 1
LEARNING_RATE = 0.75
MOMENTUM_CONSTANT = 0.0
TRANSFER_FUNCTION_VARIANT = 1
WEIGHT_RANGE_LIMIT = 19
OUTPUT_DECIMAL_PLACES = 3
HIDDEN_NEURON_COUNT = 8

INPUT_FEATURE_COUNT = 12
OUTPUT_NEURON_COUNT = 1


# -------------------------------------------------------------------------
# Categorical value mappings
# -------------------------------------------------------------------------

WORKCLASS_MAPPING = {
    "Private": 0.0,
    "Self-emp-not-inc": 1.0,
    "Self-emp-inc": 2.0,
    "Federal-gov": 3.0,
    "Local-gov": 4.0,
    "State-gov": 5.0,
    "Without-pay": 6.0,
    "Never-worked": 7.0,
}

EDUCATION_MAPPING = {
    "Bachelors": 0.0,
    "Some-college": 1.0,
    "11th": 2.0,
    "HS-grad": 3.0,
    "Prof-school": 4.0,
    "Assoc-acdm": 5.0,
    "Assoc-voc": 6.0,
    "9th": 7.0,
    "7th-8th": 8.0,
    "12th": 9.0,
    "Masters": 10.0,
    "1st-4th": 11.0,
    "10th": 12.0,
    "Doctorate": 13.0,
    "5th-6th": 14.0,
    "Preschool": 15.0,
}

MARITAL_STATUS_MAPPING = {
    "Married-civ-spouse": 0.0,
    "Divorced": 1.0,
    "Never-married": 2.0,
    "Separated": 3.0,
    "Widowed": 4.0,
    "Married-spouse-absent": 5.0,
    "Married-AF-spouse": 6.0,
}

OCCUPATION_MAPPING = {
    "Tech-support": 0.0,
    "Craft-repair": 1.0,
    "Other-service": 2.0,
    "Sales": 3.0,
    "Exec-managerial": 4.0,
    "Prof-specialty": 5.0,
    "Handlers-cleaners": 6.0,
    "Machine-op-inspct": 7.0,
    "Adm-clerical": 8.0,
    "Farming-fishing": 9.0,
    "Transport-moving": 10.0,
    "Priv-house-serv": 11.0,
    "Protective-serv": 12.0,
    "Armed-Forces": 13.0,
}

RELATIONSHIP_MAPPING = {
    "Wife": 0.0,
    "Own-child": 1.0,
    "Husband": 2.0,
    "Not-in-family": 3.0,
    "Other-relative": 4.0,
    "Unmarried": 5.0,
}

RACE_MAPPING = {
    "White": 0.0,
    "Asian-Pac-Islander": 1.0,
    "Amer-Indian-Eskimo": 2.0,
    "Other": 3.0,
    "Black": 4.0,
}

SEX_MAPPING = {
    "Female": 0.0,
    "Male": 1.0,
}

NATIVE_COUNTRY_MAPPING = {
    "United-States": 0.0,
    "Cambodia": 1.0,
    "England": 2.0,
    "Puerto-Rico": 3.0,
    "Canada": 4.0,
    "Germany": 5.0,
    "Outlying-US(Guam-USVI-etc)": 6.0,
    "India": 7.0,
    "Japan": 8.0,
    "Greece": 9.0,
    "South": 10.0,
    "China": 11.0,
    "Cuba": 12.0,
    "Iran": 13.0,
    "Honduras": 14.0,
    "Philippines": 15.0,
    "Italy": 16.0,
    "Poland": 17.0,
    "Jamaica": 18.0,
    "Vietnam": 19.0,
    "Mexico": 20.0,
    "Portugal": 21.0,
    "Ireland": 22.0,
    "France": 23.0,
    "Dominican-Republic": 24.0,
    "Laos": 25.0,
    "Ecuador": 26.0,
    "Taiwan": 27.0,
    "Haiti": 28.0,
    "Columbia": 29.0,
    "Hungary": 30.0,
    "Guatemala": 31.0,
    "Nicaragua": 32.0,
    "Scotland": 33.0,
    "Thailand": 34.0,
    "Yugoslavia": 35.0,
    "El-Salvador": 36.0,
    "Trinadad&Tobago": 37.0,
    "Peru": 38.0,
    "Hong": 39.0,
    "Holand-Netherlands": 40.0,
}

INCOME_CLASS_MAPPING = {
    ">50K": [0.0],
    "<=50K": [1.0],
}


def parse_adult_dataset_line(line, is_test_dataset=False):
    """
    Parses and converts one line from the Adult Income dataset.

    Parameters
    ----------
    line : str
        Raw dataset line.
    is_test_dataset : bool
        Indicates whether the line belongs to the test dataset.
        Test labels usually contain a final period, such as ">50K.".

    Returns
    -------
    tuple or None
        A tuple containing:
        - input_features : list
        - target_output : list

        Returns None when the line should be ignored.
    """

    # The original script only accepted lines with 14 spaces and no missing values.
    if line.count(" ") != 14 or "?" in line:
        return None

    if is_test_dataset and line.strip() == "|1x3 Cross validator":
        return None

    cleaned_line = line.replace(" ", "").replace("\n", "")

    if is_test_dataset:
        cleaned_line = cleaned_line.replace(".", "")

    values = cleaned_line.split(",")

    # Removes unused columns from the original dataset:
    # - fnlwgt
    # - education-num
    del values[2]
    del values[3]

    values[0] = float(values[0])                     # age
    values[1] = WORKCLASS_MAPPING[values[1]]         # workclass
    values[2] = EDUCATION_MAPPING[values[2]]         # education
    values[3] = MARITAL_STATUS_MAPPING[values[3]]    # marital-status
    values[4] = OCCUPATION_MAPPING[values[4]]        # occupation
    values[5] = RELATIONSHIP_MAPPING[values[5]]      # relationship
    values[6] = RACE_MAPPING[values[6]]              # race
    values[7] = SEX_MAPPING[values[7]]               # sex
    values[8] = float(values[8])                     # capital-gain
    values[9] = float(values[9])                     # capital-loss
    values[10] = float(values[10])                   # hours-per-week
    values[11] = NATIVE_COUNTRY_MAPPING[values[11]]  # native-country
    values[12] = INCOME_CLASS_MAPPING[values[12]]    # income class

    input_features = values[0:12]
    target_output = values[12]

    return input_features, target_output


def load_adult_dataset(dataset_path, is_test_dataset=False):
    """
    Loads and preprocesses the Adult Income dataset.

    Parameters
    ----------
    dataset_path : str
        Path to the dataset file.
    is_test_dataset : bool
        Indicates whether the dataset is the test dataset.

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
            parsed_line = parse_adult_dataset_line(
                line,
                is_test_dataset=is_test_dataset,
            )

            if parsed_line is None:
                continue

            input_features, target_output = parsed_line

            feature_vectors.append(input_features)
            target_outputs.append(target_output)

    return feature_vectors, target_outputs


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

    Loads the data, prepares the training/test structures, initializes the
    neural network, and starts the training/test process.
    """

    training_features, training_targets = load_adult_dataset(
        TRAINING_DATASET_PATH,
        is_test_dataset=False,
    )

    test_features, test_targets = load_adult_dataset(
        TEST_DATASET_PATH,
        is_test_dataset=True,
    )

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

    adult_income_classifier = MultiLayerPerceptron(
        INPUT_FEATURE_COUNT,
        OUTPUT_NEURON_COUNT,
        network_parameters,
        training_target_table,
    )

    try:
        adult_income_classifier.train_and_test(
            training_epoch_batches,
            test_epoch_batches,
            test_target_table,
        )
    except AttributeError:
        # Fallback for the original method name used by M_Perceptron.
        adult_income_classifier.learning_teste(
            training_epoch_batches,
            test_epoch_batches,
            test_target_table,
        )


if __name__ == "__main__":
    main()