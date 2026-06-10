Segue um README em inglês para essa pasta, já alinhado com os nomes atuais dos scripts e datasets.

# 🧬 Evolutionary Neural Networks

A collection of small experiments combining **genetic algorithms**, **neuroevolution**, **NEAT**, and **feed-forward neural networks**.

This folder explores different ways of training or evolving neural models, including:

* A custom genetic algorithm implementation.
* A simple feed-forward neural network.
* Multilayer Perceptron training scripts.
* NEAT-based classifiers.
* Neuroevolution experiments with Iris and binary datasets.
* Genetic action-sequence evolution for classic Gym environments.

---

## 📌 Overview

This project is part of a broader **Natural Computing** study repository.

The main goal is to experiment with how evolutionary techniques can be used to:

* Optimize neural network weights.
* Evolve action sequences for reinforcement learning environments.
* Train classifiers using both gradient-based and evolutionary strategies.
* Compare manual genetic encoding with NEAT-style topology evolution.

---

## 🧠 Main Concepts

This folder covers:

* **Genetic Algorithms**
* **Neuroevolution**
* **Feed-forward Neural Networks**
* **Multilayer Perceptrons**
* **NEAT**
* **Binary Classification**
* **Iris Classification**
* **Gym Environment Optimization**

---

## 📁 Folder Structure

```text
Evolutionary_Neural_Networks/
├── binary_admission_dataset.txt
├── three_feature_binary_dataset.txt
├── iris_train_dataset.txt
├── iris_test_dataset.txt
│
├── genetic_algorithm.py
├── feedforward_neural_network.py
├── neat_visualization.py
│
├── train_mlp_binary_classifier.py
├── train_mlp_iris_classifier.py
│
├── train_neat_binary_classifier.py
├── train_neat_iris_classifier.py
├── train_neat_iris_one_hot_classifier.py
│
├── neuroevolution_binary_classifier.py
├── neuroevolution_three_feature_classifier.py
├── neuroevolution_iris_classifier.py
├── neuroevolution_genetic_algorithm_demo.py
│
├── test_neuroevolution_encoding.py
│
├── evolve_mountain_car_actions.py
├── evolve_continuous_mountain_car_actions.py
├── evolve_acrobot_actions.py
│
├── neat_binary_classifier_config
├── neat_iris_classifier_config
│
└── mountain_car_genetic_algorithm_demo.m4v
```

---

## 🧩 Core Modules

### `genetic_algorithm.py`

Implements the custom genetic algorithm engine.

It includes:

* `Gene`
* `Chromosome`
* `GeneticMachine`
* Tournament selection
* Roulette selection
* Elitism
* Mutation
* Crossover
* Neural chromosome encoding

This module is used by the neuroevolution and Gym action-evolution experiments.

---

### `feedforward_neural_network.py`

Implements a simple feed-forward neural network.

It includes:

* Weight matrix initialization
* Random network creation
* Layer activation
* Transfer function using `tanh`
* Full feed-forward execution

This module is used in the custom neuroevolution scripts, where chromosomes encode network weights.

---

### `neat_visualization.py`

Utility module for visualizing NEAT experiments.

It can generate:

* Fitness evolution plots
* Species evolution plots
* Neural network topology diagrams using Graphviz

---

## 📊 Datasets

### `binary_admission_dataset.txt`

Binary classification dataset with two input features and one output class.

Format:

```text
feature_1,feature_2,class
```

Example:

```text
34.62365962451697,78.0246928153624,0
60.18259938620976,86.30855209546826,1
```

Used by:

```text
train_mlp_binary_classifier.py
neuroevolution_binary_classifier.py
train_neat_binary_classifier.py
```

---

### `three_feature_binary_dataset.txt`

Binary classification dataset with three input features.

Used by:

```text
neuroevolution_three_feature_classifier.py
```

---

### `iris_train_dataset.txt`

Training dataset for Iris classification experiments.

Used by:

```text
train_mlp_iris_classifier.py
train_neat_iris_classifier.py
train_neat_iris_one_hot_classifier.py
neuroevolution_iris_classifier.py
```

---

### `iris_test_dataset.txt`

Test dataset for Iris classification experiments.

Used by:

```text
train_mlp_iris_classifier.py
neuroevolution_iris_classifier.py
```

---

## ⚙️ Configuration Files

### `neat_binary_classifier_config`

NEAT configuration for binary classification.

Expected network structure:

```text
num_inputs  = 2
num_outputs = 1
```

Used by:

```text
train_neat_binary_classifier.py
```

---

### `neat_iris_classifier_config`

NEAT configuration for Iris classification.

Expected network structure depends on the script:

For two-output encoding:

```text
num_inputs  = 4
num_outputs = 2
```

For one-hot encoding:

```text
num_inputs  = 4
num_outputs = 3
```

Used by:

```text
train_neat_iris_classifier.py
train_neat_iris_one_hot_classifier.py
```

---

## 🚀 How to Run

From the project root, run one of the scripts:

```bash
python Evolutionary_Neural_Networks/train_mlp_binary_classifier.py
```

```bash
python Evolutionary_Neural_Networks/train_mlp_iris_classifier.py
```

```bash
python Evolutionary_Neural_Networks/train_neat_binary_classifier.py
```

```bash
python Evolutionary_Neural_Networks/train_neat_iris_classifier.py
```

```bash
python Evolutionary_Neural_Networks/neuroevolution_binary_classifier.py
```

```bash
python Evolutionary_Neural_Networks/evolve_mountain_car_actions.py
```

---

## 📦 Requirements

Install the main dependencies:

```bash
pip install numpy matplotlib neat-python graphviz gym
```

Graphviz may also require a system installation.

On Windows, install Graphviz and make sure its `bin` folder is available in the system `PATH`.

---

## 🧪 Experiments

### MLP Experiments

| Script                           | Description                                                    |
| -------------------------------- | -------------------------------------------------------------- |
| `train_mlp_binary_classifier.py` | Trains a multilayer perceptron on the binary admission dataset |
| `train_mlp_iris_classifier.py`   | Trains a multilayer perceptron on the Iris dataset             |

---

### NEAT Experiments

| Script                                  | Description                                             |
| --------------------------------------- | ------------------------------------------------------- |
| `train_neat_binary_classifier.py`       | Uses NEAT for binary classification                     |
| `train_neat_iris_classifier.py`         | Uses NEAT for Iris classification with compact encoding |
| `train_neat_iris_one_hot_classifier.py` | Uses NEAT for Iris classification with one-hot encoding |

---

### Custom Neuroevolution Experiments

| Script                                       | Description                                                    |
| -------------------------------------------- | -------------------------------------------------------------- |
| `neuroevolution_binary_classifier.py`        | Evolves feed-forward network weights for binary classification |
| `neuroevolution_three_feature_classifier.py` | Evolves network weights for a three-feature binary dataset     |
| `neuroevolution_iris_classifier.py`          | Evolves network weights for Iris classification                |
| `test_neuroevolution_encoding.py`            | Tests neural chromosome encoding and decoding                  |

---

### Gym Action Evolution

| Script                                      | Description                                                        |
| ------------------------------------------- | ------------------------------------------------------------------ |
| `evolve_mountain_car_actions.py`            | Evolves discrete action sequences for `MountainCar-v0`             |
| `evolve_continuous_mountain_car_actions.py` | Evolves continuous action sequences for `MountainCarContinuous-v0` |
| `evolve_acrobot_actions.py`                 | Evolves action sequences for `Acrobot-v1`                          |

---

## ⏱️ Complexity Overview

### Feed-forward Neural Network

For a network with layer sizes:

```text
L0, L1, L2, ..., Ln
```

The feed-forward cost is approximately:

```text
O(sum(Li * Li+1))
```

The memory usage for weights is also:

```text
O(sum(Li * Li+1))
```

---

### Genetic Algorithm

For:

```text
P = population size
C = chromosome length
G = number of generations
```

The approximate evolutionary cost is:

```text
O(G * P * C)
```

This does not include the cost of evaluating each chromosome in the environment or dataset.

---

### Neuroevolution

For:

```text
P = population size
C = chromosome length
N = number of samples
G = number of generations
```

The approximate cost is:

```text
O(G * P * N * C)
```

This is usually more expensive than direct supervised training because every chromosome must be evaluated over multiple samples.

---

### NEAT

NEAT complexity depends on:

* Population size
* Number of generations
* Number of species
* Number of nodes and connections per genome
* Dataset size

A simplified view is:

```text
O(G * P * N * E)
```

Where:

```text
G = generations
P = population size
N = number of samples
E = average number of enabled connections
```

---

## 📈 Outputs

Some scripts generate visual outputs such as:

* Fitness plots
* Species evolution plots
* Network topology graphs
* Environment renderings
* Training progress in the terminal

Graph files such as `Digraph.gv` may be generated automatically by Graphviz when using NEAT visualization utilities.

---

## 🧠 Notes

Some experiments use older exploratory implementations and may require small adjustments depending on:

* File paths
* Dataset names
* NEAT config names
* Gym version
* Graphviz installation
* Class and method names after refactoring

The project is educational and experimental. The main purpose is to study how evolutionary methods can interact with neural models and decision-making environments.

---

## 📚 References

* Kenneth O. Stanley and Risto Miikkulainen — *Evolving Neural Networks through Augmenting Topologies*
* NEAT-Python documentation
* UCI Machine Learning Repository — Iris Dataset
* OpenAI Gym / Gymnasium classic control environments
* Mitchell, M. — *An Introduction to Genetic Algorithms*
* Russell and Norvig — *Artificial Intelligence: A Modern Approach*
