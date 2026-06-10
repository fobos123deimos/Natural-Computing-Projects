Segue um `README.md` geral para a raiz do repositório `Natural-Computing-Projects`.

# 🌿 Natural Computing Projects

A collection of educational and experimental projects focused on **Natural Computing**, including neural networks, genetic algorithms, neuroevolution, NEAT, and biologically inspired computational models.

This repository explores how ideas inspired by nature can be used to solve computational problems, optimize systems, train models, and simulate adaptive behavior.

---

## 📌 Overview

Natural Computing is a field of Computer Science that studies computational models inspired by natural phenomena.

This repository includes experiments related to:

* Neural Networks
* Genetic Algorithms
* Neuroevolution
* NEAT
* Evolutionary Computation
* Bio-inspired optimization
* Machine Learning experiments
* Classic control environments
* Educational implementations from scratch

The main goal is not only to use ready-made libraries, but also to understand how these algorithms work internally.

---

## 📁 Repository Structure

```text
Natural-Computing-Projects/
├── Evolutionary_Neural_Networks/
├── Neural_Network/
└── README.md
```

---

## 🧬 Evolutionary Neural Networks

The `Evolutionary_Neural_Networks/` folder contains experiments combining genetic algorithms, neural networks, NEAT, and neuroevolution.

It includes:

* Custom genetic algorithm implementation
* Feed-forward neural network implementation
* NEAT classifiers
* Iris classification experiments
* Binary classification experiments
* Neuroevolution with custom chromosomes
* Genetic action evolution for Gym environments

Main topics:

```text
Genetic Algorithms
Neuroevolution
NEAT
Feed-forward Neural Networks
Evolutionary Optimization
Gym Classic Control
```

Example scripts:

```text
train_mlp_binary_classifier.py
train_mlp_iris_classifier.py
train_neat_binary_classifier.py
train_neat_iris_classifier.py
neuroevolution_binary_classifier.py
neuroevolution_iris_classifier.py
evolve_mountain_car_actions.py
evolve_acrobot_actions.py
```

---

## 🧠 Neural Network

The `Neural_Network/` folder contains implementations and experiments focused on neural network models.

It includes scripts related to:

* Perceptron
* Multilayer Perceptron
* Feed-forward networks
* Supervised learning
* Dataset loading and training loops
* Classification experiments

This folder is more focused on the basic structure of neural networks before combining them with evolutionary techniques.

---

## 🚀 How to Run

Clone the repository:

```bash
git clone https://github.com/fobos123deimos/Natural-Computing-Projects.git
cd Natural-Computing-Projects
```

Install the main dependencies:

```bash
pip install numpy matplotlib neat-python graphviz gym
```

Run a script from the repository root, for example:

```bash
python Evolutionary_Neural_Networks/train_mlp_binary_classifier.py
```

or:

```bash
python Evolutionary_Neural_Networks/train_neat_iris_classifier.py
```

Some scripts may require specific datasets or configuration files inside their own folders.

---

## 📦 Main Dependencies

The projects may use:

```text
numpy
matplotlib
neat-python
graphviz
gym
```

For Graphviz visualizations, the system version of Graphviz may also need to be installed and added to the system PATH.

---

## 🧪 Project Goals

This repository was created to study and experiment with:

* How neural networks process data
* How genetic algorithms evolve solutions
* How chromosomes can encode weights, actions, or neural structures
* How NEAT evolves both weights and topology
* How natural processes can inspire computational problem solving
* How different learning strategies compare in simple classification tasks

---

## ⏱️ Complexity Overview

### Feed-forward Neural Networks

For a network with layer sizes:

```text
L0, L1, L2, ..., Ln
```

The approximate feed-forward cost is:

```text
O(sum(Li * Li+1))
```

The memory cost for the weights is also:

```text
O(sum(Li * Li+1))
```

---

### Genetic Algorithms

For:

```text
P = population size
C = chromosome length
G = number of generations
```

The approximate cost is:

```text
O(G * P * C)
```

The real cost may be higher depending on how expensive the fitness evaluation is.

---

### Neuroevolution

For:

```text
P = population size
N = number of samples
C = chromosome length
G = number of generations
```

The approximate cost is:

```text
O(G * P * N * C)
```

This can be expensive because each chromosome needs to be evaluated over multiple samples.

---

### NEAT

NEAT evolves both connection weights and network topology.

A simplified cost estimate is:

```text
O(G * P * N * E)
```

Where:

```text
G = number of generations
P = population size
N = number of samples
E = average number of enabled connections per genome
```

---

## 📊 Datasets

Some experiments use classic small datasets, such as:

* Binary admission dataset
* Iris dataset
* Three-feature binary dataset

These datasets are used for classification experiments with MLPs, NEAT, and custom neuroevolution models.

---

## 🧠 Learning Notes

This repository is experimental and educational.

Some implementations are intentionally written from scratch to make the internal logic more visible, even when libraries could provide more optimized versions.

The code may include:

* Older exploratory scripts
* Refactored English versions
* Custom algorithm implementations
* Visualization utilities
* Dataset-specific experiments

---

## 📚 References

* Mitchell, M. — *An Introduction to Genetic Algorithms*
* Russell, S. and Norvig, P. — *Artificial Intelligence: A Modern Approach*
* Haykin, S. — *Neural Networks and Learning Machines*
* Stanley, K. O. and Miikkulainen, R. — *Evolving Neural Networks through Augmenting Topologies*
* NEAT-Python documentation
* UCI Machine Learning Repository — Iris Dataset
* OpenAI Gym / Gymnasium Classic Control environments

---

## 📝 Status

This repository is under active organization and refactoring.

The current focus is improving:

* File names
* Folder organization
* Documentation
* Code readability
* English naming conventions
* Consistency across neural network and genetic algorithm experiments
