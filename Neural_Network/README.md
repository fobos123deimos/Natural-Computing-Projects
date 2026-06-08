# 🧠 Neural Network From Scratch

Educational implementation of basic **neural networks from scratch** using **Python**, **NumPy**, and **Matplotlib**.

This project includes:

```text
Single-layer perceptron
Multilayer perceptron
Feedforward neural network
Iris classification
Adult Income classification
Yeast protein localization classification
```

---

## 🧠 Project Overview

This repository was created to study how neural networks work internally, without relying on high-level machine learning frameworks such as TensorFlow, PyTorch, Keras, or scikit-learn.

The focus is on:

```text
weight matrices
activation functions
transfer functions
forward propagation
basic training loops
manual dataset preprocessing
manual target encoding
```

---

## 🧰 Dependencies

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square\&logo=python\&logoColor=white)]()
[![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?style=flat-square\&logo=numpy\&logoColor=white)]()
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=flat-square)]()
[![Neural Networks](https://img.shields.io/badge/Neural%20Networks-From%20Scratch-orange?style=flat-square)]()

Install the dependencies:

```bash
pip install numpy matplotlib
```

---

## 📂 Repository Structure

```text
Neural_Network/
│
├── adult_income_mlp_training.py
├── adult_income_train.txt
├── adult_income_test.txt
│
├── feedforward_neural_network.py
│
├── iris_single_layer_perceptron_training.py
├── iris_train.txt
├── iris_evaluation.txt
│
├── perceptron_implementation.py
│
├── yeast_protein_localization_mlp_training.py
├── yeast_protein_localization.txt
│
└── README.md
```

---

## ✅ Main Files

| File                                         | Description                                                           |
| -------------------------------------------- | --------------------------------------------------------------------- |
| `perceptron_implementation.py`               | Contains the single-layer and multilayer perceptron implementations.  |
| `feedforward_neural_network.py`              | Simple feedforward neural network focused on forward propagation.     |
| `iris_single_layer_perceptron_training.py`   | Trains and evaluates a single-layer perceptron on the Iris dataset.   |
| `adult_income_mlp_training.py`               | Trains and tests a multilayer perceptron on the Adult Income dataset. |
| `yeast_protein_localization_mlp_training.py` | Trains and tests a multilayer perceptron on the Yeast dataset.        |

---

## 🌸 Iris Classification

The Iris experiment uses a **single-layer perceptron** to classify flower species.

Dataset files:

```text
iris_train.txt
iris_evaluation.txt
```

Input features:

```text
sepal length
sepal width
petal length
petal width
```

Target encoding:

| Class             | Target       |
| ----------------- | ------------ |
| `Iris-setosa`     | `[0.0, 0.0]` |
| `Iris-versicolor` | `[0.0, 1.0]` |
| `Iris-virginica`  | `[1.0, 1.0]` |

Run:

```bash
python iris_single_layer_perceptron_training.py
```

---

## 👤 Adult Income Classification

The Adult Income experiment uses a **multilayer perceptron** to classify income groups.

Dataset files:

```text
adult_income_train.txt
adult_income_test.txt
```

Input features include:

```text
age
workclass
education
marital status
occupation
relationship
race
sex
capital gain
capital loss
hours per week
native country
```

Target encoding:

| Class   | Target  |
| ------- | ------- |
| `>50K`  | `[0.0]` |
| `<=50K` | `[1.0]` |

Run:

```bash
python adult_income_mlp_training.py
```

---

## 🧬 Yeast Protein Localization

The Yeast experiment uses a **multilayer perceptron** to classify protein localization sites.

Dataset file:

```text
yeast_protein_localization.txt
```

Input:

```text
8 numeric biological features
```

Output:

```text
4-value encoded protein localization class
```

Run:

```bash
python yeast_protein_localization_mlp_training.py
```

---

## 🧮 Model Summary

| Model                   | File                            | Main Use                              |
| ----------------------- | ------------------------------- | ------------------------------------- |
| Single-layer perceptron | `perceptron_implementation.py`  | Iris classification                   |
| Multilayer perceptron   | `perceptron_implementation.py`  | Adult Income and Yeast classification |
| Feedforward network     | `feedforward_neural_network.py` | Study of forward propagation          |

---

## 📈 Complexity Overview

This section summarizes the approximate computational complexity of the main models and scripts.

Let:

```text
N = number of samples
E = number of epochs
m = number of input features
h = number of hidden neurons per hidden layer
L = number of hidden layers
o = number of output neurons
W = total number of weights in the network
```

---

### Model Complexity

| Model                   | Forward Pass | Training per Sample |    Full Training |    Space |
| ----------------------- | -----------: | ------------------: | ---------------: | -------: |
| Single-layer perceptron |     O(m × o) |            O(m × o) | O(E × N × m × o) | O(m × o) |
| Feedforward network     |         O(W) |     Not implemented |  Not implemented |     O(W) |
| Multilayer perceptron   |         O(W) |                O(W) |     O(E × N × W) |     O(W) |

---

### Script Complexity

| Script                                       | Main Cost                            | Approximate Complexity |
| -------------------------------------------- | ------------------------------------ | ---------------------: |
| `iris_single_layer_perceptron_training.py`   | Training a single-layer perceptron   |       O(E × N × m × o) |
| `adult_income_mlp_training.py`               | Training an MLP on Adult Income data |           O(E × N × W) |
| `yeast_protein_localization_mlp_training.py` | Training an MLP on Yeast data        |           O(E × N × W) |
| `feedforward_neural_network.py`              | Running forward propagation          |                   O(W) |

---

### Dataset Processing Complexity

| Step                            |                                                  Complexity |
| ------------------------------- | ----------------------------------------------------------: |
| Reading dataset files           |                                                        O(N) |
| Converting numeric features     |                                                    O(N × m) |
| Mapping categorical values      |                                                    O(N × m) |
| Creating target tables          |                                                        O(N) |
| Creating shuffled epoch batches |                                                    O(E × N) |
| Evaluating predictions          | O(N × W) for MLP / O(N × m × o) for single-layer perceptron |

---

### Summary

The most expensive operation is the training loop.

For the single-layer perceptron, the cost grows mainly with:

```text
epochs × samples × input features × output neurons
```

For the multilayer perceptron, the cost grows mainly with:

```text
epochs × samples × total network weights
```

In practice, increasing the number of hidden layers, hidden neurons, epochs, or dataset size directly increases the execution time.

---

## ▶️ How to Run

From the project folder:

```bash
cd Neural_Network
```

Install dependencies:

```bash
pip install numpy matplotlib
```

Run one of the experiments:

```bash
python iris_single_layer_perceptron_training.py
python adult_income_mlp_training.py
python yeast_protein_localization_mlp_training.py
```

On Windows, you can also use:

```powershell
py iris_single_layer_perceptron_training.py
py adult_income_mlp_training.py
py yeast_protein_localization_mlp_training.py
```

---

## ⚠️ Import Note

If the folder name is `Neural_Network`, make sure imports are consistent.

Example:

```python
from Neural_Network.perceptron_implementation import M_Perceptron
```

If running scripts directly from inside the folder, a simpler import may be used:

```python
from perceptron_implementation import M_Perceptron
```

If using the refactored class names, the import may look like this:

```python
from perceptron_implementation import SingleLayerPerceptron
from perceptron_implementation import MultiLayerPerceptron
```

---

## 🧭 Future Improvements

Possible improvements:

```text
Add requirements.txt
Add __init__.py
Normalize input data
Add confusion matrices
Add precision, recall, and F1-score
Add random seed control
Improve activation functions
Add ReLU, sigmoid, and tanh options
Save training plots
Move datasets into a data/ folder
Move scripts into an experiments/ folder
Compare results with scikit-learn
```

---

## ⚠️ Notes

This project is educational.

The goal is not to replace production machine learning libraries. The goal is to understand the internal mechanics of neural networks.

For production projects, prefer:

```text
scikit-learn
PyTorch
TensorFlow
Keras
JAX
```

---

## 📚 References

| Reference                                                          | Topic                              |
| ------------------------------------------------------------------ | ---------------------------------- |
| Simon Haykin — **Neural Networks and Learning Machines**           | Neural networks and learning rules |
| Christopher Bishop — **Pattern Recognition and Machine Learning**  | Classification and optimization    |
| Ian Goodfellow, Yoshua Bengio, Aaron Courville — **Deep Learning** | Modern deep learning               |
| Tom Mitchell — **Machine Learning**                                | Machine learning fundamentals      |
| UCI Machine Learning Repository — Iris                             | Iris dataset                       |
| UCI Machine Learning Repository — Adult                            | Adult Income dataset               |
| UCI Machine Learning Repository — Yeast                            | Yeast dataset                      |
| NumPy Documentation                                                | Numerical computing                |
| Matplotlib Documentation                                           | Plotting and visualization         |

---

## 📄 License

This project is available for educational and study purposes.
