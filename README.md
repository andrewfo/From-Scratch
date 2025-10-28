Micrograd Neural Network Extension

An extended version of [Andrej Karpathy’s Micrograd](https://github.com/karpathy/micrograd), built from scratch in pure Python.
This project adds modular neural network components (`Neuron`, `Layer`, and `MLP` classes) and a computation graph visualizer for inspecting forward and backward passes.

Features

 Custom Autograd Engine:
  Built on Micrograd’s `Value` objects with full forward and backward pass support.

 Trainable Neural Layers:
  Added `Neuron`, `Layer`, and `MLP` classes for constructing and training multilayer perceptrons.

 Graph Visualization:
  Integrated Graphviz support to visualize computation graphs, showing data flow and gradient propagation.

 Minimal, Transparent Design:
  No external ML libraries — just Python, math, and a few dozen lines of code.



Example

```python
from micrograd import Value
from nn import MLP

# Create a simple 3-layer perceptron (3 inputs → [4, 4, 1] hidden structure)
model = MLP(3, [4, 4, 1])

# Example input
x = [Value(2.0), Value(-1.0), Value(0.5)]
output = model(x)

print("Output:", output)
```

Graph Visualization

You can visualize the computation graph and gradient flow with:

```python
from graphviz import Digraph
from draw import draw_dot

dot = draw_dot(output)
dot.render('graph', view=True)
```

This generates a diagram showing how values and operations connect through the network.


 Architecture Overview

```
Value → Neuron → Layer → MLP
       ↘ autograd (backpropagation)
```

Each layer is composed of multiple neurons, each neuron holds weights and biases as `Value` objects, and gradients are computed automatically through Micrograd’s autograd engine.



Requirements

 Python ≥ 3.8
 graphviz (`pip install graphviz`)

 Future Extensions

 Add `Linear`, `Conv1D`, and simple `RNN` layers
 Implement different activation functions (ReLU, Sigmoid, etc.)
 Support training utilities (loss tracking, batching, optimizers)


 Author

Andrew 
CS Student @ UT Austin
Exploring the intersection of machine learning, systems design, and interpretability.


