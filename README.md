# Plumbed_3-Manifolds
This is the python implementation of the Graph Neural Networks studied in [arXiv:24XX.XXXXX](https://arxiv.org) based on PyTorch and PyTorch Geometric.

## Requirements
- PyTorch 2.3.1
- PyTorch Geometric 2.5.3
- NetworkX 3.3
- Numpy 1.26.4
- Matplotlib 3.9.0
## Usage
The program is composed of three main pieces:
- `utilities.py`,
- `torch_classe.py`,
- `Main_File.py`.
  
The `utilities.py` file contains functions aimed to generate a Random Plumbing graph and perform Neumann moves on it to then generate homeomorphic or non-homeomorphic graphs pairs.
In `torch_classe.py` all the Graph Neural Nework architectures are implemented together with the PyTorch dataclass we convert our graph to.
Finally, the `Main_File.py` script contains the heart of the program: a database of graph pairs is generated and multiple network structures are trained on it. The resulting comparative accuracy on the test and validation sets plus the loss function evolution over the epoch are plotted and stored respectively in `img_accuracy.pdf` and `img_loss.pdf`.
