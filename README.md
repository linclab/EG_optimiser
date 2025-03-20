# EG_optimiser
A fork of SGD and AdamW from PyTorch that implements exponentiated gradient descent (EG [Kivinen and Warmuth, 1997](https://www.sciencedirect.com/science/article/pii/S0890540196926127))
used in our paper "Brain-like learning with exponentiated gradients".

## Installation
```
git clone https://github.com/linclab/EG_optimiser.git
cd EG_optimiser
pip install -e . --no-deps # use --no-deps if pytorch is already installed (e.g. in Colab)
```

## Usage

See examples in the [example notebook](https://github.com/linclab/EG_optimiser/blob/main/example.ipynb).

## Reference
If you find this code useful, please cite our paper!
https://www.biorxiv.org/content/10.1101/2024.10.25.620272v1

```
@article {Cornford2024.10.25.620272,
	author = {Cornford, Jonathan and Pogodin, Roman and Ghosh, Arna and Sheng, Kaiwen and Bicknell, Brendan and Codol, Olivier and Clark, Beverley A and Lajoie, Guillaume and Richards, Blake},
	title = {Brain-like learning with exponentiated gradients},
	elocation-id = {2024.10.25.620272},
	year = {2024},
	doi = {10.1101/2024.10.25.620272},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Computational neuroscience relies on gradient descent (GD) for training artificial neural network (ANN) models of the brain. The advantage of GD is that it is effective at learning difficult tasks. However, it produces ANNs that are a poor phenomenological fit to biology, making them less relevant as models of the brain. Specifically, it violates Dale{\textquoteright}s law, by allowing synapses to change from excitatory to inhibitory and leads to synaptic weights that are not log-normally distributed, contradicting experimental data. Here, starting from first principles of optimisation theory, we present an alternative learning algorithm, exponentiated gradient (EG), that respects Dale{\textquoteright}s Law and produces log-normal weights, without losing the power of learning with gradients. We also show that in biologically relevant settings EG outperforms GD, including learning from sparsely relevant signals and dealing with synaptic pruning. Altogether, our results show that EG is a superior learning algorithm for modelling the brain with ANNs.Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2024/10/26/2024.10.25.620272},
	eprint = {https://www.biorxiv.org/content/early/2024/10/26/2024.10.25.620272.full.pdf},
	journal = {bioRxiv}
}
```
