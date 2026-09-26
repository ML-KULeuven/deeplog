---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Semantic Loss

To showcase how DeepLog would be included in a normal ML pipeline, we implement the [Semantic Loss](https://proceedings.mlr.press/v80/xu18h.html) framework in DeepLog.

In this experiment, we train a neural network on the MNIST dataset in a semi-supervised setting, discarding most of the labels. A constraint is added that enforces exactly one output of the neural network is 1, while all others should be 0.



+++

## Neural network

First, we define the neural network that will be regularized. We use per-class sigmoids instead of a Softmax because this is a semi-supervised setting: the constraint encourages exactly-one behavior even when labels are missing, whereas Softmax would force a distribution regardless of the constraint signal.

```{code-cell} ipython3
import torch.nn as nn


class MNISTNet(nn.Module):
    def __init__(self, n=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8 x 14 x 14
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16 x 7 x 7
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x


mlp = MNISTNet()
```

## Dataset

We now define the semi-supervised MNIST dataset.

```{code-cell} ipython3
import os
import random

import torch
import torchvision.transforms.v2 as tf
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torchvision.datasets import MNIST


def per_image_standardization(image: torch.Tensor) -> torch.Tensor:
    """
    An implementation of the per_image_standardization function of TensorFlow.
    """
    stddev, mean = torch.std_mean(image, (-3, -2, -1))
    adjusted_stddev = torch.max(
        stddev, torch.rsqrt(torch.tensor(image.numel(), dtype=float))
    )
    return (image - mean) / adjusted_stddev


class SemiSupervised(Dataset):

    def __init__(self, dataset: Dataset, max_labeled):
        self.dataset = dataset
        self.max_labeled = max_labeled

    def __getitem__(self, index):
        # In original implementation, each batch was half labeled and half unlabeled examples.
        # Here, we iterate over the unlabeled examples, and add a random labeled example for each unlabeled one.
        # That's why the batch size here is half the one used in the original implementation.
        random_labeled = self.dataset[random.randrange(self.max_labeled)]
        return (
            self.dataset[index + self.max_labeled][0],
            random_labeled[0],
            random_labeled[1],
        )

    def __len__(self):
        return len(self.dataset) - self.max_labeled


transform = tf.Compose(
    [
        tf.ToImage(),
        tf.ToDtype(torch.float32, scale=True),
        tf.Lambda(per_image_standardization),
        tf.GaussianNoise(sigma=0.3),
        tf.RandomCrop(25),
        tf.Pad([1, 1, 2, 2]),
    ]
)

test_transform = tf.Compose(
    [
        tf.ToImage(),
        tf.ToDtype(torch.float32, scale=True),
        tf.Lambda(per_image_standardization),
    ]
)

# Downloaded datasets go to DEEPLOG_DATA_DIR, or to data/ beside this notebook.
data_dir = os.getenv("DEEPLOG_DATA_DIR", "data")
train_base = MNIST(train=True, root=data_dir, download=True, transform=transform)
test_base = Subset(
    MNIST(train=False, root=data_dir, download=True, transform=test_transform),
    range(1000),
)
max_labeled = 100

train_dataset = Subset(SemiSupervised(train_base, max_labeled), range(1000))
test_dataloader = DataLoader(test_base, batch_size=512, num_workers=0)
```

## Constraint

Exactly-one is a textbook CNF: one clause saying at least one output is on, and one clause per pair saying no two are on together. Forty-six clauses over ten variables — easier to generate than to write out, so we generate DIMACS text and let `parse_dimacs_cnf` build the boolean circuit.

A CNF is a *boolean* formula, and reading it as a probability is a weighted model count. That is what `transform_expectation_to_probability` does: it knowledge-compiles the circuit and emits it into the probability semiring, so the module takes one probability per digit and returns the probability that exactly one of them is on. Its `leaf_mapping` renames the DIMACS variables `v1`…`v10` to the network outputs they stand for, so the compiled module reads the ten sigmoids directly.

```{code-cell} ipython3
from deeplog import parse_dimacs_cnf, parse_symbol, reshape, SymTensor, to_module
from deeplog import CircuitFactory
from deeplog.formula.strategies import transform_expectation_to_probability
from deeplog.symbol import with_structure


num_outputs = 10


def exactly_one_cnf(n: int) -> str:
    """DIMACS CNF for 'exactly one of v1..vn is true'."""
    at_least_one = [tuple(range(1, n + 1))]
    at_most_one = [(-i, -j) for i in range(1, n + 1) for j in range(i + 1, n + 1)]
    clauses = at_least_one + at_most_one
    header = f"p cnf {n} {len(clauses)}"
    return "\n".join([header, *(" ".join(map(str, c)) + " 0" for c in clauses)])


def network_output(variable):
    """Name DIMACS variable ``vi`` after the network output it stands for."""
    return parse_symbol(f"nn({int(variable[0][1:]) - 1})")


print(exactly_one_cnf(3))  # the encoding in miniature

cnf = parse_dimacs_cnf(exactly_one_cnf(num_outputs), CircuitFactory())
(weighted,) = transform_expectation_to_probability(cnf, leaf_mapping=network_output)
constraint_module = to_module(weighted, names=(("exactly_one",),))

# One input column per network output, in the network's order.
constraint_module = reshape(
    constraint_module,
    input=SymTensor(
        [with_structure(parse_symbol(f"nn({i})"), "probability") for i in range(num_outputs)]
    ),
)
print(constraint_module)
```

## Complete model

We now construct the complete model as a PyTorch Lightning module.

```{code-cell} ipython3
import logging

import pytorch_lightning as pl
import torchmetrics

from deeplog.util import fast_dev_run_enabled


pl.utilities.disable_possible_user_warnings()
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


class LightningSL(pl.LightningModule):
    def __init__(self, network, constraint_module, constraint_weight, learning_rate):
        super().__init__()
        self.network = network
        self.constraint = constraint_module
        self.constraint_weight = constraint_weight
        self.loss = torch.nn.BCELoss()
        self.learning_rate = learning_rate
        self.train_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=10
        )
        self.test_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=10
        )
        self.constraint_history = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def forward(self, x) -> torch.Tensor:
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x_unlabeled, x_labeled, y = batch
        x = torch.cat([x_unlabeled, x_labeled])

        y_pred = self(x)
        y_constraint = self.constraint(y_pred)
        constraint_loss = -torch.log(y_constraint).mean(0)
        self.constraint_history.append(constraint_loss.detach().cpu().item())
        y_pred_labeled = torch.chunk(y_pred, 2)[1]
        y_one_hot = torch.nn.functional.one_hot(y, 10).float()

        classification_loss = self.loss(y_pred_labeled, y_one_hot)
        loss = classification_loss + self.constraint_weight * constraint_loss

        self.log("loss", loss, prog_bar=False, logger=False)
        self.log(
            "classification_loss", classification_loss, prog_bar=False, logger=False
        )
        self.log("constraint_loss", constraint_loss, prog_bar=False, logger=False)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        self.test_accuracy(y_pred, y)
        self.log("test_acc", self.test_accuracy)
```

## Constraint evaluation

To track how well the network satisfies the semantic loss, we evaluate the constraint module over the test set after training. The helper below runs the module on a dataloader and reports the average satisfaction probability.

```{code-cell} ipython3
def evaluate_constraint_on_loader(network, constraint_module, dataloader):
    """Return the mean satisfaction probability over ``dataloader``."""
    was_training = network.training
    network.eval()
    satisfaction = []
    with torch.no_grad():
        for inputs, *_ in dataloader:
            satisfaction.append(constraint_module(network(inputs)).flatten())
    if was_training:
        network.train()
    return torch.cat(satisfaction).mean().item()
```

## Baseline vs constrained training

We'll train two identical networks: one with the semantic loss (constraint) and one without. Both use the same architecture and optimizer so we can compare constraint satisfaction and classification accuracy, highlighting how the constraint supplies the exactly-one supervision in the absence of labels.

```{code-cell} ipython3
import copy


def make_models():
    constrained = LightningSL(copy.deepcopy(mlp), constraint_module, 0.005, 0.01)
    baseline = LightningSL(copy.deepcopy(mlp), constraint_module, 0.0, 0.01)
    return constrained, baseline


regularized_network, baseline_network = make_models()
```

### Training both models

```{code-cell} ipython3
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)

trainer = pl.Trainer(
    fast_dev_run=fast_dev_run_enabled(),
    max_epochs=2,
    enable_progress_bar=False,
    enable_model_summary=False,
    logger=False,
    enable_checkpointing=False,
)
trainer.fit(model=regularized_network, train_dataloaders=train_dataloader)
trainer.fit(model=baseline_network, train_dataloaders=train_dataloader)
```

### Constraint loss during training (constrained model)

```{code-cell} ipython3
import matplotlib.pyplot as plt


plt.plot(regularized_network.constraint_history, label="constraint loss")
plt.xlabel("Batch")
plt.ylabel("Constraint loss")
plt.legend()
plt.show()
```

### Testing

Evaluate classification accuracy and constraint satisfaction after training for both models.

```{code-cell} ipython3
constrained_results = trainer.test(
    model=regularized_network, dataloaders=test_dataloader, verbose=False
)[0]
baseline_results = trainer.test(
    model=baseline_network, dataloaders=test_dataloader, verbose=False
)[0]

post_constraint_constrained = evaluate_constraint_on_loader(
    regularized_network,
    constraint_module,
    test_dataloader,
)
post_constraint_baseline = evaluate_constraint_on_loader(
    baseline_network,
    constraint_module,
    test_dataloader,
)

print(f"Constrained test_acc: {constrained_results.get('test_acc')}")
print(f"Baseline test_acc:    {baseline_results.get('test_acc')}")
print(f"Constrained constraint prob: {post_constraint_constrained:.6f}")
print(f"Baseline constraint prob:    {post_constraint_baseline:.6f}")
```
