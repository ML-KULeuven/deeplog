---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# DeepProbLog: neural probabilistic logic programs

+++

In the `problog` notebook every fact carried a *fixed* probability — `0.6::burglary` — baked straight into the compiled circuit. **DeepProbLog** keeps the logic program but lets a **neural network** supply those probabilities instead: `classifier(I,N) :: digit(I,N)` says the probability that image `I` shows digit `N` is whatever the network `classifier` predicts.

The logic is unchanged — DeepLog still proves the queries and compiles them into a differentiable circuit — but now the circuit's inputs are *network outputs*, so gradients flow from a query all the way back into the network's weights. That is what makes it *deep*.

This notebook builds up in three steps:

1. **From ProbLog to DeepProbLog** — a digit is declared by an *annotated disjunction*
   (a mutually-exclusive distribution) whose probabilities come from outside the program.
2. **A tiny example** — adding two one-bit digits, first with those probabilities supplied
   by hand, then with the network named in the program itself.
3. **MNIST addition** — the classic NeSy experiment, end to end.

+++

## From ProbLog to DeepProbLog

A network classifier outputs a *distribution*: image `I` is digit 0 **or** 1 **or** … **or** 9, with probabilities that sum to one. We tell DeepLog this with an **annotated disjunction** — the `;`-separated rule

```
classifier(i1,0) :: digit(i1,0); classifier(i1,1) :: digit(i1,1); ... .
```

which reads "image `i1` is *exactly one* of these digits." The engine takes that as one **variable** whose values are the branch atoms. Declaring a variable selects the multi-valued-SDD backend, which keeps the digits mutually exclusive; as independent probabilistic facts they could hold at once, and their probabilities would not sum to one.

The other difference from ProbLog is where the numbers come from. `0.6::burglary` is a constant, so it is folded into the compiled circuit and the module needs nothing at call time. `classifier(i1,0)` names no number in the program, so it survives compilation as a **runtime input** — a slot for a network's output. We fill that slot by hand first; naming *which* network fills it is the `nn(...)` annotation, further down.

```{code-cell}
import math

import torch
from torch import nn

from deeplog import DeepLogModuleFactory
from deeplog import SymTensor
from deeplog import get_network_predicate
from deeplog import reshape
from deeplog import symbol_to_pretty_string
from deeplog import to_dict
from deeplog import CircuitFactory
from deeplog.grounding import SimpleGrounder
from deeplog.systems.deepproblog import Solver
from deeplog.systems.deepproblog import compile_to_module
from deeplog.grounding import str_to_rules
```

### A tiny example: adding two one-bit digits

Two images, each a digit in `{0, 1}`; we query their sum. This is MNIST addition in miniature — same shape, but small enough to read every number.

Compiling takes the same two steps as in the `problog` notebook: `get_query_result` proves every `?-` query into a boolean proof, and `compile_to_module` lowers those proofs into one module.

```{code-cell}
MINI = """
classifier(i1,0) :: digit(i1,0); classifier(i1,1) :: digit(i1,1).
classifier(i2,0) :: digit(i2,0); classifier(i2,1) :: digit(i2,1).
addition(I1,I2,S) :- digit(I1,N1), digit(I2,N2), is(S,+(N1,N2)).
?- addition(i1,i2,S).
"""

program = tuple(str_to_rules(MINI))
result = Solver(SimpleGrounder()).get_query_result(program, CircuitFactory())
circuit = compile_to_module(result, DeepLogModuleFactory())

print("inputs  :", *circuit.get_input_shape())
print("outputs :", circuit.get_output_shape())
```

Unlike the ProbLog alarm network — whose module needed **no** inputs, because every probability was baked in — this module's inputs are the four `classifier` probabilities, exactly what a network will produce, and its outputs are the three possible sums.

The layout, though, is the proofs' and not ours. The probabilities arrive as four separate single-column arguments, in the order the search happened to reach them, and the sums come out in numeric order only because that is where the proofs left them — the module promises nothing about either. But both shapes are *symbolic*: every slot is named by the atom it carries, tagged with the algebra of its value, `addition(i1,i2,1) _ probability`. A name is all `reshape` needs to rewire a module, so rather than read either end positionally, we re-declare what the circuit speaks — one distribution per image going in, the sums `0, 1, 2` in that order coming out.

```{code-cell}
circuit = reshape(
    circuit,
    input=tuple(
        SymTensor([f"classifier({image},{digit}) _ probability" for digit in range(2)])
        for image in ("i1", "i2")
    ),
    output=SymTensor([f"addition(i1,i2,{total}) _ probability" for total in range(3)]),
)

print("inputs  :", *circuit.get_input_shape())
print("outputs :", circuit.get_output_shape())
```

The circuit now takes one argument per image. Before naming a network, let's play its part ourselves and supply the two distributions by hand; `to_dict` reads the result back by name.

```{code-cell}
i1 = torch.tensor([[0.2, 0.8]])  # P(i1 is a 0) = 0.2,  P(i1 is a 1) = 0.8
i2 = torch.tensor([[0.6, 0.4]])  # P(i2 is a 0) = 0.6,  P(i2 is a 1) = 0.4

to_dict(circuit(i1, i2), circuit.get_output_shape())
```

These are the exact convolution of the two digit distributions:

- `P(sum=0) = P(i1=0)·P(i2=0) = 0.2·0.6 = 0.12`
- `P(sum=1) = P(i1=0)·P(i2=1) + P(i1=1)·P(i2=0) = 0.2·0.4 + 0.8·0.6 = 0.56`
- `P(sum=2) = P(i1=1)·P(i2=1) = 0.8·0.4 = 0.32`

and they sum to one — because the annotated disjunction made each image's digits mutually exclusive. The logic program did the probabilistic bookkeeping for us.

+++

### What the disjunction declared

That mutual exclusivity is not something to take on trust — it is a **variable** the engine declared, and `result.variables` reports it. There is one variable per *image*, not one per branch atom, carrying the domain that image's digit ranges over.

Which is the whole difference. Two variables of two values are four joint assignments, every one of them a legal pair of digits. The same four atoms as *independent* probabilistic facts would be four booleans and sixteen worlds, most of them meaningless — an image that is both a `0` and a `1`, or neither — and the query probabilities would not sum to one.

```{code-cell}
variables = sorted(result.variables, key=lambda variable: variable.name)
for variable in variables:
    values = ", ".join(value[0] for value in variable.domain.values)
    print(f"{symbol_to_pretty_string(variable.name)} over {{{values}}}")

sizes = [len(variable) for variable in variables]
print(f"\n{len(sizes)} variables, {math.prod(sizes):,} joint assignments")
print(f"{sum(sizes)} independent booleans would give {2 ** sum(sizes):,}")
```

### Naming the network in the program

Filling those four slots by hand is not the point; a network should fill them. DeepProbLog says which one with a **neural annotation**:

```
nn(classifier, [X], Y, [0..1]) :: digit(X,Y).
```

That is the same annotated disjunction, written once over a free image instead of enumerated per image: the network `classifier` is applied to `X`, its output ranges over `Y` in `0..1`, and `digit(X,Y)` takes the probability the network gives row `Y`. Because the annotation declares the domain, the rule no longer has to enumerate it either.

Binding the name `classifier` to an actual module is the compiler's side of the deal. `get_network_predicate` builds the predicate that runs the module, and `DeepLogModuleFactory` accepts it in `atom_builders`, keyed by the `(functor, arity, structure)` signature it implements. The network then lives *inside* the compiled module, whose inputs are the images themselves.

```{code-cell}
NEURAL_MINI = """
nn(classifier, [X], Y, [0..1]) :: digit(X,Y).
addition(I1,I2,S) :- digit(I1,N1), digit(I2,N2), is(S,+(N1,N2)).
?- addition(i1,i2,S).
"""

torch.manual_seed(0)
tiny = nn.Sequential(nn.Linear(4, 2), nn.Softmax(dim=1))

program = tuple(str_to_rules(NEURAL_MINI))
result = Solver(SimpleGrounder()).get_query_result(program, CircuitFactory())
factory = DeepLogModuleFactory(
    atom_builders={
        ("classifier", 2, "probability"): get_network_predicate(
            "classifier", 2, "probability", tiny
        )
    }
)

model = reshape(
    compile_to_module(result, factory),
    input=(SymTensor("i1"), SymTensor("i2")),
    output=SymTensor([f"addition(i1,i2,{total}) _ probability" for total in range(3)]),
)

print("inputs  :", *model.get_input_shape())
print("outputs :", model.get_output_shape())

features1, features2 = torch.rand(1, 4), torch.rand(1, 4)
to_dict(model(features1, features2), model.get_output_shape())
```

The module runs the network itself now: four features per image in, a distribution over sums out, with the logic in between. It is untrained, so the sums are a guess — but a loss on that distribution backpropagates through the circuit into `tiny`'s weights, even though we never tell the network what either digit is. That indirect supervision is the whole idea of DeepProbLog, and the MNIST experiment below is exactly this with real images and a real CNN.

+++

## MNIST addition

The classic NeSy experiment: given a *pair* of MNIST images, predict the sum of their digits — with **no** supervision on the individual digits, only on the sum. The network learns to read digits purely from the addition signal.

+++

### The network

A small CNN that maps a `28×28` image to a distribution over the ten digits — the `classifier` the program is about to name.

```{code-cell}
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
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x


mnist_net = MNISTNet()
```

### The dataset

Each item is a pair of MNIST images and the sum of their labels — the labels themselves are only used to build the sum, never shown to the model.

```{code-cell}
import os

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset


# Downloaded datasets go to DEEPLOG_DATA_DIR, or to data/ beside this notebook.
data_dir = os.getenv("DEEPLOG_DATA_DIR", "data")


class MNISTAddition(Dataset):
    def __init__(self, subset):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        self.dataset = torchvision.datasets.MNIST(
            data_dir, subset == "train", transform=transform, download=True
        )
        self.subset = subset
        self.n = 2

    def __getitem__(self, index):
        i1 = index * self.n
        im1, l1 = self.dataset[i1]
        im2, l2 = self.dataset[i1 + 1]
        return im1, im2, l1 + l2

    def __len__(self):
        return len(self.dataset) // self.n


train_dataloader = DataLoader(
    Subset(MNISTAddition("train"), range(7500)), batch_size=32, num_workers=0
)
test_dataloader = DataLoader(
    Subset(MNISTAddition("TEST"), range(100)), batch_size=32, num_workers=0
)
```

### The program and the model

The same two lines, with ten digits per image instead of two. `nn(classifier, [X], Y, [0..9])` binds the CNN to `digit/2`, so compiling yields the entire model — a pair of images in, a distribution over sums out — and there is nothing left to wire together by hand.

It also scales the declaration the same way: still two variables, now of ten values each. Twenty `digit` atoms, but not twenty booleans.

The `reshape` still earns its place: it names the two images as separate arguments in that order, and puts the sums in numeric order so the loss can read column `i` as the probability of sum `i`.

```{code-cell}
ADDITION = """
nn(classifier, [X], Y, [0..9]) :: digit(X,Y).
addition(I1,I2,S) :- digit(I1,N1), digit(I2,N2), is(S,+(N1,N2)).
?- addition(i1,i2,S).
"""

program = tuple(str_to_rules(ADDITION))
result = Solver(SimpleGrounder()).get_query_result(program, CircuitFactory())
factory = DeepLogModuleFactory(
    atom_builders={
        ("classifier", 2, "probability"): get_network_predicate(
            "classifier", 2, "probability", mnist_net
        )
    }
)

addition_model = reshape(
    compile_to_module(result, factory),
    input=(SymTensor("i1"), SymTensor("i2")),
    output=SymTensor([f"addition(i1,i2,{total}) _ probability" for total in range(19)]),
)

variables = sorted(result.variables, key=lambda variable: variable.name)
sizes = [len(variable) for variable in variables]
print("variables :", ", ".join(
    f"{symbol_to_pretty_string(variable.name)} over {len(variable)} values"
    for variable in variables
))
print(f"worlds    : {math.prod(sizes):,}, against {2 ** sum(sizes):,} for independent booleans")
print("inputs    :", *addition_model.get_input_shape())
print("outputs   :", addition_model.get_output_shape())
```

### Training

We wrap the model in a PyTorch Lightning module. The loss is the negative log-likelihood of the *correct sum* — the only label we ever provide.

```{code-cell}
import logging

import pytorch_lightning as pl
import torchmetrics

from deeplog.util import fast_dev_run_enabled


pl.utilities.disable_possible_user_warnings()
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


class LightningDPL(pl.LightningModule):
    def __init__(self, model: nn.Module, learning_rate: float):
        super().__init__()
        self.model = model
        self.loss = nn.NLLLoss()
        self.learning_rate = learning_rate
        self.train_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=19
        )
        self.test_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=19
        )
        self.loss_history = []
        self.acc_history = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, *x) -> torch.Tensor:
        return self.model(*x)

    def training_step(self, batch, batch_idx):
        images1, images2, labels = batch
        probs = self.forward(images1, images2)
        loss = self.loss(torch.log(probs + 1e-9), labels)
        acc = self.train_accuracy(probs, labels)
        self.loss_history.append(loss.detach().cpu().item())
        self.acc_history.append(acc.detach().cpu().item())
        self.log("loss", loss, prog_bar=False, logger=False)
        self.log("train_acc", acc, prog_bar=False, logger=False)
        return loss

    def test_step(self, batch, batch_idx):
        images1, images2, labels = batch
        self.test_accuracy(self.forward(images1, images2), labels)
        self.log("test_acc", self.test_accuracy)


pl_model = LightningDPL(addition_model, learning_rate=0.001)
```

```{code-cell}
trainer = pl.Trainer(
    fast_dev_run=fast_dev_run_enabled(),
    max_epochs=1,
    enable_progress_bar=False,
    enable_model_summary=False,
    logger=False,
    enable_checkpointing=False,
)
trainer.fit(model=pl_model, train_dataloaders=train_dataloader)
```

### Training loss

Loss and accuracy across batches (smoothed).

```{code-cell}
import matplotlib.pyplot as plt
import numpy as np


def moving_average(values, window):
    values = np.array(values, dtype=float)
    if len(values) < window:
        return values
    cumsum = np.cumsum(np.insert(values, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / float(window)


window = max(1, len(pl_model.loss_history) // 20)
smoothed_loss = moving_average(pl_model.loss_history, window)
smoothed_acc = moving_average(pl_model.acc_history, window)

fig, ax1 = plt.subplots()
color = "tab:blue"
ax1.set_xlabel("Batch (smoothed)")
ax1.set_ylabel("Training loss", color=color)
ax1.plot(np.arange(len(smoothed_loss)), smoothed_loss, color=color, label="loss (smoothed)")
ax1.tick_params(axis="y", labelcolor=color)

ax2 = ax1.twinx()
color = "tab:green"
ax2.set_ylabel("Training accuracy", color=color)
ax2.plot(np.arange(len(smoothed_acc)), smoothed_acc, color=color, label="train_acc (smoothed)")
ax2.tick_params(axis="y", labelcolor=color)

fig.tight_layout()
plt.show()
```

## Testing

Finally, evaluate the trained model on held-out pairs.

```{code-cell}
trainer.test(model=pl_model, dataloaders=test_dataloader)
```
