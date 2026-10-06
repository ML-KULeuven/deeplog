Tutorial
========


.. container:: dl-section dl-section--hero

   .. rst-class:: dl-section__eyebrow

      TUTORIALS

   .. rst-class:: dl-section__title

      Pick the learning path that matches your background

   .. rst-class:: dl-section__lead

      Whether you come from symbolic AI, PyTorch-first ML, or hybrid systems, each notebook focuses on the questions you are likely to have—complete with runnable cells and structured outputs.

   .. raw:: html

      <div style="height: 0.35rem;"></div>

   .. rst-class:: tutorial-hero-grid

   .. grid:: 1 1 2 2
      :gutter: 2

      .. grid-item::

         .. card:: I'm an ML practitioner
            :link: ml_practitioner
            :link-type: doc
            :class-card: dl-card dl-card--tutorial

            Follow the practitioner flow to run DeepLog Modules, shapes, Semantic Loss, and the formula parser in runnable notebooks.

      .. grid-item::

         .. card:: I'm a NeSy developer
            :link: nesy_developer
            :link-type: doc
            :class-card: dl-card dl-card--tutorial

            Dive into symbols, shapes, predicates, formula compilation, and an end-to-end DeepProbLog-style workflow.


Welcome to the **DeepLog** tutorial! This guide will help you get started with the DeepLog framework step by step. Use the grid above to open the guided flow that best fits your current project.

DeepLog learning roadmap
++++++++++++++++++++++++

DeepLog can be learned along two complementary narratives:

- **Symbolic wrapper around Torch** — start with symbols, shapes, and :class:`~deeplog.module.deeplog_module.DeepLogModule` to add semantic validation to PyTorch workflows.
- **Tensorizing DeepLog formulas** — learn the language, predicates, and compilation that turns logic into executable modules.

Use the sections below to follow either track or mix and match.

DeepLog Modules (symbolic wrapper around Torch)
+++++++++++++++++++++++++++++++++++++++++++++++
Build the core intuition: Symbols → SymTensor → DeepLogModule.

.. card:: Symbols and shapes
    :link: examples/symbols_and_shapes
    :link-type: doc

    A symbol names a term, an atom or a value and its algebra; a :class:`~deeplog.shape.SymTensor` names
    every entry of a tensor, and the shapes modules declare are made of them.

.. card:: DeepLogModule
    :link: examples/deeplogmodule
    :link-type: doc

    DeepLog Modules are wrappers around Torch modules with two additional methods: get_input_shape() and get_output_shape().
    By providing this information, we can more easily chain together different modules, as the transformations can be calculated automatically.

.. card:: Composition
    :link: examples/composition
    :link-type: doc

    Learn how to combine DeepLog Modules using Sequential and compose_modules, explore automatic shape transformations, and handle missing producers.

.. card:: Circuits
    :link: examples/circuits
    :link-type: doc

    Circuits are computational graphs that act as an intermediate representation.
    After construction, they are typically converted into a DeepLogModule for efficient execution.


DeepLog Language (tensorizing formulas)
+++++++++++++++++++++++++++++++++++++++
How DeepLog formulas map to tensors and executable modules.

.. card:: DeepLog Language
    :link: examples/language
    :link-type: doc

    Learn the textual syntax, see how it maps onto the parser and grammar, and compile formulas directly into runnable modules.

.. card:: Predicates in DeepLog
    :link: examples/predicates
    :link-type: doc

    Learn how DeepLog predicates connect symbolic atoms to executable tensor operations, enabling the evaluation of logical formulas within DeepLog.

.. card:: Aggregation basics
    :link: examples/aggregation_basics
    :link-type: doc

    Learn the aggregation syntax, finite domains, aggregation operators, and expectations under a distribution.

.. card:: Free variables and batching
    :link: examples/free_variables_and_batching
    :link-type: doc

    Free variables become module inputs.


.. card:: From Formulas to Modules
    :link: examples/formula_to_module
    :link-type: doc

    Learn how symbolic formulas are compiled into DeepLog modules and how the resulting modules plug into differentiable training code, and how an expectation is counted, enumerated or sampled.

.. card:: The Formula AST
    :link: examples/formula_ast
    :link-type: doc

    The tree a formula becomes: how to read it, how to build it from Python, and the proofs a grounder returns as the same kind of tree.


Extending DeepLog
+++++++++++++++++
Advanced features for custom algebraic structures.

.. card:: Logic Tensor Networks (LTN)
    :link: examples/ltn
    :link-type: doc

    Implement Logic Tensor Networks with custom fuzzy logic operators, generalized mean quantifiers, and user-defined algebraic structures in DeepLog.


Full Examples
+++++++++++++
End-to-end tutorials showing DeepLog in applied settings.

.. card:: Semantic Loss
    :link: examples/semantic_loss
    :link-type: doc

    This tutorial shows how to include DeepLog in a normal ML training loop by implementing the Semantic Loss framework in DeepLog with an exactly-one constraint.

.. card:: DeepProbLog
    :link: examples/deepproblog
    :link-type: doc

    Neural predicates: a fact's probability comes from a network instead of a constant. Builds up from a tiny example to a full DeepProbLog workflow where two MNIST digits are jointly classified and summed — an end-to-end neurosymbolic example, counted exactly or estimated by sampling.


.. seealso::

   - :doc:`getstarted` ‒ installation paths and a 30-second hello-world snippet.
   - :doc:`public_api` ‒ API reference for DeepLog modules, shapes, and engines used throughout the notebooks.

.. toctree::
   :caption: Learning paths
   :hidden:

   ml_practitioner
   nesy_developer

.. toctree::
   :caption: More notebooks
   :hidden:

   examples/symbols_and_shapes
   examples/deeplogmodule
   examples/composition
   examples/formula_to_module
   examples/formula_ast
   examples/semantic_loss
   examples/predicates
   examples/problog
   examples/deepproblog
   examples/circuits
   examples/language
   examples/aggregation_basics
   examples/free_variables_and_batching
   examples/ltn
