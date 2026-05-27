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
- **Tensorizing DeepLog formulas** — learn the language, predicates, and compilation pipeline that turns logic into executable modules.

Use the sections below to follow either track or mix and match.

DeepLog Modules (symbolic wrapper around Torch)
+++++++++++++++++++++++++++++++++++++++++++++++
Build the core intuition: Symbols → SymTensor → DeepLogModule.

.. card:: Symbols
    :link: examples/symbol
    :link-type: doc

    One of the core components of DeepLog is the Symbol, which is used to identify anything of symbolic nature.

.. card:: SymTensor
    :link: examples/shape
    :link-type: doc

    One of the main concepts of DeepLog is the :class:`~deeplog.shape.SymTensor`, which carries symbolic information
    about the input and output of modules. The following notebook explains the basic structure
    and shows how to transform_circuit between shapes.

.. card:: DeepLogModule
    :link: examples/deeplogmodule
    :link-type: doc

    DeepLog Modules are wrappers around Torch modules with two additional methods: get_input_shape() and get_output_shape().
    By providing this information, we can more easily chain together different modules, as the transformations can be calculated automatically.

.. card:: Composition
    :link: examples/composition
    :link-type: doc

    Learn how to combine DeepLog Modules using Sequential and ModuleCircuit, explore automatic shape transformations, and handle missing producers.

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

.. card:: DIMACS CNF parsing
    :link: examples/dimacs_cnf
    :link-type: doc

    Parse DIMACS CNF text into DeepLog formulas and modules, choosing boolean or probabilistic structures.

.. card:: Aggregation basics
    :link: examples/01_aggregation_basics
    :link-type: doc

    Learn the aggregation syntax, finite domains, and how DeepLog builds aggregation modules.

.. card:: Free variables and batching
    :link: examples/03_free_variables_and_batching
    :link-type: doc

    Free variables become module inputs.


.. card:: From Formulas to Modules
    :link: examples/formula_to_module
    :link-type: doc

    Learn how symbolic formulas are compiled into DeepLog modules and how the resulting modules plug into differentiable pipelines.


Extending DeepLog
+++++++++++++++++
Advanced features for custom algebraic structures and circuit operations.

.. card:: Circuit Transformation
    :link: examples/circuit_transformation
    :link-type: doc

    Transform circuits between algebraic structures with automatic operator mapping, leaf remapping, and batch transformation.

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

    This tutorial shows how to include DeepLog in a normal ML pipeline by implementing the Semantic Loss framework in DeepLog with an exactly-one constraint.

.. card:: MNIST Addition
    :link: examples/mnist_addition
    :link-type: doc

    Follow a full DeepProbLog workflow where two MNIST digits are jointly classified and summed using DeepLog modules for arithmetic reasoning. This is an end-to-end example intended to show the DeepLog workflow for neurosymbolic developers.


.. seealso::

   - :doc:`getstarted` ‒ installation paths and a 30-second hello-world snippet.
   - :doc:`autoapi/index` ‒ API reference for DeepLog modules, shapes, and engines used throughout the notebooks.

.. toctree::
   :caption: Learning paths
   :hidden:

   ml_practitioner
   nesy_developer

.. toctree::
   :caption: More notebooks
   :hidden:

   examples/shape
   examples/deeplogmodule
   examples/composition
   examples/formula_to_module
   examples/semantic_loss
   examples/symbol
   examples/predicates
   examples/mnist_addition
   examples/circuits
   examples/language
   examples/01_aggregation_basics
   examples/aggregation
   examples/03_free_variables_and_batching
   examples/dimacs_cnf
   examples/circuit_transformation
   examples/ltn
