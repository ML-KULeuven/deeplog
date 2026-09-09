Circuits and transformations
============================

.. container:: dl-section dl-section--hero

   .. rst-class:: dl-section__eyebrow

      REFERENCE

   .. rst-class:: dl-section__title

      Circuits and transformations

   .. rst-class:: dl-section__lead

      Circuits are the intermediate representation that DeepLog uses to
      translate logical formulas into efficient torch modules. This page covers
      direct circuit construction and the circuit transformation API introduced
      in v2.2.0.

Circuits
--------

A :class:`~deeplog.circuit.Circuit` is a DAG of operator, leaf, and constant
nodes tied to an :class:`~deeplog.algebraic.AlgebraicStructure`. Most users
never build circuits directly because
:func:`~deeplog.formula.text_parser_lark.parse_formula_to_module` does it
automatically, but the low-level API is available when you need full control.

.. code-block:: python

   from deeplog.circuit import Circuit

   circuit = Circuit("boolean")
   a = circuit.get_leaf_node(("a",))
   b = circuit.get_leaf_node(("b",))
   or_op = circuit.get_operator("or")
   and_op = circuit.get_operator("and")
   or_node = or_op(a, b)
   and_node = and_op(a, b)

   module = circuit.to_module({or_node: ("or_root",), and_node: ("and_root",)})

``to_module`` converts the circuit into a
:class:`~deeplog.module.deeplog_module.DeepLogModule`. A circuit is evaluated as
written — its operators mean what the structure's ``operator_fns`` say they
mean.

CircuitNode
~~~~~~~~~~~

:class:`~deeplog.formula.CircuitNode` wraps a node ID together with its circuit.
Use the companion free functions to compile or transform one or more nodes:

.. code-block:: python

   from deeplog.formula import CircuitNode, to_module

   node = CircuitNode(circuit, or_node)
   module = to_module(node, names=(("or_root",),))

   # Convert multiple nodes at once
   module = to_module(node_a, node_b, names=(("a",), ("b",)))

Circuit transformation
----------------------

.. versionadded:: 2.2.0

The ``transform_circuit`` function converts a circuit from one algebraic structure to
another by rebuilding each node with the target structure's operators. This is
the mechanism behind the ``expectation`` aggregation operator, which transforms
boolean proof circuits into the probability semiring.

Basic usage
~~~~~~~~~~~

.. code-block:: python

   from deeplog.circuit import Circuit, transform_circuit
   from deeplog.formula import CircuitNode

   # Build a boolean circuit
   bool_circuit = Circuit("boolean")
   a = bool_circuit.get_leaf_node(("a",))
   b = bool_circuit.get_leaf_node(("b",))
   or_op = bool_circuit.get_operator("or")
   root = or_op(a, b)

   # Transform to probability semiring
   prob_circuit, node_map = transform_circuit(
       bool_circuit, "probability", roots=[root]
   )
   prob_module = prob_circuit.to_module({node_map[root]: ("result",)})

Automatic operator mapping
~~~~~~~~~~~~~~~~~~~~~~~~~~

Operator mapping is inferred from the roles both structures declare
(:attr:`~deeplog.algebraic.AlgebraicStructure.roles`):

* **product** maps to **product** (e.g. ``and`` |rarr| ``times``)
* **sum** maps to **sum** (e.g. ``or`` |rarr| ``plus``)
* **negation** maps to **negation** (e.g. ``not`` |rarr| ``negate``, Algebra only)
* **division** maps to **division** (``divide``, Semifield only)

A role the target does not declare is left unmapped rather than refused: whether
that matters depends on the nodes being transformed, so it raises only when one
uses that operator. A structure that declares no roles at all — or an operator
that plays none — needs an explicit mapping:

.. code-block:: python

   new_circuit, node_map = transform_circuit(
       source_circuit,
       target_structure,
       roots=[root],
       operator_mapping={"and": "times", "or": "plus"},
   )

Named constants
~~~~~~~~~~~~~~~

An identity crosses the same way — by the role it plays
(:attr:`~deeplog.algebraic.AlgebraicStructure.identities`), never by its value.
The additive identity is ``("0",)`` in ``probability`` and ``("-inf",)`` in
``logprobability``, where ``("0",)`` is the *multiplicative* one, so carrying a
constant over as a number would change which element it is. A numeric constant
names no identity and does cross as itself.

Leaf remapping
~~~~~~~~~~~~~~

Use ``leaf_mapping`` to rename symbols during transformation, for example to
distinguish boolean atoms from their probability counterparts:

.. code-block:: python

   def bool_to_prob(symbol):
       return ("_", symbol, ("probability",))

   prob_circuit, node_map = transform_circuit(
       bool_circuit, "probability", roots=[root],
       leaf_mapping=bool_to_prob,
   )

Batch transformation with transform_nodes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~deeplog.formula.transform_nodes` transforms multiple
:class:`~deeplog.formula.CircuitNode` objects from the same circuit in a single
pass:

.. code-block:: python

   from deeplog.formula import CircuitNode, transform_nodes

   node_a = CircuitNode(bool_circuit, root_a)
   node_b = CircuitNode(bool_circuit, root_b)

   transformed = transform_nodes(
       node_a, node_b,
       target_structure="probability",
   )
   # transformed is a tuple of CircuitNodes in the new circuit

This is more efficient than transforming each node individually because the
shared subgraph is only traversed once.

Per-node transformation
~~~~~~~~~~~~~~~~~~~~~~~

Transforming one :class:`~deeplog.formula.CircuitNode` is the one-argument case
of :func:`~deeplog.formula.transform_nodes`:

.. code-block:: python

   (prob_node,) = transform_nodes(bool_node, target_structure="probability")
   module = to_module(prob_node, names=(("result",),))

.. seealso::

   - :doc:`deeplog_language` for the formula language that compiles to circuits.
   - :doc:`examples/circuits` for runnable circuit construction examples.
   - :doc:`examples/circuit_transformation` for a runnable circuit transformation tutorial.

.. |rarr| unicode:: U+2192
