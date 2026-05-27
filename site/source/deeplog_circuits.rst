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
:class:`~deeplog.module.deeplog_module.DeepLogModule`. For deterministic
probability circuits (using PySDD), pass ``deterministic=True`` to the
``Circuit`` constructor:

.. code-block:: python

   circuit = Circuit("probability", deterministic=True)

CircuitNode
~~~~~~~~~~~

:class:`~deeplog.circuit.CircuitNode` wraps a node ID together with its
circuit. It is returned by factory methods and provides convenience helpers:

.. code-block:: python

   from deeplog.circuit import CircuitNode, to_module

   node = CircuitNode(circuit, or_node)
   module = node.to_module(name=("or_root",))

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

   from deeplog.circuit import Circuit, CircuitNode, transform_circuit

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

When both the source and target structures are
:class:`~deeplog.algebraic.Semiring` (or :class:`~deeplog.algebraic.Algebra`),
operator mapping is inferred automatically from the structure roles:

* **product** maps to **product** (e.g. ``and`` |rarr| ``times``)
* **sum** maps to **sum** (e.g. ``or`` |rarr| ``plus``)
* **negation** maps to **negation** (e.g. ``not`` |rarr| ``negate``, Algebra only)

For custom or non-semiring structures, provide an explicit mapping:

.. code-block:: python

   new_circuit, node_map = transform_circuit(
       source_circuit,
       target_structure,
       roots=[root],
       operator_mapping={"and": "times", "or": "plus"},
   )

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

:func:`~deeplog.circuit.transform_nodes` transforms multiple
:class:`~deeplog.circuit.CircuitNode` objects from the same circuit in a single
pass:

.. code-block:: python

   from deeplog.circuit import CircuitNode, transform_nodes

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

Individual :class:`~deeplog.circuit.CircuitNode` objects expose a
``transform_circuit()`` method for convenience:

.. code-block:: python

   prob_node = bool_node.transform_circuit("probability")
   module = prob_node.to_module()

.. seealso::

   - :doc:`deeplog_language` for the formula language that compiles to circuits.
   - :doc:`examples/circuits` for runnable circuit construction examples.
   - :doc:`examples/circuit_transformation` for a runnable circuit transformation tutorial.

.. |rarr| unicode:: U+2192
