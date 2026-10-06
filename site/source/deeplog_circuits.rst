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

Circuit transformation
----------------------

.. versionadded:: 2.2.0

The ``transform_circuit`` function converts a circuit from one algebraic structure to
another by rebuilding each node with the target structure's operators. The map
is exact only insofar as the source's operators mean in the target what they
meant in the source: reading a boolean ``or`` as a semiring sum needs a
deterministic, decomposable circuit, which is what
:func:`~deeplog.circuit.knowledge_compilation.dispatch.knowledge_compile`
produces.

Basic usage
~~~~~~~~~~~

.. code-block:: python

   from deeplog.circuit import Circuit, transform_circuit

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

Several roots at once
~~~~~~~~~~~~~~~~~~~~~

``roots`` may name several nodes of the same circuit. They are transformed in a
single pass, so a subgraph they share is traversed once, and ``node_map`` holds
each of them:

.. code-block:: python

   prob_circuit, node_map = transform_circuit(
       bool_circuit, "probability", roots=[root_a, root_b]
   )
   module = prob_circuit.to_module(
       {node_map[root_a]: ("a",), node_map[root_b]: ("b",)}
   )

.. seealso::

   - :doc:`deeplog_language` for the formula language that compiles to circuits.
   - :doc:`examples/circuits` for runnable circuits, including a weighted model count.

.. |rarr| unicode:: U+2192
