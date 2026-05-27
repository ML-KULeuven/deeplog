DeepLog language
================

.. container:: dl-section dl-section--hero

   .. rst-class:: dl-section__eyebrow

      SPECIFICATION

   .. rst-class:: dl-section__title

      DeepLog language

   .. rst-class:: dl-section__lead

      The DeepLog language is a compact syntax that maps directly to
      ``DeepLogFormulaFactory`` nodes. It is used to build differentiable
      circuits from logical formulas and supports aggregations, transforms,
      unary/binary operators, and predicate leaves.

Overview
--------

The DeepLog language is a compact formula syntax built around aggregations,
operators, transformations, and structured predicate leaves. It is designed to
be readable while still mapping directly to symbolic formula graphs.

The reference grammar lives in ``docs/textual_formula_grammar.md``.

Lexical elements
----------------

* **Identifier**: ASCII names that start with a letter or underscore and then
  use letters, digits, or underscores. These name operators, aggregations,
  structures, and variables.
* **Symbol literal**: anything accepted by
  :func:`deeplog.symbol.parse_symbol` (e.g. ``p(X,1)``, ``('=',X,true)``).
* **Structure**: an identifier such as ``boolean`` or ``probability``.
* **Variable**: an identifier that will be converted to a ``Symbol``;
  conventionally variables are capitalized.
* **Whitespace**: spaces, tabs, and newlines are insignificant.
* **Comments**: lines starting with ``#`` are ignored.

Grammar summary
---------------

.. code-block:: text

   Formula        ::= Aggregation
                    | BinaryExpr
                    | UnaryExpr
                    | Transformation
                    | Leaf

   Aggregation    ::= Identifier "(" BinderList [";" ParamList] ")" ":" Formula
   BinderList     ::= Variable { "," Variable }
   ParamList      ::= Formula { "," Formula }

   BinaryExpr     ::= UnaryExpr BinaryOp UnaryExpr { BinaryOp UnaryExpr }
   BinaryOp       ::= Identifier

   UnaryExpr      ::= PrefixOp UnaryExpr
                    | "(" Formula ")"
                    | Transformation
   PrefixOp       ::= Identifier

   Transformation ::= "(" Formula ")" "_" Structure

   Leaf           ::= SymbolLiteral "_" Structure

Key semantics
-------------

* **Aggregation scope**: ``Op(X, Y; params): body`` binds ``X`` and ``Y`` in
  the body and parameter formulas.
* **Operator precedence**: all binary operators share one precedence level and
  associate to the left; use parentheses to enforce evaluation order.
* **Unary operators**: any identifier can act as a unary prefix operator.
* **Transformations**: ``(phi)_structure`` turns a grouped formula into a
  structure transform_circuit node.
* **Leaves**: every leaf pairs a symbol literal with its structure via the
  ``_structure`` suffix.
* **Structure names**: the built-in structures are ``_boolean``,
  ``_probability``, and ``_logprobability``.

Underscores are reserved for ``_structure`` suffixes. Predicate names that
include underscores should be quoted inside the symbol literal (e.g.
``'=foo_bar'``).

Structures
----------

Structures tag each atom with the "domain" it lives in, and they determine how
operators are interpreted across a formula. The most common structures are:

* **Boolean**: truth-valued formulas (``_boolean``), typically combined with
  logical operators.
* **Probability**: weighted formulas (``_probability``) that combine numeric
  scores.
* **Log-probability**: log-space weighted formulas (``_logprobability``).

Structures are attached to leaves (``predicate(args)_structure``) and can be
changed mid-formula using transformations such as ``(expr)_probability``. This
is how you express patterns like "evaluate a Boolean condition, then turn it
into a probability and multiply by literal weights." The operator names in the
surface language stay the same; the structure tells the system how to interpret
them.

Structure classes
~~~~~~~~~~~~~~~~~

All structures are instances of :class:`~deeplog.algebraic.AlgebraicStructure`.
The built-in structures use two specialised subclasses that enable automatic
operator mapping during :doc:`circuit transformation <deeplog_circuits>`:

* :class:`~deeplog.algebraic.Semiring` — adds named ``product`` and ``sum``
  roles (with constants ``zero`` and ``one``). When both the source and target
  of a circuit transformation are semirings, operators are mapped automatically
  (product |rarr| product, sum |rarr| sum).
* :class:`~deeplog.algebraic.Algebra` — extends ``Semiring`` with a
  ``negation`` role, which is also mapped automatically.

All three built-in structures (``BOOLEAN``, ``PROBABILITY``,
``LOGPROBABILITY``) are ``Algebra`` instances. Custom structures can use any of
the three classes depending on which roles they provide.

.. code-block:: python

   from deeplog import Algebra, Semiring, AlgebraicStructure

   # Minimal: free-form operator names
   fuzzy = AlgebraicStructure(
       name="fuzzy",
       operator_fns={"and": ..., "or": ..., "not": ...},
   )

   # Semiring: enables auto-mapping of product/sum
   tropical = Semiring(
       name="tropical",
       product="plus", product_fn=lambda a, b: a + b,
       sum="min", sum_fn=lambda a, b: torch.minimum(a, b),
   )

   # Algebra: enables auto-mapping of product/sum/negation
   my_algebra = Algebra(
       name="myalgebra",
       product="times", sum="plus", negation="negate",
   )

.. |rarr| unicode:: U+2192

Examples
~~~~~~~~

.. code-block:: text

   # Boolean structure: combine truth-valued literals
   =(Burglary,true)_boolean or =(Earthquake,true)_boolean

   # Probability structure: weight a Boolean formula and combine scores
   (=(Burglary,true)_boolean or =(Earthquake,true)_boolean)_probability
     times p(Burglary)_probability

Predicates
----------

Predicates are the smallest executable units in the language. A predicate ties
symbolic atoms (like ``=(X,true)`` or ``digit(I,N)``) to tensor computations and
defines whether an atom holds (Boolean predicates) or how strongly it holds
(probability/neural predicates).

In the language syntax, predicates appear only as leaves with a structure
suffix. The functor name and arguments describe the atom, while the structure
decides the domain of the result. Built-in predicate modules cover common cases
like equality, probabilities, arithmetic constraints, neural classifiers, and
indexing. See :doc:`deeplog_predicates` for the full list and usage patterns.

Implementation hooks
--------------------

These details explain how the language connects to the underlying codebase.
They are not required to read or write formulas, but help when extending or
debugging the parser and factories.

Mapping to factory calls
~~~~~~~~~~~~~~~~~~~~~~~~

==============================  ==================================================
Syntax                          Factory call
------------------------------  --------------------------------------------------
``Op(V1, V2; p1): body``         ``create_aggregation(Op, [V1, V2], [p1], body)``
``(phi)_structure``              ``create_transformation(structure, phi)``
``lhs Op rhs``                   ``create_binary_node(Op, lhs, rhs)``
``Op phi``                       ``create_unary_node(Op, phi)``
``symbol_structure``             ``create_leaf_node(parse_symbol(symbol), structure)``
==============================  ==================================================

Examples
--------

Model counting
~~~~~~~~~~~~~~

.. code-block:: text

   sum(Burglary, Earthquake):
       =(Burglary,true)_boolean or =(Earthquake,true)_boolean

Weighted model counting
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   sum(Burglary, Earthquake):
         (=(Burglary,true)_boolean or =(Earthquake,true)_boolean)_probability
       times
         (p(Burglary)_probability times p(Earthquake)_probability)

Parsing API
~~~~~~~~~~~

Use :func:`deeplog.formula.text_parser_lark.parse_formula` to parse a formula into
factory calls, or :func:`deeplog.formula.text_parser_lark.parse_formula_to_module`
to parse and immediately build a :class:`~deeplog.module.deeplog_module.DeepLogModule`.

.. seealso::

   - :doc:`deeplog_predicates` for the built-in predicate modules.
   - :doc:`docs/deepproblog_language` for the Prolog-style language used by the
     DeepProbLog engines.

.. toctree::
   :hidden:

   deeplog_predicates
