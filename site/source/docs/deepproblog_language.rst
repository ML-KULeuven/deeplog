DeepProbLog language
====================

.. container:: dl-section dl-section--hero

   .. rst-class:: dl-section__eyebrow

      SPECIFICATION

   .. rst-class:: dl-section__title

      Logic accepted by the DeepProbLog engines

   .. rst-class:: dl-section__lead

      The engines under :mod:`deeplog.systems.deepproblog.engine` interpret a compact DeepProbLog-like language.
      This page spells out its syntax, lexical rules, and the operational semantics that ultimately produce
      differentiable formulas through :class:`deeplog.formula.deeplogformulafactory.DeepLogFormulaFactory`.

Syntactic categories
--------------------

Programs are sequences of clauses emitted by :func:`deeplog.systems.deepproblog.program.program.str_to_rules`
or built manually through the helper constructors in :mod:`deeplog.systems.deepproblog.program.program`.
Clauses are always terminated with a ``.`` and belong to one of four forms:

* **Rules** – ``H :- B.`` where ``H`` is a disjunction of one or more atoms and ``B`` is a (possibly empty)
  conjunction of literals.
* **Facts** – either ``a.`` (deterministic) or ``label :: a.`` (probabilistic) which are just rules with an
  empty body.
* **Queries** – ``?- B.`` which request the engine to solve ``B`` and produce answer substitutions.
* **Constraints** – ``:- B.`` which reject the program whenever ``B`` is provable.

Lexical conventions mirror Prolog:

* **Variables** begin with an uppercase letter or ``_``. ``_`` denotes an anonymous variable.
* **Atoms / functors** begin with a lowercase letter or are quoted; numerals are treated as atoms.
* **Terms** are atoms, variables, or functor applications ``f(t1,...,tn)`` whose arguments are themselves terms.
* **Labels** reuse the term grammar so probabilistic facts can refer to neural predicates or numeric tensors.
* ``true`` and ``false`` are reserved: the helper constructors map empty bodies to ``true`` and empty heads to
  ``false``.

Grammar
-------

The parser in :func:`deeplog.systems.deepproblog.program.program.str_to_rule` accepts the following grammar
(``{x}`` means zero or more occurrences and ``[x]`` means optional):

.. code-block:: text

   <program>         ::= { <clause> }
   <clause>          ::= <rule> | <fact> | <query> | <constraint>
   <rule>            ::= <disjunctive-head> ":-" <body> "."
   <fact>            ::= <labeled-atom> "."
   <query>           ::= "?-" <body> "."
   <constraint>      ::= ":-" <body> "."
   <disjunctive-head>::= <labeled-atom> { ";" <labeled-atom> }
   <body>            ::= "true" | <goal>
   <goal>            ::= <literal> { "," <literal> }
   <literal>         ::= <atom>
                       | "not" <atom>
                       | "(" <goal> ")"
                       | "(" <goal> ";" <goal> ")"
   <labeled-atom>    ::= [ <term> "::" ] <atom>
   <atom>            ::= <predicate> [ "(" <term> { "," <term> } ")" ]
   <term>            ::= <variable> | <constant> | <atom> | <list>
   <list>            ::= "[" [ <list-elements> ] "]"
   <list-elements>   ::= <term> { "," <term> } [ "|" <term> ]
   <predicate>       ::= <constant>

``;`` inside a head produces a disjunctive rule and is also available in bodies through explicit parenthesised
subgoals. Nested parentheses are parsed with :func:`deeplog.util.bracket_aware_split`, so constructs such as
``a :- (b ; c), d.`` are valid. List terms are syntactic sugar: the grammar above rewrites ``[t1,...,tn|tail]`` into
``cons(t1, cons(t2, ... cons(tn, tail)...))`` with ``[]`` treated as the atom ``nil``.

Parser limitations
------------------

The lightweight parser keeps parity with the helper utilities in :mod:`deeplog.symbol` and
:mod:`deeplog.systems.deepproblog.program.program`, not the full ISO Prolog grammar. Known limitations:

* Only ``#``-prefixed full-line comments are ignored. Inline ``%`` or ``/* */`` comments are treated as atoms.
* Operator declarations are not recognised. Besides the fixed connectives ``:-``, ``?-``, ``::``, ``','``, ``';'``, and
  ``not/1``, every other functor must be written in prefix form.
* Atoms are not unquoted: ``'has space'`` becomes the literal functor ``"'has space'"`` (quotes included). Escape
  sequences are not interpreted.
* Numbers are parsed greedily as atoms; there is no automatic float/integer detection beyond what built-ins do when
  evaluating arithmetic.

List support
------------

Square-bracket list syntax is supported directly by :func:`deeplog.symbol.parse_symbol`. The parser rewrites
``[t1,t2,...,tn]`` and ``[H|T]`` into nested ``cons/2`` functors that terminate in the atom ``nil/0``:

.. code-block:: python

   >>> from deeplog.symbol import parse_symbol
   >>> parse_symbol('[a,b,c]')
   ('cons', ('a',), ('cons', ('b',), ('cons', ('c',), ('nil',))))
   >>> parse_symbol('[H|T]')
   ('cons', ('H',), ('T',))

This applies recursively, so nested lists (e.g. ``[a,[b,c],d|T]``) work as expected. These ``cons`` tuples behave
exactly like canonical Prolog lists during unification and rule evaluation, so you can pattern match on ``[Head|Tail]``
in rule heads or bodies without extra boilerplate.

Operational semantics
---------------------

Inference follows memoised SLD-resolution as implemented in :class:`deeplog.systems.deepproblog.engine.simple_engine.SimpleEngine`.
For a goal ``G`` the engine:

1. Chooses the predicate at the root of ``G``.
2. Dispatches special connectives ``','/2``, ``';'/2``, ``not/1`` and ``true/0`` directly.
3. Matches built-in predicates (see below) before consulting user rules.
4. Selects every rule whose head predicate unifies with the goal, substitutes variables, and recursively proves
   the body.

Answer substitutions retain bindings only for variables that appeared in the original goal. The memoisation
layer collapses duplicate substitutions by disjoining their formulas.

Boolean proof formulas
----------------------

Engines compile proof trees through
:class:`~deeplog.systems.deepproblog.engine.engine.EngineFactory`, a thin
wrapper around a :class:`~deeplog.formula.deeplogformulafactory.DeepLogFormulaFactory`
that always builds **boolean** formulas:

* ``engine_factory.get_true()`` / ``get_false()`` create boolean constant atoms.
* ``engine_factory.conjoin()`` / ``disjoin()`` build ``and`` / ``or`` nodes via the
  underlying factory's ``create_binary_node``.
* ``engine_factory.negate()`` builds a ``not`` node via ``create_unary_node``.
* Labeled facts ``label :: atom.`` call ``engine_factory.get_boolean(goal, label)``,
  which creates a boolean leaf for the goal atom and records the label in a
  separate mapping.

The result is always a boolean proof circuit. Compilation to a probabilistic
semiring happens in a later step via the circuit transformation API (see
`Compilation pipeline`_ below and :doc:`../deeplog_circuits`).

Built-in predicates
-------------------

The default engine ships with a small built-in predicate library. See
:doc:`deepproblog_builtins` for the full list and semantics. Additional
predicates can be registered by calling :meth:`deeplog.systems.deepproblog.engine.engine.Engine.add_builtin`.

Example
-------

.. code-block:: prolog

   digit(0). digit(1). digit(2). digit(3). digit(4).
   digit(5). digit(6). digit(7). digit(8). digit(9).

   nn_is_sum(A, B) :: sum(A, B, S).

   valid_sum(A, B, S) :-
       digit(A),
       digit(B),
       between(0, 18, S),
       nn_is_sum(A, B, S).

   ?- valid_sum(D1, D2, R).

The deterministic ``digit`` facts collapse to ``true`` leaves, while the labeled ``nn_is_sum/3`` fact introduces a
learnable probability that is multiplied into every proof of ``valid_sum/3``. The final query returns all pairs of
digits together with the differentiable formulas that DeepLog executes on GPU.

Compilation pipeline
--------------------

.. versionadded:: 2.2.0

After the engine produces an
:class:`~deeplog.systems.deepproblog.engine.engine.EngineResult`, the
compilation module turns it into a single differentiable
:class:`~deeplog.module.deeplog_module.DeepLogModule`.

:func:`~deeplog.systems.deepproblog.compile_to_module` performs the full
pipeline in one call:

1. Builds a factorized probability distribution from the engine's atom labels
   internally.
2. Creates an ``expectation`` aggregation for each formula in the result, which
   transforms the boolean proof circuit to the probability semiring (see
   :doc:`../deeplog_circuits`).
3. Batch-transforms shared boolean circuits in a single pass via
   :func:`~deeplog.circuit.transform_nodes`.
4. Composes the circuit module with batched predicate modules (e.g. neural
   network predicates) to produce the final module.

.. code-block:: python

   from deeplog.systems.deepproblog import compile_to_module
   from deeplog.systems.deepproblog.engine import SimpleEngine

   result = SimpleEngine().get_query_result(program, factory)
   module = compile_to_module(result, factory)

   # module accepts input tensors and returns query probabilities
   output = module(input_tensor)

For lower-level control, you can also work with the engine result directly:

.. code-block:: python

   from deeplog.circuit import to_module, transform_nodes

   # Access formulas and labels separately
   for answer, formula in result.formulas.items():
       print(f"{answer}: {formula}")

   # Transform and compile manually
   nodes = [factory.create_aggregation("expectation", [], [], f)
            for f in result.formulas.values()]
   # ... transform_circuit and call to_module()

.. seealso::

   - :mod:`deeplog.systems.deepproblog.engine.simple_engine` for the pure Python interpreter.
   - :mod:`deeplog.systems.deepproblog.engine.janus_engine` for the Janus/SWI-Prolog backend that accepts the same language.
   - :doc:`../deeplog_circuits` for the circuit transformation API used internally.
