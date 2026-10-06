DeepProbLog language
====================

.. container:: dl-section dl-section--hero

   .. rst-class:: dl-section__eyebrow

      SPECIFICATION

   .. rst-class:: dl-section__title

      Logic accepted by the DeepProbLog solver

   .. rst-class:: dl-section__lead

      The DeepProbLog :class:`~deeplog.systems.deepproblog.Solver`, driving a plain-Prolog grounder
      from :mod:`deeplog.grounding.prolog`, interprets a compact DeepProbLog-like language.
      This page spells out its syntax, lexical rules, and the operational semantics that ultimately produce
      boolean proof formulas, which compile to differentiable modules.

Syntactic categories
--------------------

Programs are sequences of clauses emitted by :func:`deeplog.grounding.prolog.str_to_rules`
or built manually through the helper constructors in :mod:`deeplog.grounding.prolog`.
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

The parser in :func:`deeplog.grounding.prolog.str_to_rule` accepts the following grammar
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
:mod:`deeplog.grounding.prolog.program`, not the full ISO Prolog grammar. Known limitations:

* Only ``#``-prefixed full-line comments are ignored. Inline ``%`` or ``/* */`` comments are treated as atoms.
* Operator declarations are not recognised. Besides the fixed connectives ``:-``, ``?-``, ``::``, ``','``, ``';'``, and
  ``not/1``, every other functor must be written in prefix form.
* Atoms are not unquoted: ``'has space'`` becomes the literal functor ``"'has space'"`` (quotes included). Escape
  sequences are not interpreted.
* Numbers are parsed greedily as atoms; there is no automatic float/integer detection beyond what built-ins do when
  evaluating arithmetic.

List support
------------

Square-bracket list syntax is supported directly by :func:`~deeplog.parse_symbol`. The parser rewrites
``[t1,t2,...,tn]`` and ``[H|T]`` into nested ``cons/2`` functors that terminate in the atom ``nil/0``:

.. code-block:: python

   >>> from deeplog import parse_symbol
   >>> parse_symbol('[a,b,c]')
   ('cons', ('a',), ('cons', ('b',), ('cons', ('c',), ('nil',))))
   >>> parse_symbol('[H|T]')
   ('cons', ('H',), ('T',))

This applies recursively, so nested lists (e.g. ``[a,[b,c],d|T]``) work as expected. These ``cons`` tuples behave
exactly like canonical Prolog lists during unification and rule evaluation, so you can pattern match on ``[Head|Tail]``
in rule heads or bodies without extra boilerplate.

Operational semantics
---------------------

Inference follows memoised SLD-resolution as implemented in :class:`~deeplog.grounding.prolog.SimpleGrounder`.
For a goal ``G`` the grounder:

1. Chooses the predicate at the root of ``G``.
2. Dispatches special connectives ``','/2``, ``';'/2``, ``not/1`` and ``true/0`` directly.
3. Matches built-in predicates (see below) before consulting user rules.
4. Selects every rule whose head predicate unifies with the goal, substitutes variables, and recursively proves
   the body.

Answer substitutions retain bindings only for variables that appeared in the original goal. The memoisation
layer collapses duplicate substitutions by disjoining their formulas.

Boolean proof formulas
----------------------

A grounder returns each answer's proof as a **boolean** formula
(:data:`~deeplog.FormulaNode`):

* ``true`` and ``false`` are boolean constant atoms.
* A conjunction or disjunction of proofs is an ``and`` / ``or``
  :class:`~deeplog.BinaryOp`, and a negation a ``not``
  :class:`~deeplog.UnaryOp`.
* An *open*-predicate fact is a leaf :class:`~deeplog.Atom`; every
  other (closed) fact collapses to ``true``. The grounder attaches no meaning to
  those leaves.

The result is always a boolean proof formula. Probabilistic labels are reattached
afterwards by the DeepProbLog :class:`~deeplog.systems.deepproblog.Solver`
(matching each ground leaf to its ``label :: atom`` declaration), and
:func:`~deeplog.systems.deepproblog.compile_to_module` compiles the result into a
differentiable module (see `Compiling an engine result`_ below and
:doc:`../deeplog_circuits`).

Built-in predicates
-------------------

The default grounder ships with a small built-in predicate library. See
:doc:`deepproblog_builtins` for the full list and semantics. Additional
predicates can be registered by calling :meth:`~deeplog.grounding.prolog.grounder.PrologGrounder.add_builtin`.

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
       sum(A, B, S).

   ?- valid_sum(D1, D2, R).

The deterministic ``digit`` facts collapse to ``true`` leaves, while the labeled ``sum/3`` fact introduces a
learnable probability (``nn_is_sum/2``) that is multiplied into every proof of ``valid_sum/3``. The final query
returns all pairs of digits together with the proof formulas that DeepLog compiles into a differentiable module.

Compiling an engine result
---------------------------

.. versionadded:: 2.2.0

After the solver produces an
:class:`~deeplog.systems.deepproblog.EngineResult`, the
compilation module turns it into a single differentiable
:class:`~deeplog.DeepLogModule`.

:func:`~deeplog.systems.deepproblog.compile_to_module` does this in one call:

1. Gives each random atom a variable of its own: a probabilistic fact a
   two-valued variable, an annotated disjunction one variable over its values.
   Each proof becomes the ``expectation`` of its formula, with its atoms turned
   into tests of their variables' values, and its distribution the product of
   those atoms in probability, each extended with its variable.
2. Declares the variables' domains and their labelling function, built from
   the engine's atom labels, to the compiler
   (:meth:`~deeplog.formula.lowering.compiler.Compiler.declaring`). A label is a
   constant, or the value of the atom it names: a network's output when the
   compiler has a builder for that atom, and an input of the module otherwise.
3. Compiles every expectation together, which knowledge-compiles the proofs
   into the probability semiring once for all of them (see
   :doc:`../deeplog_circuits`). With evidence, each answer divides by the one
   expectation of the evidence.

.. code-block:: python

   from deeplog import Compiler
   from deeplog.grounding.prolog import SimpleGrounder
   from deeplog.systems.deepproblog import Solver, compile_to_module

   result = Solver(SimpleGrounder()).get_query_result(program)
   module = compile_to_module(result, Compiler())

   # module accepts input tensors and returns query probabilities
   output = module(input_tensor)

The engine result's formulas and labels can also be read directly:

.. code-block:: python

   for answer, formula in result.formulas.items():
       print(f"{answer}: {formula}")  # the proof, as formula text
       print(result.labels)

.. seealso::

   - :class:`~deeplog.grounding.prolog.SimpleGrounder` for the pure Python interpreter.
   - :class:`~deeplog.grounding.prolog.JanusGrounder` for the Janus/SWI-Prolog backend that accepts the same language.
   - :doc:`../deeplog_circuits` for the circuit transformation API used internally.
