Writing a prover on Janus
=========================

.. container:: dl-section dl-section--hero

   .. rst-class:: dl-section__eyebrow

      REFERENCE

   .. rst-class:: dl-section__title

      Run your own SWI-Prolog prover through DeepLog

   .. rst-class:: dl-section__lead

      A prover is an SWI-Prolog module that a JanusProver loads and runs queries in. It builds
      on DeepLog's Prolog library, whose exported predicates follow semantic versioning like
      the Python API.

The prover
----------

:class:`~deeplog.grounding.prolog.JanusProver` loads a prover module once.
:meth:`~deeplog.grounding.prolog.janus.prover.JanusProver.consult` loads a program's clauses into
a module of their own and returns its name, and
:meth:`~deeplog.grounding.prolog.janus.prover.JanusProver.query` runs a goal in the prover's
module. Builtins added with
:meth:`~deeplog.grounding.prolog.janus.prover.JanusProver.add_builtin` belong to that prover, so
pass the prover into the goals that may call them.

.. code-block:: python

   from pathlib import Path

   from deeplog.grounding.prolog import JanusProver

   prover = JanusProver(Path("my_prover.pl"))
   program = prover.consult([":- dynamic fact/1.", "fact(p(a))."])
   rows = prover.query(
       "prove(Program, Prover, Goal, Answer)",
       {"Program": program, "Prover": prover, "Goal": ("p", ("X",))},
   )
   answers = [row["Answer"] for row in rows]

The prover module declares its own module name, loads the library, and reads the program
through the module name it is given:

.. code-block:: prolog

   :- module(my_prover, []).
   :- use_module(deeplog(grounding)).

   prove(Program, Prover, GoalSymbol, AnswerSymbol) :-
       from_symbol(GoalSymbol, Goal),
       (   is_builtin(Prover, Goal)
       ->  call_builtin(Prover, Goal)
       ;   Program:fact(Goal)
       ),
       to_symbol(Goal, AnswerSymbol).

The library
-----------

.. list-table::
   :header-rows: 1
   :widths: 24 20 56

   * - Predicate
     - Arguments
     - Meaning
   * - ``to_symbol/2``
     - ``+Term, -Symbol``
     - ``Symbol`` is ``Term`` as a DeepLog symbol. A list becomes a ``cons``/``nil`` chain.
   * - ``from_symbol/2``
     - ``+Symbol, -Term``
     - ``Term`` is the DeepLog symbol ``Symbol`` as a Prolog term. A ``cons``/``nil`` chain
       becomes a list.
   * - ``is_builtin/2``
     - ``+Prover, +Goal``
     - ``Goal`` is a builtin: an SWI-Prolog builtin a program may call, or one added to
       ``Prover``.
   * - ``call_builtin/2``
     - ``+Prover, ?Goal``
     - Prove the builtin ``Goal``, binding its arguments once per answer.
   * - ``unknown_predicate/1``
     - ``+Goal``
     - Throw the error that ``JanusProver.query`` raises as
       :class:`~deeplog.grounding.prolog.UnknownPredicateException`.
   * - ``non_ground_open_predicate/1``
     - ``+Goal``
     - Throw the error that ``JanusProver.query`` raises as :class:`ValueError`: ``Goal``,
       an instance of an open predicate, is not ground after proving its rule body.

The SWI-Prolog builtins a program may call are ``between/3``, ``nth0/3``, ``member/2``,
``length/2``, ``\==/2``, ``=:=/2`` and ``is/2``.

Only these exports are public. Anything else in the library can change in any release, so a
prover never calls into the library with a module prefix.

.. seealso::

   - :doc:`deepproblog_builtins` for the builtins of the default grounder.
   - The `Janus manual <https://www.swi-prolog.org/pldoc/man?section=packages-janus>`_ for
     ``py_call/2`` and how Python values cross into Prolog.
