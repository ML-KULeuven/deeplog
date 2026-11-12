DeepProbLog built-in predicates
===============================

.. container:: dl-section dl-section--hero

   .. rst-class:: dl-section__eyebrow

      REFERENCE

   .. rst-class:: dl-section__title

      Built-in predicates for DeepProbLog-style engines

   .. rst-class:: dl-section__lead

      Built-ins are dispatched before user rules during proof search. Use this list to understand
      which predicates are available by default and how they behave on ground terms.

Built-ins
---------

The default plain-Prolog grounder exposes the following predicates (defined in
``src/deeplog/grounding/prolog/builtins.py``):

=============================  ======  ===============================================================
Predicate                      Arity   Meaning
-----------------------------  ------  ---------------------------------------------------------------
``between``                    3       Enumerate an integer interval
``=``, ``\=``                  2       Unify two terms, or test that they do not unify
``==``, ``\==``                2       Test that two terms are identical, or that they are not
``is``                         2       Evaluate the right argument and unify its number with the left
``=:=, =\=, <, >, =<, >=``     2       Evaluate both arguments and compare their numbers
=============================  ======  ===============================================================

Arithmetic is SWI-Prolog's, so :class:`~deeplog.grounding.prolog.SimpleGrounder` and
:class:`~deeplog.grounding.prolog.JanusGrounder` prove the same answers. An expression applies the operators
``+ - * / // mod rem div ^ /\ \/ xor << >>``; ``/`` of two integers is an integer where it divides exactly
(``4 / 2`` is ``2``, ``3 / 2`` is ``1.5``), and a number is written back as SWI-Prolog writes it (``1.0e-05``).

Additional predicates can be registered by calling
:meth:`~deeplog.grounding.prolog.grounder.PrologGrounder.add_builtin`.

.. seealso::

   - :doc:`deepproblog_language` for the full grammar and operational semantics.
