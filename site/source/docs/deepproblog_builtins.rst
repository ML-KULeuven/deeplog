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

The default DeepProbLog-style engine exposes the following predicates (defined in
``src/deeplog/systems/deepproblog/engine/builtins.py``):

==================  ======  ============================================
Predicate           Arity   Meaning
------------------  ------  --------------------------------------------
``between``         3       Enumerate an integer interval
``==``              2       Term equality
``\\==``            2       Term inequality
``<, >, <=, >=``    2       Numeric comparisons on ground terms
``is``              2       Arithmetic evaluation (right argument must be ground)
==================  ======  ============================================

Additional predicates can be registered by calling
:meth:`deeplog.systems.deepproblog.engine.engine.Engine.add_builtin`.

.. seealso::

   - :doc:`deepproblog_language` for the full grammar and operational semantics.
