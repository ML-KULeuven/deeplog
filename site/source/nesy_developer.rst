NeSy developer path
===================

.. container:: dl-section dl-section--hero

   .. rst-class:: dl-section__eyebrow

      NESY DEVELOPER PATH

   .. rst-class:: dl-section__title

      Understand and extend DeepLog internals

   .. rst-class:: dl-section__lead

      Trace symbols, shapes, predicates, and formula compilation to see how DeepLog assembles neurosymbolic systems. This flow emphasizes structure and extensibility for custom engines.

.. rst-class:: dl-section__switch

   Want a faster integration route? Try the :doc:`ML practitioner path <ml_practitioner>` instead.

.. toctree::
   :caption: NeSy developer path
   :hidden:

   paths/nesy/symbols_and_shapes
   paths/nesy/predicates
   paths/nesy/aggregation_basics
   paths/nesy/free_variables_and_batching
   paths/nesy/formula_to_module
   paths/nesy/problog
   paths/nesy/deepproblog

.. container:: dl-section

   .. rubric:: Who it's for

   - Developers designing or extending neurosymbolic stacks.
   - Readers interested in how DeepLog composes symbols, predicates, and modules.

   .. rubric:: Prerequisites

   - PyTorch experience and comfort with symbolic reasoning concepts.

   .. rubric:: Estimated time

   - ⏱️ 60–90 minutes start to finish.

.. container:: dl-section

   .. rubric:: Guided notebook flow

   #. :doc:`Symbols and shapes <paths/nesy/symbols_and_shapes>` (⏱️ 10 min)

      Name terms, atoms and their algebras with symbols, and the entries of a tensor with a SymTensor.

   #. :doc:`DeepLog Predicates <paths/nesy/predicates>` (⏱️ 15–20 min)

      Connect symbolic atoms to executable tensor operations.

   #. :doc:`Aggregation basics <paths/nesy/aggregation_basics>` (⏱️ 10 min)

      Learn the core aggregation syntax, domain enumeration, and expectations under a distribution.

   #. :doc:`Free variables and batching <paths/nesy/free_variables_and_batching>` (⏱️ 10 min)

      Free variables become module inputs.

   #. :doc:`From Formulas to Modules <paths/nesy/formula_to_module>` (⏱️ 10–20 min)

      Compile formulas into :class:`~deeplog.module.deeplog_module.DeepLogModule` objects ready for composition.

   #. :doc:`ProbLog programs <paths/nesy/problog>` (⏱️ 10–15 min)

      Probabilistic facts, rules, queries, and conditioning on evidence.

   #. :doc:`MNIST Addition with DeepProbLog <paths/nesy/deepproblog>` (⏱️ 20–30 min)

      Integrate perception, arithmetic, and logic in a full workflow.

.. container:: dl-section

   .. rubric:: Keep going

   - Hop back to the :doc:`tutorial overview <tutorial>` to revisit the ML practitioner flow or explore other examples.
