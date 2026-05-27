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

   paths/nesy/symbol
   paths/nesy/shape
   paths/nesy/predicates
   paths/nesy/01_aggregation_basics
   paths/nesy/03_free_variables_and_batching
   paths/nesy/formula_to_module
   paths/nesy/mnist_addition

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

   #. :doc:`Symbols <paths/nesy/symbol>` (⏱️ 10 min)

      Define symbols and understand how they anchor symbolic layouts.

   #. :doc:`Shapes <paths/nesy/shape>` (⏱️ 10–15 min)

      Work with symbolic shapes and reshape paths.

   #. :doc:`DeepLog Predicates <paths/nesy/predicates>` (⏱️ 15–20 min)

      Connect symbolic atoms to executable tensor operations.

   #. :doc:`Aggregation basics <paths/nesy/01_aggregation_basics>` (⏱️ 10 min)

      Learn the core aggregation syntax, domain enumeration, and module construction mechanics.

   #. :doc:`Free variables and batching <paths/nesy/03_free_variables_and_batching>` (⏱️ 10 min)

      Free variables become module inputs.

   #. :doc:`From Formulas to Modules <paths/nesy/formula_to_module>` (⏱️ 10–20 min)

      Compile formulas into :class:`~deeplog.module.deeplog_module.DeepLogModule` objects ready for composition.

   #. :doc:`MNIST Addition with DeepProbLog <paths/nesy/mnist_addition>` (⏱️ 20–30 min)

      Integrate perception, arithmetic, and logic in a full workflow.

.. container:: dl-section

   .. rubric:: Keep going

   - Hop back to the :doc:`tutorial overview <tutorial>` to revisit the ML practitioner flow or explore other examples.
