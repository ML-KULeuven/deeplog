ML practitioner path
====================

.. container:: dl-section dl-section--hero

   .. rst-class:: dl-section__eyebrow

      ML PRACTITIONER PATH

   .. rst-class:: dl-section__title

      Ship logic-augmented models fast

   .. rst-class:: dl-section__lead

      Use DeepLog Modules, shapes, and the formula parser to plug symbolic constraints into PyTorch pipelines without changing your modeling stack. Start with runnable cells and ship the first constraint-driven model quickly.

.. rst-class:: dl-section__switch

   Prefer to dig into internals? Switch to the :doc:`NeSy developer path <nesy_developer>`.

.. toctree::
   :caption: ML practitioner path
   :hidden:

   paths/ml/shape
   paths/ml/deeplogmodule
   paths/ml/formula_to_module
   paths/ml/semantic_loss

.. container:: dl-section

   .. rubric:: Who it's for

   - Practitioners adding semantic constraints to existing PyTorch models.
   - Readers who want runnable examples over theory.

   .. rubric:: Prerequisites

   - Basic PyTorch and notebook familiarity.

   .. rubric:: Estimated time

   - ⏱️ 45–75 minutes start to finish.

.. container:: dl-section

   .. rubric:: Guided notebook flow

   #. :doc:`Shapes <paths/ml/shape>` (⏱️ 10–15 min)

      Learn how shapes encode symbolic structure and how transformations are constructed automatically.

   #. :doc:`DeepLog Module <paths/ml/deeplogmodule>` (⏱️ 10 min)

      Run a minimal :class:`~deeplog.module.deeplog_module.DeepLogModule`, see input/output shapes, and understand how shape validation guards your pipeline.

   #. :doc:`From Formulas to Modules <paths/ml/formula_to_module>` (⏱️ 10–20 min)

      Use the parser to turn formulas into :class:`~deeplog.module.deeplog_module.DeepLogModule` objects that drop into your training code.

   #. :doc:`Semantic Loss: exactly-one constraint <paths/ml/semantic_loss>` (⏱️ 20–30 min)

      Regularize a semi-supervised MNIST model with an exactly-one constraint to see DeepLog in a standard ML loop.



.. container:: dl-section

   .. rubric:: Keep going

   - Revisit the :doc:`tutorial overview <tutorial>` to pick another path or dive into deeper internals when ready.
