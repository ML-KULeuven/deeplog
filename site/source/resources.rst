Resources
=========

.. container:: dl-section dl-section--hero

   .. rst-class:: dl-section__eyebrow

      REFERENCE MATERIAL

   .. rst-class:: dl-section__title

      Talks, papers, and engines that power DeepLog

.. rst-class:: dl-section__lead

      Catch up on the theory behind neurosymbolic systems, cite the right paper for your stack, and grab the engine docs needed to deploy DeepLog in production pipelines.

Language guide
++++++++++++++

.. container:: dl-section

   .. container:: dl-resources-grid

      .. container:: dl-resource-card

         **DeepLog language**

         Read the formula grammar that maps directly to DeepLog formula modules, including aggregations, transformations, and operator rules.

         .. raw:: html

            <footer><a class="dl-cta-button dl-cta-button--ghost" href="deeplog_language.html">Open the language guide</a></footer>

      .. container:: dl-resource-card

         **DeepLog predicate modules**

         Browse the built-in predicate modules that the formula factory wires in by default, including probability and equality predicates.

         .. raw:: html

            <footer><a class="dl-cta-button dl-cta-button--ghost" href="deeplog_predicates.html">Open predicate overview</a></footer>

      .. container:: dl-resource-card

         **Circuits and transformations**

         Learn about algebraic circuits, the intermediate representation that powers DeepLog, and the circuit transformation API for converting between structures.

         .. raw:: html

            <footer><a class="dl-cta-button dl-cta-button--ghost" href="deeplog_circuits.html">Open circuits guide</a></footer>

Implemented frameworks
++++++++++++++++++++++

.. container:: dl-section

   .. container:: dl-resources-grid

      .. container:: dl-resource-card

         **DeepProbLog-style language**

         Prolog-style rules and proof search used by DeepLog’s engine backend. This is distinct from the DeepLog formula language above.

         .. raw:: html

            <footer><a class="dl-cta-button dl-cta-button--ghost" href="docs/deepproblog_language.html">Open the engine language</a></footer>

      .. container:: dl-resource-card

         **DeepProbLog built-in predicates**

         Browse the default predicate library used by the DeepProbLog-style engines, including arithmetic, comparisons, and utility helpers that run before user rules.

         .. raw:: html

            <footer><a class="dl-cta-button dl-cta-button--ghost" href="docs/deepproblog_builtins.html">Open built-ins</a></footer>

Talks
+++++

.. container:: dl-section

   .. container:: dl-resources-grid

      .. container:: dl-resource-card

         **From Statistical Relational to Neural Symbolic Artificial Intelligence**

         Dumancic walks through the trajectory from SRL foundations to modern NeSy pipelines, highlighting where DeepLog fits and how semantic constraints anchor learning.

         .. raw:: html

            <footer><a class="dl-cta-button dl-cta-button--ghost" href="https://videolectures.net/videos/ESSAIandACAI2023_dumancic_from_statistical" target="_blank" rel="noopener">Watch the recording</a></footer>

Publications
++++++++++++

Surveys
-------

.. container:: dl-section

   .. container:: dl-resources-grid

      .. container:: dl-resource-card

         **From Statistical Relational to Neurosymbolic Artificial Intelligence: a Survey**

         Comprehensive overview of how probabilistic logic, differentiable reasoning, and neural learning interact, with taxonomy tables that help position DeepLog components in the broader landscape.

         .. raw:: html

            <footer><a class="dl-cta-button dl-cta-button--ghost" href="https://arxiv.org/pdf/2108.11451" target="_blank" rel="noopener">Read the PDF</a></footer>

Frameworks
----------

.. container:: dl-section

   .. container:: dl-resources-grid

      .. container:: dl-resource-card

         **Neural Probabilistic Logic Programming in DeepProbLog**

         Introduces DeepProbLog’s hybrid reasoning model that DeepLog builds upon; use it when you need probabilistic semantics together with neural perception.

         .. raw:: html

            <footer><a class="dl-cta-button dl-cta-button--ghost" href="https://arxiv.org/pdf/1907.08194" target="_blank" rel="noopener">Read the PDF</a></footer>

      .. container:: dl-resource-card

         **DeepStochLog: Neural Stochastic Logic Programming**

         Shows how stochastic proof search can remain differentiable—great background if you adapt DeepLog to non-deterministic symbolic programs.

         .. raw:: html

            <footer><a class="dl-cta-button dl-cta-button--ghost" href="https://ojs.aaai.org/index.php/AAAI/article/view/21248/20997" target="_blank" rel="noopener">Read the PDF</a></footer>

      .. container:: dl-resource-card

         **Soft-Unification in Deep Probabilistic Logic**

         Details how soft unification enables differentiable logic programs and motivates the abstractions implemented in DeepLog’s engines.

         .. raw:: html

            <footer><a class="dl-cta-button dl-cta-button--ghost" href="https://proceedings.neurips.cc/paper_files/paper/2023/file/bf215fa7fe70a38c5e967e59c44a99d0-Paper-Conference.pdf" target="_blank" rel="noopener">Read the PDF</a></footer>

Engines & tooling
-----------------

.. container:: dl-section

   .. container:: dl-resources-grid

      .. container:: dl-resource-card

         **Janus engine documentation**

         Official manual for the Janus bidirectional bridge that powers DeepLog’s SWI-Prolog backend. Covers installation, callbacks, and security model details.

         .. raw:: html

            <footer><a class="dl-cta-button dl-cta-button--ghost" href="https://www.swi-prolog.org/pldoc/man?section=packages-janus" target="_blank" rel="noopener">Visit docs</a></footer>

.. seealso::

   - :doc:`getstarted` ‒ installation, extras, and a hello-world snippet.
   - :doc:`tutorial` ‒ guided notebooks that reference the resources above.
   - `GitHub Discussions <https://github.com/ML-KULeuven/deeplog/discussions>`_ ‒ share papers or links that should be added to this page.

.. toctree::
   :hidden:

   deeplog_circuits
   docs/deepproblog_language
   docs/deepproblog_builtins
