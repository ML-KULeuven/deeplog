Getting started
===============

.. container:: dl-section dl-section--hero

   .. rst-class:: dl-section__eyebrow

      INSTALLATION

   .. rst-class:: dl-section__title

      Ship a NeSy baseline in under a minute

   .. rst-class:: dl-section__lead

      Install DeepLog from PyPI or from source to explore the tutorial notebooks and examples.

.. container:: dl-section dl-install-tabs

   .. tab-set::

      .. tab-item:: pip (recommended)

         Install DeepLog with pip, optionally adding extras for notebooks, tests, or the Janus backend:

         .. code-block:: console

            pip install pydeeplog

         Add extras as needed, for example ``pip install "pydeeplog[examples,tests]"``.

      .. tab-item:: Conda environments

         The recommended approach in Conda environments is to install PyTorch via Conda Forge first, then install DeepLog from PyPI:

         .. code-block:: console

            conda install pytorch -c conda-forge
            pip install pydeeplog

         This approach is handy when you need GPU-enabled PyTorch builds from Conda Forge.

      .. tab-item:: Install from source

         Clone the repository to track the latest commits or to contribute to DeepLog itself.

         .. code-block:: console

            git clone https://github.com/ML-KULeuven/deeplog.git
            cd deeplog
            pip install -e ".[tests,examples]"

         Editable installs automatically refresh when you modify the source tree.

.. container:: dl-section

   .. rubric:: Feature extras

   * ``pydeeplog[tests]`` — includes pytest, hypothesis, and assorted utilities to execute ``pytest`` across the repo.
   * ``pydeeplog[examples]`` — pulls the tutorial notebook requirements so every example in :doc:`tutorial` runs without additional setup.
   * ``pydeeplog[site]`` — installs the doc toolchain (Sphinx, myst-nb, design components) so you can run ``(cd site && make html)`` locally.
   * ``pydeeplog[janus_engine]`` — activates the SWI-Prolog powered Janus backend for lower-latency logical inference (requires a local SWI-Prolog install).

.. container:: dl-section

   .. rubric:: Hello DeepLog (30 seconds)

   .. jupyter-execute::

      import torch
      from deeplog import parse_formula_to_module

      # Compute the expected value of an implication (A → B ≡ ¬A ∨ B).
      # E[A → B] where A and B are independent boolean random variables.
      module = parse_formula_to_module("expectation(A, B): not A_boolean or B_boolean")

      # Each row: [P(A=true), P(B=true)]
      probs = torch.tensor(
          [
              [0.0, 0.0],  # A is false, so implication holds: E = 1.0
              [1.0, 1.0],  # Both true, implication holds: E = 1.0
              [1.0, 0.0],  # A true, B false, implication fails: E = 0.0
              [0.8, 0.3],  # E[A → B] = P(¬A) + P(A)P(B) = 0.2 + 0.8*0.3 = 0.44
          ]
      )
      expectations = module(probs)
      for (p_a, p_b), exp in zip(probs, expectations):
          print(f"P(A)={p_a:.1f}  P(B)={p_b:.1f}  ->  E[A → B]={float(exp):.2f}")

   This computes the probability that an implication holds given independent atom probabilities — a weighted model count. The ``expectation`` operator compiles the boolean formula into the probability semiring. Feed this into your loss to softly encourage logical constraints during training.

.. seealso::

   - :doc:`tutorial` ‒ pick the example notebook that matches your workload.
   - :doc:`autoapi/index` ‒ dive into the generated API reference when you start composing modules.
