.. DeepLog documentation master file, created by
   sphinx-quickstart on Sun Oct  6 18:09:05 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. title:: DeepLog


:html_theme.sidebar_secondary.remove:


.. raw:: html

   <body class="homepage">

.. raw:: html

   <div id="overview" class="full-width-banner section-anchor">
   <div class="banner_title_buttons">
   <div class="text-image-container-title">
   <div class="image-container">

   <img class="hero-logo" src="_static/images/deeplog-white.svg" alt="DeepLog logo" decoding="async">

   </div>
   <div class="text-container">
   <div class="hero-subtitle">DTAI LAB</div>
   <div class="hero-title">DeepLog</div>
   </div>
   </div>
   <div class="button-container">

.. grid:: 1 1 3 3

   .. grid-item::
      :child-align: center

      .. button-ref:: getstarted
         :color: warning
         :shadow:
         :align: center
         :expand:

         Get Started

   .. grid-item::
      :child-align: center

      .. button-ref:: tutorial
         :color: info
         :shadow:
         :align: center
         :expand:

         Tutorial

   .. grid-item::
      :child-align: center

      .. button-ref:: autoapi/index
         :color: info
         :shadow:
         :align: center
         :expand:

         API

.. raw:: html

   <div class="hero-description-block">
      <div class="hero-description-heading">What is DeepLog?</div>
      <p>DeepLog is a fundamental, operational framework designed to ease the implementation of new
      neurosymbolic systems. Instead of being "yet another" NeSy stack, DeepLog offers high-performance
      building blocks that integrate seamlessly with PyTorch-first workflows so you can accelerate reasoning
      without sacrificing GPU-ready tooling.</p>
   </div>

   </div>
   </div>
   </div>

.. raw:: html

   <div id="examples" class="use-case-stack section-anchor">
      <a class="use-case-row use-case-right use-case-bg-light use-case-pattern-neural" href="ml_practitioner.html">
         <p class="use-case-label">DeepLog for ML practitioners</p>
         <h2 class="use-case-heading">Inject logic into ML stacks</h2>
         <p class="use-case-description">DeepLog's building blocks drop into PyTorch-first workflows, letting you add symbolic guarantees without rewriting your training code. Follow the practitioner path to run DeepLog Modules, shapes, Semantic Loss, and the formula parser in runnable notebooks.</p>
         <span class="use-case-cta">Open the practitioner path</span>
      </a>
      <a class="use-case-row use-case-left use-case-bg-dark use-case-pattern-circuit" href="nesy_developer.html">
         <p class="use-case-label">DeepLog for neurosymbolic developers</p>
         <h2 class="use-case-heading">Implement neurosymbolic frameworks</h2>
         <p class="use-case-description">DeepLog provides the foundations for building your own neurosymbolic frameworks without reimplementing the difficult parts. Follow the NeSy developer path to trace symbols, shapes, predicates, formula compilation, and an end-to-end DeepProbLog-style workflow.</p>
         <span class="use-case-cta">Open the NeSy developer path</span>
      </a>
   </div>

.. raw:: html

   <div class="erc-logo-banner">
      <img src="_static/images/erc_logo.png" alt="European Research Council logo">
   </div>

.. toctree::
   :hidden:

   getstarted
   tutorial
   deeplog_language
   resources
