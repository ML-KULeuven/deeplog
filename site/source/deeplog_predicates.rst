DeepLog predicate modules
=========================

.. container:: dl-section dl-section--hero

   .. rst-class:: dl-section__eyebrow

      REFERENCE

   .. rst-class:: dl-section__title

      Built-in predicate modules

   .. rst-class:: dl-section__lead

      DeepLog ships a small set of predicate modules wired directly by the
      formula factory. These are distinct from the DeepProbLog engine built-ins.

Overview
--------

Predicate modules implement the leaves of a DeepLog formula graph. They
consume concrete symbol bindings and return tensors in the chosen structure
(e.g. ``boolean``, ``probability``, ``logprobability``).

Built-ins
---------

.. list-table::
   :header-rows: 1
   :widths: 20 10 25 45

   * - Predicate
     - Arity
     - Structure
     - Meaning / usage
   * - ``=``
     - 2
     - ``boolean``
     - Equality over a finite domain.
   * - ``sums``
     - 3
     - ``boolean``
     - True when ``x + y == z``.
   * - ``p``
     - 2
     - ``probability``
     - Probability label ``p(atom,label)``.
   * - ``p``
     - 2
     - ``logprobability``
     - Log-probability label ``p(atom,label)``.
   * - ``<custom>``
     - n
     - ``probability``
     - Neural predicate backed by a torch module.

Details
-------

* **EqualityPredicate**: created with functor ``=`` and compares symbols after
  mapping the provided domain to integer ids.
* **SumsPredicate**: functor ``sums`` evaluates ``x + y == z`` and returns a
  boolean tensor.
* **ProbabilityPredicate**: functor ``p`` mixes constant labels (``true``,
  ``false``, numeric values) with symbolic atoms; use ``_probability`` or
  ``_logprobability`` to select semiring.
* **NetworkPredicate** (via ``get_network_predicate``): user-defined functor and
  arity; delegates evaluation to a provided ``torch.nn.Module``. Indexing
  functionality is built in: the predicate passes the first argument through the
  module and uses subsequent integer arguments as indices into the output.

.. seealso::

   - :doc:`deeplog_language` for the formula language syntax.
   - :doc:`docs/deepproblog_builtins` for the Prolog-style engine built-ins.
