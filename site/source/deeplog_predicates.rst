DeepLog predicate modules
=========================

.. container:: dl-section dl-section--hero

   .. rst-class:: dl-section__eyebrow

      REFERENCE

   .. rst-class:: dl-section__title

      Built-in predicate modules

   .. rst-class:: dl-section__lead

      DeepLog ships a small set of predicate modules that every
      :class:`~deeplog.formula.lowering.compiler.Compiler` wires in by default.
      These are distinct from the DeepProbLog engine built-ins.

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
   * - ``p``
     - 2
     - ``probability``
     - Probability label ``p(atom,label)``.
   * - ``logp``
     - 2
     - ``logprobability``
     - Log-probability label ``logp(atom,label)``.
   * - ``<custom>``
     - 2
     - ``probability``
     - Neural predicate backed by a torch module.

Details
-------

* **EqualityPredicate**: registered under ``=``; compares symbols after
  mapping the provided domain to integer ids.
* **ProbabilityPredicate**: registered under ``p``; weighs a truth value, a
  variable's or a written ``true`` or ``false``, by its label, a number or an
  input; use ``_probability`` or ``_logprobability`` to select semiring.
* **NetworkPredicate**: registered under a binary predicate of the user's
  choosing, with its module bound, as
  ``atom_builders={("digit", 2, "probability"): partial(NetworkPredicate, module=net)}``;
  delegates evaluation to a provided ``torch.nn.Module``. The predicate
  passes the first argument through the module, and the second argument reads a
  row of its output. Given a domain, which lists the values of the module's
  output rows in row order, an atom reads the row at its value's position;
  without one, a value is its row, counting from 0.
* **Your own predicate**: declares the domain of each argument, its sort, with
  ``domains``. Over a named or a tensor domain, a value written there is read as
  a variable there holds it; over the values, ``Domain.of_values()``, the
  predicate reads an argument as it is. A variable ranges over the domain of the
  arguments it is.

.. seealso::

   - :doc:`deeplog_language` for the formula language syntax.
   - :doc:`docs/deepproblog_builtins` for the Prolog-style engine built-ins.
