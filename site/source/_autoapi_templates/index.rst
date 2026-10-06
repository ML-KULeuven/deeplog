Internals
=========

Every DeepLog module, including the machinery behind the
:doc:`public API </public_api>`. Only the names the public API lists are stable:
anything else documented here may change in any release.

{# Configure which modules appear in the quick-jump table #}
{% set quickjump_ids = [
    'deeplog.formula',
    'deeplog.circuit',
    'deeplog.module',
    'deeplog.grounding',
    'deeplog.systems',
] %}

.. list-table:: Quick jumps
   :widths: 28 72
   :class: sd-shadow-none

   * - Module
     - Summary
   {% for target in quickjump_ids %}
      {% set page = pages|selectattr("id", "equalto", target)|first %}
      {% if page %}
         {% set summary = page.summary if page.summary else "(No module summary provided.)" %}
   * - :doc:`{{ page.id }} <{{ page.include_path }}>`
     - {{ summary }}
      {% else %}
   * - ``{{ target }}``
     - (Module not found in current build.)
      {% endif %}
   {% endfor %}

.. toctree::
   :hidden:
   :titlesonly:

   {% for page in pages|selectattr("is_top_level_object") %}
   {{ page.include_path }}
   {% endfor %}

Auto-generated via `sphinx-autoapi <https://github.com/readthedocs/sphinx-autoapi>`_.
