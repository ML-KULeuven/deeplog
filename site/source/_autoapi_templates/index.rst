API Reference
=============

Browse the DeepLog API. Start with a curated set of high-level modules,
or view the full API in the sidebar.

{# Configure which modules appear in the quick-jump table #}
{% set quickjump_ids = [
    'deeplog',
    'deeplog.module',
    'deeplog.systems',
    'deeplog.util',
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
