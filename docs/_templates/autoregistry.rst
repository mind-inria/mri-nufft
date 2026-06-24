
.. list-table::
    :header-rows: 1
    :stub-columns: 1
    :widths: 20 80

    * - Key
      - Function
{% for item in items %}
    * - ``"{{ item.name }}"``
      - :py:func:`{{ item.truename }}{{ item.sig }} <{{ item.module }}.{{item.truename}}>`
{% endfor %}
