
.. tip::  

    This module has a `registry` builtin.
    
    You can get easy access to the functions below by calling ``get_{{registry_key}}(<key>)`` or with ``get_{{registry_key}}(<key>, *args, **kwargs)``. Here are the function available from the registry:

    .. list-table::
        :header-rows: 1
        :widths: 20 80

        * - Key
          - Function
    {% for item in items %}
        * - ``"{{item.name}}"``
          - :py:obj:`{{item.truename}} <{{ item.path }}>` `{{item.sig}}`
    {% endfor %}
