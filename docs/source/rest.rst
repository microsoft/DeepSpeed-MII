RESTful API
===========

With a :doc:`deployment`, a RESTful API can be created. This allows users to
send requests to the server via ``HTTP POST`` methods (e.g., using ``curl`` or
the Python ``requests`` module). The RESTful API can be enabled with the
``enable_restful_api`` option using :func:`mii.serve`:

.. code-block:: python

    client = mii.serve(
        "mistralai/Mistral-7B-v0.1",
        deployment_name="test_dep",
        enable_restful_api=True,
        restful_api_port=28080,
    )

It is useful to provide a ``deployment_name`` and ``restful_api_port`` when
enabling the RESTful API as it will be used to provide an address where requests
can be sent. The address for sending requests will be
``http://{HOST}:{RESTFUL_API_PORT}/mii/{DEPLOYMENT_NAME}``. In the above
example, this will be ``http://localhost:28080/mii/test_dep``.

To send a request to the RESTful API, use the ``HTTP POST`` method. For example, using ``curl``:

.. code-block:: console

    (.venv) $ curl --header "Content-Type: application/json" --request POST  -d '{"prompts": ["DeepSpeed is", "Seattle is"], "max_length": 128}' http://localhost:28080/mii/test_dep

or using the Python ``requests`` module:

.. code-block:: python

    import json
    import requests
    url = f"http://localhost:28080/mii/test_dep"
    params = {"prompts": ["DeepSpeed is", "Seattle is"], "max_length": 128}
    json_params = json.dumps(params)
    output = requests.post(
        url, data=json_params, headers={"Content-Type": "application/json"}
    )
