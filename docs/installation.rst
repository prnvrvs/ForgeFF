Installation
============

PyPI
----

ForgeFF is published on PyPI and can be installed directly with ``pip``:

.. code-block:: bash

    pip install ForgeFF

GitHub
------

The development version is available from `GitHub <https://github.com/prnvrvs/ForgeFF>`_.

.. code-block:: bash

    pip install git+https://github.com/prnvrvs/ForgeFF.git

For development:

- Clone the GitHub repository.
- Install build-time requirements.
- Install ``forgeff`` in the editable mode with |no-build-isolation|_.

.. code-block:: bash

    git clone git@github.com:prnvrvs/ForgeFF.git
    cd forgeff
    pip install meson-python setuptools_scm "numpy>=2,<3"
    pip install --no-build-isolation -e .

.. |no-build-isolation| replace:: ``--no-build-isolation``
.. _no-build-isolation: https://pip.pypa.io/en/stable/cli/pip_install/#cmdoption-no-build-isolation
