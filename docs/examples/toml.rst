TOML Examples
=============

These pages are the TOML-based training examples.
They show the same fitting workflow as the Python walkthroughs, but in a
command-line friendly form.

Each case below is shown as:

- the TOML file contents
- the training command
- the real output from the launcher, including the automatic error summary
- a follow-up error-only check with ``forgeff error`` when you want to reuse the
  same trained model on another dataset

.. raw:: html

   <div class="ff-example-buttons">
      <a class="ff-example-button" href="#quick-start">
         <span class="ff-example-button__icon">R</span>
         <span class="ff-example-button__text">
           <strong>Run training</strong>
           <span>Launch a real TOML training job from the command line.</span>
         </span>
      </a>
      <a class="ff-example-button" href="#pairwise">
         <span class="ff-example-button__icon">P</span>
         <span class="ff-example-button__text">
           <strong>Pairwise examples</strong>
           <span>Morse, double Morse, and custom expressions.</span>
         </span>
      </a>
      <a class="ff-example-button" href="#eam">
         <span class="ff-example-button__icon">E</span>
         <span class="ff-example-button__text">
           <strong>EAM examples</strong>
           <span>Alloy and Finnis-Sinclair layouts for unary and binary cases.</span>
         </span>
      </a>
      <a class="ff-example-button" href="#adp">
         <span class="ff-example-button__icon">A</span>
         <span class="ff-example-button__text">
           <strong>ADP examples</strong>
           <span>Binary ADP training setup.</span>
         </span>
      </a>
   </div>

Quick start
-----------

To run a real training example from TOML, use the command-line launcher:

.. code-block:: bash

   python examples/toml/train.py --setting examples/toml/eam/alloy/forgeff.train.toml

You can swap the setting file for any of the folders listed below.

If you are starting out, use the unary alloy example first, then try the
binary alloy or Finnis-Sinclair cases to see how the same workflow changes
when the species list becomes larger.

Real output from the launcher looks like this:

.. code-block:: text

   Training setting: examples/toml/eam/alloy/forgeff.train.toml
   Time (step 0: minimize): 59.9907828749856 (s)
   Error statistics:
   +------------------------------------+-------+------------+------------+------------+
   | Metric                             | Count |        MAX |        ABS |        RMS |
   +------------------------------------+-------+------------+------------+------------+
   | Energy (eV)                        |    10 |  0.0636435 |  0.0368968 |  0.0418139 |
   | Energy per atom (eV/atom)          |    10 | 0.00198886 | 0.00115303 | 0.00130669 |
   | Forces per component (eV/angstrom) |   960 |    0.21925 |  0.0335738 |  0.0506721 |
   | Stress per component (GPa)         |    90 |    10.7196 |    3.40096 |    5.85269 |
   +------------------------------------+-------+------------+------------+------------+
   Training complete.

After training, you can run the error-only CLI on any dataset:

.. code-block:: bash

   forgeff error examples/toml/eam/alloy/final.npy examples/toml/data/unary/training.cfg

The CLI prints the same compact error table:

.. code-block:: text

   Error statistics:
   +------------------------------------+-------+------------+------------+------------+
   | Metric                             | Count |        MAX |        ABS |        RMS |
   +------------------------------------+-------+------------+------------+------------+
   | Energy (eV)                        |    10 |  0.0636435 |  0.0368968 |  0.0418139 |
   | Energy per atom (eV/atom)          |    10 | 0.00198886 | 0.00115303 | 0.00130669 |
   | Forces per component (eV/angstrom) |   960 |    0.21925 |  0.0335738 |  0.0506721 |
   | Stress per component (GPa)         |    90 |    10.7196 |    3.40096 |    5.85269 |
   +------------------------------------+-------+------------+------------+------------+

.. _pairwise:

Pairwise examples
-----------------

Morse
~~~~~

Initial TOML:

.. literalinclude:: ../../examples/toml/pairwise/morse/initial.toml
   :language: toml

Training setting:

.. literalinclude:: ../../examples/toml/pairwise/morse/forgeff.train.toml
   :language: toml

Run:

.. code-block:: bash

   python examples/toml/train.py --setting examples/toml/pairwise/morse/forgeff.train.toml

Output:

.. code-block:: text

   Training setting: examples/toml/pairwise/morse/forgeff.train.toml
   Time (step 0: minimize): 17.765608499990776 (s)
   Error statistics:
   +------------------------------------+-------+------------+------------+------------+
   | Metric                             | Count |        MAX |        ABS |        RMS |
   +------------------------------------+-------+------------+------------+------------+
   | Energy (eV)                        |    10 |    0.21788 |  0.0935474 |   0.109779 |
   | Energy per atom (eV/atom)          |    10 | 0.00680876 | 0.00292336 | 0.00343061 |
   | Forces per component (eV/angstrom) |   960 |   0.524229 |  0.0687978 |   0.104469 |
   | Stress per component (GPa)         |    90 |    37.2365 |    12.3929 |    21.4484 |
   +------------------------------------+-------+------------+------------+------------+
   Training complete.

Double Morse
~~~~~~~~~~~~

Initial TOML:

.. literalinclude:: ../../examples/toml/pairwise/double_morse/initial.toml
   :language: toml

Training setting:

.. literalinclude:: ../../examples/toml/pairwise/double_morse/forgeff.train.toml
   :language: toml

Run:

.. code-block:: bash

   python examples/toml/train.py --setting examples/toml/pairwise/double_morse/forgeff.train.toml

Output:

.. code-block:: text

   Training setting: examples/toml/pairwise/double_morse/forgeff.train.toml
   Time (step 0: minimize): 10.154842459014617 (s)
   Error statistics:
   +------------------------------------+-------+-----------+------------+------------+
   | Metric                             | Count |       MAX |        ABS |        RMS |
   +------------------------------------+-------+-----------+------------+------------+
   | Energy (eV)                        |    10 |  0.375028 |   0.170571 |   0.196564 |
   | Energy per atom (eV/atom)          |    10 | 0.0117196 | 0.00533036 | 0.00614261 |
   | Forces per component (eV/angstrom) |   960 |   0.17684 |  0.0104337 |  0.0185009 |
   | Stress per component (GPa)         |    90 |   27.8814 |    9.26232 |    16.0292 |
   +------------------------------------+-------+-----------+------------+------------+
   Training complete.

Custom expression
~~~~~~~~~~~~~~~~~

Initial TOML:

.. literalinclude:: ../../examples/toml/pairwise/custom_expression/initial.toml
   :language: toml

Training setting:

.. literalinclude:: ../../examples/toml/pairwise/custom_expression/forgeff.train.toml
   :language: toml

Run:

.. code-block:: bash

   python examples/toml/train.py --setting examples/toml/pairwise/custom_expression/forgeff.train.toml

Output:

.. code-block:: text

   Training setting: examples/toml/pairwise/custom_expression/forgeff.train.toml
   Time (step 0: minimize): 7.512485041981563 (s)
   Error statistics:
   +------------------------------------+-------+-----------+------------+------------+
   | Metric                             | Count |       MAX |        ABS |        RMS |
   +------------------------------------+-------+-----------+------------+------------+
   | Energy (eV)                        |    10 |  0.365016 |   0.181931 |   0.205917 |
   | Energy per atom (eV/atom)          |    10 | 0.0114067 | 0.00568534 | 0.00643489 |
   | Forces per component (eV/angstrom) |   960 |  0.776363 |  0.0538911 |  0.0919101 |
   | Stress per component (GPa)         |    90 |   103.554 |    34.2477 |    59.1233 |
   +------------------------------------+-------+-----------+------------+------------+
   Training complete.

.. _eam:

EAM examples
------------

Alloy, unary
~~~~~~~~~~~~

Initial TOML:

.. literalinclude:: ../../examples/toml/eam/alloy/initial.toml
   :language: toml

Training setting:

.. literalinclude:: ../../examples/toml/eam/alloy/forgeff.train.toml
   :language: toml

Run:

.. code-block:: bash

   python examples/toml/train.py --setting examples/toml/eam/alloy/forgeff.train.toml

Output:

.. code-block:: text

   Training setting: examples/toml/eam/alloy/forgeff.train.toml
   Time (step 0: minimize): 59.9907828749856 (s)
   Error statistics:
   +------------------------------------+-------+------------+------------+------------+
   | Metric                             | Count |        MAX |        ABS |        RMS |
   +------------------------------------+-------+------------+------------+------------+
   | Energy (eV)                        |    10 |  0.0636435 |  0.0368968 |  0.0418139 |
   | Energy per atom (eV/atom)          |    10 | 0.00198886 | 0.00115303 | 0.00130669 |
   | Forces per component (eV/angstrom) |   960 |    0.21925 |  0.0335738 |  0.0506721 |
   | Stress per component (GPa)         |    90 |    10.7196 |    3.40096 |    5.85269 |
   +------------------------------------+-------+------------+------------+------------+
   Training complete.

Alloy, binary
~~~~~~~~~~~~~

Initial TOML:

.. literalinclude:: ../../examples/toml/eam/alloy_binary/initial.toml
   :language: toml

Training setting:

.. literalinclude:: ../../examples/toml/eam/alloy_binary/forgeff.train.toml
   :language: toml

Run:

.. code-block:: bash

   python examples/toml/train.py --setting examples/toml/eam/alloy_binary/forgeff.train.toml

Output:

.. code-block:: text

   Training setting: examples/toml/eam/alloy_binary/forgeff.train.toml
   Time (step 0: minimize): 254.79491449997295 (s)
   Error statistics:
   +------------------------------------+-------+------------+------------+------------+
   | Metric                             | Count |        MAX |        ABS |        RMS |
   +------------------------------------+-------+------------+------------+------------+
   | Energy (eV)                        |    10 |   0.163612 |   0.079643 |  0.0931796 |
   | Energy per atom (eV/atom)          |    10 | 0.00511286 | 0.00248884 | 0.00291186 |
   | Forces per component (eV/angstrom) |   960 |   0.505452 |  0.0540783 |   0.082891 |
   | Stress per component (GPa)         |    90 |    64.5989 |    15.8664 |    30.4991 |
   +------------------------------------+-------+------------+------------+------------+
   Training complete.

Finnis-Sinclair, unary
~~~~~~~~~~~~~~~~~~~~~~

Initial TOML:

.. literalinclude:: ../../examples/toml/eam/fs_unary/initial.toml
   :language: toml

Training setting:

.. literalinclude:: ../../examples/toml/eam/fs_unary/forgeff.train.toml
   :language: toml

Run:

.. code-block:: bash

   python examples/toml/train.py --setting examples/toml/eam/fs_unary/forgeff.train.toml

Output:

.. code-block:: text

   Training setting: examples/toml/eam/fs_unary/forgeff.train.toml
   Time (step 0: minimize): 60.33321720798267 (s)
   Error statistics:
   +------------------------------------+-------+------------+------------+------------+
   | Metric                             | Count |        MAX |        ABS |        RMS |
   +------------------------------------+-------+------------+------------+------------+
   | Energy (eV)                        |    10 |  0.0723266 |  0.0407481 |  0.0464472 |
   | Energy per atom (eV/atom)          |    10 | 0.00226021 | 0.00127338 | 0.00145147 |
   | Forces per component (eV/angstrom) |   960 |   0.221679 |  0.0336762 |   0.050916 |
   | Stress per component (GPa)         |    90 |    9.55173 |    3.01064 |    5.17642 |
   +------------------------------------+-------+------------+------------+------------+
   Training complete.

Finnis-Sinclair, binary
~~~~~~~~~~~~~~~~~~~~~~~

Initial TOML:

.. literalinclude:: ../../examples/toml/eam/fs/initial.toml
   :language: toml

Training setting:

.. literalinclude:: ../../examples/toml/eam/fs/forgeff.train.toml
   :language: toml

Run:

.. code-block:: bash

   python examples/toml/train.py --setting examples/toml/eam/fs/forgeff.train.toml

Output:

.. code-block:: text

   Training setting: examples/toml/eam/fs/forgeff.train.toml
   Time (step 0: minimize): 238.48720883397618 (s)
   Error statistics:
   +------------------------------------+-------+------------+------------+------------+
   | Metric                             | Count |        MAX |        ABS |        RMS |
   +------------------------------------+-------+------------+------------+------------+
   | Energy (eV)                        |    10 |   0.183574 |  0.0940871 |   0.106693 |
   | Energy per atom (eV/atom)          |    10 | 0.00573668 | 0.00294022 | 0.00333415 |
   | Forces per component (eV/angstrom) |   960 |   0.461628 |  0.0505797 |  0.0791555 |
   | Stress per component (GPa)         |    90 |    70.2181 |    16.2962 |    32.8613 |
   +------------------------------------+-------+------------+------------+------------+
   Training complete.

.. _adp:

ADP examples
------------

Binary
~~~~~~

Initial TOML:

.. literalinclude:: ../../examples/toml/adp/alcu/initial.toml
   :language: toml

Training setting:

.. literalinclude:: ../../examples/toml/adp/alcu/forgeff.train.toml
   :language: toml

Run:

.. code-block:: bash

   python examples/toml/train.py --setting examples/toml/adp/alcu/forgeff.train.toml

Output:

.. code-block:: text

   Training setting: examples/toml/adp/alcu/forgeff.train.toml
   Time (step 0: minimize): 487.10146375000477 (s)
   Error statistics:
   +------------------------------------+-------+------------+------------+-----------+
   | Metric                             | Count |        MAX |        ABS |       RMS |
   +------------------------------------+-------+------------+------------+-----------+
   | Energy (eV)                        |    10 |   0.163188 |  0.0630916 | 0.0795872 |
   | Energy per atom (eV/atom)          |    10 | 0.00509962 | 0.00197161 | 0.0024871 |
   | Forces per component (eV/angstrom) |   960 |   0.393778 |  0.0526564 | 0.0794568 |
   | Stress per component (GPa)         |    90 |     2.4349 |   0.342029 |  0.525867 |
   +------------------------------------+-------+------------+------------+-----------+
   Training complete.
