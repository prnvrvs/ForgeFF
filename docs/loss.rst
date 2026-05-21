Loss function
=============

The following shows the loss function to be optimized during the training.

Definition
----------

.. math::

    L \equiv
    w_\mathrm{e} L_\mathrm{e} +
    w_\mathrm{f} L_\mathrm{f} +
    w_\mathrm{s} L_\mathrm{s}

- :math:`L_\mathrm{e}`: contribution from energy

  - ``energy_per_atom = true``:

    .. math::
        L_\mathrm{e} \equiv \sum_{k=1}^{N_\mathrm{conf}}
        (\hat{E}_k - \hat{E}_k^\mathrm{ref})^2

    where

    .. math::
        \hat{E}_k \equiv \frac{E_k}{N_{\mathrm{atom},k}}

- :math:`L_\mathrm{f}`: Contribution from forces

  - ``forces_per_atom = true``:

    .. math::
        L_\mathrm{f} \equiv \sum_{k=1}^{N_\mathrm{conf}}
        \frac{1}{N_{\mathrm{atom},k}} \sum_{i=1}^{N_{\mathrm{atom},k}} \sum_{\alpha=1}^{3}
        (F_{k,i\alpha} - F_{k,i\alpha}^\mathrm{ref})^2

- :math:`L_\mathrm{s}`: Contribution from stress

  - ``stress_times_volume = true``:

    .. math::
        L_\mathrm{s} \equiv \sum_{k=1}^{N_\mathrm{conf}} V_k^2
        \sum_{\alpha=1}^{3} \sum_{\beta=1}^{3}
        (\sigma_{k,\alpha\beta} - \sigma_{k,\alpha\beta}^\mathrm{ref})^2

    ``energy_per_atom`` is also respected.

Default
-------

``forgeff.toml``

.. code-block:: toml

    [loss]
    energy_weight = 1.0
    forces_weight = 0.01
    stress_weight = 0.001
    energy_per_atom = true
    forces_per_atom = true
    stress_times_volume = true
    energy_per_conf = true
    forces_per_conf = true
    stress_per_conf = true

Optional reference-energy offsets
---------------------------------

If your dataset does not include isolated-atom reference energies, you can
optionally add per-species offsets in the training TOML under ``[loss]``:

.. code-block:: toml

    [loss]
    species_energy_offset_mode = "manual"
    species_energy_offsets = { Al = -4.0, Cr = -3.0 }

Use ``species_energy_offset_mode = "regression"`` if you want ForgeFF to fit
the offsets from the dataset composition and total energies. The fitted
offsets are added to calculator predictions so the loss, RMSE, and reported
energies use the same absolute-energy convention.
