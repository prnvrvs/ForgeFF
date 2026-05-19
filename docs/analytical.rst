Analytical Forms
================

ForgeFF keeps all built-in analytical equations in one registry. This page is
the place to look when you want the equation itself, the parameter order, and
the exact name you can use in TOML.

If you only need the practical rule:

- pick the form name
- copy the parameter order
- give the initial values in the same order

For the TOML syntax itself, see :doc:`toml`. For worked file examples, see
:doc:`example`.

Pairwise forms
--------------

``lj``
    Lennard-Jones.

    .. math::

       V(r) = 4 \epsilon \left[\left(\frac{\sigma}{r}\right)^{12}
       - \left(\frac{\sigma}{r}\right)^6\right]

    Parameters: ``epsilon``, ``sigma``.

``bornmayer``
    Born-Mayer repulsion.

    .. math::

       V(r) = A e^{-r / \rho}

    Parameters: ``A``, ``rho``.

``morse``
    Morse pair potential.

    .. math::

       V(r) = D_e \left[e^{-2 a (r-r_e)} - 2 e^{-a (r-r_e)}\right]

    Parameters: ``De``, ``a``, ``re``.

``double_morse``
    Two Morse terms plus an offset.

    .. math::

       V(r) = E_1 \left[e^{-2 a_1 (r-r_1)} - 2 e^{-a_1 (r-r_1)}\right]
             + E_2 \left[e^{-2 a_2 (r-r_2)} - 2 e^{-a_2 (r-r_2)}\right]
             + \delta

    Parameters: ``E1``, ``a1``, ``r1``, ``E2``, ``a2``, ``r2``, ``delta``.

``power_decay``
    Power-law decay.

    .. math::

       V(r) = \alpha r^{-\beta}

    Parameters: ``alpha``, ``beta``.

``exp_decay``
    Exponential decay.

    .. math::

       V(r) = \alpha e^{-\beta r}

    Parameters: ``alpha``, ``beta``.

``constant``
    Constant offset.

    .. math::

       V(r) = c

    Parameters: ``c``.

``coul``
    Coulomb interaction.

    .. math::

       V(r) = 14.3996454784255 \frac{q_1 q_2}{r}

    Parameters: ``q1``, ``q2``.

``exponential``
    Power form.

    .. math::

       V(r) = A r^n

    Parameters: ``A``, ``n``.

``hbnd``
    Hydrogen-bond style 12-10 form.

    .. math::

       V(r) = \frac{A}{r^{12}} - \frac{B}{r^{10}}

    Parameters: ``A``, ``B``.

``buck``
    Buckingham-style form.

    .. math::

       V(r) = A e^{-r / \rho} - \frac{C}{r^6}

    Parameters: ``A``, ``rho``, ``C``.

``eopp``
    Oscillatory inverse-power form.

    .. math::

       V(r) = \frac{C_1}{r^{\eta_1}} + \frac{C_2}{r^{\eta_2}} \cos(k r + \phi)

    Parameters: ``C1``, ``eta1``, ``C2``, ``eta2``, ``k``, ``phi``.

``csw``
    Cosine/sine weighted form.

    .. math::

       V(r) = \frac{1 + c_1 \cos(k r) + c_2 \sin(k r)}{r^{\text{power}}}

    Parameters: ``c1``, ``c2``, ``k``, ``power``.

``csw2``
    Phase-shifted cosine form.

    .. math::

       V(r) = \frac{1 + c_1 \cos(k r + \phi)}{r^{\text{power}}}

    Parameters: ``c1``, ``k``, ``phi``, ``power``.

``ms``
    Morse-like short-range form.

    .. math::

       V(r) = D_e \left[e^{a(1-r/r_0)} - 2 e^{0.5 a(1-r/r_0)}\right]

    Parameters: ``De``, ``a``, ``r0``.

``born``
    Born-style repulsion with inverse-power tail.

    .. math::

       V(r) = A e^{(r_0-r)/\sigma} - \frac{C}{r^6} + \frac{D}{r^8}

    Parameters: ``A``, ``sigma``, ``r0``, ``C``, ``D``.

``softshell``
    Soft-shell power law.

    .. math::

       V(r) = \left(\frac{\alpha}{r}\right)^{\beta}

    Parameters: ``alpha``, ``beta``.

``exp_plus``
    Exponential plus constant offset.

    .. math::

       V(r) = \alpha e^{-\beta r} + c

    Parameters: ``alpha``, ``beta``, ``c``.

``mexp_decay``
    Shifted exponential decay.

    .. math::

       V(r) = \alpha e^{-\beta (r-r_0)}

    Parameters: ``alpha``, ``beta``, ``r0``.

``strmm``
    Two-term repulsive/attractive form.

    .. math::

       V(r) = 2 \alpha e^{-\frac{\beta}{2}(r-r_0)}
             - \gamma \left[1 + \delta (r-r_0)\right] e^{-\delta (r-r_0)}

    Parameters: ``alpha``, ``beta``, ``gamma``, ``delta``, ``r0``.

``poly_5``
    Fifth-order polynomial around ``r = 1``.

    .. math::

       V(r) = p_0 + \frac{1}{2} p_1 (r-1)^2 + p_2 (r-1)^3
             + p_3 (r-1)^4 + p_4 (r-1)^5

    Parameters: ``p0``, ``p1``, ``p2``, ``p3``, ``p4``.

``zero``
    Zero-valued form.

    .. math::

       V(r) = 0

    Parameters: ``none``.
