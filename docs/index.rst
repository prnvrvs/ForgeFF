ForgeFF
=======

.. raw:: html

   <section class="ff-hero">
     <div class="ff-hero__eyebrow">ForgeFF documentation</div>
     <h1>Make the fitting workflow understandable.</h1>
     <p>
       ForgeFF is a practical toolkit for semi-empirical force-field fitting.
       It keeps the equations, parameters, and workflow explicit so the small details
       stay visible: tabulated EAM and ADP, analytical pair forms, TOML configuration,
       and fast NumPy and Numba evaluation.
     </p>
     <div class="ff-hero__actions">
       <a class="ff-btn ff-btn--primary" href="installation.html">Get started</a>
       <a class="ff-btn" href="toml.html">Read the TOML spec</a>
       <a class="ff-btn" href="example.html">Browse example</a>
     </div>
   </section>

.. raw:: html

   <section class="ff-feature-strip">
     <div class="ff-feature-card">
       <span class="ff-feature-card__eyebrow">Fits</span>
       <strong>Pairwise, EAM, and ADP models</strong>
       <span>Analytical pair functions, tabulated potentials, and user-defined expressions.</span>
     </div>
    <div class="ff-feature-card">
      <span class="ff-feature-card__eyebrow">Config</span>
      <strong>One TOML file per potential</strong>
      <span>Explicit term ordering, grids, and initial values with relative paths.</span>
    </div>
     <div class="ff-feature-card">
       <span class="ff-feature-card__eyebrow">Runtime</span>
       <strong>NumPy and Numba engines</strong>
       <span>Fast evaluation for tabulated and built-in forms.</span>
     </div>
     <div class="ff-feature-card">
       <span class="ff-feature-card__eyebrow">Guide</span>
      <strong>Built for learning and reuse</strong>
      <span>EAM/ADP theory, TOML format, and a workflow that keeps the physics visible.</span>
    </div>
  </section>

If you are opening the docs for the first time, the easiest path is:

1. Install the package.
2. Read :doc:`toml` to see what goes in the file.
3. Open :doc:`example` to see the runnable Python walkthroughs.
4. Open :doc:`analytical` if you want a built-in pair form.
5. Come back to :doc:`calculators` when you want to see which engine runs it.
6. Open :doc:`toml` for the TOML schema.

Quick links
-----------

If you want the shortest route into the package, start here:

.. rst-class:: ff-link-grid

* :doc:`installation`
* :doc:`example`
* :doc:`analytical`
* :doc:`theory`
* :doc:`performance`
* :doc:`toml`
* :doc:`calculators`
* :doc:`cli/index`
* :doc:`api/index`
* :doc:`optimizers/index`
* :doc:`loss`
* :doc:`io`
* :doc:`credit`
* :doc:`maintainer`

Learning path
-------------

If you are new to the codebase, read these in order:

1. :doc:`installation`
2. :doc:`example`
3. :doc:`analytical`
4. :doc:`theory`
5. :doc:`performance`
6. :doc:`toml`
7. :doc:`calculators`
8. :doc:`cli/index`
9. :doc:`api/index`
10. :doc:`optimizers/index`
11. :doc:`loss`
12. :doc:`io`
13. :doc:`credit`
14. :doc:`maintainer`

Reference
---------

.. toctree::
   :maxdepth: 1

   installation
   example
   analytical
   theory
   performance
   toml
   calculators
   cli/index
   api/index
   optimizers/index
   loss
   io
   credit
   maintainer
