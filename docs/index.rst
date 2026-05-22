ForgeFF
=======

.. raw:: html

   <section class="ff-hero">
     <div class="ff-hero__eyebrow">ForgeFF documentation</div>
     <h1>Make the fitting workflow understandable.</h1>
     <p>
       ForgeFF is a practical toolkit for semi-empirical force-field fitting.
       It keeps the equations, parameters, and workflow explicit so the small details
       stay visible: analytical pair forms, tabulated EAM and ADP, Stillinger-Weber,
       TOML configuration, and ASE plus ForgeFF NumPy and ForgeFF Numba
       engines.
     </p>
     <div class="ff-hero__actions">
       <a class="ff-btn ff-btn--primary" href="installation.html">Get started</a>
       <a class="ff-btn" href="toml.html">Read the TOML spec</a>
       <a class="ff-btn" href="engines.html">See engines</a>
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
       <strong>ASE, ForgeFF NumPy, and ForgeFF Numba engines</strong>
       <span>Pick the engine in the training setting file.</span>
     </div>
     <div class="ff-feature-card">
       <span class="ff-feature-card__eyebrow">Guide</span>
      <strong>Built for learning and reuse</strong>
      <span>Clear theory, TOML format, and runnable examples for the supported families.</span>
    </div>
  </section>

 If you are opening the docs for the first time, the easiest path is:

1. Install the package.
2. Read :doc:`toml` to see what goes in the file.
3. Open :doc:`engines` when you want to see which engine runs it.
4. Open :doc:`example` to see the runnable Python walkthroughs.
5. Open :doc:`analytical` if you want a built-in pair form.

Quick links
-----------

If you want the shortest route into the package, start here:

.. rst-class:: ff-link-grid

* :doc:`installation`
* :doc:`example`
* :doc:`toml`
* :doc:`engines`
* :doc:`analytical`
* :doc:`theory`
* :doc:`performance`
* :doc:`release_notes`
* :doc:`cli/index`
* :doc:`api/index`
* :doc:`optimizers/index`
* :doc:`loss`
* :doc:`io`
* :doc:`credit`
* :doc:`maintainer`

Learning path
-------------

If you are new to the codebase, start with :doc:`installation`, then
:doc:`example`, :doc:`theory`, and :doc:`toml`.

Reference
---------

.. toctree::
   :maxdepth: 1

   installation
   example
   toml
   engines
   analytical
   theory
   performance
   release_notes
   cli/index
   api/index
   optimizers/index
   loss
   io
   credit
   maintainer
