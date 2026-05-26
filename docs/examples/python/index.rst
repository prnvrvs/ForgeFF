Python Walkthroughs
===================

These pages are the standalone tutorial examples.
They are intentionally small and sequential: train, evaluate, grade, plus one
non-EAM pairwise walkthrough.

Use this section if you want to understand the code path end to end.
If you want a copyable configuration first, start with :doc:`../toml`.

What this section covers:

- how a fitted potential is created from training data
- how the trained model is evaluated on held-out structures
- how grading reports extrapolation and active-learning scores
- one analytical pairwise tutorial alongside the EAM-based path

What this section does not try to cover:

- every potential family in ForgeFF
- the interactive template wizard
- MPI deployment details
- all command-line flags and edge cases

.. raw:: html

   <div class="ff-example-buttons">
      <a class="ff-example-button" href="0.train.html">
         <span class="ff-example-button__icon">T</span>
         <span class="ff-example-button__text">
           <strong>Training</strong>
           <span>Fit a potential on a training set.</span>
         </span>
      </a>
      <a class="ff-example-button" href="1.evaluate.html">
         <span class="ff-example-button__icon">E</span>
         <span class="ff-example-button__text">
           <strong>Evaluation</strong>
           <span>Score a fitted model on held-out data.</span>
         </span>
      </a>
      <a class="ff-example-button" href="2.grade.html">
         <span class="ff-example-button__icon">G</span>
         <span class="ff-example-button__text">
           <strong>Grading</strong>
           <span>Inspect extrapolation and fit quality.</span>
         </span>
      </a>
      <a class="ff-example-button" href="6.toml_train.html">
         <span class="ff-example-button__icon">T</span>
         <span class="ff-example-button__text">
           <strong>TOML-driven training</strong>
           <span>Read a forgeff.train.toml file and train directly in Python.</span>
         </span>
      </a>
      <a class="ff-example-button" href="pairwise/morse/unary/morse_unary.html">
         <span class="ff-example-button__icon">P</span>
         <span class="ff-example-button__text">
           <strong>Pairwise Morse</strong>
           <span>Train a non-EAM analytical pair model in Python.</span>
         </span>
      </a>
   </div>

Each page shows the code, the result, and a short explanation of what the
example is doing. The TOML-driven training page is the bridge from the
configuration-first workflow to the Python API.

Suggested reading order:

1. Training
2. Evaluation
3. Grading
4. TOML-driven training
5. Pairwise Morse

For copyable training settings, jump back to :doc:`../toml`.

.. toctree::
   :hidden:

   /examples/python/0.train
   /examples/python/1.evaluate
   /examples/python/2.grade
