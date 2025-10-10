<div style="border: 2px solid #000; padding: 10px; margin-bottom: 20px;">
  <h1 align="center">Learning meta-GGA functionals with transformers</h1>

  <ul style="list-style-position: inside; text-align: center; padding: 0; margin: 10px 0;">
    <li style="margin-bottom: 8px;">
      This repository contains research code for the project (in preparation): 
      <strong>Mixture-of-experts transformers for faithfully deorbitalized meta-GGA density functionals</strong>
    </li>
    <li>
      All additional data (datasets, training scripts, model weights, etc.) has been uploaded to Figshare at the DOI: XXX
    </li>
  </ul>

  <h2 align="center" style="margin-top: 10px; color: #333;">
  Overview
  </h2>
  <p align="center">
    <img src="Meta-GGA-overview.png" width="800" />
    <br>
    <em>Summary of the physics-informed deep learning approach for reparameterizing exchange-correlation enhancement factors. The goal is to construct a surrogate model that mimics a faithfully deorbitalized meta-GGA functional, i.e., accurately predicting exchange-correlation energy densities and partial derivatives across real-space integration grids, whilst using only orbital-free input features.</em>
  </p>

<h2 align="center" style="margin-top: 10px; color: #333;">
  Requirements
</h2>

<ul style="list-style-position: outside; text-align: left; width: 80%; margin: 0 auto; padding-left: 40px;">
  <li>
    Python version 3.11.2 virtual environment
  </li>
  <li>
    Core packages: <code>pip install torch numpy pandas matplotlib e3nn</code>
  </li>
  <li>
    General density approximation: installed via  <code>pip3 install 'gda@git+https://github.com/Matematija/global-density-approximation.git'</code>
  </li>
  <li>
    Pylibxc version 7.0.0: installed via <code>conda install -c conda-forge pylibxc</code>
  </li>
  <li>
  Libxc shared libraries (<code>libxc.so</code>, <code>libxc.so.0</code>, <code>libxc.so.0.0.0</code>) and 
  Pylibxc Python bindings: copied from a Conda environment into the Python virtual environment’s 
  <code>lib/</code> and <code>site-packages/</code> directories, respectively
  </li>
  <li>
    Before activating the virtual environment, export the library path to include Libxc:
    <code>export LD_LIBRARY_PATH={virtual_environment_path/lib}:$LD_LIBRARY_PATH</code>
  </li>
</ul>

<h2 align="center" style="margin-top: 10px; color: #333;">
  Data Generation
</h2>

<ul style="list-style-position: outside; text-align: left; width: 80%; margin: 0 auto; padding-left: 40px;">
  <li>
    All training data was generated using the FHI-aims software, via post-processing of self-consistent densities (corresponding to optimised geometries from meta-GGA DFT calculations). Post-processing is performed using the <code>rho_and_derivs_on_grid</code> keyword, as detailed in the manual (see <a href="https://fhi-aims.org/" target="_blank" style="color: #1a73e8; text-decoration: none;">https://fhi-aims.org/</a>)
  </li>
  <li>
    Data to evaluate faithful deorbitalization for non-self-consistent densities was obtained via similar post-processing of meta-GGA DFT outputs, but for intermediate densities sampled along the SCF cycle. Non-self-consistent densities were generated and post-prcoessed using the keywords <code>rho_and_derivs_on_grid</code>, <code>sc_iter_limit</code> and <code>postprocess_anyway</code>. 
  </li>

</ul>

<h2 align="center" style="margin-top: 10px; color: #333;">
  Data Pre-Processing for the Mixture-of-Experts Model
</h2>

<ul style="list-style-position: outside; text-align: left; width: 80%; margin: 0 auto; padding-left: 40px;">
  <li>
    The Mixture-of-Experts (MoE) model is inferenced for a pre-computed set of inputs, which requires pre-processing in 5 steps
  </li>
  <li>
    <code>Eval_MGGA_X_MS2.py</code> inferences the General Density Approximation (GDA) kinetic energy density functional, given the FHI-aims computed inputs
  </li>
  <li>
    <code>Pylibxc_MGGA_X_MS2.py</code> inferences a Libxc-defined, semi-local meta-GGA exchange-correlation functional, given the FHI-aims computed inputs. The results of the orbital-dependent and deorbitalized Libxc functional are computed
  </li>
  <li>
    <code>Remove_rho_zero.py</code> cleans the output file from <code>Pylibxc_MGGA_X_MS2.py</code>
  </li>
  <li>
    <code>Evaluate_metaGGA_descriptors.py</code> evaluates all additional input features required for the MoE model 
  </li>

</ul>

<h2 align="center" style="margin-top: 10px; color: #333;">
  Mixture-of-Experts Model Inference
</h2>

<ul style="list-style-position: outside; text-align: left; width: 80%; margin: 0 auto; padding-left: 40px;">
  <li>
    The Mixture-of-Experts (MoE) model is inferenced by running <code>eval_full_MoE.py</code> inside a directory with a <code>normalization_constants_full.json</code> file, and two subdirectories containing the expert models and gating network (uploaded to Figshare at the DOI: XXX). Within <code>eval_full_MoE.py</code>, one can specify the path to the pre-processed input files for a given system. 
  </li>
</ul>
 
</div>
