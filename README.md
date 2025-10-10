<div style="border: 2px solid #000; padding: 10px; margin-bottom: 20px;">
  <h1 align="center">Learning meta-GGA functionals with transformers</h1>

  <ul style="list-style-position: inside; text-align: center; padding: 0; margin: 10px 0;">
    <li style="margin-bottom: 8px;">
      This repository contains research code for the project (in preparation): 
      <strong>Mixture-of-experts transformers for faithfully deorbitalized meta-GGA density functionals</strong>
    </li>
    <li>
      All additional data (datasets, training scripts, etc.) has been uploaded to Figshare at the DOI:
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
    General density approximation: installed via <code>pip3 install 'gda@git+https://github.com/Matematija/global-density-approximation.git'<code>
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



  
</div>
