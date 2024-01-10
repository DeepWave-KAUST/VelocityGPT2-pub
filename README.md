![LOGO](https://github.com/DeepWave-KAUST/VelocityGPT-dev/blob/main/asset/logo.png)

Reproducible material for **VelocityGPT: A sequential depth-wise velocity model generator in an autoregressive framework - Harsuko R., Cheng S., Alkhalifah T.**

[Click here](https://kaust.sharepoint.com/:f:/r/sites/M365_Deepwave_Documents/Shared%20Documents/Restricted%20Area/DWxxxxxxxx) to access the Project Report. Authentication to the _Restricted Area_ filespace is required.

# Project structure
This repository is organized as follows:

* :open_file_folder: **velocitygpt**: python library containing routines for VelocityGPT;
* :open_file_folder: **asset**: folder containing logo;
* :open_file_folder: **data**: folder reserved for data;
* :open_file_folder: **results**: folder reserved for storing results;
* :open_file_folder: **notebooks**: set of jupyter notebooks reproducing the experiments in the paper (see below for more details);

## Notebooks
The following notebooks are provided:

- :orange_book: ``VQVAE_training.ipynb``: notebook performing training of the VQ-VAE;
- :orange_book: ``VelocityGPT_training.ipynb``: notebook performing training of the GPT;
- :orange_book: ``Visualization.ipynb``: notebook to visualize the results


## Getting started :space_invader: :robot:
To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.

Simply run:
```
./install_env.sh
```
It will take some time, if at the end you see the word `Done!` on your terminal you are ready to go.

Remember to always activate the environment by typing:
```
conda activate velocitygpt
```

**Disclaimer:** All experiments have been carried on a Intel(R) Xeon(R) Platinum 8176 CPU @ 2.10GHz equipped with a single NVIDIA Quadro RTX 8000 GPU. Different environment 
configurations may be required for different combinations of workstation and GPU.

## Cite us 
DW0041 - Harsuko et al. (2024) VelocityGPT: A sequential depth-wise velocity model generator in an autoregressive framework.

