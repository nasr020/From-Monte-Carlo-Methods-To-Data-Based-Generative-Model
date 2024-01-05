# MCMC Project

This repository contains the code and documentation for the MCMC project, which explores various statistical and machine learning methods to model extreme fiancial market returns. The project includes implementations of Cholesky Decomposition, Gaussian Mixture Models (GMM), and Generative Adversarial Networks (GAN), and is focused on analyzing and generating synthetic financial market data.

## Project Structure

- `submission/`: This folder contains the submission materials, including generated synthetic data and possibly pre-trained models.
- `.gitignore`: Git ignore file to prevent unnecessary files from being tracked.
- `MCMC_project.ipynb`: Jupyter notebook with the main implementation of the cholesky decomposition (analytical solution), the Gaussian Mixture Model and the GAN, including training procedures and result analysis.
- `grid_search_parallelized.py`: Script for performing GAN architecture grid search for hyperparameter tuning in parallelized form.
- `reproducing_results.ipynb`: Notebook containing the steps to reproduce the results of the best GAN model
- `utils.py`: Utility functions used across the project.

## Getting Started

To get started with this project, clone the repository to your local machine using:

### Testing the GAN

To test the GAN model and other models such as Cholesky decomposition and GMM, as well as to view the generated plots, run the `MCMC_project.ipynb` notebook:

This notebook will guide you through the process of setting up the models, training the GAN, and visualizing the results.

## Reproducing GAN results

several GAN architecture (size of layer, number of layers, activation function) have been tested and the best model in term of Anderson Darling Distance have been kept, in the submission folder
To reproduce result with the best GAN? run the reproducing_result notebook

## Submission Folder

The `submission/` folder contains the outputs from the GAN, including synthetic data files that closely match the statistical properties of real financial market data. These files are the end product of the project and serve as a reference for the quality of the generated synthetic data.


## Contact

 Nasr El Hamzaoui  & Martin Boutier 
