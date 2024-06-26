# cycIF_TMAs

Analysis of multiplex imaging data from breast cancer tissue microarrays. Images and large data files available at https://www.synapse.org/#!Synapse:syn50134757/. (Free account required).

## Main Analysis Notebook

**Used to generate all figures in paper.**

**https://github.com/engjen/cycIF_TMAs/blob/main/20220912_JP-IMC-MIBI-TMAs_survival_spatial.ipynb**


## Additional Code

### Notebooks and code for image processing

Image smoothing, registration, single cell segmentation and feature extraction. 

https://github.com/engjen/cycIF_TMAs/blob/main/20201005_JP-TMA_Pipeline.py

https://github.com/engjen/cycIF_TMAs/blob/main/20220103_IMC_pipeline.ipynb

https://github.com/engjen/cycIF_TMAs/blob/main/20220315_MIBI_pipeline.ipynb

### Notebooks for single cell clustering and annotation

Cluster cells based on biomarker mean intensity using the leiden algorithm, and annotate.

**Breast cancer datasets:**

https://github.com/engjen/cycIF_TMAs/blob/main/20220118_JP-TMA_both_cluster.ipynb

https://github.com/engjen/cycIF_TMAs/blob/main/20220201_IMC_cluster_Mesmer_both.ipynb

https://github.com/engjen/cycIF_TMAs/blob/main/20220410_MIBI_cluster.ipynb

**Control datasets:**

https://github.com/engjen/cycIF_TMAs/blob/main/20240520_NP-DCIS_cluster.ipynb

https://github.com/engjen/cycIF_TMAs/blob/main/20240523_HER2-TMA_cluster.ipynb

https://github.com/engjen/cycIF_TMAs/blob/main/20240528_U54-TMA_MIBI_cluster.ipynb

https://github.com/engjen/cycIF_TMAs/blob/main/20240528_U54-TMA_cycIF_cluster.ipynb

### Notebooks for running spatstat and spatialLDA

Spatstat package used for Ripley's L, K cross, G cross, Occupancy. spatialLDA used for neighborhood analysis.

https://github.com/engjen/cycIF_TMAs/blob/main/20230419_spatstat_cycIF_TMAs.ipynb

https://github.com/engjen/cycIF_TMAs/blob/main/20220922_spatstat_IMC_MIBI.ipynb

https://github.com/engjen/cycIF_TMAs/blob/main/BC_Spatial_LDA_1.ipynb

### Code for visualization of images in Napari 

For QC and ROI selection.

https://github.com/engjen/cycIF_TMAs/blob/main/20201018_JP-TMAs_napari.py


## Analysis environment

To run the main analysis, installing python3/miniconda, and enter the following in the terminal to set up an `analysis` environment. 

`conda create -n analysis`

`conda activate analysis`

`conda install seaborn scikit-learn statsmodels numba pytables pandas ipykernel openpyxl`

`conda install -c conda-forge jupyterlab matplotlib python-igraph leidenalg scikit-image opencv tifffile libpysal shapely lifelines umap-learn napari scanpy statsmodels nodejs matplotlib-venn`

`conda install -c anaconda psutil pysal pillow`

`conda install -c bioconda anndata`

Finally, clone my repo for processing, visualization and analysis of multiplex imaging data

`git clone https://gitlab.com/engje/mplex_image.git`

## Other environments

To run image processing of cycIF images, set up environment to run our mplexable pipeline as described here:
https://gitlab.com/engje/mplexable

To run image processing of IMC and MIBI images (image smoothing, segmentation and feature extraction), set up an enviroment to run DeepCell, available here: https://pypi.org/project/DeepCell/

To run spatstat analysis, create an environment with a r kernel.
