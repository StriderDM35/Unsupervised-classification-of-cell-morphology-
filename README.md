# Self-supervised classification of cell morphometric phenotypes as a function of specific extracellular matrices
This repository contains the code for the following paper (submitted): Self-supervised classification of cell morphometric phenotypes as a function of specific extracellular matrices.

**How to use the files for your own cluster analysis**
There are 3 files in this repository:<br />
- The first file contains the deep learning model being used
- The second file is used for training a model
- The third file is used for classification and cluster analysis
The second and third files can be run directly using any Python IDE such as Spyder or Sublime Text. 

**Results from the paper**
*The figures below are extracted directly from the manuscript, and show the results obtained using the dataset as shown in figure 1.*<br />

Figure 1 shows the dataset being used for cluster analysis.<br />

Figure 2 shows the inter- and intra-class morphological distinction.  A) Scatter plot of the first two principal components demonstrates the diverse spread of morphologies between and within cell classes. B) PCA loading plot indicating how different morphological aspects of the cells influence the directions of the component values. C) t-SNE dimensionality-reduced representation, emphasizing local structure. D) HDBSCAN cluster analysis of t-SNE processed data from C) with representative images of each cluster shown.<br />

Figure 3 shows a treemap of the HDBSCAN cluster results (top) and the respective labels for each cluster (bottom).<br /><br />

Figure 1<br />
<p align="center">
<img width="451" alt="image" src="https://user-images.githubusercontent.com/56214779/179389766-295cf958-b4a5-4164-bbd2-48ad3776db63.png">
</p>

Figure 2<br />
<p align="center">
<img width="451" alt="image" src="https://user-images.githubusercontent.com/56214779/179389955-6b13aef1-4a73-4fdd-a398-a73ea86d8c54.png">
</p>

Figure 3<br />
<p align="center">
<img width="451" alt="image" src="https://user-images.githubusercontent.com/56214779/179389709-fb9ebe45-be69-4007-bb87-7a9d0c4c00e4.png">
</p>
