# FRDEEP_v2.0

The **FR-DEEP v2 Batched Dataset** is a upgrated version of FR-DEEP v1, a dataset of labeled radio galaxies suitable for use with deep learning algorithms.  The labels for the samples are compiled from the [FRICAT](https://arxiv.org/abs/1610.09376) and [CoNFIG](https://academic.oup.com/mnras/article/390/2/819/1032320) catalogs. Each sample is classified as either [Fanaroff-Riley](https://en.wikipedia.org/wiki/Fanaroff%E2%80%93Riley_classification) Class I (FRI) or Class II (FRII). This dataset forms the base training data for the paper *Transfer Learning for Radio Galaxy Classification*. If you use this dataset please cite:

[(1)](#paper) *Transfer learning for radio galaxy classification*, Tang H., Scaife A. M. M., Leahy J. P., 2019, MNRAS, 488, 3358. doi:10.1093/mnras/stz1883 (or https://arxiv.org/abs/1903.11921)  

We would also be grateful if you ar happy to cite the same paper when using our dataset foundation tutorial as the basis of your own dataset foundation pipeline:)

# Necessary Packages 

1. Pytorch (ver. used: 1.0.1.post2)
2. hashlib
3. pickle

## Updates:

1. Data object ID now have full traceability, user can print out the object ID of each sample image immediately when using dataset for training/testing/data visualization.
2. Data sample size have enlarged from 600 objects to 658 objects
3. The corresponding catalogues of the selected data samples are available (CoNFIG and FRICAT; citation required)
 
