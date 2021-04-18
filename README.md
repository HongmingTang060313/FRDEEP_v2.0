# FRDEEP_v2.0

The **FR-DEEP v2 Batched Dataset** is a upgrated version of FR-DEEP v1, a dataset of labeled radio galaxies suitable for use with deep learning algorithms.  The labels for the samples are compiled from the [FRICAT](https://arxiv.org/abs/1610.09376) and [CoNFIG](https://academic.oup.com/mnras/article/390/2/819/1032320) catalogs. Each sample is classified as either [Fanaroff-Riley](https://en.wikipedia.org/wiki/Fanaroff%E2%80%93Riley_classification) Class I (FRI) or Class II (FRII). This dataset forms the base training data for the paper *Transfer Learning for Radio Galaxy Classification*. If you use this dataset please cite:

[(1)](#paper) *Transfer learning for radio galaxy classification*, Tang H., Scaife A. M. M., Leahy J. P., 2019, MNRAS, 488, 3358. doi:10.1093/mnras/stz1883 (or https://arxiv.org/abs/1903.11921)  

We would also be grateful if you are happy to cite the same paper when using our step-by-step dataset foundation tutorial as the basis of your own dataset foundation pipeline:)

# Necessary Packages 

1. Pytorch (ver. used: 1.0.1.post2)
2. hashlib
3. pickle

## Updates:

1. Data sample size have enlarged from 600 objects to 658 objects
2. Data object ID now have full traceability, user can print out the object ID of each sample image immediately when using dataset for training/testing/data visualization.
3. Training batches are now merged into one. The dataset comprises only 1 training batch data and 1 testing batch data.
4. FRDEEP-N and FRDEEP-F are integrated as one, user could train/test only with NVSS, FIRST images, or both.
5. A step-by-step dataset foundation pipeline are now open to access (yeah!)

## The FR-DEEP v2 Batched Dataset

The [FR-DEEP v2]() is comprised of both NVSS and FIRST images for each sample. Images of the same objects taken from (1) the [NVSS survey](https://www.cv.nrao.edu/nvss/) and (2) the [FIRST survey](https://www.cv.nrao.edu/first/). The dataset contains 658 150x150 images in two classes: FR I & FR II. Images were extracted from the [Skyview Virtual Observatory](https://skyview.gsfc.nasa.gov/current/cgi/titlepage.pl), and underwent pre-processing described in [(1)](#paper). 

The angular size of the pixels for NVSS images: 15''/pixel; FIRST: 1.8''/pixel. In terms of angular scale, this means that a 150 x 150 pixel FIRST image covers the same area as an 18 x 18 pixel NVSS image.

There are 460 training images, and 198 test images. It is inspired by [CIFAR-10 Dataset](http://www.cs.toronto.edu/~kriz/cifar.html) and [HTRU1 Dataset](https://as595.github.io/HTRU1/).

The dataset is divided into 1 training batch and 1 test batch. In total the dataset contains 268 FR I objects and 390 FR II objects. Notably, In [(1)](#paper) we mentioned there ar 659 primary samples, whil we lately found that sample 4C 20.29a and 4C 20.29b are two FR II objects wit the same image centre. We therefore only retain one of the two sample image in this version.

This is an *imbalanced dataset*

NVSS set images look like:

FR I: ![a](/4_DataPickle_Generation/NVSS_IMG/1433-0239_I.png) ![b](/4_DataPickle_Generation/NVSS_IMG/1434+0158_I.png) ![c](/4_DataPickle_Generation/NVSS_IMG/1435-0268_I.png) ![d](/4_DataPickle_Generation/NVSS_IMG/1437-0025_I.png)

FR II: ![a](/4_DataPickle_Generation/NVSS_IMG/1408+0050_II.png) ![b](/4_DataPickle_Generation/NVSS_IMG/1408+0281_II.png) ![c](/4_DataPickle_Generation/NVSS_IMG/1409-0307_II.png) ![d](/4_DataPickle_Generation/NVSS_IMG/1412-0075_II.png)

FIRST set images, on the other hand, look like:

FR I: ![a](/4_DataPickle_Generation/FIRST_IMG/1433-0239_I.png) ![b](/4_DataPickle_Generation/FIRST_IMG/1434+0158_I.png) ![c](/4_DataPickle_Generation/FIRST_IMG/1435-0268_I.png) ![d](/4_DataPickle_Generation/FIRST_IMG/1437-0025_I.png)

FR II: ![a](/4_DataPickle_Generation/FIRST_IMG/1408+0050_II.png) ![b](/4_DataPickle_Generation/FIRST_IMG/1408+0281_II.png) ![c](/4_DataPickle_Generation/FIRST_IMG/1409-0307_II.png) ![d](/4_DataPickle_Generation/FIRST_IMG/1412-0075_II.png)
