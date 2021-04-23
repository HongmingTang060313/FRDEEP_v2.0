# FRDEEP v2

The **FR-DEEP v2 Batched Dataset** is an upgrade of FR-DEEP v1, a dataset of labeled radio galaxies suitable for use with deep learning algorithms.  The labels for the samples are compiled from the [FRICAT](https://arxiv.org/abs/1610.09376) and [CoNFIG](https://academic.oup.com/mnras/article/390/2/819/1032320) catalogs. Each sample is classified as either [Fanaroff-Riley](https://en.wikipedia.org/wiki/Fanaroff%E2%80%93Riley_classification) Class I (FRI) or Class II (FRII). This dataset forms the base training data for the paper *Transfer Learning for Radio Galaxy Classification*. If you use this dataset please cite:

[(1)](#paper) *Transfer learning for radio galaxy classification*, Tang H., Scaife A. M. M., Leahy J. P., 2019, MNRAS, 488, 3358. doi:10.1093/mnras/stz1883 (or https://arxiv.org/abs/1903.11921)  

We would also be grateful if you are happy to cite the same paper when using our step-by-step dataset foundation tutorial as the basis of your own dataset foundation pipeline:)

# Necessary Packages 

1. Pytorch (ver. used: 1.0.1.post2)
2. hashlib
3. pickle
4. Numpy
5. Astroquery

## Updates:

1. Data sample size have enlarged from 600 objects to 658 objects
2. Data object ID now have full traceability, user can print out the object ID of each sample image immediately when using dataset for training/testing/data visualization.
3. Training batches are now merged into one. The dataset comprises only 1 training batch data and 1 testing batch data.
4. FRDEEP-N and FRDEEP-F in the FRDEEP v1 are now integrated as one, user could train/test with only NVSS, FIRST images, or both.
5. A step-by-step dataset foundation pipeline tutorial are now open to access! Please check out the tutorial (with 4 scripts, from Step 2 to 5; Step 1 refers to data sample spreadsheet collection, which we here have done it for you).

## The FR-DEEP v2 Batched Dataset

The [FR-DEEP v2]() is comprised of both NVSS and FIRST images for each sample. Images of the same objects taken from (1) the [NVSS survey](https://www.cv.nrao.edu/nvss/) and (2) the [FIRST survey](https://www.cv.nrao.edu/first/). The dataset contains 658 150x150 images in two classes: FR I & FR II. Images were extracted from the [Skyview Virtual Observatory](https://skyview.gsfc.nasa.gov/current/cgi/titlepage.pl), and underwent pre-processing described in [(1)](#paper). 

The angular size of the pixels for NVSS images: 15''/pixel; FIRST: 1.8''/pixel. In terms of angular scale, this means that a 150 x 150 pixel FIRST image covers the same area as an 18 x 18 pixel NVSS image.

There are 460 training images, and 198 test images. It is inspired by [CIFAR-10 Dataset](http://www.cs.toronto.edu/~kriz/cifar.html) and [HTRU1 Dataset](https://as595.github.io/HTRU1/).

The dataset is divided into 1 training batch and 1 test batch. In total, the dataset contains 268 FR I objects and 390 FR II objects. Notably, In [(1)](#paper) we mentioned there ar 659 primary samples, while we lately found that sample 4C 20.29a and 4C 20.29b are two FR I objects with the same recorded image central location. We therefore only retain one of the two sample image in this version, which gives 658 samples in total.

This is an *imbalanced dataset*

NVSS set images look like:

FR I: ![a](/4_DataPickle_Generation/NVSS_IMG/1433-0239_I.png) ![b](/4_DataPickle_Generation/NVSS_IMG/1434+0158_I.png) ![c](/4_DataPickle_Generation/NVSS_IMG/1435-0268_I.png) ![d](/4_DataPickle_Generation/NVSS_IMG/1437-0025_I.png)

FR II: ![a](/4_DataPickle_Generation/NVSS_IMG/1408+0050_II.png) ![b](/4_DataPickle_Generation/NVSS_IMG/1408+0281_II.png) ![c](/4_DataPickle_Generation/NVSS_IMG/1409-0307_II.png) ![d](/4_DataPickle_Generation/NVSS_IMG/1412-0075_II.png)

FIRST set images, on the other hand, look like:

FR I: ![a](/4_DataPickle_Generation/FIRST_IMG/1433-0239_I.png) ![b](/4_DataPickle_Generation/FIRST_IMG/1434+0158_I.png) ![c](/4_DataPickle_Generation/FIRST_IMG/1435-0268_I.png) ![d](/4_DataPickle_Generation/FIRST_IMG/1437-0025_I.png)

FR II: ![a](/4_DataPickle_Generation/FIRST_IMG/1408+0050_II.png) ![b](/4_DataPickle_Generation/FIRST_IMG/1408+0281_II.png) ![c](/4_DataPickle_Generation/FIRST_IMG/1409-0307_II.png) ![d](/4_DataPickle_Generation/FIRST_IMG/1412-0075_II.png)

## Using the Dataset in PyTorch

The FRDEEPv2_foundation.py file contains an instance of the [torchvision Dataset()](https://pytorch.org/docs/stable/torchvision/datasets.html) for the FRDEEP v2, and FRDEEPv2_tutorial.ipynb provides you a quick example of loading FRDEEP v2 and use it for simple CNN model training/testing.

To use it with PyTorch in Python, first import the torchvision datasets and transforms libraries like:

```python
from torchvision import datasets
import torchvision.transforms as transforms
```

Then import the FRDEEP v2:

```python
from FRDEEPv2_foundation import FRDEEPv2
```

Define the transform (you can always perform other transform operation here):

```python
# convert data to a normalized torch.FloatTensor
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5],[0.5])])
 ```

Read the FRDEEP v2 dataset from the data saving directory (if your customized directory didn't contains FRDEEPv2 data files, please set download=True to download the dataset to the directory):

```python
# choose the training and test datasets
trainset = FRDEEPv2(root="This is the directory saving FRDEEP v2 data", train=True, download=False, transform=transform)
testset = FRDEEPv2(root="This is the directory saving FRDEEP v2 data", train=False, download=False, transform=transform)
```

 
### Jupyter Notebooks

As said, an example of classification using the FIRST images in FRDEEP v2 via PyTorch is provided as a Jupyter notebook names FRDEEPv2_tutorial.ipynb. These are examples for demonstration only. If you use them for science directly, we too would be appreciated if you could cite [(1)](#paper). For contact and collaboration, please contact hongming.tang@manchester.ac.uk
