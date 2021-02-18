#!/usr/bin/env python
# This script is to generate GRGNOM dataset, returning users the necessary training/testing
# images/metrics.
# The script is primarily generated on 2019/10/16
# Last modified: 20200820
#-------------------------------------------------
from __future__ import print_function
# Array handling
import numpy as np
# the necessary libraries for dataset foundation
from PIL import Image
import os
import os.path
import sys
from tqdm import tqdm
import pickle
import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity
import md5_batch_gen
import torch
#-------------------------------------------------
#-------------------------------------------------
# Dataset foundation
#-------------------------------------------------
upper_dir = '/Users/tanghongming/Desktop/PhD/FRDEEP_v2.0/4_DataPickle_Generation/'
# GRGNOM-N data 1
base_folder_NVSS = 'NVSS'
url_NVSS = "https://drive.google.com/file/d/1yjMxY4PfXlD5GL-DCZH7dfhsiOSLbKXr/view?usp=sharing" #(require modification)
filename_NVSS = "NVSS.tar.gz"
# md5 code for the zipped file.
nvss_tgz_dir = upper_dir + filename_NVSS
tgz_md5_NVSS = md5_batch_gen.md5_gen(nvss_tgz_dir)
# md5 code for the train/test/meta files
meta, data_batches, test_batch = md5_batch_gen.md5_data_batch_gen(upper_dir+base_folder_NVSS)
train_list_NVSS = data_batches
test_list_NVSS = [
                  test_batch,
                  ]
meta_NVSS = {'filename': 'batches.meta',
             'key': 'label_names',
             'md5': meta[1],
            }
#-------------------------------------------------
# GRGNOM-F data 2
base_folder_FIRST = 'FIRST'
url_FIRST = "https://drive.google.com/file/d/1yjMxY4PfXlD5GL-DCZH7dfhsiOSLbKXr/view?usp=sharing" #(require modification)
filename_FIRST = "FIRST.tar.gz"
# md5 code for the zipped file.
first_tgz_dir = upper_dir + filename_FIRST
tgz_md5_FIRST = md5_batch_gen.md5_gen(first_tgz_dir)
# md5 code for the train/test/meta files
meta, data_batches, test_batch = md5_batch_gen.md5_data_batch_gen(upper_dir+base_folder_FIRST)
train_list_FIRST = data_batches
test_list_FIRST = [
                   test_batch,
                   ]
meta_FIRST = {'filename': 'batches.meta',
              'key': 'label_names',
              'md5': meta[1],
             }
#-------------------------------------------------


#-------------------------------------------------
# Source las data 3
base_folder_id = 'object_id'
# redshift information of each batch.
train_list_id = [
                       'train_batch.npy',
                       ]
test_list_id = [
                      'test_batch.npy',
                      ]
#-------------------------------------------------
#-------------------------------------------------
# Integrated data class
class FRDEEPv2(data.Dataset):
    """ FRDEEP v2.0 dataset

        Inspired by `HTRU1 <https://as595.github.io/HTRU1/>`_ Dataset.

        Args:
        root1 (string): Root directory of GRGNOM-N dataset where directory
        ``NVSS_data`` exists or will be saved to if download is set to True.
        root2 (string): Root directory of GRGNOM-F dataset where directory
        ``FIRST_data`` exists or will be saved to if download is set to True.
        root3 (string): Root directory of source object id where directory
        ``object_id`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
        creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
        and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
        target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
        puts it in root directory. If dataset is already downloaded, it is not
        downloaded again.

        """
    # NVSS dataset
    base_folder1 = base_folder_NVSS
    url_1 = url_NVSS
    filename_1 = filename_NVSS
    tgz_md5_1 = tgz_md5_NVSS
    train_list_1 = train_list_NVSS
    test_list_1 = test_list_NVSS
    meta1 = meta_NVSS

    # FIRST dataset
    base_folder2 = base_folder_FIRST
    url_2 = url_FIRST
    filename_2 = filename_FIRST
    tgz_md5_2 = tgz_md5_FIRST
    train_list_2 = train_list_FIRST
    test_list_2 = test_list_FIRST
    meta2 = meta_FIRST

    # Source object id
    base_folder3 = base_folder_id
    train_list_3 = train_list_id
    test_list_3 = test_list_id

    def __init__(self, root1, root2, root3,train=True,
                 transform=None, target_transform=None,
                 download=False):

        self.root1 = os.path.expanduser(root1)
        self.root2 = os.path.expanduser(root2)
        self.root3 = os.path.expanduser(root3)

        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list_1 = self.train_list_1
            downloaded_list_2 = self.train_list_2
            downloaded_list_3 = self.train_list_3
        else:
            downloaded_list_1 = self.test_list_1
            downloaded_list_2 = self.test_list_2
            downloaded_list_3 = self.test_list_3

        self.data1 = []
        self.data2 = []
        self.data3 = []
        self.filename1 = []
        self.targets = []

        # now load the picked numpy arrays of data 1
        for file_name, checksum in downloaded_list_1:
            file_path = os.path.join(self.root1, self.base_folder1, file_name)

            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data1.append(entry['data'])

                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                    self.filename1.extend(entry['filenames'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data1 = np.vstack(self.data1).reshape(-1, 1, 150, 150)
        self.data1 = self.data1.transpose((0, 2, 3, 1))
        print(np.shape(self.data1))

        self.filename1 = np.vstack(self.filename1).reshape(-1, 1, 1)

        # now load the picked numpy arrays of data 2
        for file_name, checksum in downloaded_list_2:
            file_path = os.path.join(self.root2, self.base_folder2, file_name)

            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')

                self.data2.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data2 = np.vstack(self.data2).reshape(-1, 1, 150, 150)
        self.data2 = self.data2.transpose((0, 2, 3, 1))
        print(np.shape(self.data2))

        # Function to decode object id
        def batch_str_tensor_decode(tensor):
            "convert string tensor to strings in batches"
            arr_to_return = [np.ndarray.tostring(element).decode("utf-8").strip() for element in tensor]
            return arr_to_return   
        
        # now load the numpy arrays of data 3
        for file_name in downloaded_list_3:
            file_path = os.path.join(self.root3, self.base_folder3, file_name)

            self.data3.extend(np.load(file_path))
            if 'labels' in entry:
                self.targets.extend(entry['labels'])
            else:
                self.targets.extend(entry['fine_labels'])
        self.data3 = np.asarray([np.fromstring(x,dtype='uint8') for x in self.data3])
        self.data3 = np.vstack(self.data3).reshape(-1, 31)
        self.data3 = batch_str_tensor_decode(self.data3)

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root1, self.base_folder1, self.meta1['filename'])
        if not check_integrity(path, self.meta1['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')

            self.classes = data[self.meta1['key']]

        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
            Args:
            index (int): Index

            Returns:
            tuple: (image1, image2, metric1,metric2,metric3,target) where target is index of the target class.
        """

        img1,img2,metric1,target =self.data1[index],self.data2[index],self.data3[index],self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image (data 1)
        img1 = np.reshape(img1,(150,150))
        img1 = Image.fromarray(img1,mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image (data 2)
        img2 = np.reshape(img2,(150,150))
        img2 = Image.fromarray(img2,mode='L')
        if self.transform is not None:
            img2 = self.transform(img2)


        if self.target_transform is not None:
            target = self.target_transform(target)

        return img1,img2,metric1,target

    def __len__(self):
        return len(self.data1)

    def _check_integrity(self):
        root1 = self.root1
        for fentry in (self.train_list_1 + self.test_list_1):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root1, self.base_folder1, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        # extract file
        with tarfile.open(os.path.join(self.root1, self.filename_1), "r:gz") as tar:
            tar.extractall(path=self.root1)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root1 Location: {}\n'.format(self.root1)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
