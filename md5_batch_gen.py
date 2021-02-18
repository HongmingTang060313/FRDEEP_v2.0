from __future__ import print_function
# md5 hash generation
import hashlib
# path/dir
import re
import os,sys
from os import listdir
from os.path import isfile, join

# ------------------------------------
# Natural Sorting
def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """
    Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)
    return l

def get_file_list(datadir,file_format):
    """
        This function returns file names in selected format under given directory.

        Args:
        datadir: the directory where the selected format files are saved.
        file_format: customized file format waiting for seaching.

        Returns:
        file_list: a list of file names under the given directory.

    """
    # get list of FITS files:
    file_list = []
    # Iterate files over directory
    for filename in os.listdir(datadir):
        if filename.endswith("."+ file_format):
            name = os.path.join(filename)
            file_list.append(name)
    print('Number of file under the directory: ',len(file_list))
    return file_list

def full_file_list(datadir):
    """
        This function returns allfile names under given directory.

        Args:
        datadir: the directory where the selected format files are saved.

        Returns:
        filenames: a list of file names under the given directory.

    """
    filenames = [f for f in listdir(datadir) if isfile(join(datadir, f))]
    return filenames

def md5_gen(file_dir):
    """
        This function returns md5 sum code of selected file.

        Args:
        file_dir: filename with full directory waiting for md5 encoding.
        Returns:
        file_md5: the md5 code of the file

    """
    m = hashlib.md5(open(file_dir,'rb').read())
    file_md5 = m.hexdigest()
    return file_md5

def md5_data_batch_gen(data_dir):
    meta,data_batches,test_batch = [],[],[]
    filenames = full_file_list(data_dir)
    filenames = sort_nicely(filenames)
    for ii in range(len(filenames)):
        file_tmp_dir = data_dir + '/' + filenames[ii]
        tmp_md5 = md5_gen(file_tmp_dir)
        if filenames[ii].startswith('batches'):
            meta.append(filenames[ii])
            meta.append(tmp_md5)
        elif filenames[ii].startswith('test'):
            test_batch.append(filenames[ii])
            test_batch.append(tmp_md5)
        else:
            tmp_batch = []
            tmp_batch.append(filenames[ii])
            tmp_batch.append(tmp_md5)
            data_batches.append(tmp_batch)
    return meta, data_batches, test_batch
