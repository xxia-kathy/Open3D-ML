import numpy as np
import pandas as pd
import os, sys, glob, pickle
from pathlib import Path
from os.path import join, exists, dirname, abspath
from tqdm import tqdm
import random
from plyfile import PlyData, PlyElement
from sklearn.neighbors import KDTree
from tqdm import tqdm
import logging

from .base_dataset import BaseDataset
from ..utils import make_dir, DATASET

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
)
log = logging.getLogger(__name__)


class AHAT(BaseDataset):
    """
    AHAT dataset, used in visualizer, training, or test
    """

    def __init__(self,
                 dataset_path,
                 name='AHAT',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 num_points=65536,
                 test_result_folder='./test',
                 val_files=['AHAT_val.ply'],
                 val_label_files=['AHAT_val_labels.ply'],
                 **kwargs):
        """
        Initialize
        Args:
            dataset_path (str): path to the dataset
            kwargs:
        Returns:
            class: The corresponding class.
        """
        super().__init__(dataset_path=dataset_path,
                         name=name,
                         cache_dir=cache_dir,
                         use_cache=use_cache,
                         num_points=num_points,
                         test_result_folder=test_result_folder,
                         val_files=val_files,
                         **kwargs)

        cfg = self.cfg
        print(cfg.val_files)

        self.label_to_names = self.get_label_to_names()

        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([0])

        train_path = cfg.dataset_path + "/train/"
        self.train_files = glob.glob(train_path + "/*.ply")
        self.train_label_files = glob.glob(train_path + "*.txt")
        self.val_files = [
            f for f in self.train_files if Path(f).name in cfg.val_files
        ]
        # self.val_label_files = [
        #     f for f in self.train_label_files if Path(f).name in cfg.val_label_files
        # ]
        self.train_files = [
            f for f in self.train_files if f not in self.val_files
        ]
        # self.train_label_files = [
        #     f for f in self.train_label_files if f not in self.val_label_files
        # ]


        test_path = cfg.dataset_path + "/test/"
        self.test_files = glob.glob(test_path + '*.ply')
        self.test_label_files = glob.glob(test_path + "*.txt")


    @staticmethod
    def get_label_to_names():
        label_to_names = {
            0: 'Background',
            1: 'Target',
        }
        return label_to_names

    def get_split(self, split):
        return AHATSplit(self, split=split)

    def get_labels_split_list(self, split):
        if split in ['test', 'testing']:
            files = self.test_label_files
        elif split in ['train', 'training']:
            files = self.train_label_files
        elif split in ['val', 'validation']:
            files = self.val_label_files
        elif split in ['all']:
            files = self.val_label_files + self.train_label_files + self.test_label_files
        else:
            raise ValueError("Invalid split {}".format(split))

        return files
    def get_split_list(self, split):
        if split in ['test', 'testing']:
            files = self.test_files
        elif split in ['train', 'training']:
            files = self.train_files
        elif split in ['val', 'validation']:
            files = self.val_files
        elif split in ['all']:
            files = self.val_files + self.train_files + self.test_files
        else:
            raise ValueError("Invalid split {}".format(split))

        return files

    def is_tested(self, attr):
        cfg = self.cfg
        name = attr['name']
        path = cfg.test_result_folder
        store_path = join(path, self.name, name + '.txt')
        if exists(store_path):
            print("{} already exists.".format(store_path))
            return True
        else:
            return False

    def save_test_result(self, results, attr):
        cfg = self.cfg
        name = attr['name'].split('.')[0]
        path = cfg.test_result_folder
        make_dir(path)

        pred = results['predict_labels'] + 1
        store_path = join(path, self.name, name + '.txt')
        make_dir(Path(store_path).parent)
        np.savetxt(store_path, pred.astype(np.int32), fmt='%d')

        log.info("Saved {} in {}.".format(name, store_path))


class AHATSplit():

    def __init__(self, dataset, split='training'):
        self.cfg = dataset.cfg
        path_list = dataset.get_split_list(split)
        labels_path_list = dataset.get_labels_split_list(split)
        log.info("Found {} pointclouds for {}".format(len(path_list), split))

        self.path_list = path_list
        self.labels_path_list = labels_path_list
        self.split = split
        self.dataset = dataset

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        pc_path = self.path_list[idx]
        log.debug("get_data called {}".format(pc_path))
        data = PlyData.read(pc_path)['vertex']
        labels_raw = pd.read_csv(self.labels_path_list[idx], header=None, delim_whitespace=True).values

        points = np.zeros((data['x'].shape[0], 3), dtype=np.float32)
        points[:, 0] = data['x']
        points[:, 1] = data['y']
        points[:, 2] = data['z']

        print(labels_raw)
        if (self.split != 'test'):
            labels = np.array([self.dataset.label_to_idx[l] for l in labels_raw], dtype=np.int32).reshape((-1,))
        else:
            labels = np.zeros((points.shape[0],), dtype=np.int32)

        data = {'point': points, 'feat': None, 'label': labels}

        return data

    def get_attr(self, idx):
        pc_path = Path(self.path_list[idx])
        name = pc_path.name.replace('.ply', '')

        attr = {'name': name, 'path': str(pc_path), 'split': self.split}
        return attr


DATASET._register_module(AHAT)
