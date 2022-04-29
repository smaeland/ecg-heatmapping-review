from pathlib import Path
from glob import glob
import random
import json
from typing import Callable, Tuple, Optional, Union, List
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset


class Hdf5Dataset(Dataset):

    def __init__(
        self,
        data_dir: Union[Path, str],
        target_name: str,
        transform: Optional[Callable] = None,
        subset: Optional[str] = None,
        subset_splits: Optional[Tuple[float]] = None,
        random_seed: Optional[int] = None,
        sorted: bool = False,
        format: str = 'channels_first',
        precheck_data: bool = False,
        debug_file_limit: Optional[int] = None
    ) -> None:

        super().__init__()

        if isinstance(data_dir, str):
            data_dir = Path(data_dir)

        assert target_name in ['VentRate', 'qt', 'pr', 'qrs', 'STJ_v5', 'T_PeakAmpl_v5', 'R_PeakAmpl_v5']
        self.target_name = target_name

        assert format in ['channels_first', 'channels_last']
        self.format = format

        self.transform = transform

        # Get file list
        glob_expr = (data_dir / '*.h5').__str__()
        files = glob(glob_expr)
        assert len(files) > 0, f'No files found in {data_dir}'

        # Limit number of files for debugging purposes
        if debug_file_limit is not None:
            files = files[:debug_file_limit]
            print('Limiting number of files to', debug_file_limit)

        # Run a pre-check on all files, checking target label is in place
        if precheck_data:
            valid_files = []
            for f in files:
                with h5py.File(f, 'r') as hin:
                    d = hin.get('median')
                    if target_name in d.attrs.keys():
                        valid_files.append(f)

            print('Pre-check: {}/{} files passed'.format(len(valid_files), len(files)))
            files = valid_files

        # Do subset splits 
        assert not (bool(subset) ^ bool(subset_splits)), 'If using subsets, subset_splits must be defined'
        
        if subset is not None:
            assert subset in ['train', 'validation', 'test']
            self.subset = subset

            train_split = subset_splits[0]
            val_split = subset_splits[1]
            test_split = subset_splits[2]

            assert train_split + val_split + test_split == 1.0
            train_split_index = round(len(files)*train_split)
            val_split_index = train_split_index + round(len(files)*val_split)

            #print('dbg: subset indices: 0 - {} - {} - {}'.format(
            #    train_split_index, val_split_index, len(files)
            #))

            if random_seed is not None:
                random.seed(random_seed)            
            random.shuffle(files)
            
            if subset == 'train':
                self.files = files[:train_split_index]
            elif subset == 'validation':
                self.files = files[train_split_index: val_split_index]
            elif subset == 'test':
                self.files = files[val_split_index: ]

            print('Dataset: subset {}: initialised with {} files (of {})'.format(
                subset, len(self.files), len(files)
            ))
        
        else:
            self.files = files
            print('Dataset: initialised with {} files'.format(len(self.files)))
        
        print('Target name:', target_name)

        if sorted:
            self.files.sort()
        

    def __len__(self) -> int:
        return len(self.files)
    

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.files[idx]

        with h5py.File(filename, 'r') as fin:
            
            dataset = fin.get('median')
            target = dataset.attrs[self.target_name]
            target = np.array([target], dtype=np.float32)

            data = dataset[:]
            if self.format == 'channels_first':
                data = data.T
            
            if self.transform is not None:
                data = self.transform(data)
            
            data = data.astype(np.float32)

            return (data, target)
                

class Hdf5DatasetFromJson(Dataset):

    def __init__(
        self,
        filelist: str,
        target_name: str,
        transform: Optional[Callable] = None,
        sorted: bool = False,
        format: str = 'channels_first'
    ) -> None:

        super().__init__()

        assert target_name in ['VentRate', 'qt', 'pr', 'qrs', 'STJ_v5', 'T_PeakAmpl_v5', 'R_PeakAmpl_v5']
        self.target_name = target_name

        assert format in ['channels_first', 'channels_last']
        self.format = format

        self.transform = transform

        with open(filelist) as fin:
            self.files = json.load(fin)

        print('Dataset: initialised with {} files'.format(len(self.files)))

        if sorted:
            self.files.sort()
        

    def __len__(self) -> int:
        return len(self.files)
    

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.files[idx]

        with h5py.File(filename, 'r') as fin:
            
            dataset = fin.get('median')
            target = dataset.attrs[self.target_name]
            target = np.array([target], dtype=np.float32)

            data = dataset[:]
            if self.format == 'channels_first':
                data = data.T
            
            if self.transform is not None:
                data = self.transform(data)
            
            data = data.astype(np.float32)

            return (data, target)
                