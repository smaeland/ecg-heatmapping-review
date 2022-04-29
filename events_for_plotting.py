"""
Find good events (i.e. no missing data) for heatmap plotting
"""

from glob import glob
from pathlib import Path
from random import shuffle
import json
import h5py
from hdf5dataset import Hdf5Dataset


if __name__ == '__main__':

    num_events = 100
    good_files = []

    # Get files from test set -- get list from a Hdf5Dataset
    test_dataset = Hdf5Dataset(
        Path('../data/ECG_8lead_median_Run4_hdf5'),
        target_name='qt',
        subset='test',
        subset_splits=(0.6, 0.2, 0.2),
        random_seed=123,
    )

    files = test_dataset.files

    shuffle(files)

    targets = set(['VentRate', 'qt', 'pr', 'qrs', 'STJ_v5', 'T_PeakAmpl_v5', 'R_PeakAmpl_v5'])

    for f in files:
        with h5py.File(f, 'r') as hin:
            d = hin.get('median')
            keys = set(d.attrs.keys())
            if (targets & keys) == targets:
                good_files.append(f)
        
        if len(good_files) >= num_events:
            break
    
    foutname = 'good_events_for_plotting.json'
    with open(foutname, 'w') as fout:
        json.dump(good_files, fout)
    
    print(f'Wrote {len(good_files)} file names to {foutname}')

