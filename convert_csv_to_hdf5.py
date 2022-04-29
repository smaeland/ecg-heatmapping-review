from pathlib import Path
from multiprocessing import Pool
from glob import iglob
from typing import Dict
import numpy as np
import h5py


def convert_type(in_str):

    if in_str.lower() == 'true':
        return True
    elif in_str.lower() == 'false':
        return False
    
    try:
        return int(in_str)
    except ValueError:
        pass

    return in_str


def process_single_data_csv(input_filename: Path, output_path: Path, info_dict: Dict, recreate=False):

    outfile = output_path / input_filename.name.replace('.csv', '.h5')

    if not recreate and outfile.exists():
        print('Skipping existing file:', outfile)
        return False

    if not input_filename.exists():
        print('No such file:', input_filename)
        return False

    data = []

    with open(input_filename) as fin:

        is_median = False

        for line in fin:
            
            # Proceed until we reach the medians
            if '12 lead medians' in line:
                is_median = True
                continue

            if not is_median:
                continue
            
            # Read in rows but skip last comma
            row = line.strip().split(',')[:-1]

            if len(row) < 12:
                continue

            row = list(map(int, row))

            data.append(row)
    
    data = np.array(data)

    with h5py.File(outfile, 'w') as fout:
        dataset = fout.create_dataset('median', data=data, dtype=np.int16)
        for attr_name, attr_value in info_dict.items():
            if attr_value != '' and attr_value is not None:
                dataset.attrs[attr_name] = convert_type(attr_value)
    
    print('Wrote', outfile)

    return True



def process_info_csv(ground_truth_file: Path, csv_dir: Path, output_dir: Path):
    """
    Process the ground truth CSV file
    """

    debug = False

    fin = open(ground_truth_file)
    info_header = fin.readline().strip().split(';')

    job_list = []
    with Pool(processes=12) as pool:

        for i, line in enumerate(fin.readlines()):

            if debug and i > 10:
                break

            values = line.strip().split(';')
            info = dict(zip(info_header, values))

            _id = int(info['patid'])
            filepath = csv_dir / f'{_id}.csv'

            job_list.append(
                pool.apply_async(
                    process_single_data_csv, (filepath, output_dir, info)
                )
            )
        
        res = [j.get() for j in job_list]
        print('Saved {} / {} files to hdf5'.format(sum(res), len(res)))


def check_file_integrity(filename):

    try:
        with h5py.File(filename, 'r') as h:
            d = h.get('median')
            if d is None or not d.shape == (600, 12):
                return False
    except OSError:
        return False
    
    return True


def check_files(data_dir: Path):

    glob_pattern = (data_dir / '*.h5').__str__()
    for f in iglob(glob_pattern):
        ok = check_file_integrity(f)
        if not ok:
            print(f'Bad file: {f}, removing')
            Path(f).unlink()


if __name__ == '__main__':

    info_csv = Path('../data/pulse2pulse_150k_ground_truth.csv')
    input_dir = Path('../data/ECG 8-lead (csv) median and rhythm Run4/')
    output_dir = Path('../data/ECG_8lead_median_Run4_hdf5')

    if not output_dir.exists():
        output_dir.mkdir()

    process_info_csv(info_csv, input_dir, output_dir)

    #check_files(output_dir)
    
    
