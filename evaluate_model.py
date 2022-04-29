from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models import StevenNet
from hdf5dataset import Hdf5Dataset



def compute_metrics(model, dataloader):

    sum_abs_error = 0
    sum_squared_error = 0
    num_samples = len(dataloader.dataset)

    with torch.no_grad():
        for data, target in dataloader:
            output = model(data)
            sum_abs_error += torch.sum(torch.abs(output - target))
            sum_squared_error += torch.sum(torch.square(output - target))

    mae = sum_abs_error / num_samples
    rmse = torch.sqrt(sum_squared_error / num_samples)

    mae = mae.numpy()
    rmse = rmse.numpy()

    return mae, rmse


model_savefile_mapping = {
    'VentRate': 'model_VentRate_stevennet_2022-03-02.pt',
    'qt': 'model_qt_stevennet_2022-02-25.pt',
    'pr': 'model_pr_stevennet_2022-03-02.pt',
    'qrs': 'model_qrs_stevennet_2022-03-02.pt',
    'STJ_v5': 'model_STJ_v5_stevennet_2022-03-02.pt',
    'T_PeakAmpl_v5': 'model_T_PeakAmpl_v5_stevennet_2022-03-02.pt',
    'R_PeakAmpl_v5': 'model_R_PeakAmpl_v5_stevennet_2022-03-02.pt',
}


if __name__ == '__main__':

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    targets = ['qt', 'pr', 'qrs', 'STJ_v5', 'T_PeakAmpl_v5', 'R_PeakAmpl_v5', 'VentRate']

    output = ''

    for subset in ['train', 'validation', 'test']:

        output += f'Subset: {subset}\n'
           
        for target in targets:

            model = StevenNet()
            model.load_state_dict(torch.load('models/' + model_savefile_mapping[target], map_location=device))
            model.eval()

            run_data_precheck = False
            if target in ['pr', 'R_PeakAmpl_v5', 'T_PeakAmpl_v5', 'STJ_v5']:
                run_data_precheck = True 
            
            dataset = Hdf5Dataset(
                Path('../data/ECG_8lead_median_Run4_hdf5'),
                target_name=target,
                subset=subset,
                subset_splits=(0.6, 0.2, 0.2),
                random_seed=123,
                #debug_file_limit=10000,
                precheck_data=run_data_precheck
            )
            loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

            subset_mae, subset_rmse = compute_metrics(model, loader)

            output += ' {}: MAE = {:.4f}, RMSE = {:.4f}\n'.format(target, subset_mae, subset_rmse)

        output += '\n'

    print('')
    print(output)

