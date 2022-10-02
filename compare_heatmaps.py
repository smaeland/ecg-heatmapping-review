"""
APOC-like comparison of heatmap methods
"""
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict
import warnings
import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
from evaluate_attributions import RandomiseIteratively
from models import StevenNet
from hdf5dataset import Hdf5Dataset
from heatmaps import CaptumMethodWrapper, model_savefile_mapping1
from visualise import visualise_attr_1d, _normalize_attr


def format_timedelta(dt, iso=False):

    # Remove microseconds
    s = dt.__str__()
    s = s[:s.find('.')]

    if not iso:
        # Remove hrs/mins if zero
        if dt.seconds < 3600:
            s = s[s.find(':')+1:]
        
        if dt.seconds < 60:
            s = s[s.find(':')+1:]

    return s


def compute_iterative_errors(
    x,
    y,
    attr_method: CaptumMethodWrapper,
    num_iterations: int,
    window_size: int,
    plot: bool = False,
    plotting_kwargs: Optional[Dict] = None
):
    """
    For a given sample x, compute the errors under iterative randomisation
    """

    pred = attr_method.predict(x)
    attributions = attr_method.attribute(x, y)

    # Use absolute values of attributions
    attributions = np.abs(attributions)

    # Make sure we use absolute values 
    # For plotting, keep a copy of normalised attributions before randomisation
    # Do not clip values during normalisation
    normed_attr = _normalize_attr(
        attributions,
        sign='absolute_value',
        clip_values=False
    )

    if plotting_kwargs is None:
        plotting_kwargs = {}
    
    # Iterative randomiser
    randomizer = RandomiseIteratively(x, normed_attr, add_or_replace_noise='add')
    itr = randomizer.randomise_by_attributions(window_size=window_size, verbose=plot)

    abs_err = np.zeros(shape=(num_iterations))
    abs_perc_err = np.zeros(shape=(num_iterations))
    
    abs_err[0] = np.abs(y-pred)
    abs_perc_err[0] = np.abs(y-pred) / y

    for j in range(1, num_iterations):

        x_rnd, y_rnd = next(itr)
        _x_rnd = torch.Tensor(x_rnd).unsqueeze(0)
        pred_rnd = attr_method.predict(_x_rnd)

        abs_err[j] = np.abs(y-pred_rnd)
        abs_perc_err[j] = np.abs(y-pred_rnd) / y

        if plot:
            if j in [1, 2, 5, 10, 20, 50, 100, 200, 1000]:
                visualise_attr_1d(
                    y_rnd, x_rnd,
                    normalise=False,
                    title='Iteration {}, abs error = {:.1f}%'.format(
                        j, abs_perc_err[j]*100
                    ),
                    **plotting_kwargs
                )

    return abs_err, abs_perc_err







if __name__ == '__main__':

    observables = ['VentRate', 'qt', 'pr', 'qrs', 'STJ_v5', 'T_PeakAmpl_v5', 'R_PeakAmpl_v5']

    parser = ArgumentParser('Objective comparison of heatmaps')
    subparsers = parser.add_subparsers(dest='command')

    parser_compute = subparsers.add_parser('compute_errors')
    parser_compute.add_argument('method', type=str)
    parser_compute.add_argument('--observable', type=str, choices=observables, required=True)
    parser_compute.add_argument('--num_iterations', type=int, default=200)
    parser_compute.add_argument('--event_limit', type=int, default=None)
    parser_compute.add_argument('--plot', action='store_true')
    parser_compute.add_argument('--output_dir', type=str, default='error_analysis')

    parser_compare = subparsers.add_parser('plot_comparison')
    parser_compare.add_argument('--input_dir', type=str, default='error_analysis')
    parser_compare.add_argument('--output_dir', type=str, default='plots')
    parser_compare.add_argument('--observable', type=str, choices=observables, required=True)
    parser_compare.add_argument('--save', action='store_true')


    args = parser.parse_args()
    print(args)

    
    # Compute the errors per method and store the results to file
    if args.command == 'compute_errors':

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)

        # Load model and data
        model = StevenNet()
        model.load_state_dict(torch.load('models/' + model_savefile_mapping1[args.observable], map_location=device))
        model.eval()

        test_dataset = Hdf5Dataset(
            '../data/ECG_8lead_median_Run4_hdf5',
            target_name=args.observable,
            subset='test',
            subset_splits=(0.6, 0.2, 0.2),
            random_seed=123,
            precheck_data=True
        )

        batch_size = 1
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )

        output_dir = Path(args.output_dir)
        if not output_dir.is_dir():
            print('Creating directory:', output_dir.__str__())
            output_dir.mkdir()
        
        method = CaptumMethodWrapper(
            args.method,
            model,
            verbose=False
        )

        all_abs_errors = []
        all_abs_perc_errors = []
        
        time0 = datetime.now()
        time1 = datetime.now()

        nprocessed = 1

        # Loop over events 
        for data, target in test_dataloader:
            for i in range(batch_size):

                x, y = data[i], target[i]

                try:
                    attr = method.attribute(x, y)

                    abs_error, abs_perc_error = compute_iterative_errors(
                        x, y,
                        method,
                        num_iterations=args.num_iterations,
                        window_size=1,
                        plot=args.plot
                    )
                except Exception as exc:
                    print('ERROR:', exc)
                    print('Skipping event')
                    nprocessed += 1 # keep this?
                    continue
                
                all_abs_errors.append(abs_error)
                all_abs_perc_errors.append(abs_perc_error)

                time2 = datetime.now()
                difftime_str = format_timedelta(time2 - time1)
                elapsed_str = format_timedelta(time2 - time0, iso=True)
                print('Event {}: {} s/event, {} elapsed'.format(
                    nprocessed, difftime_str, elapsed_str),
                    flush=True
                )
                time1 = datetime.now()

                nprocessed += 1
            
            if args.event_limit is not None and nprocessed > args.event_limit:
                print('Reached event limit')
                break
        
        all_abs_errors = np.array(all_abs_errors)
        all_abs_perc_errors = np.array(all_abs_perc_errors)

        outname = output_dir / f'iterative_errors_{args.observable}_{args.method}.h5'
        with h5py.File(outname.__str__(), 'w') as fout:
            fout.create_dataset('abs_errors', data=all_abs_errors, dtype=np.float32)
            fout.create_dataset('abs_perc_errors', data=all_abs_perc_errors, dtype=np.float32)

        print('Saved output as', outname.__str__())
    


    # Load previously computed errors and plot
    if args.command == 'plot_comparison':

        warnings.simplefilter('error')

        methods = {
            'saliency': 'Saliency',
            'smoothgrad': 'SmoothGrad',
            'deconvolution': 'Deconvolution',
            'inputxgradient': 'Input x gradient',
            'deeplift': 'DeepLIFT',
            'guided_backprop': 'Guided backpropagation',
            'integrated_gradients': 'Integrated gradients',
            'gradientshap': 'Gradient SHAP',
            'random': 'Random',
            'guided_gradcam': 'Guided Grad-CAM',
        }

        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            raise RuntimeError('No such directory:', input_dir)


        #fig_mae, ax_mae = plt.subplots()
        fig_mape, ax_mape = plt.subplots(figsize=(8, 8))
        fig_mape_wide, ax_mape_wide = plt.subplots(figsize=(24, 8))
        #fig_rmse, ax_rmse = plt.subplots()
        xvals = None

        colors =  plt.cm.Set1.colors[:5] * 2

        color_cycler = (
            plt.cycler('color',  colors) + 
            plt.cycler('linestyle', (['-']*5 + ['--']*5))
        )
        #ax_mae.set_prop_cycle(color_cycler)
        ax_mape.set_prop_cycle(color_cycler)
        ax_mape_wide.set_prop_cycle(color_cycler)
        #ax_rmse.set_prop_cycle(color_cycler)


        for name, title in methods.items():
            
            infile = input_dir / f'iterative_errors_{args.observable}_{name}.h5'
            with h5py.File(infile.__str__(), 'r') as fin:
                abs_errors = fin.get('abs_errors')[:]
                abs_perc_errors = fin.get('abs_perc_errors')[:]


            assert not np.isnan(abs_errors).any(), 'abs_errors contains NaNs'
            assert not np.isnan(abs_perc_errors).any(), 'abs_perc_errors contains NaNs'

            # Remove events containing inf 
            if np.isinf(abs_perc_errors).any():
                abs_perc_errors = abs_perc_errors[~np.any(np.isinf(abs_perc_errors), axis=1), :]
                print(infile, ': removed infs, new shape =', abs_perc_errors.shape)

            mae = np.mean(abs_errors, axis=0)
            std_mae = np.std(abs_errors, axis=0)
            mape = np.mean(abs_perc_errors, axis=0)
            std_mape = np.std(abs_perc_errors, axis=0)
            rmse = np.sqrt(np.mean(np.power(abs_errors, 2), axis=0))

            if xvals is None:
                xvals = np.arange(mae.shape[0])

            #ax_mae.plot(xvals, mae, label=title, linewidth=2)
            #ax_mae.fill_between(xvals, mae - std_mae, mae + std_mae, alpha=0.1)

            ax_mape.plot(xvals, mape, label=title, linewidth=2)
            ax_mape_wide.plot(xvals, mape, label=title, linewidth=2)
            #ax_mape.fill_between(xvals, mape - std_mape, mape + std_mape, alpha=0.1)
            #ax_rmse.plot(xvals, rmse, label=title, linewidth=2)

            # Compute area under curve
            endpts = [5, 10, 15, 25]
            auc_str = ''
            for endpt in endpts:
                auc = np.trapz(mape[:endpt+1])
                auc_str += 'AUC-{} = {:.2f}  '.format(endpt, auc)

            print('{:25}: {}'.format(title, auc_str))


        def set_axes_info(ax, ylabel):
            ax.set_xlim(xvals[0], xvals[-1])
            ax.set_xlabel('Iterations')
            ax.set_ylabel(ylabel)
            ax.legend(loc='lower right')

        #set_axes_info(ax_mae, 'Mean absolute error')
        set_axes_info(ax_mape, 'Mean absolute percentage error')
        set_axes_info(ax_mape_wide, 'Mean absolute percentage error')
        #set_axes_info(ax_rmse, 'RMSE')

        fig_mape.tight_layout()
        fig_mape_wide.tight_layout()


        if args.save:
            outfile = Path(args.output_dir) / f'heapmap_comparison_{args.observable}'
            fig_mape.savefig(outfile.__str__() + '.png')
            fig_mape.savefig(outfile.__str__() + '.pdf')
            fig_mape_wide.savefig(outfile.__str__() + '_wide.png')
            fig_mape_wide.savefig(outfile.__str__() + '_wide.pdf')

        plt.show()





