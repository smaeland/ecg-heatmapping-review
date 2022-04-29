from datetime import datetime
from pathlib import Path
from argparse import ArgumentParser
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import torch
from captum.attr import (
    LayerAttribution,
    IntegratedGradients,
    Saliency, 
    DeepLift,
    InputXGradient,
    GuidedBackprop,
    GuidedGradCam,
    LayerGradCam,
    Deconvolution,
    LRP,
    NoiseTunnel,
    GradientShap
)
from captum.attr._utils.lrp_rules import Alpha1_Beta0_Rule, EpsilonRule, GammaRule, IdentityRule
import h5py
from models import StevenNet, StandardConvNet
from hdf5dataset import Hdf5Dataset, Hdf5DatasetFromJson
from visualise import visualise_attr_1d, _normalize_attr
from evaluate_attributions import RandomiseTopAttrs, RandomiseByValue, RandomiseIteratively





def set_lrp_rules(model, rule='epsilon'):

    # Composite rules
    if rule == 'composite':

        # Lower layers: gamma rule
        for _module in [model.init_block, model.block1, model.block2, model.block3]:
            for _layer in _module.children():
                if isinstance(_layer, torch.nn.Conv1d):# or isinstance(_layer, torch.nn.BatchNorm1d):
                    _layer.rule = GammaRule()
        
        # Middle layers: epsilon rule
        for _module in [model.block4, model.block5, model.block6, model.block7]:
            for _layer in _module.children():
                if isinstance(_layer, torch.nn.Conv1d):# or isinstance(_layer, torch.nn.BatchNorm1d):
                    _layer.rule = EpsilonRule(epsilon=1e-7)
        
        # Top layers: zero rule
        for _module in [model.block8, model.fc1]:
            for _layer in _module.children():
                if isinstance(_layer, torch.nn.Conv1d):# or isinstance(_layer, torch.nn.BatchNorm1d):
                    _layer.rule = EpsilonRule(epsilon=0)
        
        # BatchNorm layers
        for _module in model.children():
            for _layer in _module.children():
                if isinstance(_layer, torch.nn.BatchNorm1d):
                    _layer.rule = IdentityRule()
        
        # Pooling and flatten
        model.avgpool.rule = EpsilonRule(epsilon=0)
        model.flatten.rule = EpsilonRule(epsilon=0)


    # Epsilon rule everywhere
    elif rule == 'epsilon':

        for _module in model.children():
            for _layer in _module.children():
                #if isinstance(_layer, torch.nn.Conv1d) or isinstance(_layer, torch.nn.BatchNorm1d):
                    #_layer.rule = EpsilonRule(epsilon=0)
                if isinstance(_layer, torch.nn.Conv1d):
                    _layer.rule = EpsilonRule(epsilon=1.0e-7)
                if isinstance(_layer, torch.nn.BatchNorm1d):
                    _layer.rule = IdentityRule()
                if isinstance(_layer, torch.nn.AdaptiveAvgPool1d):
                    _layer.rule = EpsilonRule(epsilon=0)
        
        model.avgpool.rule = EpsilonRule(epsilon=0)
        model.flatten.rule = EpsilonRule(epsilon=0)
        model.fc1.rule = EpsilonRule()



class RandomAttribution:
    """
    Class that assigns attributions randomly
    """

    def __init__(self):
        
        self.name = 'random'


    def attribute(self, inputs, target):

        return torch.randn(size=inputs.shape)




class CaptumMethodWrapper:
    """
    Wrap the Captum methods to store all methods and their respective settings
    in a single place
    """

    def __init__(
        self,
        method_name: str,
        model,
        verbose=True,
    ) -> None:

        # Model under investigation
        self.model = model
        # Instance of Captum attribution method
        self.method = None
        # kwargs passed to method.attribute()
        self.attribution_kwargs = {}
        # kwags passed to visualise_attr_1d()
        self.visualisation_kwargs = {}

        self.verbose = verbose


        if method_name == 'saliency':
            self.method = Saliency(model)
            self.attribution_kwargs = {'abs': False}
            self.visualisation_kwargs = {'sign': 'all'}
            #self.visualisation_kwargs = {'sign': 'absolute_value'}

        elif method_name == 'smoothgrad':
            self.method = NoiseTunnel(
                Saliency(model)
            )
            self.attribution_kwargs = {
                'nt_type': 'smoothgrad',
                'nt_samples': 25,
                'stdevs': None  # to be set based on data values
            }
            self.visualisation_kwargs = {'sign': 'all'}
        
        elif method_name == 'gradcam':
            self.method = LayerGradCam(model, layer=model.block8.conv2)
            self.attribution_kwargs = {'relu_attributions': True}
            self.visualisation_kwargs = {'sign': 'positive'}

        elif method_name == 'guided_gradcam':
            self.method = GuidedGradCam(model, layer=model.block8.conv2)
            self.attribution_kwargs = {
                'interpolate_mode': 'linear'
            }
            self.visualisation_kwargs = {'sign': 'all'}

        elif method_name == 'deconvolution':
            self.method = Deconvolution(model)
            self.visualisation_kwargs = {'sign': 'all'}

        elif method_name == 'deeplift':
            self.method = DeepLift(model, multiply_by_inputs=False)
            self.attribution_kwargs = {'baselines': None}
            self.visualisation_kwargs = {'sign': 'all'}

        elif method_name == 'inputxgradient':
            self.method = InputXGradient(model)
            self.visualisation_kwargs = {'sign': 'all'}
        
        elif method_name == 'integrated_gradients':
            self.method = IntegratedGradients(model, multiply_by_inputs=False)
            self.attribution_kwargs = {'n_steps': 50}
            self.visualisation_kwargs = {'sign': 'all'}

        elif method_name == 'guided_backprop':
            self.method = GuidedBackprop(model)
            self.visualisation_kwargs = {'sign': 'all'}
        
        elif method_name == 'gradientshap':
            self.method = GradientShap(model, multiply_by_inputs=False)
            self.attribution_kwargs = {
                'n_samples': 50,
                'stdevs': None,
                'baselines': torch.zeros(size=(1, 12, 600))
            }
            self.visualisation_kwargs = {'sign': 'all'}
        
        elif method_name == 'lrp-epsilon':
            set_lrp_rules(model, 'epsilon')
            self.method = LRP(model)
            self.visualisation_kwargs = {'sign': 'all'}

        elif method_name == 'lrp-composite':
            set_lrp_rules(model, 'composite')
            self.method = LRP(model)
            self.visualisation_kwargs = {'sign': 'all'}

        elif method_name == 'random':
            self.method = RandomAttribution()
            self.visualisation_kwargs = {'sign': 'all'}
        
        else:
            raise ValueError(f'No such method: {method_name}')
    

    def predict(self, x, numpy=True):
        """
        Call the model
        """

        _x = x
        if isinstance(x, np.ndarray):
            _x = torch.Tensor(x)

        # Add batch dimension, if not present
        if len(_x.shape) == 2:
            _x = torch.unsqueeze(_x, axis=0)

        # Predict and return desired type
        pred = self.model(_x).squeeze().detach()
        if numpy:
            pred = pred.numpy()

        assert len(pred.shape) == 0, f'Expected single model output, got shape {pred.shape}'
        
        return pred.tolist()


    def attribute(self, x, y):
        """ 
        Compute attributions
        """

        _x = x
        if isinstance(x, np.ndarray):
            _x = torch.Tensor(x)

        # Add batch dimension, if not present
        if len(_x.shape) == 2:
            _x = torch.unsqueeze(_x, axis=0)

        assert _x.shape[0] == 1 and len(_x.shape) == 3, f'Expected input of shape (1, channels, length), got {_x.shape}'

        # Get predictions (without batch dim)
        pred = self.predict(_x)

        # Remove batch dim from y, if present
        if len(y.shape) == 0:
            pass
        elif len(y.shape) == 1 and len(y) == 1:
            y = y[0]
        else:
            raise TypeError(f'Unexpected y length/shape: {y.shape}')

        if self.verbose:
            print('true = {:.1f}, pred = {:.1f}'.format(y, pred))

        # Get noise scale from data, if needed
        if isinstance(self.method, NoiseTunnel) or isinstance(self.method, GradientShap):
            noise_scale = 0.02 * (torch.max(_x) - torch.min(_x))
            self.attribution_kwargs['stdevs'] = float(noise_scale)
            if self.verbose:
                print('smoothgrad noise_scale =', noise_scale)

        # Compute attributions
        attrs = self.method.attribute(_x, target=0, **self.attribution_kwargs)

        #print('dbg: min(attrs) = {}, max(attrs) = {}, mean(attrs) = {}'.format(
        #    torch.min(attrs),
        #    torch.max(attrs),
        #    torch.mean(attrs)
        #))

        # Upscale if needed
        input_length = _x.shape[-1]
        if attrs.shape[-1] != input_length:
            attrs = LayerAttribution.interpolate(
                attrs,
                input_length,
                interpolate_mode='linear'
            )

        # Duplicate channels if needed
        input_num_channels = _x.shape[1]
        if attrs.shape[1] != input_num_channels:
            attrs = torch.tile(attrs, (input_num_channels, 1))
        
        # Remove batch dim, convert to numpy
        attrs = torch.squeeze(attrs, axis=0)
        attrs = attrs.cpu().detach().numpy()

        return attrs


def process_event_index_from_args(arg: List[str]):

    indices = []

    for a in arg:

        # 1,2,3
        if ',' in a:
            indices = a.split(',')
            indices = list(map(int, indices))

        # 1-3
        elif '-' in a:
            pos = a.find('-')
            start, stop = a[:pos], a[pos+1:]
            start, stop = int(start), int(stop)
            indices = list(range(start, stop+1))

        # 1 2 3 
        else:
            indices.append(int(a))

    return indices


model_savefile_mapping1 = {
    'VentRate': 'model_VentRate_stevennet_2022-03-02.pt',
    'qt': 'model_qt_stevennet_2022-02-25.pt',
    'pr': 'model_pr_stevennet_2022-03-02.pt',
    'qrs': 'model_qrs_stevennet_2022-03-02.pt',
    'STJ_v5': 'model_STJ_v5_stevennet_2022-03-02.pt',
    'T_PeakAmpl_v5': 'model_T_PeakAmpl_v5_stevennet_2022-03-02.pt',
    'R_PeakAmpl_v5': 'model_R_PeakAmpl_v5_stevennet_2022-03-02.pt',
}

model_savefile_mapping2 = {
    'VentRate': 'model_VentRate_stevennet_take2_2022-04-22.pt',
    'qt': 'model_qt_stevennet_take2_2022-04-08.pt',
    'pr': 'model_pr_stevennet_take2_2022-04-22.pt',
    'qrs': 'model_qrs_stevennet_take2_2022-04-22.pt',
    'STJ_v5': 'model_STJ_v5_stevennet_take2_2022-04-22.pt',
    'T_PeakAmpl_v5': 'model_T_PeakAmpl_v5_stevennet_take2_2022-04-22.pt',
    'R_PeakAmpl_v5': 'model_R_PeakAmpl_v5_stevennet_take2_2022-04-22.pt',
}

model_savefile_mapping3 = {
    'VentRate': 'model_VentRate_stevennet_take3_2022-04-25.pt',
    'qt': 'model_qt_stevennet_take3_2022-04-25.pt',
    'pr': 'model_pr_stevennet_take3_2022-04-25.pt',
    'qrs': 'model_qrs_stevennet_take3_2022-04-25.pt',
    'STJ_v5': 'model_STJ_v5_stevennet_take3_2022-04-25.pt',
    'T_PeakAmpl_v5': 'model_T_PeakAmpl_v5_stevennet_take3_2022-04-25.pt',
    'R_PeakAmpl_v5': 'model_R_PeakAmpl_v5_stevennet_take3_2022-04-25.pt',
}

model_savefile_mapping4 = {
    'VentRate': 'model_VentRate_stevennet_take4_2022-04-26.pt',
    'qt': 'model_qt_stevennet_take4_2022-04-26.pt',
    'pr': 'model_pr_stevennet_take4_2022-04-26.pt',
    'qrs': 'model_qrs_stevennet_take4_2022-04-26.pt',
    'STJ_v5': 'model_STJ_v5_stevennet_take4_2022-04-26.pt',
    'T_PeakAmpl_v5': 'model_T_PeakAmpl_v5_stevennet_take4_2022-04-26.pt',
    'R_PeakAmpl_v5': 'model_R_PeakAmpl_v5_stevennet_take4_2022-04-26.pt',
}

model_savefile_mapping5 = {
    'VentRate': '',
    'qt': 'model_qt_stevennet_take5_2022-04-29.pt',
    'pr': '',
    'qrs': '',
    'STJ_v5': '',
    'T_PeakAmpl_v5': '',
    'R_PeakAmpl_v5': '',
}


if __name__ == '__main__': 

    attr_methods = [
        'saliency',
        'deeplift',
        'smoothgrad',
        'inputxgradient',
        'integrated_gradients',
        'deconvolution',
        'gradcam',
        'guided_gradcam',
        'guided_backprop',
        'lrp-epsilon',
        'lrp-composite',
        'gradientshap',
        'random',
    ]

    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    parser = ArgumentParser()
    parser.add_argument('-o', '--observable', choices=model_savefile_mapping1.keys())
    parser.add_argument('-m', '--method', choices=attr_methods)
    parser.add_argument('-mo', '--model', type=int, choices=[1, 2, 3, 4, 5])
    parser.add_argument('-p', '--no_show', action='store_false')
    parser.add_argument('-s', '--save', action='store_true')
    parser.add_argument('-ei', '--event_index', nargs='+')
    parser.add_argument('-mc', '--merge_channels', choices=['none', 'average', 'v5_only'], default='none')
    parser.add_argument('-od', '--output_dir', type=str, default='plots')
    parser.add_argument('-pc', '--precheck_files', action='store_true')
    args = parser.parse_args()

    print('args:', args)

    event_indices = process_event_index_from_args(args.event_index)

    if args.model == 1:
        model_savefile_mapping = model_savefile_mapping1
    elif args.model == 2:
        model_savefile_mapping = model_savefile_mapping2
    elif args.model == 3:
        model_savefile_mapping = model_savefile_mapping3
    elif args.model == 4:
        model_savefile_mapping = model_savefile_mapping4
    elif args.model == 5:
        model_savefile_mapping = model_savefile_mapping5
    print('Using model:', model_savefile_mapping[args.observable])

    model = StevenNet()
    model.load_state_dict(torch.load('models/' + model_savefile_mapping[args.observable], map_location=device))
    model.eval()

    dataset = Hdf5DatasetFromJson(
        'good_events_for_plotting.json',
        target_name=args.observable
    )


    method = CaptumMethodWrapper(
        args.method,
        model
    )

    for i in event_indices:

        x, y = dataset[i]
        attributions = method.attribute(x, y)

        outnames = None
        
        if args.save:

            outdir = Path(args.output_dir)
            if not outdir.is_dir():
                outdir.mkdir()

            savename = (
                args.observable + '-' + 
                args.method +
                '_eventindex_' + str(i)
            )
            if args.merge_channels != 'none':
                savename += ('_' + args.merge_channels)
            formats = ['.png', '.pdf']

            outnames = []
            for frmt in formats:
                outnames.append(
                    outdir / (savename + frmt)
                )
            print('Saving plot as', outnames)
        
        visualise_attr_1d(
            attributions,
            x, 
            merge_channels=args.merge_channels,
            savefile=outnames,
            show=args.no_show,
            **method.visualisation_kwargs,    
        )
        

        

