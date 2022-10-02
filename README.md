# ecg-heatmapping-review



## Usage

Convert median data and info CSV files to HDF5:
```
python convert_csv_to_hdf5.py
```

Train StevenNet model:
```
python train_medians.py -t qt -o stevennet_take5
# or
sbatch train_model.sh
```

Compute model performance metrics
```
python evaluate_model.py
# or
sbatch evluate_model_metrics.sh
```

Get a list of events from test dataset with complete metadata
```
python events_for_plotting.py
```

Create a heatmap
```
python heatmaps.py --observable qt --method saliency --model 1 --event_index 0-10
```

Create all heatmaps for all methods and observables
```
sbatch create_heatmap_plots.sh
```

Run region perturbation procedure for objective evaluation of heatmaps
```
python compare_heatmaps.py compute_errors --observable qt --num_iterations 300 --event_limit 1000 --output_dir error_analysis
```

Run region perturbation for all observables
```
sbatch compare_heatmaps.py
```

Plot results from above
```
python compare_heatmaps.py plot_comparison --observable qt --input_dir error_analysis --output_dir plots
```
