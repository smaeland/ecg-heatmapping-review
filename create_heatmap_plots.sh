#!/bin/bash
#SBATCH -J 4attr-plot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=16:00:00
####SBATCH --gres=gpu:1
####SBATCH -p a100q 
#SBATCH -p xeongold16q

set -e

echo "Loading modules"
module use /cm/shared/ex3-modules/latest/modulefiles
module load slurm/20.02.7
module load pytorch-py37-cuda11.2-gcc8/1.9.1
source venv/bin/activate

echo "python version:"
python --version

# For a100q 
if [ ! -f /usr/lib/x86_64-linux-gnu/libevent_core-2.1.so.6 ]; then
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`/lib
fi

declare -a method_names=(
    "saliency"
    "deeplift"
    "smoothgrad"
    "inputxgradient"
    "integrated_gradients"
    "deconvolution"
    "guided_gradcam"
    "guided_backprop"
    "gradientshap"
)
declare -a observables=(
    "qt"
    "pr"
    "qrs"
    "STJ_v5"
    "T_PeakAmpl_v5"
    "R_PeakAmpl_v5"
    "VentRate"
)
event_indices="0-50"
model="1"
output_dir="plots/heatmaps/apr29/model${model}"
merge_channels="none"

for method in "${method_names[@]}"; do
    for obs in "${observables[@]}"; do
        cmnd="srun python heatmaps.py \
        --observable ${obs} \
        --method ${method} \
        --model ${model} \
        --no_show \
        --save \
        --merge_channels ${merge_channels} \
        --event_index ${event_indices} \
        --output_dir ${output_dir} \
        --precheck_files
        "
        echo "${cmnd}"
        ${cmnd}
    done
done
