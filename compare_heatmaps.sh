#!/bin/bash
#SBATCH -J compare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --array=0-70%10
#SBATCH -p xeongold16q

####SBATCH --gres=gpu:1
####SBATCH -p a100q 

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
    "random"
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
num_iterations=200
output_dir="error_analysis/apr08"

# Run all under a single job:
#for method in "${method_names[@]}"; do
#    for obs in "${observables[@]}"; do
#        cmnd="srun python compare_heatmaps.py \
#        compute_errors \
#        ${method} \
#        --observable ${obs} \
#        --num_iterations ${num_iterations} \
#        --output_dir ${output_dir}
#        "
#        echo "\n${cmnd}\n"
#        starttime=${SECONDS}
#        ${cmnd}
#        duration=$(($SECONDS - $starttime))
#        echo "Duration: $(($duration / 60)) m $(($duration % 60)) s"
#    done
#done


# Submit job array 
declare -a methods_arr
declare -a observables_arr 
for method in "${method_names[@]}"; do
    for obs in "${observables[@]}"; do
        methods_arr+=($method) 
        observables_arr+=($obs) 
    done
done

cmnd="srun python compare_heatmaps.py \
compute_errors \
${methods_arr[$SLURM_ARRAY_TASK_ID]} \
--observable ${observables_arr[$SLURM_ARRAY_TASK_ID]} \
--num_iterations ${num_iterations} \
--output_dir ${output_dir} \
--event_limit 5000
"

echo ""
echo "${cmnd}"
${cmnd}
