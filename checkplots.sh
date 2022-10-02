#! /bin/bash


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

path=$1

for method in ${method_names[@]}; do
    for obs in ${observables[@]}; do
        for i in {0..50}; do
            fname1=${path}/${obs}-${method}_eventindex_${i}.pdf
            fname2=${path}/${obs}-${method}_eventindex_${i}_average.pdf
            if [ ! -f $fname1 ]; then
                echo "Missing: ${fname1}"
            fi
            if [ ! -f $fname2 ]; then
                echo "Missing: ${fname2}"
            fi
        done
    done
done
