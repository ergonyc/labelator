#!/bin/bash

# Function to run the Python CLI with given parameters
run_model() {
    local model_names=("${!1}")
    local model_path=$2
    local output_data_path=$3

    for model_name in "${model_names[@]}"
    do
    

        # echo "🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 "

        echo "########################################################################"
        echo "🚀 🚀 🚀 🚀 Running model: $model_name 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀"
        echo "## ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬"

        # Start timing
        start_time=$(date +%s)



        # python -m labelator_api \
        #     --query-path data/scdata/xylena/brain_atlas_anndata_test_cnt.h5ad \
        #     --model-path $model_path \
        #     --model-name $model_name \
        #     --output-data-path $output_data_path \
        #     --artifacts-path artifacts/ \
        #     --gen-plots \
        #     # --retrain-model


        python -m labelator_api \
            --data-path data/scdata/xylena/brain_atlas_anndata_train_cnt.h5ad \
            --query-path data/scdata/xylena/brain_atlas_anndata_test_cnt.h5ad \
            --model-path $model_path \
            --model-name $model_name \
            --output-data-path $output_data_path \
            --artifacts-path artifacts/ \
            --gen-plots 
            # --retrain-model

        # Check if the Python call was successful
        
        if [ $? -ne 0 ]; then
            echo "🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 "
            echo "🚨 🚨 🚨 🚨 🚨 Error: Model $model_name failed to run. 🚨 🚨 🚨 🚨 🚨"
            echo "🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 "
        fi

        # End timing
        end_time=$(date +%s)

        echo "##### ⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫  #############"
        echo "#  🏁 🏁 🏁  Model $model_name completed in $((end_time - start_time)) seconds. 🏁 🏁 🏁 "
        echo "## 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 "



    done
}

# Define the array of model names
# repr_model_names=("scvi_emb" "scvi_expr" "scvi_expr_pcs" "scvi_emb_xgb" "scvi_expr_xgb" "scvi_expr_pcs_xgb")
# repr_model_names=("scvi_expr_xgb")
count_model_names=("pcs_lbl8r" "raw_lbl8r" "raw_xgb" "pcs_xgb" )
transfer_model_names=("scanvi_batch_eq" "scanvi" )

# Call the function 
# run_model repr_model_names[@] "models/REPR/scvi" "data/scdata/xylena/LABELATOR/"
# run_model count_model_names[@] "models/CNT/" "data/scdata/xylena/LABELATOR/"
run_model transfer_model_names[@] "models/TRANSFER/" "data/scdata/xylena/LABELATOR/"

