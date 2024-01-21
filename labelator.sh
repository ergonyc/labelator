#!/bin/bash

# Function to run the Python CLI with given parameters
run_model() {
    local model_names=("${!1}")
    local model_path=$2
    local output_data_path=$3

    for model_name in "${model_names[@]}"
    do
    

        echo " 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 "
        echo "🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 "
        echo "########################################################################"
        echo "Running model: $model_name"
        echo "########################################################################"
        echo "⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬"

        # Start timing
        start_time=$(date +%s)

        python -m labelator_api \
            --data-path data/scdata/xylena/brain_atlas_anndata_train_cnt.h5ad \
            --query-path data/scdata/xylena/brain_atlas_anndata_test_cnt.h5ad \
            --model-path $model_path \
            --model-name $model_name \
            --output-data-path $output_data_path \
            --artifacts-path artifacts/ \
            --gen-plots \
            # --retrain-model

        # Check if the Python call was successful
        
        if [ $? -ne 0 ]; then
            echo "🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 "
            echo "Error: Model $model_name failed to run."
            echo "🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 "
        fi

        # End timing
        end_time=$(date +%s)

        echo "⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫"
        echo "########################################################################"
        echo "#### Model $model_name completed in $((end_time - start_time)) seconds."
        echo "########################################################################"
        echo "🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 "
        echo "                                                                        "


    done
}

# Define the array of LBL8R and E2E model names
lbl8r_model_names=("lbl8r_scvi_emb" "lbl8r_raw_cnt_pcs" "lbl8r_scvi_expr_pcs" "xgb_raw_cnt_pcs" "xgb_scvi_expr_pcs" "xgb_scvi_emb")
e2e_model_names=("lbl8r_raw_cnt" "lbl8r_scvi_expr" "scanvi_batch_eq" "scanvi" "xgb_raw_cnt" "xgb_scvi_expr")


# Call the function with LBL8R and E2E model names
run_model lbl8r_model_names[@] "models/LBL8R/" "data/scdata/xylena/LBL8R/"
run_model e2e_model_names[@] "models/LBL8R/" "data/scdata/xylena/LBL8R/"

