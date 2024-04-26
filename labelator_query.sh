#!/bin/bash

# Function to run the Python CLI with given parameters
query_model() {
    local query_adata=$1
    local model_path=$2
    local model_names=("${!3}")
    local output_data_path=$4
    local artifacts_path=$5


    for model_name in "${model_names[@]}"
    do
    
        # echo "🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 "
        echo "########################################################################"
        echo "🚀 🚀 🚀 🚀 Querying model: $model_name 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀"
        echo "## ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬"
        # Start timing
        start_time=$(date +%s)

        python -m query_labelator \
            --query-path $query_adata \
            --model-path $model_path \
            --model-name $model_name \
            --output-data-path $output_data_path \
            --artifacts-path $artifacts_path \
            --gen-plots \
            --labels-key "cell_type" 
            # --retrain-model
        
        if [ $? -ne 0 ]; then
            echo "🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 "
            echo "🚨 🚨 🚨 🚨 🚨 Error: Query Model $model_name failed to run. 🚨 🚨 🚨 🚨 🚨"
            echo "🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 "
        fi

        # End timing
        end_time=$(date +%s)

        echo "##### ⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫  #############"
        echo "#  🏁 🏁 🏁  Model $model_name completed in $((end_time - start_time)) seconds. 🏁 🏁 🏁 "
        echo "## 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 "


    done
}


repr_model_names=("scvi_emb" "scvi_expr" "scvi_expr_pcs")
count_model_names=("pcs_lbl8r" "raw_lbl8r")
transfer_model_names=("scanvi_batch_eq" "scanvi" )

set_names = ("10k" "5k" "3k" "2k" "1k")
model_types = ("count" "naive" "batch_eq")

# TEST 
for set_name in "${set_names[@]}"
do
    for model_type in "${model_types[@]}"
    do
        query_data="data/scdata/xylena/${set_name}/xyl2_test.h5ad"
        adata_output_path="data/scdata/xylena/${set_name}/LABELATOR/${model_type}/"
        artifacts_path="artifacts${set_name}/${model_type}/"

        # Call the function 
        model_path="models${set_name}/${model_type}/" 
        query_model $query_data $model_path repr_model_names[@] $adata_output_path $artifacts_path

    done

done


# QUERY 
for set_name in "${set_names[@]}"
do
    for model_type in "${model_types[@]}"
    do
        query_data="data/scdata/xylena/${set_name}/xyl2_query.h5ad"
        adata_output_path="data/scdata/xylena/${set_name}/LABELATOR/${model_type}/"
        artifacts_path="artifacts${set_name}/${model_type}/"

        # Call the function 
        model_path="models${set_name}/${model_type}/" 
        query_model $query_data $model_path repr_model_names[@] $adata_output_path $artifacts_path

    done

done


