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
            --labels-key "cell_type" \
            --retrain-model
        
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
        echo "##         "
        echo "##         "
        echo "##         "


    done
}

count_model_names=("raw" "pcs")
scvi_model_names=("scvi_emb" "scvi_expr" "scvi_expr_pcs" "scanvi")

# set_names=("1k" "2k" "3k" "5k" "10k")
set_names=("1k" "2k" "3k" "5k")
model_types=("naive" "count" "batch_eq")

queries=("xyl2_test" "xyl2_query")
 
queries=("xyl2_test")

for query in "${queries[@]}"
do

    for set_name in "${set_names[@]}"
    do
        for model_type in "${model_types[@]}"
        do

            if [ $model_type == "count" ]; then
                model_list=("${count_model_names[@]}")
            elif [ $model_type == "naive" ]; then
                model_list=("${scvi_model_names[@]}")
            elif [ $model_type == "batch_eq" ]; then
                model_list=("${scvi_model_names[@]}")
            fi

            query_data="scdata/xylena/${set_name}/${query}.h5ad"
            adata_output_path="scdata/xylena/${set_name}/LABELATOR/${model_type}/"
            artifacts_path="artifacts/${set_name}/${model_type}/"

            # Call the function 
            models_path="models/${set_name}/${model_type}/" 

            echo "🚀 🚀 🚀 🚀 Querying $query data for $set_name $model_type 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀"

            query_model $query_data $models_path model_list[@] $adata_output_path $artifacts_path

        done

    done

done
