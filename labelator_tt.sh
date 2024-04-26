#!/bin/bash

# Function to run the Python CLI with given parameters
train_test_model() {
    local train_adata=$1
    local model_path=$2
    local model_names=("${!3}")
    local output_data_path=$4
    local artifacts_path=$5

    for model_name in "${model_names[@]}"
    do

            # echo "🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 "
            echo "########################################################################"
            echo "🚀 🚀 🚀 🚀 Running model: $model_name 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀"
            echo "## ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬"
            # Start timing
            start_time=$(date +%s)

            python -m train_labelator \
                --train-path $train_adata \
                --model-path $model_path \
                --model-name $model_name \
                --output-data-path $output_data_path \
                --artifacts-path $artifacts_path \
                --gen-plots \
                --labels-key "cell_type" \
                --retrain-model
            
            if [ $? -ne 0 ]; then
                echo "🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 "
                echo "🚨 🚨 🚨 🚨 🚨 Error: Train $model_name failed to run. 🚨 🚨 🚨 🚨 🚨"
                echo "🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 "
            fi

            # End timing
            end_time=$(date +%s)

            echo "##### ⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫  #############"
            echo "#  🏁 🏁 🏁 Train  Model $model_name completed in $((end_time - start_time)) seconds. 🏁 🏁 🏁 "
            echo "## 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 "
    

    done
}
count_model_names=("raw" "pcs")
scvi_model_names=("scvi_emb" "scvi_expr" "scvi_expr_pcs" "scanvi")
# set_names=("10k" "5k" "3k" "2k" "1k")
set_names=("1k" "2k" "3k" "5k" "10k")
# model_types=("count" "naive" "batch_eq")
model_types=("naive" "count" "batch_eq")

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

        train_data="data/scdata/xylena/${set_name}/xyl2_train.h5ad"
        # query_data="data/scdata/xylena/${set_name}/xyl2_test.h5ad"
        adata_output_path="data/scdata/xylena/${set_name}/LABELATOR/${model_type}/"
        artifacts_path="artifacts${set_name}/${model_type}/"

        # Call the function 
        models_path="models/${set_name}/${model_type}/" 
        train_test_model $train_data $models_path model_list[@] $adata_output_path $artifacts_path

        # echo "##### $set_name $model_type $models_path "
        
        # for model in "${model_list[@]}"
        # do
        #     echo "🚀 🚀 🚀 🚀 .... $model "
        # done

        # echo "#####                        $adata_output_path $artifacts_path"


    done

done

