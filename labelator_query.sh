#!/bin/bash

# Function to run the Python CLI with given parameters
query_model() {
    local query_adata=$1
    local model_names=("${!2}")
    local model_path=$3
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

        python -m labelator_api \
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
repr_model_names=("scvi_emb" "scvi_expr" "scvi_expr_pcs")
count_model_names=("pcs_lbl8r" "raw_lbl8r")
transfer_model_names=("scanvi_batch_eq" "scanvi" )


## 1k
adata_output_path='data/scdata/xylena1k/LABELATOR/'
artifacts_path='artifacts1k/'
query_data="data/scdata/xylena1k/xyl2_query.h5ad"

model_path='models1k/REPR/scvi'  
query_model $query_data repr_model_names[@] $model_path $adata_output_path $artifacts_path

model_path='models1k/CNT'  
query_model $query_data count_model_names[@] $model_path $adata_output_path $artifacts_path

model_path='models1k/TRANSFER/'  
query_model $query_data transfer_model_names[@] $model_path $adata_output_path $artifacts_path


## 3k
adata_output_path='data/scdata/xylena3k/LABELATOR/'
artifacts_path='artifacts3k/'
query_data="data/scdata/xylena3k/xyl2_query.h5ad"

model_path='models3k/REPR/scvi'  
query_model $query_data repr_model_names[@] $model_path $adata_output_path $artifacts_path

model_path='models3k/CNT'  
query_model $query_data count_model_names[@] $model_path $adata_output_path $artifacts_path

model_path='models3k/TRANSFER/'  
query_model $query_data transfer_model_names[@] $model_path $adata_output_path $artifacts_path


# 5k 
adata_output_path='data/scdata/xylena5k/LABELATOR/'
artifacts_path='artifacts5k/'
query_data="data/scdata/xylena5k/xyl2_query.h5ad"

model_path='models5k/REPR/scvi'  
query_model $query_data repr_model_names[@] $model_path $adata_output_path $artifacts_path

model_path='models5k/CNT'  
query_model $query_data count_model_names[@] $model_path $adata_output_path $artifacts_path

model_path='models5k/TRANSFER/'  
query_model $query_data transfer_model_names[@] $model_path $adata_output_path $artifacts_path

# 10k
adata_output_path='data/scdata/xylena10k/LABELATOR/'
artifacts_path='artifacts10k/'
query_data="data/scdata/xylena10k/xyl2_query.h5ad"

model_path='models10k/REPR/scvi'  
query_model $query_data repr_model_names[@] $model_path $adata_output_path $artifacts_path

model_path='models10k/CNT'  
query_model $query_data count_model_names[@] $model_path $adata_output_path $artifacts_path

model_path='models10k/TRANSFER/'  
query_model $query_data transfer_model_names[@] $model_path $adata_output_path $artifacts_path

# 20k
adata_output_path='data/scdata/xylena20k/LABELATOR/'
artifacts_path='artifacts20k/'
query_data="data/scdata/xylena20k/xyl2_query.h5ad"

model_path='models20k/REPR/scvi'  
query_model $query_data repr_model_names[@] $model_path $adata_output_path $artifacts_path

model_path='models20k/CNT'  
query_model $query_data count_model_names[@] $model_path $adata_output_path $artifacts_path

model_path='models20k/TRANSFER/'  
query_model $query_data transfer_model_names[@] $model_path $adata_output_path $artifacts_path


