#!/bin/bash

# Function to run the Python CLI with given parameters
run_model() {
    local train_adata=$1
    local query_adata=$2
    local model_names=("${!3}")
    local model_path=$4
    local output_data_path=$5
    local artifacts_path=$6

    for model_name in "${model_names[@]}"
    do
    
        # echo "🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 "
        echo "########################################################################"
        echo "🚀 🚀 🚀 🚀 Running model: $model_name 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀"
        echo "## ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬ ⏬"
        # Start timing
        start_time=$(date +%s)

        python -m labelator_api \
            --train-path $train_adata \
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

train_data="data/scdata/xylena5k/xyl2_train.h5ad"
query_data="data/scdata/xylena5k/xyl2_test.h5ad"
adata_output_path='data/scdata/xylena5k/LABELATOR/'
# query_adata=$'/media/ergonyc/data/sc/ASAP/artifacts/07_merged_filtered_processed_integrated_clustered_annotated_anndata_object.h5ad'
# query_adata=$'/media/ergonyc/data/sc/ASAP/artifacts/07_merged_filtered_integrated_clustered_annotated_anndata_object.h5ad'
# artifacts_path='artifacts5k/'

# # Call the function 
# model_path='models5k/REPR/scvi'  
# # repr_model_names=("scvi_emb" "scvi_expr" "scvi_expr_pcs" "scvi_emb_xgb" "scvi_expr_xgb" "scvi_expr_pcs_xgb")
# repr_model_names=("scvi_emb" "scvi_expr" "scvi_expr_pcs")
# run_model $train_data $query_data repr_model_names[@] $model_path $adata_output_path $artifacts_path

# # model_path='models5k/CNT'  
# # count_model_names=("pcs_lbl8r" "raw_lbl8r" "raw_xgb" "pcs_xgb" )
# count_model_names=("pcs_lbl8r" "raw_lbl8r")
# count_model_names=("pcs_lbl8r")
# run_model $train_data $query_data count_model_names[@] $model_path $adata_output_path $artifacts_path

# model_path='models5k/TRANSFER/'  
# transfer_model_names=("scanvi_batch_eq" "scanvi" )
# # run_model $train_data $query_data transfer_model_names[@] $model_path $adata_output_path $artifacts_path


query_data="data/scdata/xylena5k/xyl2_query.h5ad"
train_data=""
# Call the function 
model_path='models5k/REPR/scvi'  
repr_model_names=("scvi_emb" "scvi_expr" "scvi_expr_pcs")
run_model $train_data $query_data repr_model_names[@] $model_path $adata_output_path $artifacts_path

# model_path='models5k/CNT'  
# count_model_names=("pcs_lbl8r" "raw_lbl8r")
# count_model_names=("pcs_lbl8r")
# run_model $train_data $query_data count_model_names[@] $model_path $adata_output_path $artifacts_path

# model_path='models5k/TRANSFER/'  
# transfer_model_names=("scanvi_batch_eq" "scanvi" )
# # run_model $train_data $query_data transfer_model_names[@] $model_path $adata_output_path $artifacts_path

################

train_data="data/scdata/xylena10k/xyl2_train.h5ad"
query_data="data/scdata/xylena10k/xyl2_test.h5ad"
adata_output_path='data/scdata/xylena10k/LABELATOR/'
# query_adata=$'/media/ergonyc/data/sc/ASAP/artifacts/07_merged_filtered_processed_integrated_clustered_annotated_anndata_object.h5ad'
# query_adata=$'/media/ergonyc/data/sc/ASAP/artifacts/07_merged_filtered_integrated_clustered_annotated_anndata_object.h5ad'
artifacts_path='artifacts10k/'

# Call the function 
model_path='models10k/REPR/scvi'  
# repr_model_names=("scvi_emb" "scvi_expr" "scvi_expr_pcs" "scvi_emb_xgb" "scvi_expr_xgb" "scvi_expr_pcs_xgb")
repr_model_names=("scvi_emb" "scvi_expr" "scvi_expr_pcs")
run_model $train_data $query_data repr_model_names[@] $model_path $adata_output_path $artifacts_path

# model_path='models10k/CNT'  
# # count_model_names=("pcs_lbl8r" "raw_lbl8r" "raw_xgb" "pcs_xgb" )
# count_model_names=("pcs_lbl8r" "raw_lbl8r")
# count_model_names=("pcs_lbl8r")
# run_model $train_data $query_data count_model_names[@] $model_path $adata_output_path $artifacts_path

# model_path='models10k/TRANSFER/'  
# transfer_model_names=("scanvi_batch_eq" "scanvi" )
# # run_model $train_data $query_data transfer_model_names[@] $model_path $adata_output_path $artifacts_path


query_data="data/scdata/xylena10k/xyl2_query.h5ad"
train_data=""
# Call the function 
model_path='models10k/REPR/scvi'  
repr_model_names=("scvi_emb" "scvi_expr" "scvi_expr_pcs")
run_model $train_data $query_data repr_model_names[@] $model_path $adata_output_path $artifacts_path

# model_path='models10k/CNT'  
# count_model_names=("pcs_lbl8r" "raw_lbl8r")
# count_model_names=("pcs_lbl8r")
# run_model $train_data $query_data count_model_names[@] $model_path $adata_output_path $artifacts_path

# model_path='models10k/TRANSFER/'  
# transfer_model_names=("scanvi_batch_eq" "scanvi" )
# # run_model $train_data $query_data transfer_model_names[@] $model_path $adata_output_path $artifacts_path


