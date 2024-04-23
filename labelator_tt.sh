#!/bin/bash

# Function to run the Python CLI with given parameters
train_test_model() {
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
        echo "#  🏁 🏁 🏁 Test (Query) Model $model_name completed in $((end_time - start_time)) seconds. 🏁 🏁 🏁 "
        echo "## 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 🏁 "


    done
}
repr_model_names=("scvi_emb" "scvi_expr" "scvi_expr_pcs")
count_model_names=("pcs_lbl8r" "raw_lbl8r")
transfer_model_names=("scanvi_batch_eq" "scanvi" )

## 2k
train_data="data/scdata/xylena2k/xyl2_train.h5ad"
query_data="data/scdata/xylena2k/xyl2_test.h5ad"
adata_output_path='data/scdata/xylena2k/LABELATOR/'
artifacts_path='artifacts2k/'

# Call the function 
# model_path='models2k/REPR/scvi'  
# train_test_model $train_data $query_data repr_model_names[@] $model_path $adata_output_path $artifacts_path

model_path='models2k/CNT'  
train_test_model $train_data $query_data count_model_names[@] $model_path $adata_output_path $artifacts_path

model_path='models2k/TRANSFER/'  
train_test_model $train_data $query_data transfer_model_names[@] $model_path $adata_output_path $artifacts_path

## 3k
train_data="data/scdata/xylena3k/xyl2_train.h5ad"
query_data="data/scdata/xylena3k/xyl2_test.h5ad"
adata_output_path='data/scdata/xylena3k/LABELATOR/'
artifacts_path='artifacts3k/'

# Call the function 
model_path='models3k/REPR/scvi'  
train_test_model $train_data $query_data repr_model_names[@] $model_path $adata_output_path $artifacts_path

model_path='models3k/CNT'  
train_test_model $train_data $query_data count_model_names[@] $model_path $adata_output_path $artifacts_path

model_path='models3k/TRANSFER/'  
train_test_model $train_data $query_data transfer_model_names[@] $model_path $adata_output_path $artifacts_path



## 5k
repr_model_names=( "scvi_expr" "scvi_expr_pcs")

train_data="data/scdata/xylena5k/xyl2_train.h5ad"
query_data="data/scdata/xylena5k/xyl2_test.h5ad"
adata_output_path='data/scdata/xylena5k/LABELATOR/'
artifacts_path='artifacts5k/'

# Call the function 
model_path='models5k/REPR/scvi'  
train_test_model $train_data $query_data repr_model_names[@] $model_path $adata_output_path $artifacts_path

model_path='models5k/CNT'  
train_test_model $train_data $query_data count_model_names[@] $model_path $adata_output_path $artifacts_path

model_path='models5k/TRANSFER/'  
train_test_model $train_data $query_data transfer_model_names[@] $model_path $adata_output_path $artifacts_path

# # 10 k
# repr_model_names=("scvi_expr" "scvi_expr_pcs")

# train_data="data/scdata/xylena10k/xyl2_train.h5ad"
# query_data="data/scdata/xylena10k/xyl2_test.h5ad"
# adata_output_path='data/scdata/xylena10k/LABELATOR/'
# artifacts_path='artifacts10k/'

# # Call the function 
# model_path='models10k/REPR/scvi'  
# train_test_model $train_data $query_data repr_model_names[@] $model_path $adata_output_path $artifacts_path

# model_path='models10k/CNT'  
# train_test_model $train_data $query_data count_model_names[@] $model_path $adata_output_path $artifacts_path

# model_path='models10k/TRANSFER/'  
# train_test_model $train_data $query_data transfer_model_names[@] $model_path $adata_output_path $artifacts_path


## 15k
# repr_model_names=("scvi_expr" "scvi_expr_pcs")

# train_data="data/scdata/xylena15k/xyl2_train.h5ad"
# query_data="data/scdata/xylena15k/xyl2_test.h5ad"
# adata_output_path='data/scdata/xylena15k/LABELATOR/'
# artifacts_path='artifacts15k/'

# # Call the function 
# model_path='models15k/REPR/scvi'  
# train_test_model $train_data $query_data repr_model_names[@] $model_path $adata_output_path $artifacts_path

# model_path='models15k/CNT'  
# train_test_model $train_data $query_data count_model_names[@] $model_path $adata_output_path $artifacts_path

# model_path='models15k/TRANSFER/'  
# train_test_model $train_data $query_data transfer_model_names[@] $model_path $adata_output_path $artifacts_path



# ## 20k
# train_data="data/scdata/xylena20k/xyl2_train.h5ad"
# query_data="data/scdata/xylena20k/xyl2_test.h5ad"
# adata_output_path='data/scdata/xylena20k/LABELATOR/'
# artifacts_path='artifacts20k/'

# # Call the function 
# model_path='models20k/REPR/scvi'  
# train_test_model $train_data $query_data repr_model_names[@] $model_path $adata_output_path $artifacts_path

# model_path='models20k/CNT'  
# train_test_model $train_data $query_data count_model_names[@] $model_path $adata_output_path $artifacts_path

# model_path='models20k/TRANSFER/'  
# train_test_model $train_data $query_data transfer_model_names[@] $model_path $adata_output_path $artifacts_path

