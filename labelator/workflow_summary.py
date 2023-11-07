


# current workflow


# step1 = PREPROCESS . preprocess.qdl / preprocess.R
#  1a. read in raw counts
#  1b. seurate object
#  1c. add percent.mt and percent.rb to metadata.(ambient rna?)   
#  1d.  scrublet doublet detection
#  1e.  add to metadata: dataset, batch, porject, batch_id


step1 = "raw_counts -> Seurat object + metadata"
"~{raw_data_path}/~{sample_id}.seurat_object.preprocessed_01.rds"

# step 2: quality control. cohort_analysis.wdl/run_quality_control.wdl


#  2a. filter and normalize. 
#  "filter.R" -> "~{raw_data_path}/~{seurat_object_basename}_filtered_02.rds"
# 
# "process.R" ->  "~{raw_data_path}/~{seurat_object_basename}_filtered_normalized_03.rds"
#    NormalizeData(), CleanVarGenes(nHVG=3000, use.sct=True), ScaleData( all.genes ), CellCycleScoring(s.genes, g2m.genes)
step2 = "subset " to pct mt,rb <50% and nGene > 500 nCell > 500?)

# step 3: cluster / aggregate:  cohort_analysis.wdl/cluster_data.wdl
#  "harmony.R" -> "~{raw_data_path}/~{cohort_id}.seurat_object.harmony_integrated_04.rds"
#  "find_neighbors.R" -> "~{raw_data_path}/~{integrated_seurat_object_basename}_neighbors_05.rds"
#  "umap.R" -> "~{raw_data_path}/~{integrated_seurat_object_basename}_neighbors_umap_06.rds"
#  "clustering.R" -> "~{raw_data_path}/~{integrated_seurat_object_basename}_neighbors_umap_cluster_07.rds"

step3 = "cluster / aggregate"




# scanpy version.