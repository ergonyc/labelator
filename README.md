# labelator
Simple framework for transfering labels to scRNAseq dataset from our favorite scRNAseq atlas.

NOTE: XGBoost variants are currently depricated.  

### overview.
We call this tool the "labelator".  The purpose of a "labelator" is to easily classify _cell types_ for out-of-sample __"Test"__ or __"Query"__ data. 

Our general approach will be to "compress" the raw count data, and generate probability of each label category.  We will do this in two ways: 

1) **naive** mode.  Or making no assumptions or attempts to account for confounding variables like "batch", "noise" (e.g. doublets, mt/rb contamination), or "library_size".   

2) **transfer** mode.  i.e. `scarches` or `scvi-tools`.  Basically, we will need to _fit_ these confounding variables for the out-of-sample data using the `scarches` _surgery_ approach.



----------------
## some details 
### dataloaders
One of the crucial decisions is how to load our scRNAseq into a pytorch model.  We prefer the `AnnData` "annotated data" format.  Several implementations of such a dataloader are available: from `scvi-tools`, `scarches`, and `anndata` itself.  The `scvi-tools` is the most complex, but we have started here to enable leveraging the `scvi-tools`.  To state our confirmation bias, we like `scvi` models so are starting here.

We will validate potential models and calibrate them with simple expectations using a typical "Train"/"Validate" and "Test"/"Probe" approach.  


### Data Definitions:
- "Train": data samples on which the model being tested is trained.  The `torch lightning` framework used by `scvi-tools` semi-automatically will "validate" to test out-of-sample prediction fidelity during training.
- "Test": held-out samples to test the fidelity of the model.  
- "Query": data generated externally,which is _probing_ the fidelity of the model to general scRNAseq data.

-----------------
### scVI data modeling specifics:
All models will be trained on the n=3000 most highly variable genes from Xylena's scRNAseq data.  

----------------
### Models:

We'll can **labelate** either a single **end-to-end** way or in two steps. 

#### 2 step: encode + categorize
In two steps:
1) _encode_: __Embed__ the scRNAseq counts into dimensinally reduced representation: 
    - a latent sub-space of a variational Model e.g.
        - scVI (a Variational Auto Encoder, VAE model)
        - scANVI (a conditional VAE which also predicts "cell_type")
    - linear embedding. i.e.
        - PCA (_naive_ linear encoding)
    
2) _categorize_: predicting creating a probability of a each category 
    - ~~Linear classifier (e.g. multinomial Logistic Regression)~~ (not implimented)
    - MLP (multi layer perceptron) non-linear classifier 
    - boosted trees (XGboost)


#### end-to-end
We will also try some _end-to-end_ approaches for comparision.  In these models a single model takes us from raw counts to category probabilities.  

- __naive__ inference
    - ~~boosted trees (e.g. xgboost)~~
    - MLP classifier

- __transfer__ learning
    - scVI/scANVI scarches "surgery"



----------------
### training & validation
Models will be trained on the "train" set from xylena's "clean" data.   Validation on a subset of the training data will ensure that overfitting is not a problem.  

Extending the `scarches` models and training classes seems to be the most straightforward.  The `scvi-tools` employs `ligntening` which _may_ be good to leverage eventually, but is overbuilt for the current state.

- scVI
    - batch/sample/depth params vs. neutered
- scANVI
    - 



## Caveats
There are several gotchas to anticipate:
- features.  Currently we are locked into the 3k genes we are testing with.  Handling subsets and supersets is TBC.
- batch.  In principle each "embedding" or decode part of the model should be able to measure a "batch-correction" parameter explicitly.  In `scVI` this is explicitly _learned_.  However in _naive_ inference mode it should just be an inferred fudge factor.
- noise.  including or _not_ including `doublet`, `mito`, or `ribo` metrics



----------------
### future changes

The _training_ and _query_ should be split up.   Different cli's for each modality.   The _query_ cli can be levraged for both the validation  __"Test"__ **and** any eadditional __"Query"_ probes.

_training_ might be split into **prep** and **train**, with **prep** creating the base `scvi` _vae_ models for the other model flavors to use.  Note that the `scanvi` variants require both _'batch_eq'_ and non batch corrected variants.


wrinkles:
- the `scvi` family of 

## SEA-AD 
model_args = {
    "n_layers": 2,
    "n_latent": 20,
    "dispersion": "gene-label"
}


## Data Preparation
It is assumed that any testing or query data will be reasonably QC-ed.  However during _dataprep_ the 'Train', 'Test', and other _query_ datasets need to be composed such that:

1. there are *not* any empty cells.   e.g. `sc.pp.filter_cells(adata, min_genes=1)`  Note that this is only sometimes the case after subsetting to _highly variable genes_.
2. PCs for the data are computed so that query data for `scvi_emb_pcs` and `count_pcs` models can be projected onto these vectors.
3. adata input objects contain raw counts in the .X field.
