# labelator
Simple framework for transfering labels to scRNAseq dataset from our favorite scRNAseq atlas.



### overview.
We call this tool the "labelator".  The purpose of a "labelator" is to easily classify _cell types_ for out-of-sample __"Test"__ or __"Probe"__ data. 

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
- "Probe": data generated externally,which is _probing_ the fidelity of the model to general scRNAseq data.

-----------------
### scVI data modeling specifics:
All models will be trained on the n=3000 most highly variable genes from Xylena's scRNAseq data.  `scVI` parameters are enumerated below.

```python 
continuous_covariate_keys = None #noise = ['doublet_score', 'percent.mt', 'percent.rb'] # aka "noise"
layer = "counts"
batch = "sample" #'batch'
categorical_covariate_keys = None #['sample', 'batch'] Currently limited to single categorical...
labels = 'cell_type'
size_factor_key = None # library size 
```
----------------
### Models:

We'll can **labelate** either a single **end-to-end** way or in two steps. 

#### 2 step: encode + categorize
In two steps:
1) _encode_: __Embed__ the scRNAseq counts into dimensinally reduced representation: 
    - a latent sub-space of a variational Model e.g.
        - VAE (e.g. MMD-VAE, infoVAE etc)
        - scVI
        - scANVI 
    - linear embedding. i.e.
        - PCA (_naive_ linear encoding)
    
2) _categorize_: predicting creating a probability of a each category 
    - Linear classifier (e.g. multinomial Logistic Regression)
    - MLP (multi layer perceptron) non-linear classifier 
    - boosted trees (XGboost)

We will use a variety of models to "embed" the scRNAseq counts into lower dimension.
- scVI latents

#### end-to-end
We will also try some _end-to-end_ approaches for comparision.  In these models a single model takes us from raw counts to category probabilities.  

- __naive__ inference
    - boosted trees (e.g. xgboost)
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

FUTURE:
- _naive_ batch correction


 Z Classifier -> y_hat


>scVI: encoder, latent_i, latent_batch_i, latent_library_i, (x_hat = vae(x))

>scANVI: encoder, latent_i, latent_batch_i, latent_library_i, (x_hat = vae(x))


----------------
### inference :: _testing_ :: _probing_
Two types of "inference" modes will be considered.  
1) batch corrected (scVI/scANVI) which requires transfer learning on the probe data
2) naive, which simply processes the examples

------------
### metrics
How ought we capture the fidelity of the __labelator__?  F1, precision, recall, accuracy?
- pct accuracy
- ?




## Caveats
There are several gotchas to anticipate:
- features.  Currently we are locked into the 3k genes we are testing with.  Handling subsets and supersets is TBC.
- batch.  In principle each "embedding" or decode part of the model should be able to measure a "batch-correction" parameter explicitly.  In `scVI` this is explicitly _learned_.  However in _naive_ inference mode it should just be an inferred fudge factor.
- noise.  including or _not_ including `doublet`, `mito`, or `ribo` metrics




### List of models
target:  8 cell types including "unknown"  also have 

1. `scVI` / `scANVI`
    - train end-to-end with `scarches` surgery for transfer learining
    - as encoder for `LBL8R` embeddings
    - as normalizer of raw counts to normalized gene _expression_ values

2. `LBL8R`
    - end-to-end
        - raw counts
        - `scVI` expression
    - encode + classify
        - `scVI` latent embedding
        - TBD _other_ VAE embeddings

3. PCA Classification, `pcaLBL8R`
    a. raw count PCA
        - directly classify from PC loadings
    b. scVI "normalized" gene expression PCA
        - directly classify from PC loadings

4. XGBoost
    a. end-to-end classificaiton
        - raw counts
        - `scVI` expression
    b. classificatin of PCA _"embedding"_
        - of raw counts
        - of `scVI` expression
    c. classification 
        - `scVI` embedding
        - TBD _other_ VAE embeddings

