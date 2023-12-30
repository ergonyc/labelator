from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import sparse

from ._utils import label_encoder, remove_sparsity

# copied from scarches TRVAE... 
class AnnotatedDataset(Dataset):
    """Dataset handler for TRVAE model and trainer.
       Parameters
       ----------
       adata: : `~anndata.AnnData`
            Annotated data matrix.
       condition_key: String
            column name of conditions in `adata.obs` data frame.
       condition_encoder: Dict
            dictionary of encoded conditions.
       cell_type_keys: List
            List of column names of different celltype hierarchies in `adata.obs` data frame.
       cell_type_encoder: Dict
            dictionary of encoded celltypes.
    """
    def __init__(self,
                 adata,
                 condition_key=None,
                 condition_encoder=None,
                 cell_type_keys=None,
                 cell_type_encoder=None,
                 labeled_array=None,
                 ):

        self.condition_key = condition_key
        self.condition_encoder = condition_encoder
        self.cell_type_keys = cell_type_keys
        self.cell_type_encoder = cell_type_encoder

        self._is_sparse = sparse.issparse(adata.X)

        # # JAH: hack in a data_key to allow for choosing X = adata.obsm['X_pca'],  or adata.X
        # #    
        # if x_loader is not None:
        #     self.data = x_loader(adata)
        #     size_factors = 1
        # else:
        self.data = adata.X if self._is_sparse else torch.tensor(adata.X)
        size_factors = np.ravel(adata.X.sum(1))


        self.size_factors = torch.tensor(size_factors)

        labeled_array = np.zeros((len(adata), 1)) if labeled_array is None else labeled_array
        self.labeled_vector = torch.tensor(labeled_array)

        # Encode condition strings to integer
        if self.condition_key is not None:
            self.conditions = label_encoder(
                adata,
                encoder=self.condition_encoder,
                condition_key=condition_key,
            )
            self.conditions = torch.tensor(self.conditions, dtype=torch.long)

        # Encode cell type strings to integer
        if self.cell_type_keys is not None:
            self.cell_types = list()
            for cell_type_key in cell_type_keys:
                level_cell_types = label_encoder(
                    adata,
                    encoder=self.cell_type_encoder,
                    condition_key=cell_type_key,
                )
                self.cell_types.append(level_cell_types)

            self.cell_types = np.stack(self.cell_types).T
            self.cell_types = torch.tensor(self.cell_types, dtype=torch.long)

    def __getitem__(self, index):
        outputs = dict()

        if self._is_sparse:
            x = torch.tensor(np.squeeze(self.data[index].toarray()))
        else:
            x = self.data[index]
        outputs["x"] = x

        outputs["labeled"] = self.labeled_vector[index]
        outputs["sizefactor"] = self.size_factors[index]

        if self.condition_key:
            outputs["batch"] = self.conditions[index]

        if self.cell_type_keys:
            outputs["celltypes"] = self.cell_types[index, :]

        return outputs

    def __len__(self):
        return self.data.shape[0]

    @property
    def condition_label_encoder(self) -> dict:
        return self.condition_encoder

    @condition_label_encoder.setter
    def condition_label_encoder(self, value: dict):
        if value is not None:
            self.condition_encoder = value

    @property
    def cell_type_label_encoder(self) -> dict:
        return self.cell_type_encoder

    @cell_type_label_encoder.setter
    def cell_type_label_encoder(self, value: dict):
        if value is not None:
            self.cell_type_encoder = value

    @property
    def stratifier_weights(self):
        conditions = self.conditions.detach().cpu().numpy()
        condition_coeff = 1. / len(conditions)

        condition2count = Counter(conditions)
        counts = np.array([condition2count[cond] for cond in conditions])
        weights = condition_coeff / counts
        return weights.astype(float)


# probably don't need this... let the model / trainer handle moving things around.


class LatentDataset(Dataset):
    """Dataset handler for fitting counts to latent space 
       Parameters
       ----------
       adata: : `~anndata.AnnData`
            Annotated data matrix.
       condition_key: String
            column name of conditions in `adata.obs` data frame.
       condition_encoder: Dict
            dictionary of encoded conditions.
       cell_type_keys: List
            List of column names of different celltype hierarchies in `adata.obs` data frame.
       cell_type_encoder: Dict
            dictionary of encoded celltypes.
    """
    def __init__(self,
                 adata,
                 condition_key=None,
                 condition_encoder=None,
                 cell_type_keys=None,
                 cell_type_encoder=None,
                 labeled_array=None,
                 ):

        self.condition_key = condition_key
        self.condition_encoder = condition_encoder
        self.cell_type_keys = cell_type_keys
        self.cell_type_encoder = cell_type_encoder

        self._is_sparse = sparse.issparse(adata.X)

        # # JAH: hack in a data_key to allow for choosing X = adata.obsm['X_pca'],  or adata.X
        # #    
        # if x_loader is not None:
        #     self.data = x_loader(adata)
        #     size_factors = 1
        # else:
        self.data = adata.X if self._is_sparse else torch.tensor(adata.X)
        size_factors = np.ravel(adata.X.sum(1))


        self.size_factors = torch.tensor(size_factors)

        labeled_array = np.zeros((len(adata), 1)) if labeled_array is None else labeled_array
        self.labeled_vector = torch.tensor(labeled_array)

        # Encode condition strings to integer
        if self.condition_key is not None:
            self.conditions = label_encoder(
                adata,
                encoder=self.condition_encoder,
                condition_key=condition_key,
            )
            self.conditions = torch.tensor(self.conditions, dtype=torch.long)

        # Encode cell type strings to integer
        if self.cell_type_keys is not None:
            self.cell_types = list()
            for cell_type_key in cell_type_keys:
                level_cell_types = label_encoder(
                    adata,
                    encoder=self.cell_type_encoder,
                    condition_key=cell_type_key,
                )
                self.cell_types.append(level_cell_types)

            self.cell_types = np.stack(self.cell_types).T
            self.cell_types = torch.tensor(self.cell_types, dtype=torch.long)

    def __getitem__(self, index):
        outputs = dict()

        if self._is_sparse:
            x = torch.tensor(np.squeeze(self.data[index].toarray()))
        else:
            x = self.data[index]
        outputs["x"] = x

        outputs["labeled"] = self.labeled_vector[index]
        outputs["sizefactor"] = self.size_factors[index]

        if self.condition_key:
            outputs["batch"] = self.conditions[index]

        if self.cell_type_keys:
            outputs["celltypes"] = self.cell_types[index, :]

        return outputs

    def __len__(self):
        return self.data.shape[0]

    @property
    def condition_label_encoder(self) -> dict:
        return self.condition_encoder

    @condition_label_encoder.setter
    def condition_label_encoder(self, value: dict):
        if value is not None:
            self.condition_encoder = value

    @property
    def cell_type_label_encoder(self) -> dict:
        return self.cell_type_encoder

    @cell_type_label_encoder.setter
    def cell_type_label_encoder(self, value: dict):
        if value is not None:
            self.cell_type_encoder = value

    @property
    def stratifier_weights(self):
        conditions = self.conditions.detach().cpu().numpy()
        condition_coeff = 1. / len(conditions)

        condition2count = Counter(conditions)
        counts = np.array([condition2count[cond] for cond in conditions])
        weights = condition_coeff / counts
        return weights.astype(float)




class LatentClassifierDataset(Dataset):
    """Dataset handler for fitting latents space classifier 
       Parameters
       ----------
       adata: : `~anndata.AnnData`
            Annotated data matrix.
       condition_key: String
            column name of conditions in `adata.obs` data frame.
       condition_encoder: Dict
            dictionary of encoded conditions.
       cell_type_keys: List
            List of column names of different celltype hierarchies in `adata.obs` data frame.
       cell_type_encoder: Dict
            dictionary of encoded celltypes.
    """
    def __init__(self,
                 adata,
                 condition_key=None,
                 condition_encoder=None,
                 cell_type_keys=None,
                 cell_type_encoder=None,
                 labeled_array=None,
                 ):

        self.condition_key = condition_key
        self.condition_encoder = condition_encoder
        self.cell_type_keys = cell_type_keys
        self.cell_type_encoder = cell_type_encoder

        self._is_sparse = sparse.issparse(adata.X)

        # # JAH: hack in a data_key to allow for choosing X = adata.obsm['X_pca'],  or adata.X
        # #    
        # if x_loader is not None:
        #     self.data = x_loader(adata)
        #     size_factors = 1
        # else:
        self.data = adata.X if self._is_sparse else torch.tensor(adata.X)
        size_factors = np.ravel(adata.X.sum(1))


        self.size_factors = torch.tensor(size_factors)

        labeled_array = np.zeros((len(adata), 1)) if labeled_array is None else labeled_array
        self.labeled_vector = torch.tensor(labeled_array)

        # Encode condition strings to integer
        if self.condition_key is not None:
            self.conditions = label_encoder(
                adata,
                encoder=self.condition_encoder,
                condition_key=condition_key,
            )
            self.conditions = torch.tensor(self.conditions, dtype=torch.long)

        # Encode cell type strings to integer
        if self.cell_type_keys is not None:
            self.cell_types = list()
            for cell_type_key in cell_type_keys:
                level_cell_types = label_encoder(
                    adata,
                    encoder=self.cell_type_encoder,
                    condition_key=cell_type_key,
                )
                self.cell_types.append(level_cell_types)

            self.cell_types = np.stack(self.cell_types).T
            self.cell_types = torch.tensor(self.cell_types, dtype=torch.long)

    def __getitem__(self, index):
        outputs = dict()

        if self._is_sparse:
            x = torch.tensor(np.squeeze(self.data[index].toarray()))
        else:
            x = self.data[index]
        outputs["x"] = x

        outputs["labeled"] = self.labeled_vector[index]
        outputs["sizefactor"] = self.size_factors[index]

        if self.condition_key:
            outputs["batch"] = self.conditions[index]

        if self.cell_type_keys:
            outputs["celltypes"] = self.cell_types[index, :]

        return outputs

    def __len__(self):
        return self.data.shape[0]

    @property
    def condition_label_encoder(self) -> dict:
        return self.condition_encoder

    @condition_label_encoder.setter
    def condition_label_encoder(self, value: dict):
        if value is not None:
            self.condition_encoder = value

    @property
    def cell_type_label_encoder(self) -> dict:
        return self.cell_type_encoder

    @cell_type_label_encoder.setter
    def cell_type_label_encoder(self, value: dict):
        if value is not None:
            self.cell_type_encoder = value

    @property
    def stratifier_weights(self):
        conditions = self.conditions.detach().cpu().numpy()
        condition_coeff = 1. / len(conditions)

        condition2count = Counter(conditions)
        counts = np.array([condition2count[cond] for cond in conditions])
        weights = condition_coeff / counts
        return weights.astype(float)

