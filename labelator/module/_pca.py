import torch
import torch.nn as nn
import torch.nn.functional as F

from scvi.nn import FCLayers

def svd_flip(u, v):
    # columns of u, rows of v
    max_abs_cols = torch.argmax(torch.abs(u), 0)
    i = torch.arange(u.shape[1]).to(u.device)
    signs = torch.sign(u[max_abs_cols, i])
    u *= signs
    v *= signs.view(-1, 1)
    return u, v

class PCA(nn.Module):
    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components

    @torch.no_grad()
    def fit(self, X):
        n, d = X.size()
        if self.n_components is not None:
            d = min(self.n_components, d)
        self.register_buffer("mean_", X.mean(0, keepdim=True))
        Z = X - self.mean_ # center
        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        Vt = Vh
        U, Vt = svd_flip(U, Vt)
        self.register_buffer("components_", Vt[:d])
        return self

    def forward(self, X):
        return self.transform(X)

    def transform(self, X):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(X - self.mean_, self.components_.t())

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Y):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(Y, self.components_) + self.mean_



class PCALoading(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X, pcs):
        return X @ self.pcs



class PCAClassifier(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_labels: int = 5,
        n_hidden: int = 128,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        logits: bool = False,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        activation_fn: nn.Module = nn.ReLU,
        **kwargs,
    ):
        super().__init__()
        self.logits = logits
        layers = []

        if n_hidden > 0 and n_layers > 0:
            layers.append(
                FCLayers(
                    n_in=n_input,
                    n_out=n_hidden,
                    n_layers=n_layers,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_rate,
                    use_batch_norm=use_batch_norm,
                    use_layer_norm=use_layer_norm,
                    activation_fn=activation_fn,
                    **kwargs,
                )
            )
        else:
            n_hidden = n_input

        layers.append(nn.Linear(n_hidden, n_labels))

        if not logits:
            layers.append(nn.Softmax(dim=-1))

        self.classifier = nn.Sequential(*layers)
        self.encoder = PCALoading()

    def forward(self, x, pcs):
        """Forward computation."""
        return self.classifier(self.encoder(x, pcs))





if __name__ == "__main__":
    import numpy as np
    from sklearn.decomposition import PCA as sklearn_PCA
    from sklearn import datasets
    iris = torch.tensor(datasets.load_iris().data)
    _iris = iris.numpy()
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    for device in devices:
        iris = iris.to(device)
        for n_components in (2, 4, None):
            _pca = sklearn_PCA(n_components=n_components).fit(_iris)
            _components = torch.tensor(_pca.components_)
            pca = PCA(n_components=n_components).to(device).fit(iris)
            components = pca.components_
            assert torch.allclose(components, _components.to(device))
            _t = torch.tensor(_pca.transform(_iris))
            t = pca.transform(iris)
            assert torch.allclose(t, _t.to(device))
        __iris = pca.inverse_transform(t)
        assert torch.allclose(__iris, iris)
    print("passed!")