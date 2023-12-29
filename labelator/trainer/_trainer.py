import torch
import torch.nn as nn
from collections import defaultdict
import numpy as np
import time
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

from ..utils._monitor import EarlyStopping
from ._utils import make_dataset, custom_collate, print_progress

from torch import optim

# copy from scarches.  adding device to model
class Trainer:
    """ScArches base Trainer class. This class contains the implementation of the base CVAE/TRVAE Trainer.

       Parameters
       ----------
       model: trVAE
            Number of input features (i.e. gene in case of scRNA-seq).
       adata: : `~anndata.AnnData`
            Annotated data matrix. Has to be count data for 'nb' and 'zinb' loss and normalized log transformed data
            for 'mse' loss.
       condition_key: String
            column name of conditions in `adata.obs` data frame.
       cell_type_keys: List
            List of column names of different celltype levels in `adata.obs` data frame.
       batch_size: Integer
            Defines the batch size that is used during each Iteration
       alpha_epoch_anneal: Integer or None
            If not 'None', the KL Loss scaling factor (alpha_kl) will be annealed from 0 to 1 every epoch until the input
            integer is reached.
       alpha_kl: Float
            Multiplies the KL divergence part of the loss.
       alpha_iter_anneal: Integer or None
            If not 'None', the KL Loss scaling factor will be annealed from 0 to 1 every iteration until the input
            integer is reached.
       use_early_stopping: Boolean
            If 'True' the EarlyStopping class is being used for training to prevent overfitting.
       reload_best: Boolean
            If 'True' the best state of the model during training concerning the early stopping criterion is reloaded
            at the end of training.
       early_stopping_kwargs: Dict
            Passes custom Earlystopping parameters.
       train_frac: Float
            Defines the fraction of data that is used for training and data that is used for validation.
       n_samples: Integer or None
            Defines how many samples are being used during each epoch. This should only be used if hardware resources
            are limited.
       use_stratified_sampling: Boolean
            If 'True', the sampler tries to load equally distributed batches concerning the conditions in every
            iteration.
       monitor: Boolean
            If `True', the progress of the training will be printed after each epoch.
       monitor_only_val: Boolean
            If `True', only the progress of the validation datset is displayed.
       clip_value: Float
            If the value is greater than 0, all gradients with an higher value will be clipped during training.
       weight decay: Float
            Defines the scaling factor for weight decay in the Adam optimizer.
       n_workers: Integer
            Passes the 'n_workers' parameter for the torch.utils.data.DataLoader class.
       seed: Integer
            Define a specific random seed to get reproducable results.
    """
    def __init__(self,
                 model,
                 adata,
                 condition_key: str = None,
                 cell_type_keys: str = None,
                 batch_size: int = 128,
                 alpha_epoch_anneal: int = None,
                 alpha_kl: float = 1.,
                 use_early_stopping: bool = True,
                 reload_best: bool = True,
                 early_stopping_kwargs: dict = None,
                 **kwargs):

        self.adata = adata
        self.model = model
        self.condition_key = condition_key
        self.cell_type_keys = cell_type_keys

        self.batch_size = batch_size
        self.alpha_epoch_anneal = alpha_epoch_anneal
        self.alpha_iter_anneal = kwargs.pop("alpha_iter_anneal", None)
        self.use_early_stopping = use_early_stopping
        self.reload_best = reload_best

        self.alpha_kl = alpha_kl

        early_stopping_kwargs = (early_stopping_kwargs if early_stopping_kwargs else dict())

        self.n_samples = kwargs.pop("n_samples", None)
        self.train_frac = kwargs.pop("train_frac", 0.9)
        self.use_stratified_sampling = kwargs.pop("use_stratified_sampling", True)

        self.weight_decay = kwargs.pop("weight_decay", 0.04)
        self.clip_value = kwargs.pop("clip_value", 0.0)

        self.n_workers = kwargs.pop("n_workers", 0)
        self.seed = kwargs.pop("seed", 2020)
        self.monitor = kwargs.pop("monitor", True)
        self.monitor_only_val = kwargs.pop("monitor_only_val", True)
        ## JAH: TODO: add beta parameter to "overregularize" the latents... 

        self.early_stopping = EarlyStopping(**early_stopping_kwargs)

        torch.manual_seed(self.seed)

        ## JAH: device hack.  this should be improved, but really isn't nescessary until its a true inference tool (and training is moot)
        ## JAH: add device
        device = kwargs.pop("device", "cpu")
        torch.manual_seed(self.seed)

        if device == "cuda":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            torch.cuda.manual_seed(self.seed)
        elif device == "mps":
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            torch.set_default_dtype(torch.float32)
        elif device is None:
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = "cpu" 
    
        self.model.to(self.device)
        
        self.epoch = -1
        self.n_epochs = None
        self.iter = 0
        self.best_epoch = None
        self.best_state_dict = None
        self.current_loss = None
        self.previous_loss_was_nan = False
        self.nan_counter = 0
        self.optimizer = None
        self.training_time = 0

        self.train_data = None
        self.valid_data = None
        self.sampler = None
        self.dataloader_train = None
        self.dataloader_valid = None

        self.iters_per_epoch = None
        self.val_iters_per_epoch = None

        self.logs = defaultdict(list)

        # Create Train/Valid AnnotatetDataset objects
        self.train_data, self.valid_data = make_dataset(
            self.adata,
            train_frac=self.train_frac,
            condition_key=self.condition_key,
            cell_type_keys=self.cell_type_keys,
            condition_encoder=self.model.condition_encoder,
            cell_type_encoder=self.model.cell_type_encoder,
        )

    def initialize_loaders(self):
        """
        Initializes Train-/Test Data and Dataloaders with custom_collate and WeightedRandomSampler for Trainloader.
        Returns:

        """
        if self.n_samples is None or self.n_samples > len(self.train_data):
            self.n_samples = len(self.train_data)
        self.iters_per_epoch = int(np.ceil(self.n_samples / self.batch_size))

        if self.use_stratified_sampling:
            # Create Sampler and Dataloaders
            stratifier_weights = torch.tensor(self.train_data.stratifier_weights, device=self.device)

            self.sampler = WeightedRandomSampler(stratifier_weights,
                                                 num_samples=self.n_samples,
                                                 replacement=True)
            self.dataloader_train = torch.utils.data.DataLoader(dataset=self.train_data,
                                                                batch_size=self.batch_size,
                                                                sampler=self.sampler,
                                                                collate_fn=custom_collate,
                                                                num_workers=self.n_workers)
        else:
            self.dataloader_train = torch.utils.data.DataLoader(dataset=self.train_data,
                                                                batch_size=self.batch_size,
                                                                shuffle=True,
                                                                collate_fn=custom_collate,
                                                                num_workers=self.n_workers)
        if self.valid_data is not None:
            val_batch_size = self.batch_size
            if self.batch_size > len(self.valid_data):
                val_batch_size = len(self.valid_data)
            self.val_iters_per_epoch = int(np.ceil(len(self.valid_data) / self.batch_size))
            self.dataloader_valid = torch.utils.data.DataLoader(dataset=self.valid_data,
                                                                batch_size=val_batch_size,
                                                                shuffle=True,
                                                                collate_fn=custom_collate,
                                                                num_workers=self.n_workers)

    def calc_alpha_coeff(self):
        """Calculates current alpha coefficient for alpha annealing.

           Parameters
           ----------

           Returns
           -------
           Current annealed alpha value
        """
        if self.alpha_epoch_anneal is not None:
            alpha_coeff = min(self.alpha_kl * self.epoch / self.alpha_epoch_anneal, self.alpha_kl)
        elif self.alpha_iter_anneal is not None:
            alpha_coeff = min((self.alpha_kl * (self.epoch * self.iters_per_epoch + self.iter) / self.alpha_iter_anneal), self.alpha_kl)
        else:
            alpha_coeff = self.alpha_kl
        return alpha_coeff

    def train(self,
              n_epochs=400,
              lr=1e-3,
              eps=0.01):

        self.initialize_loaders()
        begin = time.time()
        self.model.train()
        self.n_epochs = n_epochs

        params = filter(lambda p: p.requires_grad, self.model.parameters())

        self.optimizer = torch.optim.Adam(params, lr=lr, eps=eps, weight_decay=self.weight_decay)

        self.before_loop()

        for self.epoch in range(n_epochs):
            self.on_epoch_begin(lr, eps)
            self.iter_logs = defaultdict(list)
            for self.iter, batch_data in enumerate(self.dataloader_train):
                for key, batch in batch_data.items():
                    # JAH: mps hack
                    if self.device == "mps":
                        batch_data[key] = batch.to(self.device, dtype=torch.float32) 
                    else:
                        batch_data[key] = batch.to(self.device)
                # Loss Calculation
                self.on_iteration(batch_data)

            # Validation of Model, Monitoring, Early Stopping
            self.on_epoch_end()
            if self.use_early_stopping:
                if not self.check_early_stop():
                    break

        if self.best_state_dict is not None and self.reload_best:
            print("Saving best state of network...")
            print("Best State was in Epoch", self.best_epoch)
            self.model.load_state_dict(self.best_state_dict)

        self.model.eval()
        self.after_loop()

        self.training_time += (time.time() - begin)

    def before_loop(self):
        pass

    def on_epoch_begin(self, lr, eps):
        pass

    def after_loop(self):
        pass

    def on_iteration(self, batch_data):
        # Dont update any weight on first layers except condition weights
        if self.model.freeze:
            for name, module in self.model.named_modules():
                if isinstance(module, nn.BatchNorm1d):
                    if not module.weight.requires_grad:
                        module.affine = False
                        module.track_running_stats = False

        # Calculate Loss depending on Trainer/Model
        self.current_loss = loss = self.loss(batch_data)
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping
        if self.clip_value > 0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_value)

        self.optimizer.step()

    def on_epoch_end(self):
        # Get Train Epoch Logs
        for key in self.iter_logs:
            self.logs["epoch_" + key].append(np.array(self.iter_logs[key]).mean())

        # Validate Model
        if self.valid_data is not None:
            self.validate()

        # Monitor Logs
        if self.monitor:
            print_progress(self.epoch, self.logs, self.n_epochs, self.monitor_only_val)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        self.iter_logs = defaultdict(list)
        # Calculate Validation Losses
        for val_iter, batch_data in enumerate(self.dataloader_valid):
            for key, batch in batch_data.items():
                batch_data[key] = batch.to(self.device)

            val_loss = self.loss(batch_data)

        # Get Validation Logs
        for key in self.iter_logs:
            self.logs["val_" + key].append(np.array(self.iter_logs[key]).mean())

        self.model.train()

    def check_early_stop(self):
        # Calculate Early Stopping and best state
        early_stopping_metric = self.early_stopping.early_stopping_metric
        if self.early_stopping.update_state(self.logs[early_stopping_metric][-1]):
            self.best_state_dict = self.model.state_dict()
            self.best_epoch = self.epoch

        continue_training, update_lr = self.early_stopping.step(self.logs[early_stopping_metric][-1])
        if update_lr:
            print(f'\nADJUSTED LR')
            for param_group in self.optimizer.param_groups:
                param_group["lr"] *= self.early_stopping.lr_factor

        return continue_training


class ClassifierTrainer:
    def __init__(self, model, train_loader, val_loader, device='cpu'):
        """
        Initialize the Trainer.
        :param model: The PyTorch model to train
        :param train_loader: DataLoader for the training data
        :param val_loader: DataLoader for the validation data
        :param device: 'cuda' or 'cpu'
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.CrossEntropyLoss()  # Use Cross-Entropy Loss
        self.optimizer = optim.Adam(model.parameters())  # Use vanilla Adam optimizer
        self.device = device

    def train(self, epochs):
        """
        Train the model for a number of epochs.
        :param epochs: Number of epochs to train for
        """
        self.model.train()  # Set the model to training mode
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            print(f'Epoch {epoch + 1}/{epochs} - Loss: {running_loss/len(self.train_loader)}')
            self.validate()  # Run validation at the end of each epoch

    def validate(self):
        """
        Validate the model on the validation dataset.
        """
        self.model.eval()  # Set the model to evaluation mode
        total = 0
        correct = 0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Validation Accuracy: {accuracy}%')

# Usage example:
# model = YourModel()
# train_loader = DataLoader(...)
# val_loader = DataLoader(...)

# trainer = Trainer(model, train_loader, val_loader, device='cuda')
# trainer.train(epochs=10)


class MseTrainer:
    def __init__(self, model, train_loader, val_loader, device='cpu'):
        """
        Initialize the MSE Trainer.
        :param model: The PyTorch model to train
        :param train_loader: DataLoader for the training data
        :param val_loader: DataLoader for the validation data
        :param device: 'cuda' or 'cpu'
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.MSELoss()  # Use Mean Squared Error Loss
        self.optimizer = optim.Adam(model.parameters())  # Use vanilla Adam optimizer
        self.device = device

    def train(self, epochs):
        """
        Train the model for a number of epochs.
        :param epochs: Number of epochs to train for
        """
        self.model.train()  # Set the model to training mode
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # Backward and optimize
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            print(f'Epoch {epoch + 1}/{epochs} - Loss: {running_loss/len(self.train_loader)}')
            self.validate()  # Run validation at the end of each epoch

    def validate(self):
        """
        Validate the model on the validation dataset.
        """
        self.model.eval()  # Set the model to evaluation mode
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        print(f'Validation Loss: {avg_loss}')

# Usage example:
# model = YourRegressionModel()
# train_loader = DataLoader(...)
# val_loader = DataLoader(...)

# mse_trainer = MseTrainer(model, train_loader, val_loader, device='cuda')
# mse_trainer.train(epochs=10)