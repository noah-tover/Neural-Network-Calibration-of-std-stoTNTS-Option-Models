import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.base import BaseEstimator
import gc
from architecture import custom_loss, make_optionnet, dual_balancer
##########################################################
def tensor_standardize(X, y):
    """
    Standardizes torch tensors.
    """

    X_scaled = (X - X.mean(dim=0, keepdim=True)) / X.std(dim=0, unbiased=False, keepdim=True)
    y_scaled = (y - y.mean(dim=0, keepdim=True)) / y.std(dim=0, unbiased=False, keepdim=True)
    return X_scaled, y_scaled
##########################################################
class optionnet(BaseEstimator):
    """
    Neural network model for surrogate modelling of option pricing models. Designed to enforce arbitrage free constraints 
    without impacting training. Biases towards pareto optimality of mean squared error and constraints through dual balancing. 
    
    This class implements a multi-task learning neural network using:
    - Dual balancing (Lin et al., "Dual Balancing for Multi-Task Learning")
    - Soft constrained loss (Itkin, "Deep learning calibration of option pricing models: Some pitfalls and solutions")

    
    Parameters:
    feature_cols : list
        list of feature names where their indices correspond to their location within the tensor. necessary for custom_loss().
    lr : float, default=0.01
        learning rate
    momentum : float, default=0.3
        momentum term. reccomended to be non-zero to improve training stability as dual balancing rescales gradients by the largest magnitude.
    nodes : int, default=300
        number of hidden units per layer. must be greater than the number of features for the universal approximation theorem.
    hidden_layers : int, default=3
        number of hidden layers. must be nonzero for the universal approximation theorem.
    batch_size : int, default=30000
        minibatch size for minibatch gradient descent. it is not reccomended to use batches below 50% of training data as the gradients in the loss function become noisy.
    epochs : int, default=100
        training epochs.
    seed : int or None, default=None
        random seed. set for torch.
    
    Methods
    -------
    fit(X, y)
        Trains the neural network on the given dataset using dual-balanced, soft-constrained loss. Uses mini batch gradient descent and normalizes inputs and outputs per batch.
        Compatible with scikit-learn.
        Parameters:
        X, y: tensors
    fit_with_curves(X_train, y_train, X_val, y_val, model=None)
        Trains the network using both training and validation sets, tracking train/validation losses. Slower than .fit(X, y) due to tracking. Uses mini-batch gradient descent
        and normalizes inputs and outputs per batch.
        Parameters:
        X_train, y_train, X_val, y_val: tensors
        Torch tensors containing the training and validation datasets.
        Model: PyTorch model, default None
        Returns:
        train_losses: list
        val_losses: list
        model: PyTorch model
        Specifies a model to train. By default one is created from scratch.
    score(X, y)
        Returns the negative log loss of the model on the provided data (for compatibility with scikit-learn).
    
    Notes
    - Requires `custom_loss` and `combine_and_scale_gradients` for loss computation and gradient handling.
    - The model is designed for GPU acceleration if tensors are on CUDA devices.
    - Intermediate tensors are deleted and CUDA cache cleared to avoid memory leaks and accumulation. For .fit(X, y), intermediate tensors deleted only after .score(X, y)
"""
    #custom_loss is a major dependency.
    def __init__(
        self,
        feature_cols,
        lr=1e-2,
        momentum=0.3,
        nodes=300,
        hidden_layers=3,
        batch_size=30000,
        epochs=100, 
        seed = None
    ):
        self.lr = lr
        self.momentum = momentum
        self.nodes = nodes
        self.hidden_layers = hidden_layers
        self.batch_size = batch_size
        self.epochs = epochs
        self.feature_cols = feature_cols
        self.seed = seed
        if seed != None:
            torch.manual_seed(seed)

    def fit(self, X, y):
        
        X, y = tensor_standardize(X, y)

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_ = make_optionnet(
            input_dim=X.shape[1],
            nodes=self.nodes,
            hidden_layers=self.hidden_layers
        ).to(X.device)

        optimizer = torch.optim.SGD(
            self.model_.parameters(),
            lr=self.lr,
            momentum=0 # EMA is tracked inside of balancer.
        )
        
        balancer = dual_balancer(self.model_, momentum = self.momentum)
        self.model_.train() # This is just for future proofing in case the model architecture changes.
        for _ in range(self.epochs):
            for X_batch, y_batch in loader:                

                mse_loss, reg1, reg2, reg3 = custom_loss(
                    self.model_,
                    X_batch,
                    y_batch,
                    self.feature_cols,
                    output_loss=False,
                )

                optimizer.zero_grad()
                balancer.backward([mse_loss, reg1, reg2, reg3])
                optimizer.step()


    def fit_with_curves(self, X_train, y_train, X_val, y_val, model = None):
        
        X_train, y_train = tensor_standardize(X_train, y_train)
        X_val, y_val = tensor_standardize(X_val, y_val)

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        if model == None:
            self.model_ = make_optionnet(
                input_dim=X_train.shape[1],
                nodes=self.nodes,
                hidden_layers=self.hidden_layers
            ).to(X_train.device) # TODO: Make device an input later
        else:
            self.model_ = model

        optimizer = torch.optim.SGD(
            self.model_.parameters(),
            lr=self.lr,
            momentum=0 # EMA controlled by balancer
        )
        train_losses = []
        val_losses = []
        balancer = dual_balancer(self.model_, momentum = self.momentum)
        for _ in range(self.epochs):
            epoch_train_loss = 0.0
            epoch_val_loss = 0.0 
            
            ## TRAIN ##
            
            self.model_.train() # Not necessary, but just future proofing for if batch norm/other methods used.
            
            for X_batch, y_batch in train_loader:                

                mse_loss, reg1, reg2, reg3 = custom_loss(
                    self.model_,
                    X_batch,
                    y_batch,
                    self.feature_cols,
                    output_loss=False,
                )
                
                with torch.no_grad():
                    total_loss = torch.log(mse_loss + reg1 + reg2 + reg3)
                    epoch_train_loss += total_loss * X_batch.size(0)
                    
                
                optimizer.zero_grad()
                balancer.backward([mse_loss, reg1, reg2, reg3])
                optimizer.step()
                
            epoch_train_loss /= len(train_dataset)
            train_losses.append(epoch_train_loss.item())
            
            ## TEST ##
            self.model_.eval()
            
            for X_batch, y_batch in val_loader:
                # This tracks losses without influencing training as gradients part of loss.
                X_batch = X_batch.clone().detach().requires_grad_(True)
                y_batch = y_batch.clone().detach()
                batch_loss = custom_loss(self.model_, X_batch, y_batch, self.feature_cols, output_loss=True)
                epoch_val_loss += batch_loss * X_batch.size(0)
            
            epoch_val_loss /= len(val_dataset)
            val_losses.append(epoch_val_loss.item())

            
        del batch_loss, mse_loss, reg1, reg2, reg3, 
        del optimizer
        del train_dataset, train_loader, val_dataset, val_loader
        del X_train, y_train, X_val, y_val
        gc.collect()
        torch.cuda.empty_cache()
                    
        return train_losses, val_losses, self.model_
 


    def score(self, X, y, negative = True):
        """
        Returns loss. Default log loss and negative for sklearn compatibility.
        """
        self.model_.eval()
        
        # Detaching X and y in order to avoid learning on test set. (Since gradients are needed for custom_loss)
        X, y = tensor_standardize(X, y)
        X = X.clone().detach().requires_grad_(True)
        y = y.clone().detach()
        
    
        loss = custom_loss(
            self.model_,
            X,
            y,
            self.feature_cols,
            output_loss=True
        )
        if negative == True:
            with torch.no_grad():
                loss = -1 * loss

            
        del self.model_
        del X, y
        gc.collect()
        torch.cuda.empty_cache()

        return loss.item()
#################################################
class calibrationnet(BaseEstimator):
    def __init__(
        self,
        lr=1e-2,
        momentum=0.0,
        nodes=58,
        hidden_layers=2,
        batch_size=70000,
        epochs=100,
        seed=None
    ):
        self.lr = lr
        self.momentum = momentum
        self.nodes = nodes
        self.hidden_layers = hidden_layers
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed

        if self.seed is not None:
            torch.manual_seed(self.seed)

    def fit(self, X, y):
        X, y = tensor_standardize(X, y)

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_ = make_optionnet(
            input_dim=X.shape[1],
            nodes=self.nodes,
            hidden_layers=self.hidden_layers
        ).to(X.device)

        optimizer = torch.optim.SGD(
            self.model_.parameters(),
            lr=self.lr,
            momentum=self.momentum
        )

        criterion = nn.MSELoss()

        self.model_.train()
        for _ in range(self.epochs):
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                preds = self.model_(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()

        return self

    def fit_with_curves(self, X_train, y_train, X_val, y_val, model=None):
        X_train, y_train = tensor_standardize(X_train, y_train)
        X_val, y_val = tensor_standardize(X_val, y_val)

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )

        if model is None:
            self.model_ = make_optionnet(
                input_dim=X_train.shape[1],
                nodes=self.nodes,
                hidden_layers=self.hidden_layers
            ).to(X_train.device)
        else:
            self.model_ = model

        optimizer = torch.optim.SGD(
            self.model_.parameters(),
            lr=self.lr,
            momentum=self.momentum
        )
        criterion = nn.MSELoss()

        train_losses = []
        val_losses = []

        for _ in range(self.epochs):
            self.model_.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    preds = self.model_(X_batch)
                    batch_loss = criterion(preds, y_batch)
                    val_loss += batch_loss * X_batch.size(0)

            val_loss /= len(val_dataset)
            val_losses.append(val_loss.item())

            self.model_.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                preds = self.model_(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss * X_batch.size(0)

            train_loss /= len(train_dataset)
            train_losses.append(train_loss.item())

        del optimizer, train_dataset, val_dataset, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()

        return train_losses, val_losses, self.model_

    def score(self, X, y):

        self.model_.eval()

        X, y = tensor_standardize(X, y)

        criterion = nn.MSELoss()

        with torch.no_grad():
            preds = self.model_(X)
            mse = criterion(preds, y)
            score = -mse # For scikit learn compatibility - maximizes.

        del X, y
        gc.collect()
        torch.cuda.empty_cache()

        return score.item()
