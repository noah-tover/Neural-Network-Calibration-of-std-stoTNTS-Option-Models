import torch
import torch.nn as nn
import torch.nn.functional as F
############################################################################################
class MELU(nn.Module):
    '''
    Modified linear exponential unit function. Sourced from the paper "Deep learning calibration of option pricing models: some pitfalls and solutions" by Andrew Itkin.
    Intended to overcome dying neuron issue of softplus while being twice differentiable. 
    '''
    def __init__(self, alpha=.49): # Fixed alpha at.49 ensures the function is monotonic. Check it out in desmos to see!
        super().__init__()
        self.alpha = alpha
        self.a = 1 - 2 * alpha
        self.b = -2 + 1 / alpha
    # Masking is used as a singularity exists outside of the piecewise functions bounds. torch.where() computes this anyway resulting in nans that propagate throughout the network. 
    def forward(self, z):
        out = torch.empty_like(z)

        pos_mask = z > 0
        neg_mask = ~pos_mask

        # compute pos only where z > 0
        z_pos = z[pos_mask]
        out[pos_mask] = ((0.5 * z_pos ** 2) + self.a * z_pos) / (z_pos + self.b)

        # compute neg only where z <= 0
        z_neg = z[neg_mask]
        out[neg_mask] = self.alpha * (torch.exp(z_neg) - 1)

        return out
############################################################################################
def phi_lambda(x, lam, m):
    return lam * (torch.clamp(x, min = 0) ** m)
#############################################################################################
def custom_loss(
    model,
    inputs,
    call_true,
    feature_cols,
    lam1=1.0, m1=3,
    lam2=1.0, m2=3,
    lam3=1.0, m3=3,
    output_loss = True, 
):
    '''
    Soft constrained loss function sourced from "Deep learning calibration of option pricing models: some pitfalls and solutions" by Andrew Itkin. Contains piecewise functions which are 0 when constraints are not violated, and nonzero otherwise.
    NOTE: M_i must be at least 3, as the penalty functions are at least m-1 times differentiable and is differentiated at most twice. 
    feature_cols: the column names corresponding to the input tensor. the index of the column names coincides to the column index of their respective data in the tensor.
    output_loss: specifies whether to return the loss itself or the components of the loss. used for gradient scaling.
    '''
    inputs = inputs.requires_grad_(True)
    call_pred = model(inputs)

    mse_loss = F.mse_loss(call_pred, call_true)

    moneyness_idx = feature_cols.index("moneyness")
    tao_idx = feature_cols.index("tao")

    grads = torch.autograd.grad(
        outputs=call_pred,
        inputs=inputs,
        grad_outputs=torch.ones_like(call_pred),
        create_graph=True
    )[0]

    dC_dm = grads[:, moneyness_idx]
    dC_dt = grads[:, tao_idx]

    d2C_dm2 = torch.autograd.grad(
        outputs=dC_dm.sum(),
        inputs=inputs,
        create_graph=True
    )[0][:, moneyness_idx]

    reg1 = phi_lambda(-d2C_dm2, lam1, m1).mean()
    reg2 = phi_lambda(-dC_dt, lam2, m2).mean()
    reg3 = phi_lambda(dC_dm, lam3, m3).mean()
    if output_loss == True:
        loss = torch.log(mse_loss + reg1 + reg2 + reg3)
        return loss
       
    else:
        return mse_loss, reg1, reg2, reg3


############################################################################################
class make_optionnet(nn.Module):
    """
    Creates a neural network with a user specified activation function.
    input_dim: Dimension of the input.
    nodes: Number of hidden nodes (must be >= input_dim) for UAT.
    hidden_layers: Number of hidden layers.
    activation: Activation function (class or instance) to use in hidden layers.
    """
    def __init__(self, input_dim, nodes=None, hidden_layers=2, activation=nn.MELU, output_activation=nn.MELU):
        super().__init__()
        
        if nodes is None:
            nodes = input_dim + 1  # Universal approximation theorem does not work for skinny networks per "Deep, Skinny Neural Networks Are Not Universal Approximators" by Jesse Johnson
        if nodes <= input_dim:
            raise ValueError("Nodes must be greater than input dimension for universal approximation.")
        if hidden_layers < 1:
            raise ValueError("Hidden layers must be at least 1 for universal approximation.")

        if isinstance(activation, type):
            act_fn = activation()
        else:
            act_fn = activation
        
        if isinstance(output_activation, type):
            output_act = output_activation()
        else:
            output_act = output_activation

        layers = []
        in_dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, nodes))
            layers.append(act_fn)
            in_dim = nodes

        layers.append(nn.Linear(nodes, 1))
        layers.append(output_act)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
############################################################################################
class dual_balancer:
    """
    Dual balancing algorithm. 
    """

    def __init__(self, model, momentum=0.9, eps=1e-12):
        self.model = model
        self.momentum = momentum
        self.eps = eps

        # EMA storage: list (per loss) of lists (per parameter)
        self.ema_grads = None

    def _init_ema(self, grad_list):
        self.ema_grads = [
            [torch.zeros_like(g) for g in grads]
            for grads in grad_list
        ]

    def _update_ema(self, grad_list):
        if self.ema_grads is None:
            self._init_ema(grad_list)

        updated = []
        for loss_idx, grads in enumerate(grad_list):
            ema_for_loss = []
            for p_idx, g in enumerate(grads):
                ema = self.ema_grads[loss_idx][p_idx]
                ema.mul_(self.momentum).add_(g, alpha=1 - self.momentum)
                ema_for_loss.append(ema)
            updated.append(ema_for_loss)

        return updated

    def _normalize(self, grads):
        norm = torch.sqrt(sum((g**2).sum() for g in grads) + self.eps)
        return [g / norm for g in grads], norm

    def _combine(self, grad_list):
        norms = []
        normalized = []

        for grads in grad_list:
            normed, norm = self._normalize(grads)
            normalized.append(normed)
            norms.append(norm)

        max_norm = max(norms)

        scaled = [
            [g * max_norm for g in grads]
            for grads in normalized
        ]

        return [sum(params) for params in zip(*scaled)]


    def backward(self, loss_components):
        """
        loss_components: iterable of scalar losses
        """

        grad_list = [
            torch.autograd.grad(
                loss,
                self.model.parameters(),
                retain_graph=(i < len(loss_components) - 1)
            )
            for i, loss in enumerate(loss_components)
        ]

        grad_list = self._update_ema(grad_list)

        with torch.no_grad():
            grads_final = self._combine(grad_list)

            total_loss = sum(loss_components)
            # Manual gradient of the log loss. Via the chain rule, 1/total_loss * sum of individual gradients. 
            # This is made manual because pytorch doesnt allow you to take the gradient of each individual term in a composite loss.
            grads_final = [g / total_loss for g in grads_final]

        # Assign gradients
        for p, g in zip(self.model.parameters(), grads_final):
            p.grad = g
