from fipy.tools import float32
import torch
from torch import dtype, nn, autograd, Tensor
from torch.nn import functional as F

def calc_grad(y: Tensor, x: Tensor) -> Tensor:
    """
    Calculate the gradient of `y` with respect to `x`.

    Args:
        y (Tensor): The output tensor.
        x (Tensor): The input tensor.

    Returns:
        Tensor: The gradient of `y` with respect to `x`.
    """

    grad = autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]
    return grad


class FfnBlock(nn.Module):
    def __init__(self, dim: int):
        """
        Initialize the Feed-Forward Network Block.

        Args:
            dim (int): The dimensionality of the input and output.
        """

        super().__init__()
        inter_dim = 4 * dim
        self.fc1 = nn.Linear(dim, inter_dim)
        self.fc2 = nn.Linear(inter_dim, dim)
        self.act_fn = nn.Softplus()
        self.dropout = nn.Dropout(0.1)
        self.c = torch.log(torch.tensor(2., dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the FFN block.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """

        x0 = x
        x = self.fc1(x)
        x = self.act_fn(x) - self.c
        x = self.fc2(x)
        x = self.dropout(x)
        return x + x0

class BaseNet(nn.Module):
    """
    Base neural network model for predicting values.

    `forward` returns a tensor of shape (D, 3), where D is the number of
    data points, and the 2nd dimension contains the predicted values of p, u, v.
    """

    def __init__(self, hidden_dim: int = 128, num_blocks: int = 8, name: str=''):
        super().__init__()
        """
        Initialize the BaseNet.

        Args:
            hidden_dim (int): The hidden dimension size. Defaults to 128.
            num_blocks (int): The number of FFN blocks. Defaults to 8.
            name (str): The name of the network. Defaults to an empty string.
        """

        self.name = name
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.first_map = nn.Linear(3, self.hidden_dim)
        self.last_map = nn.Linear(self.hidden_dim, 1)
        self.ffn_blocks = nn.ModuleList([
            FfnBlock(self.hidden_dim) for _ in range(self.num_blocks)
        ])

        self.diffusion_coeff = torch.tensor([0])

        self.init_weights()

    def init_weights(self):
        """
        Initialize weights of the network.
        """

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def ffn(self, inputs: Tensor) -> Tensor:
        """
        Pass inputs through the feed-forward network.

        Args:
            inputs (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """

        x = self.first_map(inputs)
        for blk in self.ffn_blocks:
            x = blk(x)
        x = self.last_map(x)
        return x

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        t: Tensor,
        u: Tensor = None,
    ) -> dict:
        """
        Forward pass through the network.

        Args:
            x (Tensor): x-coordinates of the input data.
            y (Tensor): y-coordinates of the input data.
            t (Tensor): Time values of the input data.
            u (Tensor, optional): Ground truth values. Defaults to None.

        Returns:
            dict: Dictionary containing predictions, loss, and individual losses.
        """

        inputs = torch.stack([x, y, t], dim=1)
        
        hidden_output = self.ffn(inputs)
        u_pred = hidden_output[:, 0]

        loss, losses = self.loss_fn(u, u_pred)
        return {
            "preds": u_pred,
            "loss": loss,
            "losses": losses
        }

    def loss_fn(self, u: Tensor, u_pred: Tensor) -> tuple: 
        """
        Calculate the loss.

        Args:
            u (Tensor): Ground truth values.
            u_pred (Tensor): Predicted values.

        Returns:
            tuple: Total loss and individual losses.
        """

        u_loss = F.mse_loss(u_pred, u)
        f_u_loss = F.mse_loss(u_pred[0], u_pred[0])
        loss = u_loss + f_u_loss
        return loss, {
            "u_loss": u_loss,
            "f_u_loss": f_u_loss,
        }

class Pinn(nn.Module):
    """
    Physics-Informed Neural Network (PINN) model.

    `forward` returns a tensor of shape (D, 3), where D is the number of
    data points, and the 2nd dimension contains the predicted values of p, u, v.
    """

    def __init__(self, hidden_dim: int = 128, 
        num_blocks: int = 8, name: str=''
        ):
        """
        Initialize the PINN.

        Args:
            hidden_dim (int): The hidden dimension size. Defaults to 128.
            num_blocks (int): The number of FFN blocks. Defaults to 8.
            name (str): The name of the network. Defaults to an empty string.
        """
        
        super().__init__()

        self.name = name
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.first_map = nn.Linear(3, self.hidden_dim)
        self.last_map = nn.Linear(self.hidden_dim, 1)
        self.ffn_blocks = nn.ModuleList([
            FfnBlock(self.hidden_dim) for _ in range(self.num_blocks)
        ])

        self.diffusion_coeff = nn.Parameter(torch.tensor(1.0))

        self.init_weights()

    def init_weights(self):
        """
        Initialize weights of the network.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def ffn(self, inputs: Tensor) -> Tensor:
        """
        Pass inputs through the feed-forward network.

        Args:
            inputs (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        x = self.first_map(inputs)
        for blk in self.ffn_blocks:
            x = blk(x)
        x = self.last_map(x)
        return x

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        t: Tensor,
        u: Tensor = None,
    ) -> dict:
        """
        Forward pass through the network.

        Args:
            x (Tensor): x-coordinates of the input data.
            y (Tensor): y-coordinates of the input data.
            t (Tensor): Time values of the input data.
            u (Tensor, optional): Ground truth values. Defaults to None.

        Returns:
            dict: Dictionary containing predictions, loss, and individual losses.
        """

        inputs = torch.stack([x, y, t], dim=1)
        
        hidden_output = self.ffn(inputs)
        u_pred = hidden_output[:, 0]
        
        u_yy_pred = calc_grad(calc_grad(u_pred, y), y)
        u_xx_pred = calc_grad(calc_grad(u_pred, x), x)
        u_t_pred = calc_grad(u_pred, t)

        f_u = (
            self.diffusion_coeff * (u_xx_pred + u_yy_pred ) - u_t_pred
        )

        loss, losses = self.loss_fn(u, u_pred, f_u)

        return {
            "preds": u_pred,
            "loss": loss,
            "losses": losses,
        }

    def loss_fn(self, u: Tensor, u_pred: Tensor, f_u_pred: Tensor) -> tuple:
        """
        Calculate the loss.

        Args:
            u (Tensor): Ground truth values.
            u_pred (Tensor): Predicted values.
            f_u_pred (Tensor): Predicted PDE residuals.

        Returns:
            tuple: Total loss and individual losses.
        """

        u_loss = F.mse_loss(u_pred, u)
        f_u_loss = F.mse_loss(f_u_pred, torch.zeros_like(f_u_pred))

        loss = u_loss + 2*f_u_loss
        return loss, {
            "u_loss": u_loss,
            "f_u_loss": f_u_loss,
        }

class Pinn2(nn.Module):
    """
    Enhanced Physics-Informed Neural Network (PINN) model with domain sampling.

    `forward` returns a tensor of shape (D, 3), where D is the number of
    data points, and the 2nd dimension contains the predicted values of p, u, v.
    """

    def __init__(self, hidden_dim: int = 128, 
        num_blocks: int = 8, name: str='', 
        maxes: Tensor =torch.ones(4),
        mins: Tensor =torch.zeros(4),
        means: Tensor =torch.zeros(4),
        stds: Tensor =torch.ones(4),
        num_samples: int =20
        ):
        super().__init__()

        self.name = name
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.first_map = nn.Linear(3, self.hidden_dim)
        self.last_map = nn.Linear(self.hidden_dim, 1)
        self.ffn_blocks = nn.ModuleList([
            FfnBlock(self.hidden_dim) for _ in range(self.num_blocks)
        ])
        self.maxes = maxes
        self.mins = mins
        self.means = means
        self.stds = stds
        self.diffusion_coeff = nn.Parameter(torch.tensor(1.0))
        self.num_samples = num_samples//3 
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def ffn(self, inputs: Tensor) -> Tensor:
        x = self.first_map(inputs)
        for blk in self.ffn_blocks:
            x = blk(x)
        x = self.last_map(x)
        return x

    def sample_domain(self, device: str):
        domain_inside = self.mins[:-1] + (self.maxes[:-1] - self.mins[:-1]) * torch.rand(self.num_samples, 3, device=device) 
        domain_inside = (domain_inside-self.means[:-1])/self.stds[:-1]

        domain_boundary = torch.rand(self.num_samples, 3, device=device)
        indices = torch.stack([
            torch.arange(self.num_samples, device=device), 
            torch.randint(0, 2, (self.num_samples,), device=device)
        ], dim=1)

        # Create values to assign
        values_to_assign = torch.randint(0, 2, (self.num_samples,), device=device, dtype=torch.float)

        # Use index_put_ to assign values to domain_boundary at specified indices
        domain_boundary.index_put_((indices[:,0], indices[:,1]), values_to_assign, accumulate=False)

        domain_boundary = self.mins[:-1] + (self.maxes[:-1] - self.mins[:-1]) * domain_boundary
        domain_boundary = (domain_boundary-self.means[:-1])/self.stds[:-1]

        return domain_inside, domain_boundary


    def forward(
        self,
        x: Tensor,
        y: Tensor,
        t: Tensor,
        u: Tensor = None,
    ) -> dict:
        """
        All shapes are (b,)

        inputs: x, y, t
        labels: u

        """
        device = x.device
        inputs = torch.stack([x, y, t], dim=1)
        inputs_inside, inputs_boundary = self.sample_domain(device)
        l = inputs.shape[0]
        inputs = torch.concatenate([inputs, inputs_inside], dim=0)
        u_pred = self.ffn(inputs)[:, 0]
        
        u_yy_pred = calc_grad(calc_grad(u_pred, y), y)
        u_xx_pred = calc_grad(calc_grad(u_pred, x), x)
        u_t_pred = calc_grad(u_pred, t)

        f_u = (
            self.diffusion_coeff * (u_xx_pred + u_yy_pred ) - u_t_pred
        )

        u_pred_bound = self.ffn(inputs_boundary)[:, 0]

        loss, losses = self.loss_fn(u, u_pred[:l], u_pred_bound, f_u)

        return {
            "preds": u_pred[:l],
            "loss": loss,
            "losses": losses,
        }

    def loss_fn(self, u: Tensor, u_pred: Tensor, u_pred_bound: Tensor, f_u_pred: Tensor) -> tuple:
        """
        u: (b, 1)
        """
        u_loss = F.mse_loss(u_pred, u)
        f_u_loss = F.mse_loss(f_u_pred, torch.zeros_like(f_u_pred))
        T = (100-self.means[-1])/self.stds[-1]
        f_u_loss = f_u_loss + F.mse_loss(u_pred_bound, T * torch.ones_like(u_pred_bound))
        loss = u_loss + 2*f_u_loss
        return loss, {
            "u_loss": u_loss,
            "f_u_loss": f_u_loss,
        }