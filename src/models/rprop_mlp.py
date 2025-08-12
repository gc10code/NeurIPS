import torch
import torch.nn as nn
from typing import List, Union

from src.utils.activations import ACTIVATION_FUNCTIONS, validate_activations

class RPropMLP(nn.Module):
    """Resilient Propagation Multi-Layer Perceptron with improved robustness."""
    
    def __init__(self, input_size: int, hidden_layers: List[int], output_size: int, 
                 activations: Union[str, List[str]], dropout_prob: float = 0.3, 
                 batch_norm: bool = False, problem_type: str = 'regression'):
        super(RPropMLP, self).__init__()
        
        if input_size <= 0 or output_size <= 0:
            raise ValueError("input_size and output_size must be positive")
        
        if not hidden_layers:
            raise ValueError("hidden_layers cannot be empty")
        
        if any(h <= 0 for h in hidden_layers):
            raise ValueError("All hidden layer sizes must be positive")
        
        if not (0 <= dropout_prob <= 0.5):
            raise ValueError(f"dropout_prob must be in [0,0.5], got {dropout_prob}")
        
        if problem_type not in ['regression', 'classification']:
            raise ValueError(f"problem_type must be 'regression' or 'classification', got {problem_type}")
        
        if isinstance(activations, str):
            activations = [activations] * len(hidden_layers)
        elif len(activations) != len(hidden_layers):
            raise ValueError(f"Number of activations ({len(activations)}) must match hidden layers ({len(hidden_layers)})")
        
        validate_activations(activations)
        
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.activations = activations
        self.dropout_prob = dropout_prob
        self.batch_norm = batch_norm
        self.problem_type = problem_type
        
        self.model = self._build_network()
    
    def _build_network(self) -> nn.Sequential:
        layers = []
        sizes = [self.input_size] + self.hidden_layers + [self.output_size]
        
        for i in range(len(sizes) - 1):
            layer = nn.Linear(sizes[i], sizes[i + 1])
            self._initialize_weights(layer, i, sizes)
            layers.append(layer)
            
            if i < len(sizes) - 2:
                if self.batch_norm:
                    layers.append(nn.BatchNorm1d(sizes[i + 1]))
                
                activation_name = self.activations[i]
                if activation_name == 'prelu':
                    layers.append(nn.PReLU(num_parameters=sizes[i + 1]))
                else:
                    layers.append(ACTIVATION_FUNCTIONS[activation_name])
                
                if self.dropout_prob > 0:
                    layers.append(nn.Dropout(self.dropout_prob))
            elif self.problem_type == 'classification':
                layers.append(nn.Softmax(dim=1))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self, layer: nn.Linear, layer_idx: int, sizes: List[int]) -> None:
        if layer_idx < len(sizes) - 2:
            activation = self.activations[layer_idx]
            if activation in ['relu', 'gelu', 'leaky_relu', 'elu', 'selu', 'mish', 'swish']:
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            elif activation in ['sigmoid', 'tanh']:
                nn.init.xavier_uniform_(layer.weight)
            else:
                nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        else:
            nn.init.xavier_uniform_(layer.weight, gain=0.1)
            nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"Input must be 2D tensor, got {x.dim()}D")
        
        if x.size(1) != self.input_size:
            raise ValueError(f"Input size mismatch: expected {self.input_size}, got {x.size(1)}")
        
        return self.model(x)
    
    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)