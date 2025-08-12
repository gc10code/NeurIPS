import torch.nn as nn
from typing import List, Union

ACTIVATION_FUNCTIONS = {
    'relu': nn.ReLU(),
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh(),
    'elu': nn.ELU(),
    'selu': nn.SELU(),
    'gelu': nn.GELU(),
    'leaky_relu': nn.LeakyReLU(),
    'prelu': nn.PReLU(),
    'softplus': nn.Softplus(),
    'swish': nn.SiLU(),
    'mish': nn.Mish(),
    'linear': nn.Identity()
}

def validate_activations(activations: List[Union[str, List[str]]]) -> None:
    available = ', '.join(ACTIVATION_FUNCTIONS.keys())
    for act in activations:
        if isinstance(act, str):
            if act not in ACTIVATION_FUNCTIONS:
                raise ValueError(f"Unknown activation '{act}'. Available: {available}")
        elif isinstance(act, (list, tuple)):
            if not act:
                raise ValueError("Activation list cannot be empty")
            for sub_act in act:
                if not isinstance(sub_act, str):
                    raise ValueError(f"Activation must be a string, got {sub_act}")
                if sub_act not in ACTIVATION_FUNCTIONS:
                    raise ValueError(f"Unknown activation '{sub_act}'. Available: {available}")
        else:
            raise ValueError(f"Activation must be a string or list of strings, got {act}")