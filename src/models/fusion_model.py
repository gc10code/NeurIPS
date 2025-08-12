import torch
import torch.nn as nn
from typing import List
import logging

class FusionModel(nn.Module):
    def __init__(self, teacher_output_sizes: List[int], hidden_layers: List[int], 
                 activations: List[str], dropout_prob: float = 0.3, batch_norm: bool = False):
        """
        Initialize the FusionModel to integrate outputs from heterogeneous pre-trained models.
        
        Args:
            teacher_output_sizes (List[int]): List of output sizes from teacher models.
            hidden_layers (List[int]): List of sizes for hidden layers in the fusion network.
            activations (List[str]): List of activation functions for each layer.
            dropout_prob (float): Dropout probability for regularization.
            batch_norm (bool): Whether to include batch normalization layers.
        """
        super(FusionModel, self).__init__()
        try:
            logger = logging.getLogger(__name__)
            logger.info("Initializing FusionModel with parameters: "
                       f"teacher_output_sizes={teacher_output_sizes}, hidden_layers={hidden_layers}, "
                       f"activations={activations}, dropout_prob={dropout_prob}, batch_norm={batch_norm}")
            
            if not teacher_output_sizes or any(size <= 0 for size in teacher_output_sizes):
                logger.error("Invalid teacher_output_sizes: must be non-empty and contain positive integers")
                raise ValueError("teacher_output_sizes must be non-empty and contain positive integers")
            if not hidden_layers or any(size <= 0 for size in hidden_layers):
                logger.error("Invalid hidden_layers: must be non-empty and contain positive integers")
                raise ValueError("hidden_layers must be non-empty and contain positive integers")
            if len(activations) != len(hidden_layers):
                logger.error(f"Number of activations ({len(activations)}) must match number of hidden layers ({len(hidden_layers)})")
                raise ValueError("Number of activations must match number of hidden layers")
            if not (0 <= dropout_prob <= 1):
                logger.error(f"Invalid dropout_prob: {dropout_prob}, must be between 0 and 1")
                raise ValueError("dropout_prob must be between 0 and 1")

            input_size = sum(teacher_output_sizes)  # Concatenated outputs from teachers
            logger.debug(f"Calculated input size (sum of teacher outputs): {input_size}")

            # Build fusion layer
            layers = []
            current_size = input_size
            for i, (hidden_size, activation) in enumerate(zip(hidden_layers, activations)):
                layers.append(nn.Linear(current_size, hidden_size))
                logger.debug(f"Layer {i+1}: Added Linear layer: {current_size} -> {hidden_size}")

                if batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_size))
                    logger.debug(f"Layer {i+1}: Added BatchNorm1d for hidden size: {hidden_size}")

                # Add activation function
                if activation.lower() == 'relu':
                    layers.append(nn.ReLU())
                    logger.debug(f"Layer {i+1}: Added ReLU activation")
                elif activation.lower() == 'prelu':
                    layers.append(nn.PReLU())
                    logger.debug(f"Layer {i+1}: Added PReLU activation")
                elif activation.lower() == 'tanh':
                    layers.append(nn.Tanh())
                    logger.debug(f"Layer {i+1}: Added Tanh activation")
                else:
                    logger.error(f"Unsupported activation function: {activation}")
                    raise ValueError(f"Unsupported activation function: {activation}")

                layers.append(nn.Dropout(dropout_prob))
                logger.debug(f"Layer {i+1}: Added Dropout layer with probability: {dropout_prob}")
                current_size = hidden_size

            self.fusion_layer = nn.Sequential(*layers)
            logger.debug(f"Fusion layer constructed with {len(layers)} components")

            # Output heads (one for each teacher model's output size)
            self.heads = nn.ModuleList([nn.Linear(hidden_layers[-1], size) for size in teacher_output_sizes])
            logger.debug(f"Created {len(self.heads)} output heads with sizes: {teacher_output_sizes}")

            # Initialize weights
            for i, layer in enumerate(self.fusion_layer):
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                    nn.init.zeros_(layer.bias)
                    logger.debug(f"Layer {i//4 + 1}: Initialized Linear layer weights with Kaiming uniform and zero bias")
            for i, head in enumerate(self.heads):
                nn.init.xavier_uniform_(head.weight, gain=0.1)
                nn.init.zeros_(head.bias)
                logger.debug(f"Head {i+1}: Initialized weights with Xavier uniform (gain=0.1) and zero bias")

            logger.info("FusionModel initialization completed successfully")
        except Exception as e:
            logger.error(f"Error in FusionModel initialization: {str(e)}")
            raise

    def forward(self, teacher_outputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass to combine teacher outputs and produce final predictions.
        
        Args:
            teacher_outputs (List[torch.Tensor]): List of output tensors from teacher models.
        
        Returns:
            List[torch.Tensor]: List of output tensors from each head.
        """
        try:
            logger = logging.getLogger(__name__)
            logger.debug("Starting FusionModel forward pass")

            if not teacher_outputs:
                logger.error("Empty teacher_outputs list provided")
                raise ValueError("teacher_outputs list cannot be empty")
            if any(not isinstance(t, torch.Tensor) for t in teacher_outputs):
                logger.error("All teacher_outputs must be torch.Tensor")
                raise ValueError("All teacher_outputs must be torch.Tensor")
            if any(t.ndim != 2 for t in teacher_outputs):
                logger.error("All teacher_outputs must be 2D tensors")
                raise ValueError("All teacher_outputs must be 2D tensors")

            # Concatenate teacher outputs
            fused_input = torch.cat(teacher_outputs, dim=1)
            logger.debug(f"Concatenated teacher outputs, shape: {fused_input.shape}")

            # Pass through fusion layer
            fused_features = self.fusion_layer(fused_input)
            logger.debug(f"Fusion layer output shape: {fused_features.shape}")

            # Generate outputs from each head
            outputs = [head(fused_features) for head in self.heads]
            logger.debug(f"Generated {len(outputs)} outputs from heads: {[out.shape for out in outputs]}")

            logger.debug("Forward pass completed successfully")
            return outputs
        except Exception as e:
            logger.error(f"Error in FusionModel forward pass: {str(e)}")
            raise