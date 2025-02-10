import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(
        self,
        d_input: int,
        d_model: int,
        d_hidden: int,
        d_output: int,
        n_layer: int,
        device: str,
        dropout: float = 0.1,
    ):
        super(NeuralNetwork, self).__init__()
        
        self.encode = nn.Sequential(nn.Linear(d_input, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_model))

        self.mlp = nn.ModuleList([nn.Sequential(nn.Linear(d_model, d_model), nn.Dropout(dropout)) for _ in range(n_layer)])
        
        self.outputs = nn.ModuleList([nn.Sequential(nn.Linear(d_model, 1)) for _ in range(d_output)])


    def forward(self, x):
        
        x = self.encode(x)
        
        for layer in self.mlp:
            x = layer(x)
            
        outputs = []
        
        m = nn.Sigmoid()
            
        for layer in self.outputs:
            
            output = m(layer(x))
            outputs.append(output)
                    
        outputs = torch.cat(outputs, dim=1)

        return outputs