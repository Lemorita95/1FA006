import torch.nn as nn
from torch_geometric.nn import DynamicEdgeConv, global_mean_pool


class MLP(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) model implemented using PyTorch's nn.Module.
    Attributes:
        layer1 (nn.Linear): The first linear layer of the MLP, mapping input_dim to hidden_dim.
        activtion1 (nn.ReLU): The activation function applied after the first layer.
        layer2 (nn.Linear): The second linear layer of the MLP, mapping hidden_dim to output_dim.
    Methods:
        forward(x):
            Performs a forward pass through the MLP model.
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            Returns:
                torch.Tensor: Output tensor of shape (batch_size, output_dim).
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()

        # Define the MLP architecture
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activtion1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.activtion2 = nn.ReLU()
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Forward pass through the model
        x = self.layer1(x)
        x = self.activtion1(x)

        # x = self.layer2(x)
        # x = self.activtion2(x)

        x = self.output(x)

        return x


class GNNEncoder(nn.Module):
    """
    Defintion of the GNN model
    Use the DynamicEdgeConv layer from the pytorch geometric package like this:
    MLP is a Multi-Layer Perceptron that is used to compute the edge features, you still need to define it.
    The input dimension to the MLP should be twice the number of features in the input data (i.e., 2 * n_features),
    because the edge features are computed from the concatenation of the two nodes that are connected by the edge.
    The output dimension of the MLP is the new feauture dimension of this graph layer.

    GNNEncoder is a graph neural network model that uses DynamicEdgeConv layers 
    from the PyTorch Geometric library to process graph-structured data. It is 
    designed to encode node features into a graph-level representation.
    Attributes:
        layer_list (nn.ModuleList): A list of DynamicEdgeConv layers, where each 
            layer computes edge features using an MLP and aggregates them using 
            a specified aggregation method (e.g., 'mean').
        final_mlp (nn.Sequential): A final MLP that maps the graph-level features 
            to the desired output dimension.
    Args:
        n_features (int): The number of features in the input data for each node.
        hidden_dim (int): The hidden dimension used in the MLPs within the 
            DynamicEdgeConv layers.
        output_dim (int): The output dimension of the final graph-level 
            representation.
        k_neighbors (int): The number of nearest neighbors to consider for 
            constructing edges in the DynamicEdgeConv layers.
        num_layers (int): The number of DynamicEdgeConv layers in the model.
    Methods:
        forward(data):
            Processes a batch of graph data and returns the graph-level 
            representation.
            Args:
                data (torch_geometric.data.Data): A batch of graph data containing 
                    node features (data.x) and batch indices (data.batch).
            Returns:
                torch.Tensor: A tensor of shape (n_batch, output_dim), where 
                n_batch is the number of graphs in the batch and output_dim is the 
                dimension of the graph-level representation.
    """
    def __init__(self, n_features, hidden_dim, output_dim, k_neighbors, num_layers):
        super(GNNEncoder, self).__init__()

        # Create a list of `num_layers` DynamicEdgeConv layers
        self.layer_list = nn.ModuleList()
        for i in range(num_layers):
            input_dim = 2 * n_features if i == 0 else 2 * hidden_dim
            self.layer_list.append(
                DynamicEdgeConv(
                    MLP(input_dim, hidden_dim, hidden_dim),
                    aggr='mean', k=k_neighbors
                )
            )
        
        # Final MLP to map graph-level features to output dimension
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data):
        # data is a batch graph item. it contains a list of tensors (x) and how the batch is structured along this list (batch)
        x = data.x
        batch = data.batch

        # loop over the DynamicEdgeConv layers:
        for layer in self.layer_list:
            x = layer(x, batch)

        # the output of the last layer has dimensions (n_batch, n_nodes, graph_feature_dimension)
        # where n_batch is the number of graphs in the batch and n_nodes is the number of nodes in the graph
        # i.e. one output per node (i.e. the hits in the event).
        # To combine all node feauters into single prediction, we recommend to use global pooling
        x = global_mean_pool(x, batch) # -> (n_batch, output_dim)
        # x is now a tensor of shape (n_batch, output_dim)

        # either your the last graph feature dimension is already the output dimension you want to predict
        # or you need to add a final MLP layer to map the output dimension to the number of labels you want to predict
        x = self.final_mlp(x)

        return x