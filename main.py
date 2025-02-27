import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GAE
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path):
    """Loads the manufacturing data from CSV."""
    df = pd.read_csv(file_path)
    df.replace({'Running': 1, 'Stopped': 0}, inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')  # Explicitly convert to numeric
    return df

def create_graph_from_data(df):
    """Creates a dynamic graph from manufacturing data."""
    G = nx.Graph()
    
    for index, row in df.iterrows():
        timestamp = row['Timestamps']
        
        for i in range(1, 4):
            G.add_node(f"M{i}_{timestamp}", status=row[f"M{i}_Status"], workers=row[f"M{i}_Worker_Count"], type='machine')
        
        for i in range(1, 3):
            G.add_edge(f"M{i}_{timestamp}", f"M{i+1}_{timestamp}")
        
    return G

def plot_graph(G):
    """Plots the graph using NetworkX."""
    plt.figure(figsize=(12, 6))
    pos = nx.spring_layout(G)  # Layout for positioning nodes
    
    node_colors = ["lightblue" if G.nodes[n]['type'] == 'machine' else "lightgreen" for n in G.nodes()]
    nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color="gray", node_size=2000, font_size=10)
    
    plt.title("Factory Machine Graph")
    plt.show()

def convert_to_pyg_graph(G):
    """Converts NetworkX graph to PyTorch Geometric format."""
    node_map = {node: i for i, node in enumerate(G.nodes())}
    edge_index = torch.tensor([[node_map[u], node_map[v]] for u, v in G.edges()], dtype=torch.long).t().contiguous()
    
    node_features = torch.tensor([
        [G.nodes[n]['status'], G.nodes[n]['workers']]
        for n in G.nodes()
    ], dtype=torch.float)
    
    return Data(x=node_features, edge_index=edge_index)

class GNNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

def train(model, data, epochs=600, lr=0.003):  # Increase epochs, reduce learning rate
    """Trains the GAE model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)
        loss = model.recon_loss(z, data.edge_index)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

def detect_anomalies(model, data, threshold=1.3):  # Adjust threshold slightly
    """Detects anomalies based on reconstruction loss."""
    model.eval()
    z = model.encode(data.x, data.edge_index)
    loss = model.recon_loss(z, data.edge_index).item()
    print(f"Anomaly Score: {loss:.4f}")
    print("Potential anomaly detected!" if loss > threshold else "No significant anomaly detected.")

# Load and process dataset
file_path = "manufacturing_test_data.csv"
df = load_data(file_path)
G = create_graph_from_data(df)

# Plot the graph before training
plot_graph(G)

data = convert_to_pyg_graph(G)

# Initialize and train model
out_channels = 8
model = GAE(GNNEncoder(in_channels=data.x.shape[1], out_channels=out_channels))
train(model, data)

# Detect anomalies
detect_anomalies(model, data)
