# gnn_anomaly_detection

This project utilizes Graph Neural Networks (GNN) with Graph Autoencoders (GAE) to detect anomalies in a manufacturing setup by modeling machine status and worker count as a dynamic graph.

## Features
- Converts manufacturing data into a dynamic graph
- Uses PyTorch Geometric for graph-based learning
- Trains a Graph Autoencoder (GAE) for anomaly detection
- Visualizes the factory machine network

## Installation
### Prerequisites
Ensure you have Python installed. Then, install the required dependencies using:
```bash
pip install -r requirements.txt
```

## Usage
1. Place your manufacturing data in `manufacturing_test_data.csv`.
2. Run the script:
```bash
python main.py
```
3. The program will:
   - Load and process data
   - Create and visualize a factory machine graph
   - Train a Graph Autoencoder
   - Detect anomalies based on reconstruction loss

## File Structure
```
├── main.py                  # Main script for running the model
├── manufacturing_test_data.csv # Sample dataset
├── requirements.txt         # Dependencies
├── README.md                # Project documentation
```

## Model Details
- **Graph Representation**: Machines and worker data are represented as nodes, with edges indicating machine connections.
- **GNN Encoder**: Uses two GCNConv layers to extract graph embeddings.
- **Anomaly Detection**: Reconstruction loss helps identify deviations from normal behavior.

## Example Output
```
Epoch 0, Loss: 0.7345
Epoch 10, Loss: 0.5123
...
Anomaly Score: 1.4321
Potential anomaly detected!
```

## Dependencies
- Python
- PyTorch Geometric
- NetworkX
- Pandas
- Matplotlib



