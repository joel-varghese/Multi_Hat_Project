import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import SAGEConv
from sklearn.metrics import f1_score

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_edge_labels, dropout=0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = Linear(out_channels, num_edge_labels)  # for edge label prediction

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x  # node embeddings

    def classify(self, x_i, x_j):
        return self.classifier((x_i + x_j) / 2)

def train(model, data, optimizer, class_weights=None):
    model.train()
    optimizer.zero_grad()
    
    x = data['skill'].x
    edge_index = data['skill', 'related_to', 'skill'].edge_index
    edge_type = data['skill', 'related_to', 'skill'].edge_type
    
    out = model(x, edge_index)
    src, dst = edge_index
    pred = model.classify(out[src], out[dst])

    loss = F.cross_entropy(pred, edge_type, weight=class_weights)
    loss.backward()
    optimizer.step()

    return loss.item()

def evaluate(model, data):
    model.eval()
    x = data['skill'].x
    edge_index = data['skill', 'related_to', 'skill'].edge_index
    edge_type = data['skill', 'related_to', 'skill'].edge_type
    
    with torch.no_grad():
        out = model(x, edge_index)
        src, dst = edge_index
        pred = model.classify(out[src], out[dst])
        predicted_labels = pred.argmax(dim=1)
        f1 = f1_score(edge_type.numpy(), predicted_labels.numpy(), average='weighted')
    return f1

def load_model(path, in_channels, hidden_channels, out_channels, num_edge_labels):
    model = GraphSAGE(in_channels, hidden_channels, out_channels, num_edge_labels)
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"✅ Model loaded from {path}")
    return model