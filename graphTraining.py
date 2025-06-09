import json
from keybert import KeyBERT
from itertools import combinations
from model_graphsage import GraphSAGE, train, evaluate
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import torch
import pickle
from torch_geometric.data import HeteroData
import numpy as np
from collections import Counter

kw_model = KeyBERT()
embedder = SentenceTransformer('all-MiniLM-L6-v2')

with open('graphData.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

occupation_data = data['occupation_data']

def extract_keywords(text, top_n=30):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=top_n)
    return list(set([kw for kw, _ in keywords]))

def build_edges(keywords, occupation_label):
    return [(a, b, occupation_label) for a, b in combinations(keywords, 2) if a != b]

# Step 1: Extract keywords and build edges
all_keywords = set()
all_edges = []

for item in occupation_data:
    keywords = extract_keywords(item['literal'])
    all_keywords.update(keywords)
    all_edges.extend(build_edges(keywords, item['label']))

# Step 2: Encode nodes and edge labels
skill_list = sorted(list(all_keywords))
skill2idx = {skill: idx for idx, skill in enumerate(skill_list)}
with open("skill2idx.pkl", "wb") as f:
    pickle.dump(skill2idx, f)

# Encode edge types (occupations)
edge_index = []
edge_labels = []
le = LabelEncoder()
le.fit([x['label'] for x in occupation_data])

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

for src, dst, label in all_edges:
    edge_index.append([skill2idx[src], skill2idx[dst]])
    edge_labels.append(le.transform([label])[0])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
edge_type = torch.tensor(edge_labels, dtype=torch.long)

# Check edge label distribution
label_counts = Counter(edge_labels)
print("Edge label distribution:", {le.inverse_transform([k])[0]: v for k, v in label_counts.items()})

# Semantic features
embedder = SentenceTransformer('all-MiniLM-L6-v2')
x = torch.tensor(embedder.encode(skill_list), dtype=torch.float)
x = torch.nn.functional.normalize(x, p=2, dim=1)

# Train-validation split
edge_index_np = edge_index.t().numpy()
edge_type_np = edge_type.numpy()
train_idx, val_idx = train_test_split(range(len(edge_type_np)), test_size=0.2, stratify=edge_type_np, random_state=42)

train_data = HeteroData()
train_data['skill'].x = x
train_data['skill', 'related_to', 'skill'].edge_index = torch.tensor(edge_index_np[train_idx], dtype=torch.long).t().contiguous()
train_data['skill', 'related_to', 'skill'].edge_type = torch.tensor(edge_type_np[train_idx], dtype=torch.long)

val_data = HeteroData()
val_data['skill'].x = x
val_data['skill', 'related_to', 'skill'].edge_index = torch.tensor(edge_index_np[val_idx], dtype=torch.long).t().contiguous()
val_data['skill', 'related_to', 'skill'].edge_type = torch.tensor(edge_type_np[val_idx], dtype=torch.long)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(edge_labels), y=edge_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float)

# Initialize and train model
model = GraphSAGE(in_channels=x.size(1), hidden_channels=32, out_channels=16, num_edge_labels=len(le.classes_), dropout=0.3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

best_val_f1 = 0
patience = 10
counter = 0

for epoch in range(100):
    train_loss = train(model, train_data, optimizer, class_weights)
    val_f1 = evaluate(model, val_data)
    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f} | Val F1-Score = {val_f1:.4f}")
    
    # Early stopping
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        counter = 0
        torch.save(model.state_dict(), "graphsage_model.pth")
    else:
        counter += 1
    if counter >= patience:
        print("Early stopping triggered")
        break

def save_model(model, path="graphsage_model.pth"):
    torch.save(model.state_dict(), path)
    print(f"✅ Model saved to {path}")

save_model(model, "graphsage_model.pth")