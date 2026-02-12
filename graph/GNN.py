import torch
import pandas as pd
from supabase import create_client
from google.colab import userdata
import os

supabase = create_client(
    userdata.get("SUPABASE_URL"),
    userdata.get("SUPABASE_SERVICE_ROLE_KEY")
)

def fetch_all(table, select="*", page_size=1000):
    all_rows = []
    offset = 0

    while True:
        res = (
            supabase
            .table(table)
            .select(select)
            .range(offset, offset + page_size - 1)
            .execute()
        )

        batch = res.data
        if not batch:
            break

        all_rows.extend(batch)
        offset += page_size

    return all_rows


jobs_df = pd.DataFrame(fetch_all("jobs", "id, embedding, title"))
skills_df = pd.DataFrame(fetch_all("skills", "id, embedding"))
edges_df = pd.DataFrame(fetch_all("job_skills", "job_id, skill_id"))

print(len(jobs_df))
print(len(skills_df))
print(len(edges_df))

import ast

job_id_map = {jid: i for i, jid in enumerate(jobs_df["id"])}
skill_id_map = {sid: i for i, sid in enumerate(skills_df["id"])}

def build_features(df, dim=768):
    embeddings = df["embedding"].apply(ast.literal_eval).tolist()
    return torch.tensor(embeddings, dtype=torch.float)

job_x = build_features(jobs_df)
skill_x = build_features(skills_df)

src = edges_df["job_id"].map(job_id_map).values
dst = edges_df["skill_id"].map(skill_id_map).values

job_x = build_features(jobs_df)
skill_x = build_features(skills_df)
edge_index = torch.tensor([src, dst], dtype=torch.long)


import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.transforms import ToUndirected, RandomLinkSplit
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

# =======================
# 1️⃣ Prepare Hetero Data
# =======================

data = HeteroData()
data["job"].x = job_x            # tensor of shape [num_jobs, job_emb_dim]
data["skill"].x = skill_x        # tensor of shape [num_skills, skill_emb_dim]
data["job", "has_skill", "skill"].edge_index = edge_index  # [2, num_edges]

# Make graph undirected (adds reverse edges)
data = ToUndirected()(data)

# Optional: freeze skill embeddings (pretrained)
data["skill"].x.requires_grad = False


transform = RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    is_undirected=True,
    add_negative_train_samples=True,
    edge_types=("job", "has_skill", "skill"),
    rev_edge_types=("skill", "rev_has_skill", "job"),
)

train_data, val_data, test_data = transform(data)

class GNNEncoder(nn.Module):
    def __init__(self, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_dim)
        self.conv2 = SAGEConv((hidden_dim, hidden_dim), hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict).relu()
        x_dict = {k: self.dropout(v) for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict

model = to_hetero(GNNEncoder(hidden_dim=128), metadata=train_data.metadata(), aggr="sum")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
train_data = train_data.to(device)
val_data = val_data.to(device)
test_data = test_data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def decode_hetero(z_dict, edge_label_index):
    src, dst = edge_label_index
    z_src = z_dict["job"][src]
    z_dst = z_dict["skill"][dst]
    return (z_src * z_dst).sum(dim=-1)


def train_epoch():
    model.train()
    optimizer.zero_grad()

    z_dict = model(train_data.x_dict, train_data.edge_index_dict)
    edge_index = train_data["job", "has_skill", "skill"].edge_label_index
    edge_label = train_data["job", "has_skill", "skill"].edge_label

    pred = decode_hetero(z_dict, edge_index)
    loss = nn.functional.binary_cross_entropy_with_logits(pred, edge_label)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(data_split):
    model.eval()
    z_dict = model(data_split.x_dict, data_split.edge_index_dict)
    edge_index = data_split["job", "has_skill", "skill"].edge_label_index
    edge_label = data_split["job", "has_skill", "skill"].edge_label
    pred = decode_hetero(z_dict, edge_index)
    pred_sigmoid = pred.sigmoid()

    # Simple metric: AUC (requires sklearn)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(edge_label.cpu().numpy(), pred_sigmoid.cpu().numpy())
    return auc


epochs = 50
for epoch in range(1, epochs + 1):
    loss = train_epoch()
    val_auc = evaluate(val_data)
    print(f"Epoch {epoch:02d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}")

test_auc = evaluate(test_data)
print(f"Test AUC: {test_auc:.4f}")


