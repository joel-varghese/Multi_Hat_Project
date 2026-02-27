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


from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

MODEL_NAME = "BAAI/bge-base-en-v1.5"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)

class JobSkillDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         self.labels[idx]
        }

BATCH_SIZE = 32

train_dataset = JobSkillDataset(X_train, y_train, tokenizer)
val_dataset   = JobSkillDataset(X_val,   y_val,   tokenizer)
test_dataset  = JobSkillDataset(X_test,  y_test,  tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE)


# Model Definition

import torch.nn as nn
from transformers import AutoModel

class JobSkillClassifier(nn.Module):
    def __init__(self, model_name, num_labels, dropout=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size  = self.encoder.config.hidden_size  # 384 for MiniLM

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_labels)   # No sigmoid here — handled in loss
        )

    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = output.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).float()
        pooled = (token_embeddings * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)
        return self.classifier(pooled)


NUM_LABELS = len(skills_df)
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model      = JobSkillClassifier(MODEL_NAME, NUM_LABELS).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Output size: {NUM_LABELS} skills | Device: {device}")

# Training Loop

from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score

EPOCHS    = 10
LR        = 2e-5
THRESHOLD = 0.5

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler   = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)
criterion = nn.BCEWithLogitsLoss()

def evaluate(loader):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels)
            total_loss += loss.item()

            preds = (torch.sigmoid(logits) > THRESHOLD).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    all_preds  = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    f1_micro   = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    f1_macro   = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return total_loss / len(loader), f1_micro, f1_macro


best_val_f1 = 0

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0

    for batch in train_loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss   = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # prevent exploding gradients
        optimizer.step()
        scheduler.step()
        total_train_loss += loss.item()

    avg_train_loss       = total_train_loss / len(train_loader)
    val_loss, f1_mi, f1_ma = evaluate(val_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | F1-micro: {f1_mi:.4f} | F1-macro: {f1_ma:.4f}")

    # Save best checkpoint
    if f1_mi > best_val_f1:
        best_val_f1 = f1_mi
        torch.save(model.state_dict(), "best_model.pt")
        print(f"  ✓ Saved best model (F1-micro: {f1_mi:.4f})")