from model_graphsage import load_model
import joblib
from keybert import KeyBERT
import torch
from itertools import combinations
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

kw_model = KeyBERT()
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def extract_keywords(text, top_n=30):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=top_n)
    return list(set([kw for kw, _ in keywords]))

def map_unmatched_keywords(keywords, skill_list, skill2idx, threshold=0.7):
    """Map unmatched keywords to the closest training keyword using semantic similarity."""
    keyword_embeddings = embedder.encode(keywords)
    skill_embeddings = embedder.encode(skill_list)
    similarities = cosine_similarity(keyword_embeddings, skill_embeddings)
    
    mapped_keywords = []
    for i, kw in enumerate(keywords):
        if kw in skill2idx:
            mapped_keywords.append(kw)
        else:
            max_sim_idx = np.argmax(similarities[i])
            max_sim = similarities[i][max_sim_idx]
            if max_sim >= threshold:
                mapped_kw = skill_list[max_sim_idx]
                print(f"🔄 Mapped '{kw}' to '{mapped_kw}' (similarity: {max_sim:.3f})")
                mapped_keywords.append(mapped_kw)
            else:
                print(f"❌ Dropped '{kw}' (max similarity: {max_sim:.3f} < {threshold})")
    
    return list(set(mapped_keywords))

def predict_edges_from_text(text, model_path, le, skill2idx, skill_list, use_semantic_mapping=True):
    # Extract keywords
    keywords = extract_keywords(text)
    print("🔍 Extracted Keywords:", keywords)

    # Filter valid keywords and optionally map unmatched ones
    valid_keywords = [kw for kw in keywords if kw in skill2idx]
    # print("✅ Valid Keywords (in skill2idx):", valid_keywords)
    # print("❌ Invalid Keywords:", [kw for kw in keywords if kw not in skill2idx])

    if use_semantic_mapping:
        valid_keywords = map_unmatched_keywords(keywords, skill_list, skill2idx)
        print("🔍 Keywords after semantic mapping:", valid_keywords)

    if len(valid_keywords) < 2:
        print("⚠️ Not enough valid keywords to form edges. Need at least 2, got:", valid_keywords)
        return

    # Create node indices for valid keywords
    keyword_indices = [skill2idx[kw] for kw in valid_keywords]
    edge_index = torch.tensor(list(combinations(range(len(valid_keywords)), 2)), dtype=torch.long).t().contiguous()

    # Use full feature matrix from training
    x_full = torch.tensor(embedder.encode(skill_list), dtype=torch.float)
    x_full = torch.nn.functional.normalize(x_full, p=2, dim=1)

    # Load model
    model = load_model(model_path, in_channels=x_full.size(1), hidden_channels=32, out_channels=16, num_edge_labels=len(le.classes_))

    # Subset features for valid keywords
    x_subset = x_full[keyword_indices]

    # Run model
    with torch.no_grad():
        embeddings = model(x_subset, edge_index)
        src, dst = edge_index
        preds = model.classify(embeddings[src], embeddings[dst])
        pred_labels = preds.argmax(dim=1)

    theskills = []

    # Print predictions
    for (i, j), label_idx in zip(edge_index.t().numpy(), pred_labels):
        role_label = le.inverse_transform([label_idx.item()])[0]
        print(f"💡 Edge: ({valid_keywords[i]}, {valid_keywords[j]}) → Predicted Role: {role_label}")
        theskills.extend([valid_keywords[i], valid_keywords[j]])

    return list(set(theskills))

# # Example usage
# new_summary = """

# """

# le = joblib.load("label_encoder.pkl")
# skill2idx = joblib.load("skill2idx.pkl")
# skill_list = sorted(list(skill2idx.keys()))

# predict_edges_from_text(new_summary, "graphsage_model.pth", le, skill2idx, skill_list, use_semantic_mapping=True)