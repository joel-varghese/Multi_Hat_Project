# Evaluation and Inference

model.load_state_dict(torch.load("best_model.pt"))
test_loss, test_f1_micro, test_f1_macro = evaluate(test_loader)
print(f"\nTest | Loss: {test_loss:.4f} | F1-micro: {test_f1_micro:.4f} | F1-macro: {test_f1_macro:.4f}")

def predict_skills(title: str, top_k: int = 10, threshold: float = 0.5):
    """Given a job title, return the rpedicted skill names"""

    model.eval()
    encoding = tokenizer(
        title,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs  = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # (num_skills,)

    top_indices = np.argsort(probs)[::-1][:top_k]
    results = [
        {"skill": skills_df.iloc[i]["name"], "confidence": float(probs[i])}
        for i in top_indices
        if probs[i] >= threshold
    ]
    return results



# Generating prediction and example usage