from model.transformer import JobSkillClassifier
import json, pickle, torch
import numpy as np
from transformers import AutoModel, AutoTokenizer

class Predictor:
    def __init__(self, model_dir: str = "models/v1"):
        config = json.load(open(f"{model_dir}/model_config.json"))
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.config    = config
        self.tokenizer = AutoTokenizer.from_pretrained(config["base_model"])

        self.model = JobSkillClassifier(config["base_model"], config["num_labels"])
        self.model.load_state_dict(
            torch.load(f"{model_dir}/best_model.pt", map_location=self.device)
        )
        self.model.eval()
        self.model.to(self.device)

        with open(f"{model_dir}/mlb.pkl", "rb") as f:
            self.mlb = pickle.load(f)

        skills_raw  = json.load(open(f"{model_dir}/skills_lookup.json"))
        self.skills = {s["id"]: s["name"] for s in skills_raw}

    def predict(self, title: str, description: str = "",
                top_k: int = 10, threshold: float = None):
        
        threshold = threshold or self.config["threshold"]
        text      = f"{title} [SEP] {description}" if description else title
        # bge_input = f"Represent this job title for skill classification: {text}"

        enc = self.tokenizer(
            text,
            max_length=self.config["max_len"],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            logits = self.model(
                enc["input_ids"].to(self.device),
                enc["attention_mask"].to(self.device)
            )
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        top_idx = np.argsort(probs)[::-1][:top_k]

        return [
            {
                "skill_id":   self.mlb.classes_[i],
                "skill_name": self.skills.get(self.mlb.classes_[i], "unknown"),
                "confidence": round(float(probs[i]), 4),
            }
            for i in top_idx if probs[i] >= threshold
        ]