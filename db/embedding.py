from sentence_transformers import SentenceTransformer
import torch
from google.colab import userdata
from supabase import create_client
from tqdm import tqdm
import os

url = userdata.get("SUPABASE_URL")
role = userdata.get("SUPABASE_SERVICE_ROLE_KEY")

supabase = create_client(
    url,
    role
)


model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

def embed_texts(texts, batch_size=32):

  return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,  # strongly recommended for similarity + GNNs
        show_progress_bar=True
  ).tolist()

# Inserting embeddings to jobs table


jobs = (
    supabase
    .table("jobs")
    .select("id, title, description")
    .is_("embedding", None)
    .execute()
).data

job_texts = [
    f"Job title: {j['title']}\nJob description: {j['description'] or ''}"
    for j in jobs
]

job_embeddings = embed_texts(job_texts)

updates = [
    {
        "id": job["id"],
        "embedding": emb
    }
    for job, emb in zip(jobs, job_embeddings)
]

for row in tqdm(updates):
  supabase.table("jobs").update(
      {"embedding": row["embedding"]}
  ).eq("id", row["id"]).execute()


#   Chunking strategy for skills table (>1000 rows)

def fetch_all_skills_without_embeddings(page_size=1000):
    all_rows = []
    offset = 0

    while True:
        res = (
            supabase
            .table("skills")
            .select("id, name")
            .is_("embedding", None)
            .range(offset, offset + page_size - 1)
            .execute()
        )

        batch = res.data
        if not batch:
            break

        all_rows.extend(batch)
        offset += page_size

    return all_rows

def chunked(lst, size=100):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]



skills = fetch_all_skills_without_embeddings()

print(f"Found {len(skills)} skills to embed")

for skill_batch in chunked(skills, size=100):
    # Prepare texts
    texts = [f"Skill: {s['name']}" for s in skill_batch]

    # Embed
    embeddings = embed_texts(texts)

    # Update each row
    for skill, emb in zip(skill_batch, embeddings):
        (
            supabase
            .table("skills")
            .update({"embedding": emb})
            .eq("id", skill["id"])
            .execute()
        )

