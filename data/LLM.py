from groq import Groq
import os
import pandas as pd
from tqdm import tqdm
from google.colab import userdata
from supabase import create_client
import os

GROQ_API_KEY = userdata.get("GROQ_API")

client = Groq(api_key=GROQ_API_KEY)

def generate_job_description(title: str) -> str:
  response = client.chat.completions.create(
      model="llama-3.3-70b-versatile",
      messages=[
          {"role": "system", "content": SYSTEM_PROMPT},
          {"role": "user", "content": USER_PROMPT.format(job_title=title)}
      ],
      temperature=0.2,
      max_tokens=50,
  )
  return response.choices[0].message.content.strip()


SYSTEM_PROMPT = """You are an expert job taxonomy assistant.
Your task is to write a single factual sentence describing the job role.
Do not add skills, tools, or requirements not implied by the title.
Be concise and neutral.
"""

USER_PROMPT = """
Job title: "{job_title}"

Write exactly one sentence describing what this role generally does.
"""

unique_titles = df_it["vacancy_job_title"].dropna().unique()
print(f"Unique job titles: {len(unique_titles)}")

title_to_description = {}

for title in tqdm(unique_titles):
  title_to_description[title] = generate_job_description(title)

df_it["job_description"] = df_it["vacancy_job_title"].map(title_to_description)

# Uploading into supabase DB

url = userdata.get("SUPABASE_URL")
role = userdata.get("SUPABASE_SERVICE_ROLE_KEY")
supabase = create_client(
    url,
    role  # service role for inserts
)

# Jobs Table : holding uuid, title, description, embedding

jobs_df = (
    df_it[["vacancy_job_title", "job_description"]]
    .drop_duplicates()
    .rename(columns={
        "vacancy_job_title": "title",
        "job_description": "description"
    })
)

job_rows = jobs_df.to_dict(orient="records")

supabase.table("jobs").upsert(
    job_rows,
    on_conflict="title"
).execute()

job_titles = jobs_df["title"].tolist()

job_records = (
    supabase
    .table("jobs")
    .select("id, title")
    .in_("title", job_titles)
    .execute()
)

job_id_map = {r["title"]: r["id"] for r in job_records.data}


# Skills table : holding job name and skill

skills_df = (
    df_it[["skill"]]
    .dropna()
    .assign(skill=lambda x: x["skill"].str.strip().str.lower())
    .drop_duplicates()
    .rename(columns={"skill": "name"})
)

skill_rows = skills_df.to_dict(orient="records")

supabase.table("skills").upsert(
    skill_rows,
    on_conflict="name"
).execute()

def chunked(lst, size=100):
  for i in range(0, len(lst), size):
    yield lst[i:i+size]

skill_id_map = {}

skill_names = skills_df["name"].tolist()


for batch in chunked(skill_names, size=100):
  res = (
      supabase
      .table("skills")
      .select("id, name")
      .in_("name", batch)
      .execute()
  )
  for r in res.data:
    skill_id_map[r["name"]] = r["id"]

# Job Skills table : mapping job id to skill id

edges = []

for _, row in df_it.iterrows():
    edges.append({
        "job_id": job_id_map[row["vacancy_job_title"]],
        "skill_id": skill_id_map[row["skill"].strip().lower()]
    })

supabase.table("job_skills").upsert(
    edges,
    on_conflict="job_id,skill_id"
).execute()


