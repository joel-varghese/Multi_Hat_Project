from groq import Groq
import os
import pandas as pd
from tqdm import tqdm
from google.colab import userdata

GROQ_API_KEY = userdata.get("GROQ_API")

client = Groq(api_key=GROQ_API_KEY)


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


title_to_description = {}

for title in tqdm(unique_titles):
  title_to_description[title] = generate_job_description(title)

df_it["job_description"] = df_it["vacancy_job_title"].map(title_to_description)

df_it = df_it.explode("tagged_esco_skills")
df_it.rename(columns={"tagged_esco_skills": "skill"}, inplace=True)