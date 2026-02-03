import pandas as pd


df = pd.read_parquet('/content/sample_data/job_skill_val.parquet', engine='pyarrow')

include_keywords = ["software", "engineer", "it", "manager", "data",
            "scientist","analyst","technical","tech",
            "cloud","devops","design","program"]

exclude_keywords = [
    "nurse",
    "nursing",
    r"\brn\b",
    "registered nurse",
    "clinical care",
    "community care",
    "healthcare",
    "medical"
]

include_pattern = r"\b(" + "|".join(include_keywords) + r")\b"
exclude_pattern = r"\b(" + "|".join(exclude_keywords) + r")\b"

include_mask = df["vacancy_job_title"].str.contains(
    include_pattern,
    case=False,
    na=False,
    regex=True
)

exclude_mask = df["vacancy_job_title"].str.contains(
    exclude_pattern,
    case=False,
    na=False,
    regex=True
)

df_it = df[include_mask & ~exclude_mask].copy()

