import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

# Group skills per job
job_to_skills = edges_df.groupby("job_id")["skill_id"].apply(list).reset_index()
job_to_skills.columns = ["job_id", "skill_ids"]

# Merge with jobs to get title + description
data_df = jobs_df.merge(job_to_skills, left_on="id", right_on="job_id", how="inner")

print(f"Jobs with at least one skill: {len(data_df)}")
print(f"Total unique skills: {len(skills_df)}")

mlb = MultiLabelBinarizer(classes=skills_df["id"].tolist())
label_matrix = mlb.fit_transform(data_df["skill_ids"])

print(f"Lable matrix shape: {label_matrix.shape}")

# --- Train / Val / Test split ---
X = data_df["title"].tolist()          # input text — title only for now
y = label_matrix

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test     = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")