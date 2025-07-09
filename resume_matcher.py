import pandas as pd
from utils.matcher import ResumeMatcher

job_df = pd.read_csv('/content/Job_descriptions.csv')
job_df.columns = job_df.columns.str.strip()

matcher = ResumeMatcher()
matcher.fit(job_df['Job Description'], job_df['Job Title'])

resume_df = pd.read_csv('/content/Sample_resumes.csv')
results = matcher.predict(resume_df['Resume Text'])

print(results)

import os
os.makedirs('outputs', exist_ok=True)

results.to_csv('outputs/matches_output.csv', index=False)
print("✅ Saved to outputs/matches_output.csv")
