import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv("data/patients.csv")

print("Shape:", df.shape)
print("Columns:", list(df.columns))

profile = ProfileReport(df, title="Pandas Profiling Report")
profile.to_file("output/eda/patients_profile.html")
