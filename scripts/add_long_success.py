import pandas as pd

# Load your dataset
df = pd.read_csv(r"C:\Trading\Projects\ES_AI_Project\data\labels\labeled.csv")

# Create LongSuccess label
df["LongSuccess"] = (df["FutureRet"] > 0).astype(int)

# Save updated dataset
df.to_csv(r"C:\Trading\Projects\ES_AI_Project\data\labels\labeled.csv", index=False)

print("LongSuccess added successfully.")
