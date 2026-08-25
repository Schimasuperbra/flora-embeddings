import re
import pandas as pd

txt_path = "your_log_0618.txt"

with open(txt_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

records = []
current_err = None

for line in lines:
    err_match = re.search(r"Test Err:\s*([0-9.]+)", line)
    if err_match:
        current_err = float(err_match.group(1))

    iter_match = re.search(r"iter:\s*(\d+)", line)
    if iter_match and current_err is not None:
        records.append({
            "iter": int(iter_match.group(1)),
            "test_err": current_err
        })
        current_err = None

df = pd.DataFrame(records)
print(df)

df.to_csv("extracted_err_iter.csv", index=False)