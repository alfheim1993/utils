import pandas as pd
import os,glob,shutil

data = pd.read_csv(r"E:\0-data\bur\gpm\20200203-gpm\TFT1111.csv", header=None)
codes = list(set(data[1]))
from_dir = r'E:\0-data\bur\gpm\20200203-gpm\code\TFT(OK&NG)\TFT1111'
to_dir = r'E:\0-data\bur\gpm\20200203-gpm\to'
for code in codes:
	os.makedirs(os.path.join(to_dir, code), exist_ok=True)

for i, row in data.iterrows():
	print(i)
	name = row[0]
	code = row[1]
	shutil.copy(os.path.join(from_dir, name), os.path.join(to_dir, code, name))