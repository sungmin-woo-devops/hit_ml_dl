import os
import pandas as pd

path = os.getcwd() + "/data"
files = os.listdir(path)

for file in files:
    if file.split('.')[1] == "csv":
        csv_path = os.path.join(path, file)
        df = pd.read_csv(csv_path)
        pkl_filename = file.replace('.csv', '.pkl')
        pkl_path = os.path.join(path, pkl_filename)

        df.to_pickle(pkl_path)
