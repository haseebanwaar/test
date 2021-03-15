
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


dfs = []
files = [x for x in os.listdir(r'G:\d\ds') if str(x).endswith('.xlsx')]
for file in files:
    df = pd.read_excel("G:/d/ds/" + file, sheet_name=None, skiprows=1, header=1,
                       index_col=1)
    if len(df) < 8:
        d = df[8]
    else:
        print(file)
    f = d.dropna(how='all', axis=1)
    f.columns = [x.replace('\n', ' ') for x in f.columns]

    q = f.copy()
    # q.drop_duplicates(subset=['Sample Code'], keep=False, inplace=True)
    dfs.append(q)