from zipfile import ZipFile

import pandas as pd

from sklearn.model_selection import train_test_split

with ZipFile("data/raw/AIchallenger2017.zip", 'r') as z:
    with z.open("AIchallenger2017/cmn.txt") as f:
        data = pd.read_table(f, sep="\t", header=None, usecols=[0, 1], names=["en", "cn"], encoding="utf-8")


train_df, test_df = train_test_split(data, test_size=0.2)
