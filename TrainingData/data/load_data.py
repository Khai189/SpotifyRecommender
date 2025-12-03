import os
import torchaudio
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import text
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt
import psycopg2
files = [file for file in os.listdir("mp3_files")]
df = pd.read_csv('dataset.txt', sep=',', header=0)
print(df.columns)

engine = create_engine('postgresql://husamkm:#Kha704293@localhost/spotify', echo=False)
df.to_csv('dataset.csv', sep=',',index=False, encoding='utf-8')
