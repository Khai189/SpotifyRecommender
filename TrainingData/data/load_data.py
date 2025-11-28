import os
import torchaudio
import pandas as pd
files = [file for file in os.listdir("mp3_files")]
data = pd.DataFrame(files)
print(data)