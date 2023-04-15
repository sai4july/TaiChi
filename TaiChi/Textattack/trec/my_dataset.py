import textattack
import pandas as pd
from transformers import BertTokenizer

data_path = ""
df = pd.read_csv(data_path,"\t")
dataset = []
for sen,label in zip(df['sentence'],df['label']):
    dataset.append((sen,label))
print(dataset[:5])
dataset = textattack.datasets.Dataset(dataset)