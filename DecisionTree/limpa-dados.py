import matplotlib.pyplot as plt
import kagglehub
import pandas as pd
from kagglehub import KaggleDatasetAdapter
from io import StringIO
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

#Load DataBase
df = pd.read_csv('https://raw.githubusercontent.com/EnricoDiGioia/Machine-Learning/refs/heads/main/data/audi.csv')
df = df.sample(n=10, random_state=42)

#Display the first few rows
print(df.to_markdown(index=False))