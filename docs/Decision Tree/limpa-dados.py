import matplotlib.pyplot as plt
import kagglehub
import pandas as pd
from kagglehub import KaggleDatasetAdapter
from io import StringIO
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

file_path = kagglehub.dataset_download("adityadesai13/used-car-dataset-ford-and-mercedes")

df = pd.read_csv(file_path + "/audi.csv")  # Adjust filename as needed
df = df.sample(n=10, random_state=10)

print(df.to_markdown(index=False))