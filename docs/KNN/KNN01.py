import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.calibration import LabelEncoder
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns

plt.figure(figsize=(10, 6))

# Limpar e preparar os dados
def preprocess(df):
    # Fill missing values
    df['tax'].fillna(df['tax'].median(), inplace=True)
    df['mpg'].fillna(df['mpg'].median(), inplace=True)
    df['price'].fillna(df['price'].median(), inplace=True)

    # Convert categorical variables
    label_encoder = LabelEncoder()
    df['transmission'] = label_encoder.fit_transform(df['transmission'])
    df['fuelType'] = label_encoder.fit_transform(df['fuelType'])
    return df

# Generate synthetic dataset
df = pd.read_csv('https://raw.githubusercontent.com/EnricoDiGioia/Machine-Learning/refs/heads/main/data/audi.csv')
df = preprocess(df)
X = df[['price', 'tax', 'mpg', 'engineSize', 'mileage', 'year']]
y = df['fuelType']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")


df_plot = pd.DataFrame()
df_plot['price'] = (X['price']-X['price'].min())/(X['price'].max()-X['price'].min())
df_plot['tax'] = (X['tax']-X['tax'].min())/(X['tax'].max()-X['tax'].min())
df_plot['fuelType'] = y
sns.scatterplot(data=df_plot, x='price', y='tax', hue='fuelType')

