import matplotlib.pyplot as plt
import kagglehub
import pandas as pd
from kagglehub import KaggleDatasetAdapter
from io import StringIO
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
import seaborn as sns
import numpy as np


plt.figure(figsize=(16, 12))

# Preprocess the data
def preprocess(df):
    # Fill missing values
    df['tax'].fillna(df['tax'].median(), inplace=True)
    df['mpg'].fillna(df['mpg'].median(), inplace=True)
    df['price'].fillna(df['price'].median(), inplace=True)

    # Convert categorical variables
    label_encoder = LabelEncoder()
    df['transmission'] = label_encoder.fit_transform(df['transmission'])
    df['fuelType'] = label_encoder.fit_transform(df['fuelType'])

    # Select features
    features = ['model', 'year', 'price', 'transmission', 'mileage', 'fuelType', 'tax', 'mpg', 'engineSize']
    return df[features]

# Load the Audi dataset
df = pd.read_csv('https://raw.githubusercontent.com/EnricoDiGioia/Machine-Learning/refs/heads/main/data/audi.csv')

# Preprocessing
df = preprocess(df)

# Display the first few rows of the dataset
#print(df.sample(n=10, random_state=42).to_markdown(index=False))

# Carregar o conjunto de dados
x = df[['price', 'tax', 'mpg', 'engineSize', 'mileage', 'year']]
y = df['fuelType']

# Dividir os dados em conjuntos de treinamento e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

# Criar e treinar o modelo de árvore de decisão
classifier = tree.DecisionTreeClassifier()
classifier.fit(x_train, y_train)

# Fazer predições
y_pred = classifier.predict(x_test)

# Calcular métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1-Score: {f1:.4f}")

# Plotar a árvore de decisão
plt.figure(figsize=(16, 12))
tree.plot_tree(classifier, feature_names=['price', 'tax', 'mpg', 'engineSize', 'mileage', 'year'], 
               class_names=['Diesel', 'Hybrid', 'Petrol'], filled=True, rounded=True)
plt.title('Árvore de Decisão', fontsize=16, fontweight='bold')


# Para imprimir na página HTML
buffer = StringIO()
plt.savefig(buffer, format="svg", bbox_inches='tight')
print(buffer.getvalue())