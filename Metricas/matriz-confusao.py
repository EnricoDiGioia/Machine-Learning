import matplotlib.pyplot as plt
import kagglehub
import pandas as pd
from kagglehub import KaggleDatasetAdapter
from io import StringIO
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
import numpy as np

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

# Criar matriz de confusão
cm = confusion_matrix(y_test, y_pred)
class_names = ['Diesel', 'Hybrid', 'Petrol']

print("\nMatriz de Confusão:")
print("Linhas: Condição Real (Valores Verdadeiros)")
print("Colunas: Condição Predita (Valores Preditos pelo Modelo)")
print("Diagonal: Verdadeiros Positivos | Fora da Diagonal: Falsos Positivos/Negativos")
print(f"{'':>10} {'Diesel':>8} {'Hybrid':>8} {'Petrol':>8}")
for i, actual in enumerate(class_names):
    print(f"{actual:>10} ", end="")
    for j in range(len(class_names)):
        print(f"{cm[i,j]:>8}", end="")
    print()

# Plotar matriz de confusão
plt.figure(figsize=(10, 8))
im = plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title('Matriz de Confusão\n(Diagonal = Verdadeiros Positivos, Fora = Falsos Positivos/Negativos)', 
          fontsize=14, fontweight='bold', pad=20)
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
plt.ylabel('Condição Real\n(Valores Verdadeiros)', fontweight='bold', fontsize=12)
plt.xlabel('Condição Predita\n(Valores Preditos pelo Modelo)', fontweight='bold', fontsize=12)

# Adicionar valores na matriz
thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    # Determinar se é VP, FP ou FN
    if i == j:
        label = f"{cm[i, j]}\n(VP)"  # Verdadeiro Positivo
    else:
        label = f"{cm[i, j]}\n(FP/FN)"  # Falso Positivo/Falso Negativo
    
    plt.text(j, i, label,
             ha="center", va="center",
             color="white" if cm[i, j] > thresh else "black",
             fontweight='bold', fontsize=10)

# Adicionar legenda explicativa
legend_text = ("VP = Verdadeiro Positivo (predição correta)\n"
               "FP = Falso Positivo (predito como positivo, mas é negativo)\n"
               "FN = Falso Negativo (predito como negativo, mas é positivo)")
plt.figtext(0.02, 0.02, legend_text, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

plt.tight_layout()

# Para imprimir na página HTML
buffer = StringIO()
plt.savefig(buffer, format="svg", bbox_inches='tight')
print(buffer.getvalue())