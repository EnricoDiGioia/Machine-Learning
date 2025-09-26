import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

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
recall = recall_score(y_test, y_pred, average='weighted')

# Criar tabela de métricas
metrics_data = {
    'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Valor': [accuracy, precision, recall, f1],
    'Percentual': [f'{accuracy*100:.2f}%', f'{precision*100:.2f}%', f'{recall*100:.2f}%', f'{f1*100:.2f}%'],
    'Descrição': [
        'Proporção de predições corretas',
        'Proporção de predições positivas corretas',
        'Proporção de positivos reais identificados',
        'Média harmônica entre Precision e Recall'
    ]
}

metrics_df = pd.DataFrame(metrics_data)

print("=== TABELA DE MÉTRICAS DE AVALIAÇÃO ===")
print(metrics_df.to_string(index=False, formatters={'Valor': '{:.4f}'.format}))
print("\n" + "="*50)

# Criar visualização da tabela
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')

# Criar tabela visual
table_data = []
for _, row in metrics_df.iterrows():
    table_data.append([row['Métrica'], f"{row['Valor']:.4f}", row['Percentual'], row['Descrição']])

table = ax.table(cellText=table_data,
                colLabels=['Métrica', 'Valor', 'Percentual', 'Descrição'],
                cellLoc='center',
                loc='center',
                colWidths=[0.15, 0.15, 0.15, 0.55])

# Estilizar tabela
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2)

# Cores do cabeçalho
for i in range(4):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Cores alternadas nas linhas
colors = ['#F2F2F2', '#FFFFFF']
for i in range(1, 5):
    for j in range(4):
        table[(i, j)].set_facecolor(colors[(i-1) % 2])

# Destacar F1-Score
for j in range(4):
    table[(4, j)].set_facecolor('#E7F3FF')
    table[(4, j)].set_text_props(weight='bold')

plt.title('Métricas de Avaliação - Árvore de Decisão', 
          fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()

# Para imprimir na página HTML
buffer = StringIO()
plt.savefig(buffer, format="svg", bbox_inches='tight')
print(buffer.getvalue())