import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Carregar o dataset audi.csv diretamente da URL
url = "https://raw.githubusercontent.com/EnricoDiGioia/Machine-Learning/refs/heads/main/data/audi.csv"
df = pd.read_csv(url)

# Remover espaços em branco dos nomes das colunas (se houver)
df.columns = df.columns.str.strip()

# Exemplo: prever o modelo do carro ('model') a partir das demais colunas
# Separar X e y
y = df['model']
X = df.drop('model', axis=1)

# Codificar variáveis categóricas
for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].str.strip()  # remover espaços
    X[col] = LabelEncoder().fit_transform(X[col])

# Codificar o alvo
le_y = LabelEncoder()
y_encoded = le_y.fit_transform(y)

# Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Treinar o modelo
rf = RandomForestClassifier(n_estimators=100, max_depth=5, max_features='sqrt', random_state=42)
rf.fit(X_train, y_train)

# Avaliar
predictions = rf.predict(X_test)
acc = accuracy_score(y_test, predictions)
print(f"Accuracy: {acc}")

# Importância das features
importances = rf.feature_importances_
print(f"Feature Importances: {importances}")