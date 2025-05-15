import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report
from pickle import dump
import nltk
import os

# Configurar NLTK
nltk.data.path.append(os.path.join(os.getcwd(), "nltk_data"))

# Funções de pré-processamento
def preprocessar_texto(texto):
    texto = str(texto)
    texto = re.sub(r'[^a-zA-Z0-9\s!$]', " ", texto, flags=re.IGNORECASE)
    texto = re.sub(r'\s+', " ", texto.lower().strip())
    return texto.split()

def lematizar_texto(palavras):
    lematizador = WordNetLemmatizer()
    palavras_ignoradas = set(stopwords.words("english"))
    tokens = [lematizador.lemmatize(palavra) for palavra in palavras]
    tokens = [palavra for palavra in tokens if palavra not in palavras_ignoradas and len(palavra) > 3]
    return tokens

# Carregar dados
dados = pd.read_csv("data/spam.csv", encoding='latin1')
dados = dados[['v1', 'v2']]
dados['v1'] = dados['v1'].map({'ham': 0, 'spam': 1})

# Verificar balanceamento
print("Distribuição das classes:\n", dados['v1'].value_counts())

# Pré-processar textos
dados["tokens"] = dados["v2"].apply(preprocessar_texto).apply(lematizar_texto)
lista_tokens = [" ".join(tokens) for tokens in dados["tokens"]]

# Criar vetorizador TF-IDF com bigrams
vetorizador = TfidfVectorizer(max_features=3000, max_df=0.8, min_df=5, ngram_range=(1, 2))
X_text = vetorizador.fit_transform(lista_tokens).toarray()

# Adicionar features de comprimento e URLs
url_length = dados["tokens"].apply(len)
word_count = dados["tokens"].apply(len)
has_url = dados["v2"].apply(lambda x: 1 if 'http' in x.lower() or 'www' in x.lower() else 0)
X_length = np.array([url_length, word_count, has_url]).T
X = np.hstack((X_text, X_length))
y = dados["v1"].values

# Balancear dataset com SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Otimizar hiperparâmetros com GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20]
}
modelo = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(modelo, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
modelo = grid_search.best_estimator_
print("Melhores parâmetros:", grid_search.best_params_)

# Avaliar modelo
previsoes = modelo.predict(X_test)
print("Acurácia:", accuracy_score(y_test, previsoes))
print("Relatório de Classificação:\n", classification_report(y_test, previsoes))

# Salvar modelo e vetorizador
os.makedirs("models", exist_ok=True)
dump(modelo, open("models/detector_spam.sav", "wb"))
dump(vetorizador, open("models/tfidf_vectorizer.sav", "wb"))

# Salvar X_test
os.makedirs("data/processed", exist_ok=True)
pd.DataFrame(X_test).to_csv("data/processed/X_teste.csv", index=False)

print("Modelo, vetorizador e X_teste salvos com sucesso!")
