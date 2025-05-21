import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from pickle import dump
import nltk
import os

# Configurar NLTK
nltk.download('stopwords', quiet=True)
nltk.data.path.append(os.path.join(os.getcwd(), "nltk_data"))

# Funções de pré-processamento
def preprocessar_texto(texto):
    texto = str(texto)
    texto = re.sub(r'[^a-zA-Z0-9\s]', " ", texto, flags=re.IGNORECASE)  # Remove pontuação
    texto = re.sub(r'\s+', " ", texto.lower().strip())
    return texto.split()

def limpar_texto(palavras):
    stemmer = SnowballStemmer('english')
    stop_words = set(stopwords.words('english'))
    tokens = [stemmer.stem(palavra) for palavra in palavras if palavra not in stop_words and len(palavra) > 2]
    return " ".join(tokens)

# Carregar dados
dados = pd.read_csv("data/spam.csv", encoding='latin1')
dados = dados[['v1', 'v2']]  # Ajustado para colunas do spam.csv (v1=type, v2=text)
dados = dados.rename(columns={'v1': 'type', 'v2': 'text'})  # Renomear para compatibilidade
dados['type'] = dados['type'].map({'ham': 0, 'spam': 1})

# Verificar balanceamento
print("Distribuição das classes:\n", dados['type'].value_counts())

# Pré-processar textos
dados['clean_text'] = dados['text'].apply(preprocessar_texto).apply(limpar_texto)

# Criar vetorizador (equivalente à DocumentTermMatrix com termos frequentes)
vetorizador = CountVectorizer(max_features=5000, binary=True)
X = vetorizador.fit_transform(dados['clean_text']).toarray()
y = dados['type'].values

# Dividir dados (70/30 como no script R)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

# Treinar modelo Naive Bayes com suavização Laplace
modelo = MultinomialNB(alpha=1.0)  # alpha=1.0 para suavização Laplace
modelo.fit(X_train, y_train)

# Avaliar modelo
previsoes = modelo.predict(X_test)
acuracia = accuracy_score(y_test, previsoes)
print("Acurácia:", acuracia)
print("Relatório de Classificação:\n", classification_report(y_test, previsoes))
print("Matriz de Confusão:\n", confusion_matrix(y_test, previsoes))

# Salvar modelo e vetorizador
os.makedirs("models", exist_ok=True)
dump(modelo, open("models/detector_spam.sav", "wb"))
dump(vetorizador, open("models/tfidf_vectorizer.sav", "wb"))  # Nome mantido por compatibilidade

# Salvar X_test e y_test
os.makedirs("data/processed", exist_ok=True)
pd.DataFrame(X_test).to_csv("data/processed/X_teste.csv", index=False)
pd.DataFrame(y_test, columns=["type"]).to_csv("data/processed/y_teste.csv", index=False)

print("Modelo, vetorizador, X_teste e y_teste salvos com sucesso!")
