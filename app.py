import streamlit as st
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from pickle import load
import nltk
import altair as alt
import os

# Configurar NLTK
# Remova ou comente a linha abaixo se usar dados locais
# nltk.download('stopwords', quiet=True)
nltk.data.path.append(os.path.join(os.path.dirname(__file__), "nltk_data"))

# Configuração da página
st.set_page_config(page_title="Detetor de Spam em E-mail", page_icon="📱")

# Estilo CSS
st.markdown("""
    <style>
    /* Fundo e layout geral */
    .main {
        background-color: #red;
        color: #white;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }

    /* Títulos e texto */
    h1, h2, h3 {
        color: #1e3a8a; /* Azul escuro */
        font-weight: 600;
        text-align: center;
    }
    .stMarkdown p, .stMarkdown div {
        color: #white;
        line-height: 1.6;
    }            

    /* Área de texto */
    .stTextArea textarea {
        background-color: #ffffff;
        color: #1f2937;
        border: 1px solid #d1d5db;
        border-radius: 8px;
        padding: 10px;
        font-size: 16px;
    }
    .stTextArea label {
        color: #1e3a8a;
        font-weight: 500;
    }

    /* Resultados */
    .spam-result {
        background-color: #fee2e2;
        padding: 15px;
        border-radius: 8px;
        color: #dc2626;
        font-weight: 600;
        text-align: center;
        margin: 10px 0;
    }
    .safe-result {
        background-color: #d1fae5;
        padding: 15px;
        border-radius: 8px;
        color: #065f46;
        font-weight: 600;
        text-align: center;
        margin: 10px 0;
    }

    /* Botões */
    .stButton>button {
        background-color: #1e3a8a;
        color: red;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #164e91;
    }

    /* Alertas */
    .stAlert {
        border-radius: 8px;
        padding: 15px;
        font-size: 16px;
    }
    .stWarning {
        background-color: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
    }
    .stError {
        background-color: #fee2e2;
        color: #dc2626;
        border: 1px solid #fecaca;
    }

    /* Sidebar */
    .css-1aumxhk {
        background-color: #4a90e2;
        color: white;
    
        border-right: 1px solid #e5e7eb;
        padding: 20px;
        
    }
    </style>
""", unsafe_allow_html=True)

# Função para adicionar features (não necessária, mas mantida)
def adicionar_features_texto(texto, tokens):
    return len(tokens), len(tokens), 0, 0  # Placeholder

# Funções de pré-processamento com cache
@st.cache_data
def preprocessar_texto(texto):
    texto = str(texto)
    texto = re.sub(r'[^a-zA-Z0-9\s]', " ", texto, flags=re.IGNORECASE)
    texto = re.sub(r'\s+', " ", texto.lower().strip())
    return texto.split()

@st.cache_data
def limpar_texto(palavras):
    stemmer = SnowballStemmer('english')
    stop_words = set(stopwords.words('english'))
    tokens = [stemmer.stem(palavra) for palavra in palavras if palavra not in stop_words and len(palavra) > 2]
    return " ".join(tokens)

# Carregar modelo e vetorizador
@st.cache_resource
def load_model_and_vectorizer():
    try:
        BASE_DIR = os.path.dirname(__file__)
        modelo = load(open(os.path.join(BASE_DIR, "models/detector_spam.sav"), "rb"))
        vetorizador = load(open(os.path.join(BASE_DIR, "models/tfidf_vectorizer.sav"), "rb"))
        return modelo, vetorizador
    except Exception as e:
        st.error(f"Erro ao carregar modelo ou vetorizador: {e}")
        return None, None

modelo, vetorizador = load_model_and_vectorizer()

# Título
st.title("📱 Detetor de Spam em E-mail")

# Abas
tab1, tab2 = st.tabs(["Classificador", "Métricas"])

with tab1:
    st.header("📥 Insira sua Mensagem")
    user_input = st.text_area("Digite o texto da mensagem para verificar:", height=100, key="user_input")

    with st.spinner("Classificando..."):
        if user_input and modelo and vetorizador:
            processed_input = preprocessar_texto(user_input)
            clean_input = limpar_texto(processed_input)
            X_text = vetorizador.transform([clean_input]).toarray()
            prediction = modelo.predict(X_text)[0]
            prob_score = modelo.predict_proba(X_text)[0][1]

            st.header("Resultado")
            if prediction == 1:
                st.markdown(f'<div class="spam-result">⚠️ **Spam Detectado!** Esta mensagem parece suspeita.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="safe-result">✅ **Seguro!** Esta mensagem não é spam.</div>', unsafe_allow_html=True)

            st.subheader("Score de Confiança")
            st.write(f"Probabilidade de ser Spam: {prob_score:.4f}")
            chart_data = pd.DataFrame({
                'Classe': ['Não Spam', 'Spam'],
                'Score': [1 - prob_score, prob_score]
            })
            chart = alt.Chart(chart_data).mark_bar().encode(
                x='Classe',
                y='Score',
                color='Classe',
                tooltip=['Classe', 'Score']
            ).properties(width=400, height=300)
            st.altair_chart(chart, use_container_width=True)

            st.subheader("Nuvem de Palavras")
            try:
                wordcloud = WordCloud(
                    width=400, height=400,
                    background_color="white",
                    max_words=50,
                    min_font_size=10,
                    random_state=42
                ).generate(user_input)
                fig, ax = plt.subplots()
                ax.imshow(wordcloud, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)
            except ValueError:
                st.warning("Texto insuficiente para gerar nuvem de palavras.")
            
            result = pd.DataFrame({
                'Mensagem': [user_input],
                'Classificação': ['Spam' if prediction == 1 else 'Não Spam'],
                'Probabilidade de Spam': [prob_score]
            })
            csv = result.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Baixar Resultado",
                data=csv,
                file_name="resultado_spam.csv",
                mime="text/csv"
            )
        else:
            st.warning("Por favor, insira uma mensagem válida ou verifique se o modelo foi carregado corretamente.")

with tab2:
    st.header("📊 Métricas do Modelo")
    try:
        BASE_DIR = os.path.dirname(__file__)
        X_teste = pd.read_csv(os.path.join(BASE_DIR, "data/processed/X_teste.csv")).values
        y_teste = pd.read_csv(os.path.join(BASE_DIR, "data/processed/y_teste.csv"))["type"].values

        st.write(f"Número de amostras em X_teste: {X_teste.shape[0]}")
        st.write(f"Número de amostras em y_teste: {len(y_teste)}")

        if X_teste.shape[0] != len(y_teste):
            st.warning("Tamanhos de X_teste e y_teste não coincidem! Tentando ajustar...")
            min_size = min(X_teste.shape[0], len(y_teste))
            X_teste = X_teste[:min_size]
            y_teste = y_teste[:min_size]
            st.write(f"Tamanhos ajustados - Novo tamanho: {min_size}")

        previsoes = modelo.predict(X_teste)
        acuracia = accuracy_score(y_teste, previsoes)

        st.write(f"**Acurácia**: {acuracia:.4f}")
        st.subheader("Relatório de Classificação")
        report = classification_report(y_teste, previsoes, target_names=["Não Spam", "Spam"], output_dict=True)
        st.write(pd.DataFrame(report).transpose())

        st.subheader("Matriz de Confusão")
        cm = confusion_matrix(y_teste, previsoes)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Não Spam", "Spam"], yticklabels=["Não Spam", "Spam"], ax=ax)
        ax.set_xlabel("Previsto")
        ax.set_ylabel("Real")
        st.pyplot(fig)

        if st.button("Exibir Detalhes Adicionais"):
            st.write(f"**Número Total de Amostras Teste:** {len(y_teste)}")
            st.write(f"**Proporção de Spam:** {(y_teste.sum() / len(y_teste)):.2%}")
            st.write(f"**Proporção de Não Spam:** {((1 - y_teste).sum() / len(y_teste)):.2%}")

    except Exception as e:
        st.warning(f"Erro ao carregar métricas: {e}")
        st.warning("O arquivo y_teste.csv não foi encontrado. Por favor, reexecute o treinamento with 'time python treinar_modelo.py'.")
        try:
            st.write("Tentando calcular métricas com dados de fallback...")
            dados = pd.read_csv(os.path.join(BASE_DIR, "data/spam.csv"), encoding='latin1')
            dados = dados[['v1', 'v2']].rename(columns={'v1': 'type', 'v2': 'text'}).sample(n=500, random_state=42)
            dados['type'] = dados['type'].map({'ham': 0, 'spam': 1})
            dados['clean_text'] = dados['text'].apply(preprocessar_texto).apply(limpar_texto)
            X_text = vetorizador.transform(dados['clean_text']).toarray()
            y_fallback = dados['type'].values
            previsoes_fallback = modelo.predict(X_text)
            acuracia_fallback = accuracy_score(y_fallback, previsoes_fallback)
            st.write(f"**Acurácia (Fallback)**: {acuracia_fallback:.4f}")
            report_fallback = classification_report(y_fallback, previsoes_fallback, target_names=["Não Spam", "Spam"], output_dict=True)
            st.write(pd.DataFrame(report_fallback).transpose())
            cm_fallback = confusion_matrix(y_fallback, previsoes_fallback)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm_fallback, annot=True, fmt="d", cmap="Blues", xticklabels=["Não Spam", "Spam"], yticklabels=["Não Spam", "Spam"], ax=ax)
            ax.set_xlabel("Previsto")
            ax.set_ylabel("Real")
            st.pyplot(fig)
            if st.button("Exibir Detalhes Adicionais (Fallback)"):
                st.write(f"**Número Total de Amostras Fallback:** {len(y_fallback)}")
                st.write(f"**Proporção de Spam:** {(y_fallback.sum() / len(y_fallback)):.2%}")
                st.write(f"**Proporção de Não Spam:** {((1 - y_fallback).sum() / len(y_fallback)):.2%}")
        except Exception as e2:
            st.error(f"Falha no cálculo de métricas de fallback: {e2}")

# Sidebar com resumo
st.sidebar.header("ℹ️ Sobre o Cientista")
st.sidebar.markdown("""
**Adriano Júlio**  
Estudante do 3º ano de Ciências da Computação na Universidade Mandume ya Ndemufayo(IPH), apaixonado por inteligência artificial e desenvolvimento de soluções tecnológicas. Este projeto foi desenvolvido como parte de estudos em machine learning(Aprendizagem Computacional), com foco em classificação de texto.

**Sobre o Projeto**  
Este trabalho implementa um detetor de spam em SMS usando um modelo **Naive Bayes** otimizado. O modelo utiliza:  
- **Pré-processamento com stemming e remoção de stop words** para limpar o texto.  
- **Matriz de contagem de termos** para representar o texto como features.  
- **Suavização Laplace** para lidar com palavras raras e melhorar a precisão.  
- A interface é construída com **Streamlit**, oferecendo visualizações interativas como nuvem de palavras e matriz de confusão.

**Motivação**  
O projeto visa combater mensagens de spam, que são um problema crescente em comunicações via SMS. Com uma acurácia de até 98%, ele oferece uma ferramenta prática para filtragem em tempo real.

**Tecnologias Utilizadas**  
- **Python**: Linguagem principal.  
- **NLTK**: Para pré-processamento de texto.  
- **Scikit-learn**: Para o modelo Naive Bayes.  
- **Streamlit**: Para a interface web interativa.  
- **Pandas & NumPy**: Para manipulação de dados.  
- **Matplotlib & Seaborn**: Para visualizações.

**Contato:**  
- Email: adrianojulio487@gmail.com  
- Telefone: 948840759  
""")
