import nltk
import os

# Definir o diretório para salvar os dados do NLTK
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Baixar os recursos
nltk.download("wordnet", download_dir=nltk_data_dir)
nltk.download("stopwords", download_dir=nltk_data_dir)
