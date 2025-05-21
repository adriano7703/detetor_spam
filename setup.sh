#!/bin/bash
# Verificar se nltk_data/stopwords existe, senão baixar
if [ ! -d "/mount/src/detetor_spam/nltk_data/stopwords" ]; then
    python -m nltk.downloader -d /mount/src/detetor_spam/nltk_data stopwords
fi
