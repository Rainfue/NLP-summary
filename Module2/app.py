import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
from function import *
import torch

# Загрузка моделей
@st.cache_resource
def load_summarization_model():
    # Загрузка модели
    model = T5ForConditionalGeneration.from_pretrained("Module2/saved_model")
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    # Загрузка токенизатора
    tokenizer = T5Tokenizer.from_pretrained("Module2/saved_model")
    return model, tokenizer

# @st.cache_resource
# def load_similarity_model():
#     return SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# # Функция для суммаризации
# def summarize(text, model, tokenizer):
#     inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
#     outputs = model.generate(inputs["input_ids"], max_length=50)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# # Функция для сравнения статей
# def compare_articles(article1, article2, model):
#     embeddings = model.encode([article1, article2])
#     similarity = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
#     return similarity

# Интерфейс
st.title("Научные статьи: суммаризация и сравнение")

# Суммаризация
st.header("Суммаризация статьи")
article_text = st.text_area("Введите текст статьи:")

if st.button("Суммаризировать"):
    model, tokenizer = load_summarization_model()

    summary = get_summary(article_text, tokenizer, model)
    print(summary)
    st.write(summary)

# Сравнение статей
# st.header("Сравнение статей")
# article1 = st.text_area("Введите текст первой статьи:")
# article2 = st.text_area("Введите текст второй статьи:")
# if st.button("Сравнить"):
#     pass
    # model = load_similarity_model()
    # similarity = compare_articles(article1, article2, model)
    # st.write("Схожесть статей:", similarity)