# ----------------------------------------------
# импортирование библиотек
# для реализации интерфейса
import streamlit as st
# для работы с cuda
import torch
# для работы с трансформенными нейросетями
from transformers import T5ForConditionalGeneration, T5Tokenizer
# вспомогательные функции
from function import get_summary, get_similarity, find_top_similar
# модель сравнивания
from gensim.models.doc2vec import Doc2Vec
# для датафреймов
import pandas as pd

# ----------------------------------------------
# дополнительные переменные
# устройство для вычисления
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# путь к папке с данными по модели
model_path = 'Module2/saved_model'
# модель сравнивания
model_similarity = Doc2Vec.load('Module2/doc2vec.model')
# датафрейм с эмбеддингами
df = pd.read_pickle('Module2/all_embs.pkl')

# кэшируем модель и токенизатор
@st.cache_resource
def load_summarization():
    # загружаем модель суммаризации
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    # приводим модель к устройству вычисления
    model.to(device)
    # загружаем токенизатор модели
    tokenizer = T5Tokenizer.from_pretrained(model_path)

    # возвращаем полученные модель и токенизатор
    return model, tokenizer

# ----------------------------------------------
# реализация интерфейса
# главное название страницы
st.title('Суммаризация и сравнивание статей')

# создаю две колонки
col1, col2 = st.columns(2)

# добавляем контент в первую колонку
with col1:
    text1 = st.text_area("Введите текст первой статьи:")

# добавляем контент во вторую колонку
with col2:
    text2 = st.text_area("Введите текст второй статьи:")


similarity_button = st.button(
    label = 'Сравнить',
)

if similarity_button:
    # проверяем, ввел ли пользователь текста:
    if not text1 or not text2:
        # выводим то, что пользователь должен ввести второй текст
        st.write('Пожалуйста, введите обе статьи!')
    # TODO сделай чтобы для пропущенной 1 и 2х статей был разный вывод
    # если пользователь ввел обе статьи:
    else:
        # проверяем, ввел ли пользователь достаточно длинные тексты
        if len(text1) < 100 or len(text2) < 100:
            # выводим результат
            st.write('Пожалуйста, введите статьи не менее 500 символов длиной!')
        # если ввод корректный
        else:
            # получаем модель и токенизатор из кэша
            model, tokenizer = load_summarization()
            # получаем суммаризацию первой статьи
            with col1:
                summary1 = get_summary(text1, tokenizer, model, device)
                # выводим суммаризацию
                st.write(summary1)
                # получаем топ 3 самых похожих статей
                top_similarities, headers = find_top_similar(
                    df=df,
                    text=summary1,
                    model=model_similarity,
                )
                # выводим топ 3 самых похожих статей
                st.bar_chart(dict(zip(headers, top_similarities)))

            # получаем суммаризацию второй статьи
            with col2:
                summary2 = get_summary(text2, tokenizer, model, device)
                # выводим суммаризацию
                st.write(summary2)
                # получаем топ 3 самых похожих статей
                top_similarities, headers = find_top_similar(
                    df=df,
                    text=summary2,
                    model=model_similarity,
                )
                # выводим топ 3 самых похожих статей
                st.bar_chart(dict(zip(headers, top_similarities)))


            similarity = get_similarity(summary1, summary2, model_similarity)
            st.write(f'Схожесть статей: {similarity*100:.2f}%')


            if similarity:
                # Столбчатая диаграмма
                st.bar_chart(
                    [[similarity*100, 100]], 
                    x_label='Схожесть', 
                    color=['#4bd4ff', '#0E1117'], 
                    stack="layered", 
                    horizontal=True, 
                    use_container_width=True
                    )


