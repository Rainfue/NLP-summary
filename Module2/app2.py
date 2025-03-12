# ----------------------------------------------
# импортирование библиотек
# для реализации интерфейса
import streamlit as st
# для работы с cuda
import torch
# для работы с трансформенными нейросетями
from transformers import T5ForConditionalGeneration, T5Tokenizer
# вспомогательные функции
from function import get_summary, get_similarity
# модель сравнивания
from gensim.models.doc2vec import Doc2Vec

# ----------------------------------------------
# дополнительные переменные
# устройство для вычисления
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# путь к папке с данными по модели
model_path = 'Module2/saved_model'
# модель сравнивания
model_similarity = Doc2Vec.load('Module2/doc2vec.model')

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
# подзаголовок странциы
# st.header('Модель суммаризации')
# # поле для ввода текста
# text_area = st.text_area(label='Введите текст (не более 30.000 символов)', max_chars=30000)
# # если нажата кнопка
# if st.button('Суммаризировать'):
#     # проверяем введен ли текст
#     if text_area:
#         # получаем модель и токенизатор
#         model, tokenizer = load_summarization()
#         # получаем суммаризацию статьи
#         summary = get_summary(text_area, tokenizer, model, device)
#         # выводим суммаризацию на экран
#         st.write(summary)
#     # если пользователь не ввел текст, но нажал на кнопку
#     else:
#         # пишем пользователю чтобы ввел текст
#         st.write('Пожалуйста, введите текст статьи!')


# # Сравнение статей
# st.header("Сравнение статей")








# создаем две колонки
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
        if len(text1) < 500 or len(text2) < 500:
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

            # получаем суммаризацию второй статьи
            with col2:
                summary2 = get_summary(text2, tokenizer, model, device)
                # выводим суммаризацию
                st.write(summary2)

            similarity = get_similarity(summary1, summary2, model_similarity)
            st.write(f'Схожесть статей: {similarity*100:.2f}')


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


