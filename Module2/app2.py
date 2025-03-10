# ----------------------------------------------
# импортирование библиотек
# для реализации интерфейса
import streamlit as st
# для работы с cuda
import torch
# для работы с трансформенными нейросетями
from transformers import T5ForConditionalGeneration, T5Tokenizer
# вспомогательные функции
from function import get_summary

# ----------------------------------------------
# дополнительные переменные
# устройство для вычисления
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# путь к папке с данными по модели
model_path = 'saved_model'

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
st.header('Модель суммаризации')
# поле для ввода текста
text_area = st.text_area(label='Введите текст (не более 30.000 символов)', max_chars=30000)
# если нажата кнопка
if st.button('Суммаризировать'):
    # проверяем введен ли текст
    if text_area:
        # получаем модель и токенизатор
        model, tokenizer = load_summarization()
        # получаем суммаризацию статьи
        summary = get_summary(text_area, tokenizer, model, device)
        # выводим суммаризацию на экран
        st.write(summary)
    # если пользователь не ввел текст, но нажал на кнопку
    else:
        # пишем пользователю чтобы ввел текст
        st.write('Пожалуйста, введите текст статьи!')