# ----------------------------------------------
# импортирование библиотек
# для реализации интерфейса
import streamlit as st
# для работы с cuda
import torch
# объект реализации
from backend import Realization
# для логирования времени
from time import time

summarization_path = 'saved_model'      # путь к модели суммаризации
similarity_path = 'doc2vec_e50_w2_mc1_dw0_dm1_.model'       # путь к модели сравнения
df_path = 'all_embs.pkl'                # путь к датафрейму с эмбеддингами
# девайс (CUDA или CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MyGUI:
    # метод инициализации
    def __init__(self):
        '''
        Иициализация приложения
        '''
        # устройство вычисления
        self.device = device
        # объект реализации
        self.realizer = Realization(summarization_path, similarity_path, df_path, device)
        # заголовок приложения
        self.title = st.title(r'$$\text{Приложение для сравнения статей}$$', 
                              help='Решение тестового задания для подготовки к чемпионату "Профессионалы"')
        # разделение на 2 колонки
        self.col1, self.col2 = st.columns(spec=[0.5, 0.5], gap='large', border=True)
        # переменные для хранения текста 
        self.text_area1 = None
        self.text_area2 = None
        # для хранения результатов
        self.view_results1 = None
        self.view_results2 = None
        # для кнопки
        self.compare_button = None
        # для хранения результата сравнения
        self.similarity = None

    def run(self):
        '''
        Основной метод для запуска приложения
        '''
        # поля для ввода текста
        with self.col1:
            self.text_area1 = st.text_area('**Введите первую статью**', height=150, max_chars=10000)
        # поля для ввода текста
        with self.col2:
            self.text_area2 = st.text_area('**Введите вторую статью**', height=150, max_chars=10000)

        # кнопка сравнения
        self.compare_button = st.button('**Сравнить**')

        # обработка нажатия кнопки
        if self.compare_button:
            # запоминаем стартовое время
            start_time = time()
            # проверяем заполнены ли обе статьи
            if self.text_area1 and self.text_area2:
                # выводим значок загрузки
                with st.spinner('Обработка...'):
                    # вычисляем суммаризацию
                    summary1 = self.realizer.get_summary(self.text_area1)
                    summary2 = self.realizer.get_summary(self.text_area2)

                # вывод результатов
                with self.col1:
                    # выводим суммаризацию
                    st.write_stream(self.realizer.stream_data(summary1))
                    # выводим топ 3 наиболее похожих статей
                    top_sims, headers = self.realizer.find_top_similar(
                        self.realizer.dataframe,
                        summary1
                    )
                    # визуализируем результат
                    st.bar_chart(dict(zip(headers, top_sims)))


                with self.col2:
                    # выводим суммаризацию
                    st.write_stream(self.realizer.stream_data(summary2))
                    # выводим топ 3 наиболее похожих статей
                    top_sims, headers = self.realizer.find_top_similar(
                        self.realizer.dataframe,
                        summary2
                    )
                    # визуализируем результат
                    st.bar_chart(dict(zip(headers, top_sims)))
                    
                # подсчет сходства
                self.similarity = self.realizer.get_similarity(summary1, summary2)
                st.write(f'Схожесть статей: {self.similarity*100:.2f}%')
                if self.similarity:
                    # Столбчатая диаграмма
                    st.bar_chart(
                        [[self.similarity*100, 100]], 
                        x_label='Схожесть', 
                        color=['#4bd4ff', '#0E1117'], 
                        stack="layered", 
                        horizontal=True, 
                        use_container_width=True
                )
                # вывод затраченного времени
                st.write_stream(self.realizer.stream_data(f'Время инференса: {(time()-start_time):.2f}s'))

            # если введены не оба текста
            else: 
                # выводим предупреждения
                st.warning('Пожалуйста, введите текст в оба поля.', icon='⚠️')

# Запуск приложения
if __name__ == '__main__':
    app = MyGUI()
    app.run()