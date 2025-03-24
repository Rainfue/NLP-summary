# Импортирование библиотек
# --------------------------------------------------------------------------
# для работы с датафреймами
import pandas as pd

# для визуализации результатов
import matplotlib.pyplot as plt

# для работы с массивами
import numpy as np

# для работы с файловой системой
import os, shutil

# модуль со словарями
from collections import defaultdict

# для создания прогресс-бара
from tqdm import tqdm

# nlp библиотека
import nltk
# модуль со стоп словами
from nltk.corpus import stopwords

# для создания облаков слов
from wordcloud import WordCloud

# модуль с регулярными приложениями
import re

# для работы со строками
import string

# для работы со слоучайными значениями
import random

# для работы с датасетами
from datasets import Dataset, DatasetDict

# для обработки текста
from pymystem3 import Mystem

# Реализация функций
# --------------------------------------------------------------------------

# процедура для генерации облака слов
def generate_wordcoud(text: str, color: str = 'black'):
    '''Процедура для создания облака слов'''
    # создаем объект - облако слов
    wordcloud = WordCloud(
        width=800,              # ширина
        height=400,             # высота
        background_color=color  # цвет фона
    ).generate(text)
    
    # визуализация облака слова
    plt.figure(figsize=(10,5))                          # размер фигуры
    plt.imshow(wordcloud, interpolation='bilinear')     # загружаем объект для вывода
    plt.axis('off')                                     # отключаем сетку
    plt.show()                                          # вывод изображения
# --------------------------------------------------------------------------

# функция для чистки стоп слов
def clean_text(text: str, stop_words: set) -> str:
    '''
    Функция для чистки текста от стоп слов
        Args:
            - test (str): текст, который нужно почистить от стоп слов
            - stop_words (set): множество стоп слов
        Returns:
            - отфильтрованный текст (str)
    '''
     # создаем список слов используя split()
    words = text.split()
    # фильтруем слова используя заданное множество
    filtered_words = [word for word in words if word.lower() not in stop_words]
    # возвращаем почищенный текст
    return ' '.join(filtered_words)
# --------------------------------------------------------------------------

# функция для лемматизации предложения
def text_lemmatize(text: str, mystem: Mystem) -> str:
    '''
    Функция для лемматизации текста
        Args:
            - text (str): текст в строковом формате
            - mystem (Mystem): объект лемматизации из библиотеки pymystem3

        Returns:
            - processed_text (str): обработанный текст в строковом формате
    
    Пример использования:

    ```python
    >>  from pymystem3 import Mystem
    >> 
    >> # объект Mystem
    >> mystem = Mystem()
    >> 
    >>  print(text_lemmatize('мама мыла раму', mystem))
    ```
    ```Вывод:
    мама мыть рама
    ```
    '''
        
    return ''.join(mystem.lemmatize(text.strip()))
# --------------------------------------------------------------------------


# функция для построения графиков
def see_distribution(data_stats: dict, 
                     title: str = 'Values distribution',
                     xlabel: str = 'Keys',
                     ylabel: str = 'Values',
                     color: str = 'lightblue',
                     graph_type: str = 'bar',
                     grid: bool = True,
                     meta: bool = True
                     ):
    '''
    Процедура для построения графика распределения данных
        Args:
        - data_stats (dict): словарь с собранной статистикой
        - title (str): название графика
        - xlabel (str): название икса
        - ylabel (str): название игрика
        - color (str): цвет графика
        - grid (bool): включение сетки
        - meta (bool): включение метаинформации
    '''
    # парсим входной словарь
    x, y = data_stats.keys(), data_stats.values()
    
    plt.figure(figsize=(10,5))      # размер фигуры
    # в зависимости от типа графика, строим нужный
    match graph_type:
        # если тип графика - bar
        case 'bar':
            plt.bar(x, y, color=color)      # тип графика
        # если тип графика - plot   
        case 'plot':
            plt.plot(x, y, color=color)     # тип графика
        # если тип графика - plot   
        case 'scatter':
            plt.scatter(x, y, color=color)     # тип графика


    # настраиваем график
    plt.title(title)                # название графика
    plt.xlabel(xlabel)              # подпись к иксу
    plt.ylabel(ylabel)              # подпись к игрику

    # если включена сетка
    if grid:
        # включаем сетку
        plt.grid(True)

    plt.show()                      # вывод графика

    # если включен вывод метаинформации
    if meta:
        # всего значений
        print(f'Counts: {len(x)}')
        # вывод минимального значения
        print(f'Min: {min(y)}')
        # вывод среднего значения
        print(f'Mean: {sum(y)/len(y):.2f}')
        # вывод максмиального значения
        print(f'Max: {max(y)}')
# --------------------------------------------------------------------------

# функция для преобразования формата датасета 
def reorganize_dataset(dataset_path: str):
    '''
    Функция, преобразующая папки в датасет (DataseDict)
    Args:
        - dataset_path (str): путь к папке с данными в строковом формате
    Returns:
        - DatasetDict()
    '''

    # список для всех примеров
    samples = []
    # собираем все примеры
    for folder in os.listdir(dataset_path):
        # путь к папке
        folder_path = os.path.join(dataset_path, folder)
        # проверяем, папка ли это
        if os.path.isdir(folder_path):
            # получаем путь к тексту и аннотации
            text_path = os.path.join(folder_path, 'text.txt')
            annotation_path = os.path.join(folder_path, 'annotation.txt')
            tags_path = os.path.join(folder_path, 'tags.txt')
            

            # проверяем, существуют ли эти файлы
            if os.path.exists(text_path) and os.path.exists(annotation_path):
                # открываем и читаем файл с текстом
                with open(text_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()

                # открываем и читаем файл с аннотацией
                with open(annotation_path, 'r', encoding='utf-8') as f:
                    summary = f.read().strip()

                # открываем и читаем файл с тэгами
                with open(tags_path, 'r', encoding='utf-8') as f:
                    tag = f.read().strip()
                
                # сохраняем элемент в список
                samples.append({
                    'text_path': text_path,
                    'annotation_path': annotation_path,
                    'tags_path': tags_path,

                    'text': text,
                    'summary': summary,
                    'tag': tag,

                    'text_all_symb': len(text),
                    'summary_all_symb': len(summary),
                    'tag_all_symb': len(tag),

                    'text_clean': len(re.sub(r'[{}]'.format(string.punctuation), '', text)),
                    'summary_clean': len(re.sub(r'[{}]'.format(string.punctuation), '', summary)),
                    'tag_clean': len(re.sub(r'[{}]'.format(string.punctuation), '', tag)),

                    'text_words': len(text.split()),
                    'summary_words': len(summary.split()),
                    'tag_words': len(tag.split()),

                    'id': folder,
                })

    return samples

# --------------------------------------------------------------------------   
# функция для разделения датасета на выборки
def split_dataset(samples):
    # создаем словарь для датасета
    data = {'train': [], 'validation': [], 'test': []}
    # разделяем на тестовую, валидационную и тестовую выборки (80/10/10)
    random.shuffle(samples)

    # колво элементов
    n = len(samples)
    # разбиваем элементы
    # тренировочная выборка (80%)
    data['train'] = samples[:int(0.8*n)]
    # валидационная выборка (10%)
    data['validation'] = samples[int(0.8*n):int(0.9*n)]
    # тестовая выборка (10%)
    data['test'] = samples[int(0.9*n):]

    # возвращаем DatasetDict
    return DatasetDict({
        split: Dataset.from_list(data[split])
        for split in ['train', 'validation', 'test']
    })

# --------------------------------------------------------------------------
# функция для фильтрации
def df_filter(df: pd.DataFrame,
              upper: float = 0.95, 
              lower: float = 0.05,
              max_text_length: int = 10000,
              text_columns: list = None):
    '''
    Функция для фильтрации датасета по квартилям и обрезки текстов
    =======
        **Args**:
            - df (pd.DataFrame): датасет для фильтрации
            - column (str): название колонки для фильтрации по квартилям
            - upper (float = 0.95): верхний квартиль
            - lower (float = 0.05): нижний квартиль
            - max_text_length (int = 10000): максимальная длина текста
            - text_columns (list = None): список колонок с текстами для обрезки
            
        **Returns**:
            - filtered_df (pd.DataFrame): отфильтрованный датасет с обрезанными текстами
    
    Пример использования:
    ===

    ```python
    >>  from function import df_filter
    >> 
    >>  # Фильтрация по квартилям и обрезка текстов
    >>  filtered_df = df_filter(df, 'text_words', text_columns=['text', 'summary'])
    >>
    >>  # Только фильтрация по квартилям для числовых колонок
    >>  for column in df.columns:
    >>      if 'int64' == df[column].dtype:
    >>          filtered_df = df_filter(filtered_df, column)
    ```
    '''
    # логирование
    print(f'Размер датасета до: {df.shape}')
    # копируем датасет
    filtered_df = df.copy(deep=True)
    # проходимся по каждой колонке в датафрейме
    for column in filtered_df.columns:
    # проверяем тип данных в таблице
        if filtered_df[column].dtype == 'int64':
            # верхняя граница
            upper_bound = filtered_df[column].quantile(upper)
            # нижняя граница
            lower_bound = filtered_df[column].quantile(lower)

            # фильтрую датасет
            filtered_df = filtered_df[
                (filtered_df[column] >= lower_bound) & (filtered_df[column] <= upper_bound)
            ]
    
    # Обрезка текстов, если указаны текстовые колонки
    if text_columns:
        for text_col in text_columns:
            if text_col in filtered_df.columns:
                # Создаем копию датафрейма для безопасного изменения
                filtered_df = filtered_df.copy()
                # Обрезаем тексты, которые длиннее max_text_length
                filtered_df[text_col] = filtered_df[text_col].apply(
                    lambda x: x[:max_text_length] if isinstance(x, str) and len(x) > max_text_length else x
                )
                
                
    # убираем нули, если они есть         
    filtered_df = filtered_df.dropna()
    # логирование
    print(f'Размер датасета после: {filtered_df.shape}')
    # возвращаем отфильтрованный датасет
    return filtered_df

# --------------------------------------------------------------------------

