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
# токенизатор слов
from nltk.tokenize import word_tokenize
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

# для использования видеокарты
import torch

# для работы с трансформерами
from transformers import T5ForConditionalGeneration, T5Tokenizer

# модель и объект для поиска схожести текстов
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# метрика косинусной схожести
from sklearn.metrics.pairwise import cosine_similarity

# импортирование библиотек
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer

# для передачи функции как параметр
from typing import Callable

# --------------------------------------------------------------------------
# скачиваем стоп-слова для русского языка
nltk.download('stopwords')
# пунктуация 
nltk.download('punkt')
stop_words = set(stopwords.words('russian'))
# объект для лемматизации
mystem = Mystem()
# устройство для вычисления
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# предобученая модель для сравнения статей
model_similarity = SentenceTransformer('all-MiniLM-L6-v2')

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
              column: str, 
              upper: float = 0.95, 
              lower: float = 0.05):
    '''
    Функция для фильтрации датасета по квартилям
    =======
        **Args**:
            - df (pd.DataFrame): датасет для фильтрации
            - column (str): название колонки
            - upper (float = 0.95) верхний квартиль
            - lower (float = 0.05) нижний квартиль
        **Returns**:
            - filtered_df (pd.DataFrame): отфильтрованный датасет
    
    Пример использования:
    ===

    ```python
    >>  from function import df_filter
    >> 
    >>  for column in df.columns:
    >>      if 'int64' == df[column].dtype:
    >>          filtered_df = df_filter(filtered_df, column)

    ```
    '''
    # верхняя граница
    upper_bound = df[column].quantile(upper)
    # нижняя граница
    lower_bound = df[column].quantile(lower)

    # фильтрую датасет
    filtered_df = df[
        (df[column] >= lower_bound) & (df[column] <=upper_bound)
    ]
    
    # возвращаем отфильтрованный датасет
    return filtered_df.dropna()

# --------------------------------------------------------------------------
# функция для соединения абзацев текста
def remove_empty_line(text: str) -> str:
    '''
    Функция для удаления пустых слов в тексте
        Args: 
            - text (str): текст для убирания пустых строк
        Returns:
            - str: текст с убранными пустыми строками
    '''
    # разделяем текст на строки
    lines = [line for line in text.splitlines() if line.strip() != '']
    # оставляем только не пустые строки

    return ' '.join(lines)

# --------------------------------------------------------------------------
# функция для обработки входящего текста
def get_input(text: str) -> str:
    '''Функция для обработки входящего текста'''
    # убираю пустые строки
    text = remove_empty_line(text).lower()
    # убираю стоп-слова и лемматизирую
    text = text_lemmatize(clean_text(text, stop_words), mystem)
    # возвращаю обработанный текст
    return text
     
# --------------------------------------------------------------------------
# функция для суммаризации
def get_summary(text: str, 
                tokenizer: T5Tokenizer, 
                model: T5ForConditionalGeneration, 
                device: torch.device | str = device,
                show_output: bool = False) -> str:
    '''
    **Функция для суммаризации текста**
    ==
        **Args:**
            - **text** (`str`): входной текст для суммаризации 
            - **tokenizer** (`T5Tokenizer`): токенизатор для модели
            - **model** (`T5ForConditionalGeneration`): модель для суммаризации
            - **device** (`torch.device`): устройство для вычислений
            - **show_output** (`bool`): метка для показа результатов внутри функции
        
        **Returns:**
            - **str**: суммаризация входного текста


    Пример использования:
    ====

    ```python
    >> from function import get_summary
    >> 
    >> model = T5ForConditionalGeneration.from_pretrained("./saved_model")
    >> tokenizer = T5Tokenizer.from_pretrained("./saved_model")
    >> 
    >> text = "yout text for summary"
    >> 
    >> get_summary(text=text,
    >>             tokenizer=tokenizer,
    >>             model=model,
    >>             device='cuda',
    >>             show_output=True,
    >>             )
    >>             
    >>  # данный код выведет суммаризацию для вашего текста
        
    '''

    # обработка ошибок
    # если неправильно передан текст
    if not isinstance(text, str):
        raise TypeError(f'Текст (text) должен быть в формате str. Сейчас: {type(text)}')
    
    # если неправильно передан токенизатор
    if not isinstance(tokenizer, T5Tokenizer):
        raise TypeError(f'Токенизатор (tokenizer) должен быть в формате transformer.T5Tokenizer. Сейчас: {type(tokenizer)}')
    
    # если неправильно передана модель
    if not isinstance(model, T5ForConditionalGeneration):
        raise TypeError(f'Модель (model) должна быть в формате transformer.T5ForConditionalGeneration. Сейчас: {type(model)}')
    
    # если неправильно передано вычислительное устройство
    if not isinstance(device, (torch.device, str)):
        raise TypeError(f'Устройство (device) должен быть в формате torch.device или str. Сейчас: {type(device)}')
    
    # если неправильно передана метка показа результатов
    if not isinstance(show_output, bool):
        raise TypeError(f'show_output должен быть в формате bool (True или False). Сейчас: {type(show_output)}')
    
    # обработка того, как был передан device
    else:
        match type(device):
            case str():
                # проверка значения переменной
                if device == 'cpu' or device == 'cuda':
                    # если значение подходит
                    device = torch.device(device)
                # если значение не подходит:
                else:
                    raise ValueError(f'Устройство (device) должен быть "cuda" или "cpu", не {device}')
            case torch.device:
                device = device

    # обрабатываем входящий текст
    input_text = get_input(text)
    # получаем токены входящего текста
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids.to(device)
    # генерируем суммаризацию
    outputs = model.generate(
                            input_ids=input_ids,
                            max_length=500,
                            min_length=20,
                            num_beams=5,
                            temperature=0.7,
                            top_k=100,
                            top_p=0.95,
                            do_sample=True,
                            repetition_penalty=1.2,
                            no_repeat_ngram_size=3,
                            num_return_sequences=3,
                            early_stopping=True
                        )
    # декодируем получившийся текст
    gen_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # выводим результат (если указано)
    if show_output:
        # вывод входного текста
        print(f'>> Original Text: {text}')
        # вывод сгенерированного изложения
        print(f'\n\n>> Generated summary: {gen_summary}')
        # возвращем суммаризацию
        return gen_summary
    
    # если нет, возвращаем суммаризацию без вывода
    else:
        return gen_summary
    
# --------------------------------------------------------------------------
# функция для сравнения двух текстов
def get_similarity(text1: str, 
                   text2: str, 
                   model: str | Doc2Vec, 
                   prep_flag: bool = True, 
                   preprocess: Callable[[str], str] = get_input):
    '''
    Функция для получения схожести двух текстов
    ===
        Args:
            - text1 (str): первый текст в строковом формате
            - text2 (str): второй текст в строковом формате
            - model (str|Doc2Vec): модель в формате doc2vec объекта 
                                  либо путь к ней в строковом формате

        Returns:
            - float: сходство между текстами (от 1 до -1)
    '''
    # проверка формата текста 1
    if not isinstance(text1, str):
        raise TypeError(f'text1 должен быть в строковом формате, а не {type(text1)}')
    # проверка формата текста 2
    if not isinstance(text1, str):
        raise TypeError(f'text1 должен быть в строковом формате, а не {type(text1)}')
    # проверка формата модели
    if not isinstance(model, (str, Doc2Vec)):
        raise TypeError(f'model должна быть в формате Doc2Vec, либо в виде строкового путя, а не {type(model)}')
    # если модель в виде путя
    if type(model) == str:
        # проверяем, что файл существует
        if os.path.exists(model):
            # пробуем загрузить модель через try: except
            try:
                # загружаем модель из файла
                model = Doc2Vec.load(model)
            # если не получилось загрузить, выводим ошибку
            except Exception as e:
                raise ValueError(f'Ошибка! не удалось загрузить модель, проверьте ваш файл!\n{e}')
        # если путь не существует:
        else:
            # выводим ошибку
            raise ValueError('Ошибка! Файла с моделью не существует, проверьте правильность написания!')
    # если стоит метка о обработке данных
    if prep_flag:
        text1 = preprocess(text1)
        text2 = preprocess(text2)
    # вычисляем эмбеддинг
    inferred_vector1 = model.infer_vector(word_tokenize(text1)).reshape(1,-1)
    inferred_vector2 = model.infer_vector(word_tokenize(text2),).reshape(1,-1)
    # # получаем сходство
    return cosine_similarity(inferred_vector1, inferred_vector2).item()

# --------------------------------------------------------------------------
# функция для извлечения эмбеддингов из всего датасета
def extract_all_embeddings(dataset, part, column, model: Doc2Vec):
    df = pd.DataFrame(columns=['summary', 'embedding'])
    # проходимся по каждой суммаризации в датасете
    for i in tqdm(range(len(dataset[part])), desc='Извлечение эмбеддингов..', unit='text'):
        # получаем суммаризацию
        summary = dataset[part][column][i]
        # получаем эмбеддинг
        embedding = model.infer_vector(word_tokenize(get_input(summary))).reshape(1,-1)
        # сохраняем эмбеддинг
        df.loc[i] = [summary, embedding]
    df.head()

# функция для поиска топ 3 схожих статей
def find_top_similar(df: pd.DataFrame, text: str, model: Doc2Vec, top_n: int = 3) -> pd.DataFrame:
    '''
    Находит топ-N наиболее схожих эмбеддингов и их summary.

    Параметры:
        df (pd.DataFrame): Датафрейм с колонками 'summary' и 'embedding'.
        input_embedding (np.ndarray): Входной эмбеддинг для сравнения.
        top_n (int): Количество наиболее схожих результатов (по умолчанию 3).

    Возвращает:
        tuple
    '''
    # получаем эмбеддинг
    input_embedding = model.infer_vector(word_tokenize(get_input(text))).reshape(1,-1)
    # список всех схожестей
    similarities = []
    # проходимся по всему датафрейму
    for i in range(df.shape[0]):
        similarities.append(cosine_similarity(input_embedding, df['embedding'].loc[i]).item()*100)
    # Добавляем столбец с косинусной схожестью в датафрейм
    df['similarity'] = similarities
    
    # Сортируем датафрейм по убыванию схожести и выбираем топ-N
    top_similar = df.sort_values(by='similarity', ascending=False).head(3)
    summaries = top_similar['summary'].tolist()
    top_similarities = [round(sim, 2) for sim in top_similar['similarity'].tolist()]
    headers = [' '.join(word_tokenize(summary)) for summary in summaries]

    # Возвращаем только нужные колонки
    return top_similarities, headers