# импортирование библиотек
# для работы с cuda
import torch

# для модели суммаризации и токенизатора
from transformers import T5Tokenizer, T5ForConditionalGeneration
# для модели сравнения
from gensim.models.doc2vec import Doc2Vec

# библиотека для nlp
import nltk
# модуль со стоп словами
from nltk.corpus import stopwords
# токенизатор слов
from nltk import word_tokenize
# для лемматизации
from pymystem3 import Mystem

# косинусное сходство
from sklearn.metrics.pairwise import cosine_similarity

# для передачи функции как параметр
from typing import Callable

# для работы с датафреймом
import pandas as pd

# для работы с файлами
import os

# для работы с временем
from time import sleep
# дополнительные данные для предобработки
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# создаем класс суммаризации
class Realization:
    # инициализатор
    # -------------------------------------------------------------------------- 
    def __init__(
            self,
            summarization_path: str,
            similiarity_path: str,
            df_path: str,
            device: torch.device | str | None = None,
    ):
        # обработка девайса
        # если девайс задан:
        if device:
            # если неправильный формат
            if not isinstance(device, (str, torch.device)):
                raise TypeError(f'device должен быть в строковом формате, а не {type(device)}')
            # проверяем на формат
            match type(device):
                case str():
                    # проверка значения переменной
                    if device == 'cpu' or device == 'cuda':
                        # если значение подходит
                        self.device = torch.device(device)
                    # если значение не подходит:
                    else:
                        raise ValueError(f'Устройство (device) должен быть "cuda" или "cpu", не {device}')
                case torch.device:
                    self.device = device
        # если девайс не задан
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # обработка неправильных значений путя к модели суммаризации
        # проверка формата
        if not isinstance(summarization_path, str):
            raise TypeError(f'model_path должен быть в строковом формате, а не {type(summarization_path)}')
        # проверка путя
        if os.path.exists(summarization_path):
            # пробуем открыть модель и токенизатор
            try:
                # получаем модель
                self.summarization = T5ForConditionalGeneration.from_pretrained(summarization_path).to(device)
                # получаем токенизатор
                self.tokenizer = T5Tokenizer.from_pretrained(summarization_path)
            except Exception as e:
                raise RuntimeError(f'Не удалось загрузить модель или токенизатор: {e}')
        # если путя не существует
        else:
            raise FileNotFoundError(f'Путь {summarization_path} не был найден, попробуйте другой')
        
        # обработка неправильных значений путя к модели сравнения
        # проверка формата
        if not isinstance(similiarity_path, str):
            raise TypeError(f'summarization_path должен быть в строковом формате, а не {type(similiarity_path)}')
        # проверка путя
        if os.path.exists(similiarity_path):
            # пробуем открыть модель 
            try:
                # получаем модель
                self.similarity = Doc2Vec.load(similiarity_path)

            except Exception as e:
                raise RuntimeError(f'Не удалось загрузить модель: {e}')
        # если путя не существует
        else:
            raise FileNotFoundError(f'Путь {similiarity_path} не был найден, попробуйте другой')

        # обрабатываю путь к датафрейму
        # проверка на формат
        if not isinstance(df_path, str):
            raise TypeError(f'df_path должен быть в строковом формате, а не {type(df_path)}')
        # если неверный формат
        else:
            # проверка на существование
            if os.path.exists(df_path):
                # пробуем получить дф
                try:
                    # получаем датафрейм
                    self.dataframe = pd.read_pickle(df_path)
                except Exception as e:
                    raise RuntimeError(f'Не удалось загрузить датафрейм с эмбеддингами')
            # если нет
            else:
                raise FileNotFoundError(f'Путь {df_path} не был найден, попробуйте другой')
        
        # получаем стоп слова
        self.stop_words = set(stopwords.words('russian'))
        # получаем объект лемматизации
        self.mystem = Mystem()

    # --------------------------------------------------------------------------   
    # метод для чистки стоп слов
    def clean_text(self, text: str) -> str:
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
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        # возвращаем почищенный текст
        return ' '.join(filtered_words)

    # --------------------------------------------------------------------------
    # метод для соединения абзацев текста
    @staticmethod
    def remove_empty_line(text: str) -> str:
        '''
        Функция для удаления пустых слов в тексте
            Args: 
                - text (str): текст для убирания пустых строк
            Returns:
                - str: текст с убранными пустыми строками
        '''
        # разделяем текст на строки и оставляем только не пустые строки
        lines = [line for line in text.splitlines() if line.strip() != '']
        # возвращаем новый текст
        return ' '.join(lines)
    
    # функция для создания потокового текста
    @staticmethod
    def stream_data(text: str):
        '''Функция для создания потокового текста'''
        for word in text.split():
            yield word + ' '
            sleep(0.02)

    # функция для лемматизации предложения
    def text_lemmatize(self, text: str) -> str:
        '''
        Функция для лемматизации текста
            Args:
                - text (str): текст в строковом формате

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
        # объект лемматизации

        return ''.join(self.mystem.lemmatize(text.strip()))
    
    # --------------------------------------------------------------------------
    # функция для обработки входящего текста
    def get_input(self, text: str) -> str:
        '''Функция для обработки входящего текста'''
        # убираю пустые строки
        text = self.remove_empty_line(text).lower()
        # убираю стоп-слова и лемматизирую
        text = self.text_lemmatize(
            self.clean_text(text)
            )
        # возвращаю обработанный текст
        return text
    
    # --------------------------------------------------------------------------
    # функция для суммаризации
    def get_summary(self,
                    text: str, 
                    show_output: bool = False) -> str:
        '''
        **Функция для суммаризации текста**
        ==
            **Args:**
                - **text** (`str`): входной текст для суммаризации 
                - **show_output** (`bool`): метка для показа результатов внутри функции
            
            **Returns:**
                - **str**: суммаризация входного текста


        Пример использования:
        ====

        ```python
        >>> from function import get_summary
        >>> 
        >>> model = T5ForConditionalGeneration.from_pretrained("./saved_model")
        >>> tokenizer = T5Tokenizer.from_pretrained("./saved_model")
        >>> 
        >>> text = "yout text for summary"
        >>> 
        >>> get_summary(text=text,
        >>>             tokenizer=tokenizer,
        >>>             model=model,
        >>>             device='cuda',
        >>>             show_output=True,
        >>>             )
        >>>             
        >>>  # данный код выведет суммаризацию для вашего текста
            
        '''
        # обработка ошибок
        # если неправильно передан текст
        if not isinstance(text, str):
            raise TypeError(f'Текст (text) должен быть в формате str. Сейчас: {type(text)}')
        
        # если неправильно передана метка показа результатов
        if not isinstance(show_output, bool):
            raise TypeError(f'show_output должен быть в формате bool (True или False). Сейчас: {type(show_output)}')

        # обрабатываем входящий текст
        if len(text) > 10000:
            input_text = self.get_input(text[10000])
        input_text = self.get_input(text)
        # получаем токены входящего текста
        input_ids = self.tokenizer(input_text, return_tensors='pt').input_ids.to(self.device)
        # освобождаем память перед генерацией
        torch.cuda.empty_cache()
        # отключаю расчет градиентов
        with torch.no_grad():
            # генерируем суммаризацию
            outputs = self.summarization.generate(
                                    input_ids=input_ids,        # токенизированый входной текст
                                    max_length=100,             # максимальная длина генерированной последовательности
                                    min_length=10,              # минимальная длина генерированной последовательности
                                    num_beams=1,                # количество лучей для поиска с использованием Beam Search
                                    do_sample=False,            # ограничивает вариативность
                                    repetition_penalty=2.5,     # штраф за повторение токенов
                                    no_repeat_ngram_size=3,     # запрещает повторение n-грамм указанного размера
                                    num_return_sequences=1,     # количество возвращаемых последовательностей
                                    # early_stopping=True,        # генерация остановится, как только завершаться гипотезы
                                    use_cache=True,             # использование кэширования для ускорения генерации
                                    length_penalty=1.0,         # штраф за длину в beam search
                                )
        # декодируем получившийся текст
        gen_summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

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
        

    # функция для сравнения двух текстов
    def get_similarity(
            self,
            text1: str, 
            text2: str, 
            prep_flag: bool = True, 
            preprocess: Callable[[str], str] = get_input
                    ):
        '''
        Функция для получения схожести двух текстов
        ===
            Args:
                - text1 (str): первый текст в строковом формате
                - text2 (str): второй текст в строковом формате
                - prep_flag (bool = True): флаг, нужна ли обработка входного текста
                - preprocess (func(str) -> str: = lambda x: x): функция для обработки текста

            Returns:
                - float: сходство между текстами (от 1 до -1)
        '''
        # проверка формата текста 1
        if not isinstance(text1, str):
            raise TypeError(f'text1 должен быть в строковом формате, а не {type(text1)}')
        # проверка формата текста 2
        if not isinstance(text2, str):
            raise TypeError(f'text1 должен быть в строковом формате, а не {type(text1)}')
        # проверка формата флага для обработки
        if not isinstance(prep_flag, bool):
            raise TypeError(f'prep_flag должен быть в булевом формате, а не {type(prep_flag)}')
        # проверка формата функции обработки текста
        if not isinstance(preprocess, Callable):
            raise TypeError(f'preprocess должна быть ф-ей, принимающей и возвращающей str')

        # если стоит метка о обработке данных
        if prep_flag:
            text1 = preprocess(self, text1)
            text2 = preprocess(self, text2)
        # вычисляем эмбеддинг
        inferred_vector1 = self.similarity.infer_vector(word_tokenize(text1)).reshape(1,-1)
        inferred_vector2 = self.similarity.infer_vector(word_tokenize(text2),).reshape(1,-1)
        # # получаем сходство
        return cosine_similarity(inferred_vector1, inferred_vector2).item()
    

    # функция для поиска топ 3 схожих статей
    def find_top_similar(
            self,
            df: pd.DataFrame, 
            text: str, 
            top_n: int = 3
            ) -> pd.DataFrame:
        '''
        Находит топ-N наиболее схожих эмбеддингов и их summary.

        Параметры:
            - df (pd.DataFrame): Датафрейм с колонками 'summary' и 'embedding'.
            - input_embedding (np.ndarray): Входной эмбеддинг для сравнения.
            - top_n (int): Количество наиболее схожих результатов (по умолчанию 3).

        Возвращает:
            pd.DataFrame: Датафрейм с топ-N наиболее схожими эмбеддингами и их summary.
        '''
        # получаем эмбеддинг
        input_embedding = self.similarity.infer_vector(word_tokenize(self.get_input(text))).reshape(1,-1)
        # список всех схожестей
        similarities = []
        # проходимся по всему датафрейму
        for i in range(df.shape[0]):
            similarities.append(cosine_similarity(input_embedding, df['embedding'].loc[i]).item()*100)
        # Добавляем столбец с косинусной схожестью в датафрейм
        df['similarity'] = similarities
        
        # Сортируем датафрейм по убыванию схожести и выбираем топ-N
        top_similar = df.sort_values(by='similarity', ascending=False).head(top_n)
        summaries = top_similar['summary'].tolist()
        top_similarities = [round(sim, 2) for sim in top_similar['similarity'].tolist()]
        headers = [' '.join(word_tokenize(summary)) for summary in summaries]

        # Возвращаем только нужные колонки
        return top_similarities, headers



