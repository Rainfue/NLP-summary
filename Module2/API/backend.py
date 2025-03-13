# импортирование библиотек
# для работы с cuda
import torch

# для модели суммаризации и токенизатора
from transformers import T5Tokenizer, T5ForConditionalGeneration
# для модели сравнения
from gensim.models.doc2vec import Doc2Vec

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
            raise TypeError(f'model_path должен быть в строковом формате, а не {type(model_path)}')
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
            raise TypeError(f'model_path должен быть в строковом формате, а не {type(model_path)}')
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
            raise TypeError(f'model_path должен быть в строковом формате, а не {type(model_path)}')
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
        # # возвращаю обработанный текст
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
        
        # если неправильно передана метка показа результатов
        if not isinstance(show_output, bool):
            raise TypeError(f'show_output должен быть в формате bool (True или False). Сейчас: {type(show_output)}')


        # обрабатываем входящий текст
        input_text = self.get_input(text)
        # получаем токены входящего текста
        input_ids = self.tokenizer(input_text, return_tensors='pt').input_ids.to(self.device)
        # генерируем суммаризацию
        outputs = self.summarization.generate(
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

            Returns:
                - float: сходство между текстами (от 1 до -1)
        '''
        # проверка формата текста 1
        if not isinstance(text1, str):
            raise TypeError(f'text1 должен быть в строковом формате, а не {type(text1)}')
        # проверка формата текста 2
        if not isinstance(text1, str):
            raise TypeError(f'text1 должен быть в строковом формате, а не {type(text1)}')

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
            model: Doc2Vec, 
            top_n: int = 3
            ) -> pd.DataFrame:
        '''
        Находит топ-N наиболее схожих эмбеддингов и их summary.

        Параметры:
            df (pd.DataFrame): Датафрейм с колонками 'summary' и 'embedding'.
            input_embedding (np.ndarray): Входной эмбеддинг для сравнения.
            top_n (int): Количество наиболее схожих результатов (по умолчанию 3).

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


