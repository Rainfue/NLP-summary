# импортирование библиотек
from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer
from function import *

# путь к папке с моделью
model_dir = './saved_model'
# загрузка модели
model = T5ForConditionalGeneration.from_pretrained(model_dir)
# загрузка токенизатора
tokenizer = T5Tokenizer(model_dir)

