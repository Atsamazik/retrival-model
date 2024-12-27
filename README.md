# Retrieval model
Retrieval model tuning

В данном проекте мы попытались дообучить модель e5-small-v2.

В качестве метрик были взяты contrastive_loss и triplet_margin_loss




1. Предобработка и сохранение данных
```commandline
python src/process_data.py preprocess-dataset
```

2. Посмотреть данные
```commandline
python src/process_data.py show-saved-data --number [number]
```
3. Запуск обучения
```commandline
    python src/loop.py
```
4. Использование
```commandline
    python src/usage.py
```


