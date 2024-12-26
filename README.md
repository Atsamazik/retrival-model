# Retrieval model
Retrieval model tuning

В данном проекте мы попытались дообучить модель e5-small-v2.

В качестве метрик были взяты contrastive_loss и triplet_margin_loss

Для запуска 

    
    Предобработка и сохранение данных
    
    ```
        python -m src.process_data
    ```
    Запуск обучения 
    
    ```
        python -m src.loop main

    ```
    Использование 
    
    ```
        python -m src.loop main

    ```


