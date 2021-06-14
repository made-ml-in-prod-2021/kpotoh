Homework 3
-------------------

## Запуск airflow
```
docker-compose up --build
```

## Тестирование кода дагов
```
docker exec -it airflow_ml_dags_scheduler_1 pytest -v tests
```

## Выполненная работа

0 Поднял локально airflow

1 Сделал DAG, который генерирует данные каждый день и пишет их в нужное место `data/raw/` (5 баллов)

2 Модель обучается еженедельно (10 баллов) 

3 Модель используется для предсказаний ежедневно (5 баллов) 

3а Реализуйте сенсоры на то, что данные готовы для дагов тренировки и обучения (3 доп балла)

4 все даги реализованы только с помощью DockerOperator (10 баллов)

5 Протестируйте ваши даги (5 баллов) https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html 

8 Настройте alert в случае падения дага (3 доп. балла)
https://www.astronomer.io/guides/error-notifications-in-airflow

9 самооценка (1 балл)

**Итого:** 5 + 10 + 5 + 3 + 10 + 5 + 3 + 1 = 42