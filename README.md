# image_tools
Зависимости:
```
pip install -U ultralytics pyodm opencv-python
```
## Первая задача
### Описание
Для сшивания изображений была использована библиотека OpenDroneMap. В качестве выхода скрипт генерирует папку odm_media. В папке ortophoto будет находится сшитый ортофотоплан.В пайплайне используется dspsift, dsm, flann.
### Запуск
Для запуска первой задачи нужно запусить 2 скрипта:
1. Нода ODM для обработки:
``` 
docker run -ti -p 3000:3000 opendronemap/nodeodm
``` 
Так как нода требовательна к вычислительным ресурсам, её можно запусить удаленно. Задать адрес и порт можно в аргументах к запуску скрипта задачи first.py:
``` 
--node_adress адресс на котором запущен odm node
--node_port порт на котором запущен odm node
``` 

2. Скрипт рещения задачи:

```
python3 src/first.py <Директория с изображениями>
``` 
Также есть возможность настроить параметры сшивания:
```
--scale_factor коэффициент сжатия изображений
--orthophoto_resolution маштабирования пикселя ортофотоплана(чем больше, тем детализированее)

``` 

## Вторая задача
### Описание
Функционал второй и третьей задач описан в image_tools_api класс ImageProcessor. Сами же скрпты носят утилитарный характер.
Скрипт поддерживает следующие параметры(и выбор метода), которые можно настроить под свой датасет первичной предобработки изображений:
```
--scale_factor (float) Factor by which to scale down images
--contrast_method (str) Contrast enhancement method ("hist_eq" for histogram equalization, "clahe" for CLAHE)"
--white_balance (bool) Flag to enable white balance correction
--clip_limit Clip limit for CLAHE
--tile_grid_size_1 Tile grid size for CLAHE (tile_grid_size_1, tile_grid_size_2)
--tile_grid_size_2 Tile grid size for CLAHE (tile_grid_size_1, tile_grid_size_2)
```
Выходом алгоритма является директория с суффиксом _processed с обработанными изображениями.
### Запуск
Пример запуска 2 скрипта:
```
python3 src/second.py ./data --contrast_method="clahe" --white_balance=True
``` 

## Третья задача
### Описание
Скрипт использует предобученную сеть YOLOv8n зарекомендовавшую себя в промышленных задачах. В ходе детекции одноэтапным методом алгоритм получает баундинг боксы всех обработанных объектов. Также есть возможность вывести отчет по каждому изображению в txt файл добавив следующий аргумент:
```
--save_txt=True
```
### Запуск
Пример запуска 2 скрипта:
```
python3 src/third.py ./data
``` 