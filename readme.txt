Инструкция по запуску:
1. (необязательно) Для генерации новых зашумленных данных необходимо в в файле FilePathResearch переменной NEED_GENERATE_NOISY_FILES задать значение True. В папке DATA/noise_inputs уже хранятся сгенерированные зашумленные данные, которые перезапишутся.
2. (необязательно) Для запуска алгоритмов кластеризации на исходных и зашумленных данных запустить файл FilePathResearch.py или FilePathResearch.ipynb. В папках DATA/path_processed и DATA/noise_clusters уже хранятся обработанные исходные и зашумленные данные соответственно с признаком PathClusterID, которые перезапишутся.
3. Для запуска модели RandomForest на всех данных запустить файл RandomForest.py или RandomForest.ipynb.


Для запуска из Google Colab (рекомендуется):
1. Скачать и разархивировать архив path-prediction;
2. Загрузить папку path-prediction на Google Диск;
3. Открыть и запустить в Google Colab блокноты FilePathResearch.ipynb, RandomForest.ipynb

Для запуска из MacOS/Linux (возможно, потребуется использование VPN для корректной работы SentenceTransformer)
1. Скачать и разархивировать архив path-prediction;
2. В терминале перейти в папку path-prediction: выполнить команду
cd /your/path/.../path-prediction
3. Проверить версию python: выполнить команду
python3 --version #3.12
4. Настроить окружение: выполнить команды
python3 -m venv venv
source venv/bin/activate
pip install --quiet pandas scikit-learn tqdm sentence-transformers python-Levenshtein rapidfuzz scikit-learn-extra optuna kmedoids gensim scipy numpy matplotlib
5. Открыть в IDE и запустить файлы FilePathClustering.py, FilePathResearch.py, RandomForest.py, или запустить из терминала командой, например:
python python FilePathResearch.py && python RandomForest.py
