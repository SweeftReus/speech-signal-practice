Представление речевого сигнала — от звука к признакам
Поляков Александр Андреевич · ИСТ02

БАЗОВЫЙ ЗАПУСК
В каталоге с файлами speech_pipeline.py и experiments.py выполните
    python -m pip install numpy scipy matplotlib
    python speech_pipeline.py --synthetic
    python experiments.py

Проверенное окружение: Python 3.14.3, NumPy 2.5.4, SciPy 1.18.1,
Matplotlib 3.11.2. Все рисунки сохраняются без интерактивного окна.

В results/ находятся 11 графиков и матрица features.npy (float32, 39 × 198).
test_signal.wav содержит точный входной сигнал базового запуска, float32, 16 кГц.
Тот же сигнал даёт встроенный генератор с зерном 42.
run.txt — вывод базового запуска. verification.txt — итоги численной проверки.

В experiments/ находятся результаты трёх опытов с параметрами, опыт с разрядностью,
сравнение окон и мел-банков. Сводные числа сохранены в CSV, графики — в PNG.

Пример отдельного запуска с другим шагом и сохранением в собственную папку:
    python speech_pipeline.py --synthetic --hop-ms 15 --out results_hop15

ОБРАБОТКА СОБСТВЕННОГО ГОЛОСА
voice.wav — собственная запись голоса из файла «Свободный проспект.m4a».
Декодирована из AAC 48 кГц, каналы усреднены, частота приведена к 16 кГц.
В WAV сохранены 80 875 отсчётов. Этой записи соответствуют 503 полных кадра.
voice_results/ и voice_run.txt содержат 11 графиков, матрицу 39 × 503 и протокол запуска.
experiments/formants.png и experiments/formants.csv — спектрограмма с оценками F1, F2, F3 и таблица по кадрам.
Для повторного расчёта:
    python speech_pipeline.py --wav voice.wav --out voice_results
    python experiments.py --voice voice.wav
