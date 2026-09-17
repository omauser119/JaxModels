# Jax DecisionBench 1.0

Воспроизводимый бенчмарк для будущего сравнения **Jev-1 и Jax-1-Abel** по общим динамическим Choice/Score/Noul. Основной результат — набор показателей по областям, а не один рекламный балл. Jev здесь ещё не запускался; его результатов нет.

## Данные

2 560 запросов: 640 исходных сценариев × 4 варианта. 8 семейств: наличие товара, точное членство в списке, явная/отрицательная/гипотетическая просьба, последнее событие, тип документа, динамическое назначение команд, завершённость, диапазон дедлайна. Английский и русский. Варианты: исходный; смена ID; перестановка Choice и отвлекающая инструкция внутри данных; дополнительный независимый вопрос.

Dev — 640, test — 1920. Все варианты сценария находятся в одном split. Это авторские синтетические задания MIT с программно определяемыми ответами, **не 2560 независимых реальных наблюдений**. Шаблоны похожи между splits; перенос на внешние источники ими не доказан. В заданиях не содержатся ни эталон, ни split, ни ID сценария: runner передаёт только `request.state`/`request.questions`. Нейтральный `case_reference` — часть данных без связи с ответом.

`requests.jsonl` не содержит target. `references.jsonl` читает только evaluator, runner его не открывает. Оба файла зафиксированы SHA256 в manifest. Данные версии Abel-v1 не включены. Новое обучение должно использовать отдельный корпус. После просмотра test он уже не подходит для следующего независимого итогового теста: создайте новую версию benchmark, старую продолжайте показывать как regression suite.

## Запуск Abel

```bash
# Быстрая диагностическая проверка: 16 исходных dev-сценариев, без вариантов.
python3 benchmarks/run.py --backend abel --split dev --suite smoke --output benchmarks/runs/abel-smoke
# Полный test: 1920 запросов, на этом CPU это несколько часов.
python3 benchmarks/run.py --backend abel --split test --suite full --output benchmarks/runs/abel-test
# Явное продолжение: тот же checkpoint, executable, dataset и настройки.
python3 benchmarks/run.py --backend abel --split test --suite full --output benchmarks/runs/abel-test --resume
python3 benchmarks/evaluate.py benchmarks/runs/abel-test --output benchmarks/runs/abel-test/report.json
```

Не запускайте одновременно с нагрузкой на локальный API, если измеряете скорость. `abel` использует постоянный C++ worker без HTTP. Для честного end-to-end сравнения с удалённым API запускайте Abel через HTTP:

```bash
python3 benchmarks/run.py --backend http --model Jax-1-Abel \
  --endpoint http://127.0.0.1:8091/v1/decide --split test --suite full \
  --output benchmarks/runs/abel-http-test
```

## Будущий запуск Jev

Опциональный Python-адаптер опирается на официальный [TypeSafeClient](https://docs.typesafe.ai/sdk/python/api/clients/sync/client) и [форматы ответов](https://docs.typesafe.ai/sdk/python/api/types/responses). Установите `typesafe-sdk` в отдельное окружение и зафиксируйте версию; runner сохраняет её в run.json. Требуется ваш `TYPESAFE_API_KEY`. На данном этапе адаптер проверен по документации, **не живым API**.

```bash
# Подставить реальный зафиксированный ID Jev из models.list(), не плавающий latest.
python3 benchmarks/run.py --backend jev --model ACTUAL_JEV_MODEL_ID \
  --split test --suite full --output benchmarks/runs/jev-test
python3 benchmarks/evaluate.py benchmarks/runs/abel-http-test \
  --compare benchmarks/runs/jev-test --output benchmarks/runs/comparison.json
```

Никаких автоматических обращений к TypeSafe при запуске Abel. Ключи не сохраняются в результатах. Оба backend получают одни и те же критерии и state, без различных подсказок или промежуточных текстовых рассуждений. Настройки/ID и хеши сохраняются. `latest` не даёт воспроизводимого сравнения — фиксируйте provider model version и дату.

## Метрики

- **accuracy_all**: все ожидаемые задания; сбой/пропуск/невалидный ответ считается ошибкой. Отдельно success rate, причины отказов и accuracy_valid.
- **NLL**, multiclass **Brier**, **ECE-10**: только валидные ответы с явным знаменателем; p для NLL ограничено снизу 1e-12. Не смешивать хороший NLL на малой успешной подвыборке с надёжностью всей системы.
- **Score**: MAE, RMSE и MAE/(число уровней−1). Categorical accuracy Score — argmax распределения, а не округление математического ожидания.
- **Coverage/accuracy** при порогах 0.5/0.7/0.9/0.95/0.99. Единый показатель уверенности здесь max(probabilities), а не разные proprietary `confidence`.
- Разрезы по семейству, языку, примитиву и варианту. Macro-family accuracy не позволяет большому семейству скрывать маленькое.
- Устойчивость пар: смена решения, среднее/максимальное total variation. Число невалидных пар видно отдельно.
- Парное сравнение: разница accuracy, разница NLL на совместно валидных случаях, wins/losses, 95%-интервалы bootstrap по исходным сценариям (2000 повторов, фиксированный seed). Шаблонная зависимость между сценариями остаётся; интервалы не доказывают универсальное превосходство.
- Задержки end-to-end p50/p95/p99, startup отдельно. Concurrency=1, retries=0. Отказы входят в задержки с отдельной статистикой валидности. Полный load/throughput/стоимость — отдельный трек: не сравнивайте локальное CPU-время с provider latency как чистую скорость архитектуры.

Evaluator проверяет ID, хеш запроса, полный набор ответов, диапазоны, нормировку, согласованность Choice с argmax и Score с математическим ожиданием. Не нормализует и не «ремонтирует» ошибочные ответы. Повторные/чужие ID — ошибка оценки. Пропуски не исчезают из знаменателя. Resume не повторяет уже записанные отказы; новая попытка — отдельный run. Усечённая строка после аварии требует явного ремонта журнала.

## Статус опубликованной модели

Полный DecisionBench для checkpoint 0.2.0-pilot.1 ещё не выполнен. Его результат 7/10 относится к отдельному тесту пилотного обучающего корпуса, а не к DecisionBench. Результатов Jev нет. Старые локальные smoke-отчёты другого checkpoint в этот репозиторий не включены.
