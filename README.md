FinABYSS (Financial Aspect-Based Hybrid Semantic System)
---
- [Functionality](#functionality)
  - [Semantic Map](#semantic-map)
    - [Interactivity](#interactivity)
    - [Semantic Similarity](#semantic-similarity)
    - [Search for texts](#search-for-texts)
    - [Timeline infographic](#timeline-infographic)
    - [Sample word cloud](#sample-word-cloud)
  - [Predictor](#predictor)
- [Key dependencies](#key-dependencies)
- [Corpus of financial news articles](#corpus-of-financial-news-articles)
- [Notes](#notes)
  - [Project Structure](#project-structure)
  - [P.S.](#ps)

# Functionality
## Semantic Map
Семантическая карта может стать неотъемлемой частью рабочего процесса финансового аналитика, инвестора или кого-угодно еще, интересующегося финансами.

![semantic_map](docs/semantic_map.png)
### Interactivity
На интерактивной семантической карте нас встречают кластеры, представляющие темы. Каждая **точка является уникальной статьей**, при этом **размер точки указывает на относительную длину статьи**. Более того, каждую статью, мы можем с легкостью найти в Google (в дальнейшем переадресация будет доработана до прямой ссылки).

![Interactivity](docs/redirect.gif)

Итак, мы открываем статью, и во-первых видим насколько она большая — предыдущие модели не смогли бы обработать настолько длинный текст. И во-вторых диоксид углерода действительно упоминается в данной статье.

### Semantic Similarity
Как отмечалось ранее, данная карта достаточно хорошо сохраняет семантическую связь как кластеров, так и самих текстов между собой. Давайте посмотрим детальнее.

![semantic_similarity](docs/semantic_similarity.gif)

Мы видим группу кластеров, связанных со здравоохранением, все они располагаются кучно, но каждый является уникальным. Далее мы можем наблюдать, что *Sustainable Finance*, *Cybersecurity* и *Green Energy* тоже располагаются крайне близко. То же касается и *Politics* с *Monetary Policy*, но данные два кластера, имеют немного большую дистанцию, что вполне оправдано.

### Search for texts
Карта также предоставляет интерфейс к точечному обнаружению необходимых новостей по ключевым словам.

![search](docs/geopolitics.gif)

Так, Индонезию чаще всего можно встретить среди растущих рынков и политики, то же относится и к России, но Россия все же превалирует именно в политике.

### Timeline infographic
Что примечательно — мы можем **совмещать поиск по ключевым словам с распределением по датам** публикации или любым другим количественным признаком.

![timeline_infographic](docs/trump_by_dates.gif)

Так, мы можем наблюдать, что перед выборами в США, новостей о Трампе было меньше, чем после его победы. **Эта функция позволяет быстро и крайне просто выявлять исторические события и триггеры**.

### Sample word cloud
Наконец, самое интересное, что мы можем изучить, о чем говорят в новостях того или иного кластера, или просто выбранной группы.

![wordcloud](docs/lasso.gif)

Вполне резонно, что в *Sustainable Finance* чаще говорят об устойчивости, климате и углероде. Напротив, кластер с криптовалютой визуально подразделяется на два. В нижнем больше говорят о конкретных технологиях, а в верхнем скорее общеобразовательный контент на тему криптовалют.

## Predictor
Данная система вовсе не ограничивается лишь семантической картой, которая на самом деле представляет собой интепретируемый интерфейс к более закрытому процессу — прогнозированию стоимости финансовых активов с использованием тематических оценок тональностей.

# Key dependencies
- [Стилевое оформление ВКР](https://github.com/itonik/spbu_diploma/tree/master) с LaTeX-шаблоном для ВКР по ГОСТам;
- [BERTopic](https://github.com/MaartenGr/BERTopic);
- [alpha_vantage](https://github.com/RomelTorres/alpha_vantage);
- [PyTorch](https://github.com/pytorch/pytorch);
- [TensorBoard](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html) (доки по работе с PyTorch);
- [Polars](https://docs.pola.rs/) для ускорения работы с данными;
- [CUDA 12.1](https://developer.nvidia.com/cuda-toolkit) для ускорения обучения моделей;
- [cuML](https://docs.rapids.ai/api/cuml/stable/) для ускорения обучения моделей;

# Corpus of financial news articles
Датасет со всеми статьями расположен в [репозитории](https://huggingface.co/datasets/denisalpino/YahooFinanceNewsRaw) на HuggingFace.

# Notes
## Project Structure
<details>
<summary>
FinABYSS
</summary>

```bash
├── README.md
├── data
│   ├── preprocessed
│   │   └── articles.parquet
│   └── raw
│       ├── articles.parquet
│       ├── news_urls.parquet
│       └── ohlcv.parquet
├── docs
│   ├── geopolitics.gif
│   ├── lasso.gif
│   ├── redirect.gif
│   ├── semantic_map.png
│   ├── semantic_similarity.gif
│   └── trump_by_dates.gif
├── notebooks
│   ├── aspects
│   │   ├── aspects.ipynb
│   │   ├── data
│   │   │   ├── data.csv
│   │   │   └── embeddings.npy
│   │   ├── hpo.ipynb
│   │   ├── img
│   │   │   ├── docs_and_topics.png
│   │   │   └── intertopic_distance_map.png
│   │   └── models
│   │       ├── v1
│   │       │   ├── aspect.pickle
│   │       │   └── hpo.csv
│   │       ├── v2
│   │       │   ├── hdbscan.pkl
│   │       │   └── umap.pkl
│   │       ├── v3
│   │       │   ├── hdbscan.pkl
│   │       │   └── umap.pkl
│   │       ├── v4
│   │       │   ├── hdbscan.pkl
│   │       │   ├── hp_importance.jpg
│   │       │   ├── hpo.jpg
│   │       │   └── umap.pkl
│   │       └── v5
│   │           ├── README.md
│   │           ├── hdbscan.pkl
│   │           └── umap.pkl
│   ├── data_collecting
│   │   ├── ohlcv.ipynb
│   │   └── yahoo_articles.ipynb
│   └── data_preprocessing
│       ├── articles_preprocessing.ipynb
│       ├── articles_vizualization.ipynb
│       ├── feature_extraction.ipynb
│       └── img
│           ├── dark
│           │   ├── articles_dist_by_dates.png
│           │   ├── articles_dist_by_dates_nvidia.png
│           │   ├── articles_dist_by_dates_tail.png
│           │   ├── top30_sources.png
│           │   ├── top30_tickers.png
│           │   └── wordcloud.png
│           └── light
│               ├── articles_dist_by_dates.png
│               ├── articles_dist_by_dates_nvidia.png
│               ├── articles_dist_by_dates_tail.png
│               ├── top30_sources.png
│               ├── top30_tickers.png
│               └── wordcloud.png
├── paper
│   ├── bibliography.bib
│   ├── img
│   │   ├── articles_dist_by_dates.png
│   │   ├── articles_dist_by_dates_tail.png
│   │   ├── datamap.png
│   │   ├── top30_sources.png
│   │   ├── top30_tickers.png
│   │   └── wordcloud.png
│   ├── main.pdf
│   ├── main.tex
│   ├── preamble.tex
│   ├── struct
│   │   ├── 00_title.tex
│   │   ├── 01_statement.tex
│   │   ├── 02_introduction.tex
│   │   ├── 03_conclusion.tex
│   │   ├── 10_theoretical_part
│   │   │   ├── 10_theoretical_part.tex
│   │   │   ├── 11_ ai_in_finance.tex
│   │   │   ├── 12_ml_algos.tex
│   │   │   ├── 13_deep_neural_networks.tex
│   │   │   └── 14_evaluation.tex
│   │   ├── 20_practical_part
│   │   │   ├── 20_practical_part.tex
│   │   │   ├── 21_limitations.tex
│   │   │   ├── 22_data_governance.tex
│   │   │   ├── 23_domain_adaptation.tex
│   │   │   └── 24_clustering_task.tex
│   │   └── 30_results_part
│   │       ├── 30_results_part.tex
│   │       ├── 31_benchmarks.tex
│   │       ├── 32_aspect_based_representation.tex
│   │       ├── 33_invented_architecture.tex
│   │       └── 34_semantic_deduplication_solution.tex
│   └── tab
│       ├── flue.tex
│       └── glue.tex
├── parsers
│   └── yahoo_parser.py
├── requirements.txt
├── semmap.html
└── utils
    ├── api_key_manager.py
    ├── custom_tqdm.py
    ├── metrics.py
    ├── proxy_manager.py
    └── vizualization.py
```
</details>

## P.S.
После того, как проект будет собран, необходимо установить пакет `pipreqsnb` и запустить из окружения команду `pipreqsnb --ignore .venv,venv --force`, которая автоматически просканирует проект, включая ноутбуки, и сформирует файл `requirements.txt`.
Опционально, если управление проектом осуществляется через WSL, тогда имеет смысл создать следующим образом алиас:

```alias pipreqsnb='pipreqsnb --ignore .venv,venv --force'```

[def]: #finabyss-financial-aspect-based-hybrid-semantic-system