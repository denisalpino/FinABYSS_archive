После того, как проект будет собран, необходимо установить пакет `pipreqsnb` и запустить из окружения команду `pipreqsnb --ignore .venv,venv --force`, которая автоматически просканирует проект, включая ноутбуки, и сформирует файл `requirements.txt`.
Опционально, если управление проектом осуществляется через WSL, тогда имеет смысл создать следующим образом алиас:

```alias pipreqsnb='pipreqsnb --ignore .venv,venv --force'```

[Репозиторий](https://github.com/itonik/spbu_diploma/tree/master) с LaTeX-шаблоном для ВКР по ГОСТам

Ключевые зависимости:
- [BERTopic](https://github.com/MaartenGr/BERTopic);
- [alpha_vantage](https://github.com/RomelTorres/alpha_vantage);
- [PyTorch](https://github.com/pytorch/pytorch);
- [TensorBoard](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html) (доки по работе с PyTorch)
- [FireDucks](https://fireducks-dev.github.io/docs/get-started/) -- для ускорения работы с данными (такой же API, как и у `pandas`)

Последовательность установок:
1. ```pip install bertopic```
2. ```pip install transformers```
3. ```pip install torch torchvision``` # для использования TensorBoard
4. ```pip install tensorboard```
5. ```pip install alpha_vantage``` # для OHLCV
6. ```pip install -U ipywidgets widgetsnbextension``` # для работы TQDM в Jupyter Notebooks
7. ```pip install polars``` # для работы с даннными (вместо pandas)
8. ```pip install aiohttp``` # для скрапинга
9. ```pip install selectolax``` # для парсинга
10. ```pip install tqdm``` # для отображения прогресса