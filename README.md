После того, как проект будет собран, необходимо установить пакет `pipreqsnb` и запустить из окружения команду `pipreqsnb --ignore .venv,venv --force`, которая автоматически просканирует проект, включая ноутбуки, и сформирует файл `requirements.txt`.
Опционально, если управление проектом осуществляется через WSL, тогда имеет смысл создать следующим образом алиас:

```alias pipreqsnb='pipreqsnb --ignore .venv,venv --force'```

[Репозиторий](https://github.com/itonik/spbu_diploma/tree/master) с LaTeX-шаблоном для ВКР по ГОСТам

Ключевые зависимости:
- [BERTopic](https://github.com/MaartenGr/BERTopic);
- [alpha_vantage](https://github.com/RomelTorres/alpha_vantage);
- [PyTorch](https://github.com/pytorch/pytorch);
- [TensorBoard](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html) (доки по работе с PyTorch)
- [Polars](https://docs.pola.rs/) -- для ускорения работы с данными (похожий API на `pandas`)