from typing import List


# TODO: Расширить класс `APIKeyManager` таким образом, чтобы он мог управлять ключами для разных API одновременно
# TODO: Нужно подумать стоит ли реализовать паттерн синглтона (переопределить дандер __new__)
class APIKeyManager:
    '''Класс для управления API ключами'''

    def __init__(self, keys: List[str]):
        self.keys = keys.copy()
        self._index = 0
        self._current_service = None

    def get_next_key(self) -> str:
        '''Метод возвращает следующий API-ключ, если таковой имеется, либо возбуждает исключение'''
        if self.is_switchable:
            print(f"API-ключ для {self._current_service} больше недействителен. "
                  "Переходим к следующему ключу.")
            self._index += 1
            return self.keys[self._index]
        raise Exception("Закончились доступные API-ключи. "
                        "Добавьте новые или дождитесь сбрасывния лимитов.")

    def get_current_key(self) -> str:
        '''Метод возвращает текущий используемый API-ключ'''
        return self.keys[self._index]

    @property
    def is_switchable(self) -> bool:
        '''Свойство дает понять остались ли еще неиспользованные API-ключи'''
        if self._index + 1 < len(self.keys):
            return True
        return False