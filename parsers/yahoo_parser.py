# Standart library
from asyncio import sleep, gather, Semaphore, TimeoutError, create_task
from dataclasses import dataclass, field
from random import uniform
from time import time
import os
import datetime
from typing import Collection, Dict, Iterable, List, Literal, Never, NoReturn, Set, Optional, Union

# External libraries
from aiohttp import ClientSession
from selectolax.parser import HTMLParser
from sympy import false
from ua_generator import generate

import polars as pl
from tqdm.notebook import tqdm


schema = {
    "title": pl.Utf8,
    "source": pl.Categorical,
    "datetime": pl.Utf8,
    "assets": pl.List(pl.Utf8),
    "text": pl.Utf8,
    "url": pl.Utf8
}


HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.5",
    "Cache-Control": "no-cache",
    "DNT": "1",
    "Pragma": "no-cache",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Upgrade-Insecure-Requests": "1"
}


def format_seconds(seconds):
    """Convert seconds to hours/minutes/seconds for TQDM widget"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours} h. {minutes} min. {seconds} sec."


@dataclass(slots=True)
class ParserState:
    pages_failed: Set[str] = field(default_factory=set)
    pages_in_progress: Set[str] = field(default_factory=set)
    articles_failed: Set[str] = field(default_factory=set)
    articles_in_progress: Set[str] = field(default_factory=set)
    news_urls: Set[str] = field(default_factory=set)
    session: Optional[ClientSession] = None


@dataclass(slots=True)
class Selectors:
    # ==================================== #
    #          SITEMAP SELECTORS           #
    # ==================================== #
    NEWS_LIST: str = "main > div > div > div > ul"
    NEWS_LIST_ELEMENT: str = "li > a"
    ERROR_404: str = "body > table"
    MAIN_CONTENT: str = "div.sitemapcontent"
    BUTTON_ELEMENT: str = "a"
    # ==================================== #
    #          ARTICLE SELECTORS           #
    # ==================================== #
    ARTICLE_FIGURE: str = "figure.yf-8xybrv"
    ARTICLE_TABLE: str = "div.table-container.yf-t4vsm6"
    ARTICLE_ADS: str = "div.wrapper.yf-eondll"
    ARTICLE_VISIBLE_TEXT: str = "div.atoms-wrapper"
    ARTICLE_HIDDEN_TEXT: str = "div.read-more-wrapper"
    ARTICLE_ASSETS_WRRAPPER: str = "div.ticker-list.yf-pqeumq > div > div > div.scroll-carousel.yf-r5lvmz"
    ARTICLE_ASSETS_LEVEL: str = "div.carousel-top.yf-pqeumq > div"
    ARTICLE_ASSET: str = "span.symbol.yf-1fqyif7"
    ARTICLE_TITLE: str = "div.cover-title.yf-1rjrr1"
    ARTICLE_DATETIME_TAG: str = "time.byline-attr-meta-time"
    ARTICLE_DATETIME_ATTR: str = "datetime"
    ARTICLE_SOURCE_TAG: str = "div.cover-wrap.yf-1rjrr1 > div.top-header.yf-1rjrr1 > a.subtle-link"
    ARTICLE_SOURCE_ATTR: str = "title"
    ARTICLE_SOURCE_TAG_ADDITIONAL: str = "div.byline-attr.yf-1k5w6kz > div > div.byline-attr-author.yf-1k5w6kz"
    ARTICLE_PREMIUM: str = "div.top-header.yf-1rjrr1 > a.topic-link"

class YahooFinanceParser:
    # TODO: make article parser
    # TODO: escape premium news
    def __init__(
            self, max_requests: int = 14, iter_delay: int = 30,
            req_delay_range: Iterable[int] | Iterable[float] = (4, 5)
    ) -> None:
        self.state = ParserState()
        self.selectors = Selectors()
        self.max_requests = Semaphore(max_requests)
        self.iter_delay = iter_delay
        self.req_delay_range = req_delay_range

    # ==================================== #
    #          SITEMAP FUNCTIONS           #
    # ==================================== #

    def __is_empty_page(self, tree: HTMLParser) -> bool:
        """
        Definition
        ---
        The function checks if the page is empty. With code 200,
        a blank page may come up. This can happen for various reasons,
        such as a large number of hits from the current client, high
        workload of the site or regional restrictions

        Parameters
        ---
        tree: selectolax.parser.HTMLParser
            Takes a tag tree as input

        Returns
        ---
        out: bool
            whether the page is the empty page

        Notes
        ---
        Also, this function additionally checks if the text that always
        comes with a 404 code is not located on the page. It is **very
        likely that this check is redundant**.
        """

        not_found = tree.css_matches(self.selectors.ERROR_404)
        empty_page = not tree.css_matches(self.selectors.MAIN_CONTENT)
        return not_found or empty_page

    def __is_end_page(self, tree: HTMLParser) -> bool:
        """
        Definition
        ---
        The function checks if the page is not the end page for the branch of the day.

        The page is considered the final page in two scenarios:
            - The page clearly contains the text “No Results found during this period.
            Please click Site Index to go back”
            - The page contains only the “Start” button

        Parameters
        ---
        tree: selectolax.parser.HTMLParser
            Takes a tag tree as input

        Returns
        ---
        out: bool
            whether the page is the end page for the branch of the day
        """

        end_text = "No Results found during this period. Please click Site Index to go back"
        start_button_text = "Start"
        content = tree.css_first(self.selectors.MAIN_CONTENT)

        if end_text not in content.text():
            news_urls_level = tree.css_first(self.selectors.NEWS_LIST)
            button_level = news_urls_level.next
            if button_level:
                last_button = button_level.css(self.selectors.BUTTON_ELEMENT)[-1]
                return start_button_text in last_button.text()
        return True

    def __parse_links(self, tree: HTMLParser) -> Union[Set[str], Set[Never]]:
        news_list = tree.css_first(self.selectors.NEWS_LIST)

        return {
            node.attributes.get("href")
            for node in news_list.css(self.selectors.NEWS_LIST_ELEMENT)
            if node and node.attributes.get('href')
        }  # type: ignore

    def __get_next_page(self, tree: HTMLParser) -> Optional[str]:
        button_level = tree.css_first(self.selectors.NEWS_LIST).next

        # То есть всегда, потому что наличие кнопок проверяли ранее
        if button_level:
            next_button = button_level.css(self.selectors.BUTTON_ELEMENT)[-1]
            return next_button.attributes.get('href') if next_button else None
        return None

    async def _process_page_of_links(self, url: str) -> None:
        """
        Функция

        Parameters
        ---
        url: str
            url to page with links to news articles (i.e. "https://finance.yahoo.com/sitemap/YYYY_MM_DD"
            may have an additional termination "_start" with 17 digits)

        Raises
        ---
        Exception
            "There is no links on this page: {url}"  or "There is no button level or 'Next' button on this page: {url}":
            if it wasn't detected in `fetch_page()`, `_is_empty_page()` or `_is_end_page()`.
        """
        # Скорее всего данная проверка лишняя, но пускай будет, чтобы просто перестраховаться
        if url in self.state.pages_in_progress:
            return None

        self.state.pages_in_progress.add(url)

        # Get the HTML tree of the passed `url`
        tree = await self.fetch_page(url)
        if tree is None:
            # Couldn't get the tree
            self.state.pages_in_progress.discard(url)
            self.state.pages_failed.add(url)
            return None
        elif self.__is_empty_page(tree):
            # Страница пустая
            self.state.pages_failed.add(url)
            return None
        elif self.__is_end_page(tree):
            # Страница последняя для ветки дня
            self.state.pages_in_progress.discard(url)
            return None

        if links := self.__parse_links(tree):
            self.state.news_urls.update(links) # type: ignore
        else:
            raise Exception(f"There is no links on this page: {url}")

        # Если кнопок не будет, то страница отлетит сразу после `_is_end_page()` выше
        if next_url := self.__get_next_page(tree):
            await self._process_page_of_links(next_url)
        else:
            raise Exception(f"There is no button level or 'Next' button on this page: {url}")

        self.state.pages_in_progress.discard(url)
        self.state.pages_failed.discard(url)
        return None

    async def get_all_news_in_range(
            self, start: str, end: str,
            retry: Union[Literal["non_stop"], int] = -1,
            iter_delay: Union[int, float] = 30,
            req_delay_range: Iterable[Union[float, int]] = (4, 5),
            max_requests: int = 14
    ) -> Set[str]:
        """
        Parameters
        ---
        start: str
            Parsing start date in YYYY-MM-DD format
        end: str:
            Parsing end date in YYYY-MM-DD format (inclusively)
        retry: typing.Literal["non_stop"] | int:
            When there is an initial problem accessing a page, it is placed in the `pages_failed`
            set. The `retry` parameter defines the maximum number of iterations for parsing links
            from url's in `pages_failed`. If the value “non-stop” is passed, the function will
            try to parse links from the `pages_failed` set until the set remains empty (highly
            discouraged because of the high possibility of an infinite loop). If the value -1
            is passed, then the function will not try to parse links from the `pages_failed`
            set at all.
        iter_delay: int | float = 30
            Defines the delay between iterations on parsing url's from the `pages_failed`
        req_delay_range: typing.Iterable[float | int] = (4, 5)
            Determines the delay before the GET-request is executed
        max_requests: int = 14
            Maximum number of requests that can be executed asynchronously (for parsing links
            to news articles, a value of 14 is highly recommended, as otherwise 404 and 429 error
            codes will become more frequent).

        Notes
        ---
        If setting retry does not help and the links are not all parsed,
        it means that Yahoo! Finance is blocking your IP address,
        change/enable your proxy.
        """

        # Convert dates to the required format
        start = datetime.date(*map(int, start.split("-"))) # type: ignore
        end = datetime.date(*map(int, end.split("-"))) # type: ignore
        urls = [
            f"https://finance.yahoo.com/sitemap/{dt.strftime("%Y_%m_%d")}"
            for dt in pl.date_range(start, end, closed="left", eager=True)
        ]

        results = await self.get_all_news(
            urls, retry=retry, iter_delay=iter_delay,
            req_delay_range=req_delay_range, max_requests=max_requests
        )
        return results

    async def get_all_news(
            self, urls: Collection, retry: Union[Literal["non_stop"], int] = -1,
            iter_delay: Union[int, float] = 30,
            req_delay_range: Iterable[Union[float, int]] = (4, 5),
            max_requests: int = 14
    ) -> Set[str]:
        """
        Parameters
        ---
        urls: typing.Collection
            Collection of urls, that need to be parsed.
        retry: typing.Literal["non_stop"] | int:
            When there is an initial problem accessing a page, it is placed in the `pages_failed`
            set. The `retry` parameter defines the maximum number of iterations for parsing links
            from url's in `pages_failed`. If the value “non-stop” is passed, the function will
            try to parse links from the `pages_failed` set until the set remains empty (highly
            discouraged because of the high possibility of an infinite loop). If the value -1
            is passed, then the function will not try to parse links from the `pages_failed`
            set at all.
        iter_delay: int | float = 30
            Defines the delay between iterations on parsing url's from the `pages_failed`
        req_delay_range: typing.Iterable[float | int] = (4, 5)
            Determines the delay before the GET-request is executed
        max_requests: int = 14
            Maximum number of requests that can be executed asynchronously (for parsing links
            to news articles, a value of 14 is highly recommended, as otherwise 404 and 429 error
            codes will become more frequent).
            """
        self.iter_delay = iter_delay
        self.req_delay_range = req_delay_range
        self.max_requests = Semaphore(max_requests)

        size = 8190 * 2

        # Opening the session
        async with ClientSession(max_line_size=size, max_field_size=size) as session:
            self.state.session = session
            # Variables for tracking progress and stopping work
            iter_counter, current_retry, was_progress = 1, 0, True

            # Create TQDM widget for Jupyter Notebooks launching
            # ========================================================================================== #
            pbar = tqdm(
                total=len(urls),
                desc=f"Iteration {iter_counter}",
                bar_format="{desc}: {percentage:.0f}%|{bar}| {n_fmt}/{total_fmt} [It's been: {postfix}]",
                postfix="0 h. 0 min. 0 sec."
            )
            start_time = time()

            def update_task(_, pbar=pbar, start=start_time):
                elapsed = time() - start
                pbar.set_postfix_str(format_seconds(elapsed))
                pbar.update(1)

            tasks = []
            for url in urls:
                task = create_task(self._process_page_of_links(url))
                task.add_done_callback(update_task)
                tasks.append(task)
            # ========================================================================================== #
            # Run the first main wave to complete the `pages_failed` set
            await gather(*tasks)
            pbar.close()

            first_wave_results = len(self.state.news_urls)
            print(
                f"At iteration №{iter_counter} were collected {first_wave_results} URLs\n"
                "=================================================================="
            )

            if retry == -1:
                self.state.session = None
                return self.state.news_urls

            # Run the second additional wave
            while self.state.pages_failed:
                await sleep(self.iter_delay)
                iter_counter += 1
                previous_iter_pages_failed = self.state.pages_failed.copy()
                previous_iter_results = len(self.state.news_urls)
                # Create TQDM widget for Jupyter Notebooks launching
                # ========================================================================================== #
                pbar = tqdm(
                    total=len(self.state.pages_failed),
                    desc=f"Iteration {iter_counter}",
                    bar_format="{desc}: {percentage:.0f}%|{bar}| {n_fmt}/{total_fmt} [It's been: {postfix}]",
                    postfix="0 h. 0 min. 0 sec."
                )
                start_time = time()
                def update_task(_, pbar=pbar, start=start_time):
                    elapsed = time() - start
                    pbar.set_postfix_str(format_seconds(elapsed))
                    pbar.update(1)

                tasks = []
                for failed_link in self.state.pages_failed:
                    task = create_task(self._process_page_of_links(failed_link))
                    task.add_done_callback(update_task)
                    tasks.append(task)
                # ========================================================================================== #
                await gather(*tasks)
                pbar.close()

                current_iter_progress = len(previous_iter_pages_failed - self.state.pages_failed)
                current_iter_results = len(self.state.news_urls)

                print(
                    f"At iteration №{iter_counter} URLs were processed: "
                    f"{current_iter_progress} out of {len(previous_iter_pages_failed)}\n"
                    "Collected URLs per iteration: "
                    f"{current_iter_results - previous_iter_results}\n"
                    "=================================================================="
                )

                # Exit condition
                if retry != "non_stop":
                    if current_iter_progress == 0:
                        if was_progress == True:
                            current_retry, was_progress = 0, False
                        current_retry += 1
                        if current_retry >= retry:
                            break
                    else:
                        was_progress = True

        self.state.session = None
        print(f"Total сollected URLs {len(self.state.news_urls)}")

        return self.state.news_urls

    # ==================================== #
    #           GENERAL FUNCTION           #
    # ==================================== #

    async def fetch_page(self, url: str) -> Optional[HTMLParser]:
        """
        Description
        ---
        The function tries to make a GET request on `url` and get the HTML tree of the page.

        Observed errors:
            - Page with links to the day's news:
                - `404`
                "Will be right back...
                Thank you for your patience.
                Our engineers are working quickly to resolve the issue."
                - `429`
                "too many requests"
                - `502`
                    ...
            - Article pages:
                - ...

        For each GET request, the function **generates a random User-Agent** and pauses
        randomly asynchronously within `self.req_delay_range` to reduce the likelihood
        of exceeding an **unknown request rate limit** from a single IP address.

        All errors occurring within this function, except for session errors,
        are output as text to the terminal

        Parameters
        ---
        url: str
            any link to Yahoo! Finance (for example "https://finance.yahoo.com/sitemap/YYYY_MM_DD"
            or "https://finance.yahoo.com/news/ARTICLE-ID.html")

        Raises
        ---
        Exception("There is no opened session.")
            if for some reasons the asynchronous client session was not opened
        asyncio.TimeoutError
            the reasons can be varied, but they are almost always related
            to your or proxy server slow internet connection
        """

        if self.state.session is None:
            raise Exception("There is no opened session.")

        headers = {
            **HEADERS,
            "User-Agent": generate(
                device='desktop',
                browser='chrome',
                platform=['windows', 'macos']
            ).text
        }

        async with self.max_requests:
            await sleep(uniform(*self.req_delay_range))

            try:
                response = await self.state.session.get(url, headers=headers)
            except TimeoutError as error:
                print(error, url)
                return None

            if response.status == 200:
                html = await response.text()
                tree = HTMLParser(html)
                return tree
            # elif response.status not in (404, 429, 502, 500):
            print(response.status, url)
            return None

    # ==================================== #
    #          ARTICLE FUNCTIONS           #
    # ==================================== #

    def __get_source(self, tree: HTMLParser) -> str:
        """Функция для извлечения заголовка новости"""
        if source_tag := tree.css_first(self.selectors.ARTICLE_SOURCE_TAG, default=None):
            return source_tag.attributes.get(self.selectors.ARTICLE_SOURCE_ATTR) # type: ignore
        return tree.css_first(self.selectors.ARTICLE_SOURCE_TAG_ADDITIONAL).text(strip=True)

    def __get_datetime(self, tree: HTMLParser) -> str:
        """Функция для извлечения заголовка новости. Возвращает в часовом поясе GMT-0"""
        dt = (
            tree
            .css_first(self.selectors.ARTICLE_DATETIME_TAG)
            .attributes
            .get(self.selectors.ARTICLE_DATETIME_ATTR)
        )
        return dt # type: ignore

    def __get_title(self, tree: HTMLParser) -> str:
        """Функция для извлечения заголовка новости"""
        return tree.css_first(self.selectors.ARTICLE_TITLE).text(strip=True)

    def __get_assets(self, tree) -> Union[List[str], List[Never]]:
        """Функция для извлечения тикеров из-под заголовка"""

        assets: Union[List[str], List[Never]] = []
        # Проверяем есть ли тикеры
        if assets_wrapper := tree.css_first(self.selectors.ARTICLE_ASSETS_WRRAPPER, default=None):
            # Итерируемся по каждому соответствующему тикеру <div>
            for asset_node in assets_wrapper.css(self.selectors.ARTICLE_ASSETS_LEVEL):
                assets.append(
                    asset_node
                    .css_first(self.selectors.ARTICLE_ASSET)
                    .text(strip=True)
                )
        return assets

    def __prepare_text(self, tree: HTMLParser) -> NoReturn: # type: ignore
        """
        Данная функция убирает из дерева `tree` все элементы, которые не нужно парсить:
            - изображения;
            - рекламу;
            - таблицы.
        """
        for selector in (self.selectors.ARTICLE_FIGURE, self.selectors.ARTICLE_TABLE, self.selectors.ARTICLE_ADS):
            for node in tree.css(selector):
                node.decompose() # Удаляем из всего HTML-документа блоки

    def __parse_text(self, tree: HTMLParser) -> str:
        # Удаляем все ненужное
        self.__prepare_text(tree)

        # Собираем видимый текст
        visible_text = "\n".join(
            [
                paragraph.text()
                for paragraph in tree.css_first(self.selectors.ARTICLE_VISIBLE_TEXT).iter()
            ]
        )
        # Собираем скрытый текст
        if hidden_text := tree.css_first(self.selectors.ARTICLE_HIDDEN_TEXT, default=None):
            # Собираем текст по абзацам, добавляя переносы строк
            hidden_text = "\n".join([paragraph.text() for paragraph in hidden_text.iter()])
            # Собираем видимую часть со скрытой и возвращаем
            return "\n".join((visible_text, hidden_text))
        return visible_text

    def parse_article(self, tree: HTMLParser, url: str) -> Dict[str, Union[str, Union[List[str], List[Never]]]]:
        return {
            "title": self.__get_title(tree),
            "source": self.__get_source(tree),
            "datetime": self.__get_datetime(tree),
            "assets": self.__get_assets(tree),
            "text": self.__parse_text(tree),
            "url": url
        }

    def __is_premium_article(self, tree: HTMLParser) -> bool:
        """Checks if article is available only with PERMIUM access"""
        return True if tree.css_first(self.selectors.ARTICLE_PREMIUM) else False

    async def get_article(self, url: str):

        self.state.articles_in_progress.add(url)

        # Get the HTML tree of the passed `url`
        tree = await self.fetch_page(url)
        if tree is None:
            # Couldn't get the tree
            self.state.articles_in_progress.discard(url)
            self.state.articles_failed.add(url)
            print("no tree", url)
            print("================================================")
            return {
                "title": '',
                "source": '',
                "datetime": '',
                "assets": [''],
                "text": '',
                "url": url
            }

        if self.__is_premium_article(tree):
            self.state.articles_in_progress.discard(url)
            print("Платная статья", url)
            print("================================================")
            return {
                "title": '',
                "source": '',
                "datetime": '',
                "assets": [''],
                "text": '',
                "url": url
            }

        try:
            article_data = self.parse_article(tree, url)
        except Exception as e:
            print("Ошибка", e, url)
            print("================================================")
            self.state.pages_in_progress.discard(url)
            self.state.pages_failed.discard(url)
            return {
                "title": '',
                "source": '',
                "datetime": '',
                "assets": [''],
                "text": '',
                "url": url
            }
        print("Все ок:", article_data)
        print("================================================")
        self.state.pages_in_progress.discard(url)
        self.state.pages_failed.discard(url)

        return article_data

    async def get_all_articles(self, file_path: str, urls: Iterable[str], max_requests: int = 14) -> pl.DataFrame:
        # TODO: Refine this function
        self.max_requests = Semaphore(max_requests)

        size = 8190 * 2

        # Opening the session
        async with ClientSession(max_line_size=size, max_field_size=size) as session:
            self.state.session = session
            news_data = await gather(*[self.get_article(url) for url in urls])

        self.state.session = None

        return pl.DataFrame(data=news_data, schema=schema, infer_schema_length=None) # infer_schema_length=None во избежание ошбки ComputeError: could not append value: [] of type: list[null] to the builder; make sure that all rows have the same schema or consider increasing `infer_schema_length`


#    def write_batch(self, batch_df: pl.DataFrame, batch_index: int, output_dir: str = "batches"):
#        if not os.path.exists(output_dir):
#            os.makedirs(output_dir)
#        filename = os.path.join(output_dir, f"batch_{batch_index:05d}.parquet")
#        # Можно выбрать желаемый алгоритм сжатия, например, "zstd"
#        batch_df.write_parquet(filename, compression="zstd")
#        print(f"Батч {batch_index} сохранён: {batch_df.shape[0]} строк, файл: {filename}")
