# Standart library
import os
import datetime
from random import uniform
from dataclasses import dataclass, field
from asyncio import (
    Task,
    sleep,
    gather,
    create_task,
    Semaphore,
    TimeoutError
)
from typing import (
    Any,
    Collection,
    Sequence,
    Iterable,
    Dict,
    List,
    Set,
    Literal,
    Never,
    NoReturn,
    Optional,
    Union
)

# External libraries
from aiohttp import ClientSession, BasicAuth, ClientConnectorError
from selectolax.parser import HTMLParser
from ua_generator import generate
import polars as pl
from tqdm.notebook import tqdm

from utils.custom_tqdm import wrap_with_tqdm
from utils.proxy_manager import ProxyManager, ProxyDense


schema = {
    "title": pl.Utf8,
    "source": pl.Utf8,
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


@dataclass(slots=True)
class ParserState:
    pages_failed: Set[str] = field(default_factory=set)
    pages_in_progress: Set[str] = field(default_factory=set)
    # TODO: Разработать функционал кэширования для страниц (по факту переименовать все из news_urls)
    pages_cache: Set[str] = field(default_factory=set)
    articles_failed: Set[str] = field(default_factory=set)
    articles_in_progress: Set[str] = field(default_factory=set)
    articles_cache: List[Dict] = field(default_factory=list)
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


async def fetch_page(url: str, session: ClientSession, semaphore: Semaphore, delay: Union[int, float]) -> Optional[HTMLParser]:
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
            - `404`
            - `429`
            "too many requests"
            - `500`

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

    headers = {
        **HEADERS,
        "User-Agent": generate(
            device='desktop',
            browser='chrome',
            platform=['windows', 'macos']
        ).text
    }

    async with semaphore:
        await sleep(delay)
        try:
            response = await session.get(url, headers=headers)
        except TimeoutError as error:
            print(f"Cannot fetch the page {url=} because of INTERNET CONNECTION or REGIONAL RESTRICTIONS: {error.__repr__()}")
            return None
        except ClientConnectorError as error:
            print(f"Cannot fetch the page {url=} because of REDIRECTION: {error.__repr__()}")
            return None

        if response.status == 200:
            try:
                html = await response.text()
            except RuntimeError as error:
                print(f"Cannot fetch a text from the page {url=} because of PROXY BLOCKING: {error.__repr__()}")
                return None
            except Exception as error:
                print(f"Cannot fetch a text from the page {url=} UNKNOWN REASON: {error.__repr__()}")
                return None

            tree = HTMLParser(html)
            return tree
        # elif response.status not in (404, 429, 502, 500):
        print(f"Cannot process the page {url=} because of STATUS CODE {response.status}.")
        return None


class YahooFinanceParser:

    # ==================================== #
    #           GENERAL FUNCTIONS          #
    # ==================================== #

    def __init__(
            self, max_requests: int = 14, iter_delay: int = 30,
            req_delay_range: Iterable[int] | Iterable[float] = (4, 5)
    ) -> None:
        self.state = ParserState()
        self.selectors = Selectors()
        self.proxy_manager = ProxyManager()
        self.max_requests = Semaphore(max_requests)
        self.iter_delay = iter_delay
        self.req_delay_range = req_delay_range

    def add_proxy(
            self,
            url: Optional[str] = None,
            proxy: Optional[ProxyDense] = None,
            protocol: str = "",
            login: str = "", password: str = "",
            ip_adress: str = "", port: str = ""
    ) -> None:
        self.proxy_manager.add_proxy(url, proxy, protocol, login, password, ip_adress, port)

    async def do_with_proxy(self, tasks: List[Task], proxy: ProxyDense) -> List[Any]:
        # TODO: Решить куда и как сохранять сессии
        size = 8190 * 20

        # Opening asyncronious session
        async with ClientSession(max_line_size=size, max_field_size=size, proxy=str(proxy)):
            return await gather(*tasks)

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

    async def _process_page_of_links(self, url: str, session: ClientSession, max_requests: Semaphore) -> None:
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
        tree = await fetch_page(
            url, session, max_requests,
            uniform(*self.req_delay_range)
        )
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
            await self._process_page_of_links(next_url, session, self.max_requests)
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
            max_requests: int = 14,
            logging: bool = True
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
            # Variables for tracking progress and stopping work
            iter_counter, current_retry, was_progress = 1, 0, True
            tasks = [create_task(self._process_page_of_links(url, session, self.max_requests)) for url in urls]

            # Create TQDM widget for Jupyter Notebooks launching
            pbar = wrap_with_tqdm(
                desc=f"Iteration {iter_counter}",
                tasks=tasks,
                hide=False
            )
            # Run the first main wave to complete the `pages_failed` set
            await gather(*tasks)
            pbar.close()

            first_wave_results = len(self.state.news_urls)
            if logging:
                print(
                    f"At iteration №{iter_counter} were collected {first_wave_results} URLs\n"
                    "=================================================================="
                )

            if retry == -1:
                return self.state.news_urls

            # Run the second additional wave
            while self.state.pages_failed:
                await sleep(self.iter_delay)
                iter_counter += 1
                previous_iter_pages_failed = self.state.pages_failed.copy()
                previous_iter_results = len(self.state.news_urls)

                tasks = [create_task(self._process_page_of_links(url, session, self.max_requests)) for url in self.state.pages_failed]

                # Create TQDM widget for Jupyter Notebooks launching
                pbar = wrap_with_tqdm(
                    desc=f"Iteration {iter_counter}",
                    tasks=tasks,
                    hide=False
                )
                await gather(*tasks)
                pbar.close()

                current_iter_progress = len(previous_iter_pages_failed - self.state.pages_failed)
                current_iter_results = len(self.state.news_urls)

                if logging:
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

        print(f"Total сollected URLs {len(self.state.news_urls)}")

        return self.state.news_urls

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
        return sorted(assets)

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
        # Delete all redundant elements from html-tree
        self.__prepare_text(tree)

        # Parse visible text (before the button "Show story")
        visible_text = "\n".join(
            [
                paragraph.text()
                for paragraph in tree.css_first(self.selectors.ARTICLE_VISIBLE_TEXT).iter()
            ]
        )
        # Parse hidden text (after the button "Show story") if it exists
        if hidden_text := tree.css_first(self.selectors.ARTICLE_HIDDEN_TEXT, default=None):
            # Putting textual parts of the hidden text together using line breaks
            hidden_text = "\n".join([paragraph.text() for paragraph in hidden_text.iter()])
            # Putting hidden and visible parts together
            return "\n".join((visible_text, hidden_text))
        return visible_text

    def parse_article(self, tree: HTMLParser, url: str) -> Dict[str, Union[str, Union[List[str], List[Never]]]]:
        """
        # TODO: Write docstring
        """
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

    async def get_article(self, url: str, session: ClientSession, max_requests: Semaphore) -> None:
        """
        # TODO: Write docstring
        """
        self.state.articles_in_progress.add(url)

        # Get the HTML tree of the passed URL
        tree = await fetch_page(
            url, session, max_requests,
            uniform(*self.req_delay_range)
        )
        if tree is None:
            # Couldn't get the tree
            self.state.articles_in_progress.discard(url)
            self.state.articles_failed.add(url)
            return

        # Check if article is available only for PREMIUM users
        if self.__is_premium_article(tree):
            self.state.articles_in_progress.discard(url)
            return

        # Try to parse html-tree
        try:
            article_data = self.parse_article(tree, url)
        except AttributeError as error:
            # 'NoneType' object has no attribute 'text' -- из-за банера с cookies для некоторых стран
            # 'NoneType' object has no attribute 'text' -- из-за того, что ссылка была не на домене Yahoo! Finance
            # 'NoneType' object has no attribute 'iter' -- из-за задержек инфраструктуры Yahoo! Finance пришел битый текст
            print(f"AttributeError occured during PARSING TEXT on {url=}: {error.__repr__()}")
            self.state.articles_failed.add(url)
            self.state.articles_in_progress.discard(url)
            return
        except Exception as error:
            print(f"Unknown error occured during PARSING TEXT on {url=}: {error.__repr__()}")
            self.state.articles_failed.add(url)
            self.state.articles_in_progress.discard(url)
            return

        # Delete url from both sets
        self.state.articles_failed.discard(url)
        self.state.articles_in_progress.discard(url)

        # Add article in cache
        self.state.articles_cache.append(article_data)

    async def get_all_articles(
            self,
            urls: Sequence[str],
            *,
            file_path: Optional[str] = None,
            chunk_size: int = 1000,
            tqdm_progress: bool = True,
            use_proxies: bool = False,
            logging: bool = False,
    ) -> pl.DataFrame:
        """
        For proxies chunking is not implemented.
        ---
        Parameters:
        file_path: Optional[str]
            starts without "/", contain target file name (example: "data/raw/articles.parquet")
        """
        # TODO: Возвращать то, что спаршено именно сейчас, а не вместе с исходным содержимым целевого файла

        if use_proxies:
            df = await self._get_articles_with_proxies(
                urls, file_path=file_path, chunk_size=chunk_size,
                logging=logging, tqdm_progress=tqdm_progress
            )
            return df

        size = 8190 * 20
        # Open asyncronious session
        async with ClientSession(max_line_size=size, max_field_size=size) as session:
            tasks = [create_task(self.get_article(url, session, self.max_requests)) for url in urls]

            if tqdm_progress: pbar = wrap_with_tqdm("Processing articles", tasks, hide=False)

            if file_path:
                chunk_index = 0
                chunk_dir = f"{"/".join(file_path.split("/")[:-1])}/chunks/".lstrip("/")
                chunk_file_paths = []

                # Create folder for chunks if it's not exists
                if not os.path.isdir(chunk_dir):
                    os.mkdir(chunk_dir)

                try:
                    for start in range(0, len(urls), chunk_size):
                        # If something will go wrong during concatenation, parsed articles will save in cache
                        self.state.articles_cache = list()
                        # Take slice of tasks
                        chunk = tasks[start:start+chunk_size]
                        # Run article parsing on current chunk
                        await gather(*chunk)
                        # Write chunk to the Parquet file
                        self.__save_chunk(chunk_dir, chunk_index, chunk_file_paths, logging)
                        chunk_index += 1
                    # Clear cache
                    self.state.articles_cache = list()
                except Exception as error:
                    raise error
                finally:
                    if tqdm_progress: pbar.close()
                    # Grab all temporary chunked files (that was written during current function call),
                    # all cached data (if it exists), previously written data (if it exists)
                    # for merge and write it all to Parquet file
                    df = self.__save_articles(file_path, chunk_dir, chunk_file_paths, logging)
                    return df

            # Run all article parsing in once
            await gather(*tasks)
        if tqdm_progress: pbar.close()

        df = pl.DataFrame(data=self.state.articles_cache, schema=schema, infer_schema_length=None)

        if logging:
            print(
                "==============================================================================\n"
                f"URLs parsed successfully: {df.shape[0]} rows.\n"
                f"Size (estimated): {df.estimated_size("mb"):.2f} MB.\n"
                "=============================================================================="
            )
        return df

    async def _get_articles_with_proxies(
            self,
            urls: Sequence[str],
            *,
            file_path: Optional[str] = None,
            chunk_size: int = 1000,
            logging: bool = False,
            tqdm_progress: bool = True
    ):
        """..."""
        # Ckeck whether any proxy exists
        if not self.proxy_manager.proxies:
            raise Exception("There is no proxies. Add proxies using method `add_proxy()`")

        chunk_size, residue = divmod(len(urls), len(self.proxy_manager.proxies))
        all_tasks = []
        size = 8190 * 20

        for i, proxy in enumerate(self.proxy_manager.proxies):
            # Create proxy related asyncronious session
            proxy.session = ClientSession(
                max_line_size=size, max_field_size=size,
                proxy=str(proxy)
            )
            # Forming a link chunk
            proxy_urls = urls[i * chunk_size + min(i, residue): (i + 1) * chunk_size + min(i + 1, residue)]
            # Convert links into tasks
            for url in proxy_urls:
                task = create_task(self.get_article(url, proxy.session, proxy.max_requests))
                all_tasks.append(task)

        if tqdm_progress: pbar = wrap_with_tqdm("Processing articles", all_tasks, hide=False)
        # Run all article parsing in once
        await gather(*all_tasks)
        if tqdm_progress: pbar.close()

        # Close each proxy related asyncronious session
        for proxy in self.proxy_manager.proxies:
            await proxy.session.close() # type: ignore
            proxy.session = None

        if file_path:
            # Grab all temporary chunked files (that was written during current function call),
            # all cached data (if it exists), previously written data (if it exists)
            # for merge and write it all to Parquet file
            df = self.__save_articles(file_path, logging=logging)
            return df

        df = pl.DataFrame(data=self.state.articles_cache, schema=schema, infer_schema_length=None)

        if logging:
            print(
                "==============================================================================\n"
                f"URLs parsed successfully: {df.shape[0]} rows.\n"
                f"Size (estimated): {df.estimated_size("mb"):.2f} MB.\n"
                "=============================================================================="
            )
        return df

    def __save_chunk(
            self,
            chunk_dir: str,
            chunk_index: int,
            chunk_file_paths: List[str],
            logging: bool
    ) -> None:
        """
        Parameters
        ---
        chunk_file_paths: str
            weakref to list
        """
        chunk_df = pl.DataFrame(
            data=self.state.articles_cache,
            schema=schema,
            infer_schema_length=None
        )
        # Save DataFrame with chunk to Parquet file
        chunk_file_path = chunk_dir + "_".join(["chunk", str(chunk_index)]) + ".parquet"
        chunk_df.write_parquet(file=chunk_file_path)

        chunk_file_paths.append(chunk_file_path)

        if logging:
            print(
                "==============================================================================\n"
                f"Chunk {chunk_index} saved successfully: {chunk_df.shape[0]} rows.\n"
                f"File: {chunk_file_path}.\n"
                f"Size (estimated): {chunk_df.estimated_size("mb"):.2f} MB.\n"
                f"Size (real): {os.path.getsize(chunk_file_path) / (1024 * 1024):.2f} MB.\n"
                "=============================================================================="
            )

    def __save_articles(
            self,
            file_path: str,
            chunk_dir: Optional[str] = None,
            chunk_file_paths: Optional[List[str]] = None,
            logging: bool = False
    ) -> pl.DataFrame:
        """
        Если есть chunk_dir, то и chunk_file_paths должен быть
        Parameters
        ---
        chunk_file_paths: str
            weakref to list
        """
        dfs = []

        if chunk_dir:
            # Grab all temporary chunked files, that was written during current function call
            dfs = [
                pl.read_parquet(chunk_file_path)
                for chunk_file_path in chunk_file_paths # type: ignore
            ]

        # Grab all cached data if it exists
        if self.state.articles_cache:
            cache = pl.DataFrame(
                data=self.state.articles_cache,
                schema=schema,
                infer_schema_length=None
            )
            dfs.append(cache)

        # Grab all previously written data if it exists
        if os.path.isfile(file_path):
            dfs.append(pl.read_parquet(file_path))

        # Merge and write all collected data
        merged_df = pl.concat(dfs)
        merged_df.write_parquet(file_path)

        if logging:
            print(
                "==============================================================================\n"
                f"Articles were saved successfully: {merged_df.shape[0]} rows.\n"
                f"File: {file_path}.\n"
                f"Size (estimated): {merged_df.estimated_size("mb"):.2f} MB.\n"
                f"Size (real): {os.path.getsize(file_path) / (1024 * 1024):.2f} MB.\n"
                "=============================================================================="
            )

        if chunk_dir:
            # Delete all temporary files
            for chunk_file_path in chunk_file_paths: # type: ignore
                os.remove(chunk_file_path)
            os.rmdir(chunk_dir)

        return merged_df