# Standart library
from asyncio import sleep, gather, Semaphore, TimeoutError, create_task
from dataclasses import dataclass, field
from random import uniform
from time import time
from typing import Any, Coroutine, Iterable, List, Literal, Set, Optional

# External libraries
from aiohttp import ClientSession
from selectolax.parser import HTMLParser
from ua_generator import generate

import pandas as pd
import polars as pl
from tqdm.notebook import tqdm


schema = {
    "title": pl.String,
    "publication_datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
    "publication_type": pl.Categorical,
    "assets": pl.List(pl.String),
    "resource": pl.Categorical,
    "source": pl.Categorical,
    "text": pl.String
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
    parse_failed: Set[str] = field(default_factory=set)
    in_progress: Set[str] = field(default_factory=set)
    news_urls: Set[str] = field(default_factory=set)
    session: Optional[ClientSession] = None


@dataclass(slots=True)
class Selectors:
    NEWS_LIST: str = "main > div > div > div > ul"
    NEWS_LIST_ELEMENT: str = "li > a"
    ERROR_404: str = "body > table"
    MAIN_CONTENT: str = "div.sitemapcontent"
    BUTTON_ELEMENT: str = "a"


class YahooFinanceParser:
    # TODO: unite with article parser
    def __init__(self) -> None:
        self.state = ParserState()
        self.selectors = Selectors()
        self.max_requests = Semaphore(14)
        self.iter_delay = 30
        self.req_delay_range = (4, 5)

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
                self.state.parse_failed.add(url)
                print(error, url)
                return None

            if response.status == 200:
                http = await response.text()
                tree = HTMLParser(http)
                return tree
            else:
                self.state.parse_failed.add(url)
                if response.status not in (404, 429, 502):
                    print(response.status, url)
                return None

    def __parse_links(self, tree: HTMLParser) -> Set[str | None]:
        news_list = tree.css_first(self.selectors.NEWS_LIST)

        return {
            node.attributes.get("href")
            for node in news_list.css(self.selectors.NEWS_LIST_ELEMENT)
            if node and node.attributes.get('href')
        }

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
        if url in self.state.in_progress:
            return None

        self.state.in_progress.add(url)

        # Получаем HTML-дерево переданного `url`
        tree = await self.fetch_page(url)

        if tree is None:
            # Не смогли получить дерево
            self.state.in_progress.discard(url)
            return None
        elif self.__is_empty_page(tree):
            # Страница пустая
            self.state.parse_failed.add(url)
            return None
        elif self.__is_end_page(tree):
            # Страница последняя для ветки дня
            self.state.in_progress.discard(url)
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

        self.state.in_progress.discard(url)
        self.state.parse_failed.discard(url)
        return None

    async def get_all_news_in_range(
            self, start: str, end: str,
            retry: Literal["non_stop"] | int,
            iter_delay: int | float = 30, req_delay_range: Iterable[float | int] = (4, 5)
    ) -> Set[str]:
        """
        Parameters
        ---
        start: str
            Parsing start date in YYYY-MM-DD format
        end: str:
            Parsing end date in YYYY-MM-DD format (inclusively)
        retry: typing.Literal["non_stop"] | int:
            When there is an initial problem accessing a page, it is placed in the `parse_failed`
            set. The `retry` parameter defines the maximum number of iterations for parsing links
            from url's in `parse_failed`.
        iter_delay: int | float = 30
            Defines the delay between iterations on parsing url's from the `parse_failed` set
        req_delay_range: typing.Iterable[float | int] = (4, 5)
            Determines the delay before the GET-request is executed

        Notes
        ---
        If setting retry does not help and the links are not all parsed,
        it means that Yahoo! Finance is blocking your IP address,
        change/enable your proxy.
        """

        self.iter_delay = iter_delay
        self.req_delay_range = req_delay_range

        # Convert dates to the required format
        dates = [
            f"https://finance.yahoo.com/sitemap/{date.strftime("%Y_%m_%d")}"
            for date in pd.date_range(start, end, inclusive="left")
        ]
        size = 8190 * 2

        # Opening the session
        async with ClientSession(max_line_size=size, max_field_size=size) as session:
            self.state.session = session
            # Variables for tracking progress and stopping work
            iter_counter, current_retry, was_progress = 1, 0, True

            # Create TQDM widget for Jupyter Notebooks launching
            # ========================================================================================== #
            pbar = tqdm(
                total=len(dates),
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
            for date in dates:
                task = create_task(self._process_page_of_links(date))
                task.add_done_callback(update_task)
                tasks.append(task)
            # ========================================================================================== #
            # Run the first main wave to complete the `parse_failed` set
            await gather(*tasks)
            pbar.close()

            first_wave_results = len(self.state.news_urls)
            print(
                f"At iteration №{iter_counter} were collected {first_wave_results} URLs\n"
                "=================================================================="
            )

            # Run the second additional wave
            while self.state.parse_failed:
                await sleep(self.iter_delay)
                iter_counter += 1
                previous_iter_parse_failed = self.state.parse_failed.copy()
                previous_iter_results = len(self.state.news_urls)
                # Create TQDM widget for Jupyter Notebooks launching
                # ========================================================================================== #
                pbar = tqdm(
                    total=len(self.state.parse_failed),
                    desc=f"Iteration {iter_counter}",
                    bar_format="{desc}: {percentage:.0f}%|{bar}| {n_fmt}/{total_fmt} [It's been: {postfix}]",
                    postfix="0 h. 0 min. 0 sec."
                )
                start_time = time()
                tasks = []
                for failed_link in self.state.parse_failed:
                    task = create_task(self._process_page_of_links(failed_link))
                    task.add_done_callback(update_task)
                    tasks.append(task)
                # ========================================================================================== #
                await gather(*tasks)
                pbar.close()

                current_iter_progress = len(previous_iter_parse_failed - self.state.parse_failed)
                current_iter_results = len(self.state.news_urls)

                print(
                    f"At iteration №{iter_counter} URLs were processed: "
                    f"{current_iter_progress} out of {len(previous_iter_parse_failed)}\n"
                    "Collected URLs per iteration: "
                    f"{current_iter_results - previous_iter_results}\n"
                    "=================================================================="
                )

                # Exit condition
                if retry != "non_stop":
                    if current_iter_progress == 0:
                        if was_progress == True:
                            current_retry, was_progress = 0, False
                        elif current_retry == retry:
                            break
                        current_retry += 1
                    else:
                        was_progress = True

        self.state.session = None
        print(f"Total сollected URLs {len(self.state.news_urls)}")

        return self.state.news_urls
