from asyncio import Task
from time import time
from typing import List
from tqdm.notebook import tqdm


def format_seconds(seconds):
    """Convert seconds to hours/minutes/seconds for TQDM widget"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours} h. {minutes} min. {seconds} sec."


def wrap_with_tqdm(desc: str, tasks: List[Task], hide: bool):
    """This function is a wrapper for coroutines to launch tqdm in Jupyter Notebooks"""
    pbar = tqdm(
        total=len(tasks),
        desc=desc,
        bar_format="{desc}: {percentage:.0f}%|{bar}| {n_fmt}/{total_fmt} [It's been: {postfix}]",
        postfix="0 h. 0 min. 0 sec.",
        leave=not hide  # To hide widget for chunk after it's compited
    )
    start_time = time()

    def update_task(_, pbar=pbar, start=start_time):
        elapsed = time() - start
        pbar.set_postfix_str(format_seconds(elapsed))
        pbar.update(1)

    for task in tasks:
        task.add_done_callback(update_task)

    return pbar
