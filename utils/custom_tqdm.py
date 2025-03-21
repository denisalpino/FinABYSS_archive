from time import time
from tqdm.notebook import tqdm
from typing import Any, List


def format_seconds(seconds):
    """Convert seconds to hours/minutes/seconds for TQDM widget"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours} h. {minutes} min. {seconds} sec."


def wrap_with_tqdm(desc: str, func: Any, tasks_args: Any, hide: bool):
    """This function is a wrapper for coroutines to launch tqdm in Jupyter Notebooks"""
    pbar = tqdm(
        total=len(tasks_args),
        desc=desc,
        bar_format="{desc}: {percentage:.0f}%|{bar}| {n_fmt}/{total_fmt} [It's been: {postfix}]",
        postfix="0 h. 0 min. 0 sec.",
        leave=not hide # To hide widget for chunk after it's compited
    )
    start_time = time()

    def update_task(_, pbar=pbar, start=start_time):
        elapsed = time() - start
        pbar.set_postfix_str(format_seconds(elapsed))
        pbar.update(1)

    tasks: List = []
    for item in tasks_args:
        task = create_task(func(item)) # type: ignore
        task.add_done_callback(update_task)
        tasks.append(task)

    return tasks, pbar