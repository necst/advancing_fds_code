from multiprocessing import Pool
from tqdm import tqdm
from psutil import cpu_count
from typing import Callable, Dict, List


def run_with_n_jobs(
    fn: Callable, n_jobs: int, fn_params: List, tqdm_params: Dict
):
    """
    If n_jobs is 0 then fn will be called without multiprocessing, otherwise n_jobs
    will be the number of threads for the pool. If n_jobs is -1 then the pool
    will have as many threads as the cores available.
    """
    use_pool = n_jobs != 0 and n_jobs != 1
    n_jobs = cpu_count() if n_jobs == -1 else n_jobs
    if use_pool:
        print("starting pool with %d jobs" % n_jobs)
        with Pool(n_jobs) as p:
            return list(
                tqdm(
                    p.imap_unordered(
                        fn,
                        zip(*fn_params),
                    ),
                    **tqdm_params,
                )
            )
    else:
        return list(
            map(fn, tqdm(zip(*fn_params), **tqdm_params)),
        )


def get_commit():
    """
    Returns first seven chars of commit hash on current branch
    """
    try:
        from os import path
        from config import config as c

        repo_path = c.PROJECT_PATH
        git_folder = path.join(repo_path, ".git")
        head_name = path.join(git_folder, "HEAD")
        with open(head_name, "r") as f:
            head_name = f.read().split("\n")[0].split(" ")[-1]
        head_ref = path.join(git_folder, head_name)

        with open(head_ref, "r") as f:
            commit = f.read().replace("\n", "")
        return commit[:7]
    except FileNotFoundError:
        return "not_found"
