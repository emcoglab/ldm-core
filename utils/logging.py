from sys import stdout

log_message = '%(asctime)s | %(levelname).2s | %(filename)s:%(lineno)d | \t%(message)s'
date_format = "%Y-%m-%d %H:%M:%S"


def print_progress(iteration: int, total: int,
                   prefix: str = '', suffix: str = '',
                   *,
                   decimals: int = 1,
                   bar_length: int = 100,
                   clear_on_completion: bool = False):
    """
    Call in a loop to create terminal progress bar.
    Based on https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a
    @params:
        iteration           - Required  : current iteration (Int)
        total               - Required  : total iterations (Int)
        prefix              - Optional  : prefix string (Str)
        suffix              - Optional  : suffix string (Str)
        decimals            - Optional  : positive number of decimals in percent complete (Int)
        bar_length          - Optional  : character length of bar (Int)
        clear_on_completion - Optional  : clear the bar when it reaches 100% (bool)
    """
    full_char  = '█'
    empty_char = '-'

    portion_complete = iteration / float(total)
    percents = f"{100 * portion_complete:.{decimals}f}%"
    filled_length = int(round(bar_length * portion_complete))
    bar = (full_char * filled_length) + (empty_char * (bar_length - filled_length))

    stdout.write(f'\r{prefix}|{bar}| {percents}{suffix}'),

    if iteration == total:
        stdout.write("\r" if clear_on_completion else "\n")

    stdout.flush()
