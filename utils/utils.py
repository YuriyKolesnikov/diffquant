# utils/utils.py
import math


def millify(n, precision=1):
    millnames = ['', 'K', 'M', 'B', 'T', 'P']
    n = float(n)
    millidx = max(0, min(len(millnames) - 1, int(math.floor(math.log10(abs(n))) / 3))) if n != 0 else 0
    return f"{n / 10 ** (3 * millidx):.{precision}f}{millnames[millidx]}"

