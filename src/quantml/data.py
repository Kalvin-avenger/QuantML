"""Price data loading with a local cache."""

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import yfinance as yf

DEFAULT_CACHE_DIR = Path("data/cache")


def load_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    downloader: Callable[..., pd.DataFrame] = yf.download,
) -> pd.DataFrame:
    """Load adjusted daily close prices for several tickers.

    Parameters
    ----------
    tickers:
        Yahoo Finance symbols, e.g. ["AAPL", "MSFT"].
    start, end:
        Date range as "YYYY-MM-DD". ``end`` is exclusive, matching yfinance.
        ``None`` means up to the latest available date.
    cache_dir:
        Folder for cached data. Data already in the cache is not downloaded again.
    downloader:
        Function with the same interface as ``yf.download``.
        Tests pass a fake here so they never touch the network.

    Returns
    -------
    pd.DataFrame
        Wide table of split- and dividend-adjusted close prices.
        Index: sorted, unique ``DatetimeIndex`` (timezone-naive).
        Columns: one per ticker, in the order requested.

    Raises
    ------
    ValueError
        If ``tickers`` is empty, or no data is returned for any ticker.

    Notes
    -----
    Tickers that return no data are dropped with a warning.  # ← 换成你在 DEVLOG 里定的规则
    """
    raise NotImplementedError