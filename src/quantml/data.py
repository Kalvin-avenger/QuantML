"""Price data loading with a local cache."""

import warnings
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
        Yahoo Finance symbols, e.g. ["AAPL", "MSFT"]. Duplicates are ignored.
    start, end:
        Date range as "YYYY-MM-DD". ``end`` is exclusive, matching yfinance.
        Today is always excluded, even when ``end`` is later or ``None``,
        because today's bar may be incomplete while the market is open.
        ``None`` therefore means up to the last completed trading day.
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
        If ``tickers`` is empty, or none of the tickers returns any data.

    Warns
    -----
    UserWarning
        If some tickers return no data at all. Those tickers are dropped
        and listed in the warning message. Tickers with only partial
        history (e.g. listed later) are kept.
    """
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        raise ValueError("tickers list is empty")

    cache_dir = Path(cache_dir)
    today = pd.Timestamp.today().normalize()
    end_ts = min(pd.Timestamp(end), today) if end else today

    to_download = [t for t in tickers if not _cache_path(cache_dir, t).exists()]
    if to_download:
        raw = downloader(
            to_download,
            start=start,
            end=end_ts.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )
        close_prices = _extract_close(raw, to_download)

        for ticker in to_download:
            series = close_prices[ticker]
            series = series[series.index < today].dropna()
            if not series.empty:
                _write_cache(cache_dir, series)

    available = [t for t in tickers if _cache_path(cache_dir, t).exists()]
    if not available:
        raise ValueError("No data available for any of the requested tickers.")

    no_data = [t for t in tickers if t not in available]
    if no_data:
        warnings.warn(f"No data available for tickers: {no_data}", UserWarning, stacklevel=2)

    data = pd.concat([_read_cache(cache_dir, ticker) for ticker in available], axis=1)
    data = data.sort_index()
    data = data.loc[(data.index >= pd.Timestamp(start)) & (data.index < end_ts)]
    return data


def _extract_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """From a yf.download result, return close prices: dates x tickers."""
    if raw is None or raw.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([]), columns=tickers, dtype=float)

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:  # some yfinance versions return flat columns for a single ticker
        close = raw[["Close"]].set_axis(tickers[:1], axis=1)

    close = close.reindex(columns=tickers).astype(float)
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)
    return close


def _cache_path(cache_dir: Path, ticker: str) -> Path:
    return cache_dir / f"{ticker}.csv"


def _write_cache(cache_dir: Path, series: pd.Series) -> None:
    """Save one ticker's prices. series.name must be the ticker."""
    path = _cache_path(cache_dir, str(series.name))
    path.parent.mkdir(parents=True, exist_ok=True)
    series.to_csv(path, index_label="date")


def _read_cache(cache_dir: Path, ticker: str) -> pd.Series:
    """Read one ticker's prices back, with a real DatetimeIndex."""
    frame = pd.read_csv(_cache_path(cache_dir, ticker), index_col="date", parse_dates=True)
    series = frame[ticker].astype(float)
    series.index.name = None
    return series
