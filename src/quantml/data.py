"""Price data loading with a local cache."""

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import yfinance as yf

import warnings

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
        If ``tickers`` is empty, or none of the tickers returns any data.

    Warns
    -----
    UserWarning
        If some tickers return no data at all. Those tickers are dropped
        and listed in the warning message. Tickers with only partial
        history (e.g. listed later) are kept.
    """

    # 第 1 步：参数检查
    # TODO: tickers 为空时抛出 ValueError
    # 提示：顺手去掉重复代码并保持原顺序：tickers = list(dict.fromkeys(tickers))

    cache_dir = Path(cache_dir)

    tickers = list(dict.fromkeys(tickers))

    if not tickers:
        raise ValueError("tickers list is empty")

    # 第 2 步：只下载缓存里还没有的股票
    # TODO: 用 _cache_path(cache_dir, t).exists() 找出还没缓存的代码，存进 to_download
    # TODO: 如果 to_download 不为空：
    #       - 调用 downloader(to_download, start=start, end=end, auto_adjust=True, progress=False)
    #       - 用 _extract_close(...) 得到"日期 × 股票"的收盘价宽表
    #       - 逐只股票取出一列，.dropna() 后如果不为空，就用 _write_cache(...) 写入缓存
    #         （整列都是 NaN 的股票不写，它们就是"没有数据"的股票）

    to_download = [t for t in tickers if not _cache_path(cache_dir, t).exists()]

    if to_download:
        raw = downloader(to_download, start=start, end=end, auto_adjust=True, progress=False)
        close_prices = _extract_close(raw, to_download)

        for ticker in to_download:
            series = close_prices[ticker].dropna()
            if not series.empty:
                _write_cache(cache_dir, series)

    # 第 3 步：判断哪些股票有数据，按 DEVLOG 的规则处理
    # TODO: available = 缓存文件存在的代码（保持请求顺序）；no_data = 其余的代码
    # TODO: available 为空 → 抛出 ValueError
    # TODO: no_data 不为空 → warnings.warn(f"... {no_data}", UserWarning, stacklevel=2)

    available = [t for t in tickers if _cache_path(cache_dir, t).exists()]
    if not available:
        raise ValueError("No data available for any of the requested tickers.")

    no_data = [t for t in tickers if t not in available]
    if no_data:
        warnings.warn(f"No data available for tickers: {no_data}", UserWarning, stacklevel=2)

    # 第 4 步：统一从缓存读出，拼成宽表返回
    # TODO: 对 available 里每只股票调用 _read_cache(...)，用 pd.concat(..., axis=1) 横向拼接
    # TODO: 按日期排序，并只保留 [start, end) 范围内的行

    data = pd.concat([_read_cache(cache_dir, ticker) for ticker in available], axis=1)
    data = data.sort_index()
    data = data.loc[data.index >= pd.Timestamp(start)]
    if end is not None:
        data = data.loc[data.index < pd.Timestamp(end)]

    # raise NotImplementedError
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
