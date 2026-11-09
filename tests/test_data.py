import pandas as pd
import pytest

from quantml.data import load_prices


def make_fake_downloader(prices):
    """造一个假的 yf.download。prices 形如 {"AAPL": [100.0, 101.0], ...}"""
    calls = []  # 记录每一次调用，测试里用来检查"调用了几次、传了什么"

    def fake_download(tickers, start=None, end=None, **kwargs):
        # 1) 把这次调用的参数记下来。**kwargs 会收集 auto_adjust 这类额外参数
        calls.append({"tickers": tickers, "start": start, "end": end, **kwargs})

        # 2) yfinance 既接受 "AAPL" 也接受 ["AAPL"]，这里统一成列表
        if isinstance(tickers, str):
            tickers = [tickers]

        # 3) 造日期：用价格列表的长度决定有几天，bdate_range 只取工作日
        n_days = len(next(iter(prices.values())))
        dates = pd.bdate_range("2024-01-02", periods=n_days)

        # 4) 造和 yfinance 一样的两层列名：(字段, 股票代码)
        #    请对照你在第 ① 步看到的 df.columns 检查这里是否一致
        columns = pd.MultiIndex.from_product(
            [["Close", "Volume"], tickers], names=["Price", "Ticker"]
        )
        df = pd.DataFrame(index=dates, columns=columns, dtype=float)

        # 5) 填数据。prices 里没有的代码保持 NaN，模仿 yfinance 对不存在的代码的表现
        for ticker in tickers:
            if ticker in prices:
                df[("Close", ticker)] = prices[ticker]
                df[("Volume", ticker)] = 1_000_000

        return df

    fake_download.calls = calls  # 把记录挂在函数上，测试里用 fake.calls 读取
    return fake_download


def test_fake_downloader_mimics_yfinance():
    fake = make_fake_downloader({"AAPL": [100.0, 101.0, 102.0]})

    df = fake(["AAPL", "ZZZZ"], start="2024-01-01", auto_adjust=True)

    assert df[("Close", "AAPL")].tolist() == [100.0, 101.0, 102.0]
    assert df[("Close", "ZZZZ")].isna().all()  # 不存在的代码是 NaN
    assert len(fake.calls) == 1  # 被调用了 1 次
    assert fake.calls[0]["auto_adjust"] is True  # 记下了 auto_adjust 参数


def test_returns_requested_tickers_as_columns(tmp_path):
    # Arrange：准备假数据
    fake = make_fake_downloader({"AAPL": [100.0, 101.0, 102.0], "MSFT": [200.0, 201.0, 202.0]})

    # Act：调用要测试的函数，把假函数传进去
    prices = load_prices(["AAPL", "MSFT"], start="2024-01-01", cache_dir=tmp_path, downloader=fake)

    # Assert：检查结果
    assert list(prices.columns) == ["AAPL", "MSFT"]


def test_empty_ticker_list_raises(tmp_path):
    fake = make_fake_downloader({"AAPL": [100.0]})

    with pytest.raises(ValueError):
        load_prices([], start="2024-01-01", cache_dir=tmp_path, downloader=fake)


def test_index_is_date_and_sorted_and_unique(tmp_path):
    fake = make_fake_downloader({"AAPL": [100.0, 101.0, 102.0]})

    prices = load_prices(["AAPL"], start="2024-01-01", cache_dir=tmp_path, downloader=fake)

    assert isinstance(prices.index, pd.DatetimeIndex)
    assert prices.index.is_monotonic_increasing
    assert prices.index.is_unique


def test_all_prices_greater_than_zero(tmp_path):
    fake = make_fake_downloader({"AAPL": [100.0, 101.0, 102.0]})

    prices = load_prices(["AAPL"], start="2024-01-01", cache_dir=tmp_path, downloader=fake)

    assert (prices > 0).all().all()


# def test_tickers_with_no_data_are_dropped(tmp_path):
#     fake = make_fake_downloader({"AAPL": [100.0, 101.0, 102.0]})

#     prices = load_prices(["AAPL", "ZZZZ"], start="2024-01-01", cache_dir=tmp_path, downloader=fake)

#     assert "ZZZZ" not in prices.columns
#     assert "AAPL" in prices.columns


def test_auto_adjust_parameter_passed_to_downloader(tmp_path):
    fake = make_fake_downloader({"AAPL": [100.0, 101.0, 102.0]})

    load_prices(["AAPL"], start="2024-01-01", cache_dir=tmp_path, downloader=fake)

    assert len(fake.calls) == 1
    assert fake.calls[0]["auto_adjust"] is True


def test_cached_data_is_used_for_subsequent_calls(tmp_path):
    fake = make_fake_downloader({"AAPL": [100.0, 101.0, 102.0]})

    # 第一次调用，应该下载数据
    first = load_prices(["AAPL"], start="2024-01-01", cache_dir=tmp_path, downloader=fake)
    assert len(fake.calls) == 1

    # 第二次调用，应该从缓存读取，不再下载
    second = load_prices(["AAPL"], start="2024-01-01", cache_dir=tmp_path, downloader=fake)
    assert len(fake.calls) == 1  # 调用次数没有增加
    assert first.equals(second)  # 两次返回的 DataFrame 相同


def test_cached_data_is_used_for_subsequent_calls_with_different_tickers(tmp_path):
    fake = make_fake_downloader({"AAPL": [100.0, 101.0, 102.0], "MSFT": [200.0, 201.0, 202.0]})

    # 第一次调用，应该下载 AAPL 数据
    first = load_prices(["AAPL"], start="2024-01-01", cache_dir=tmp_path, downloader=fake)
    assert len(fake.calls) == 1

    # 第二次调用，应该下载 MSFT 数据，但 AAPL 数据从缓存读取
    second = load_prices(["AAPL", "MSFT"], start="2024-01-01", cache_dir=tmp_path, downloader=fake)
    assert len(fake.calls) == 2  # 调用次数增加了一次
    assert "AAPL" in second.columns and "MSFT" in second.columns


def test_tickers_with_no_data_are_dropped_with_warning(tmp_path):
    fake = make_fake_downloader({"AAPL": [100.0, 101.0, 102.0]})

    with pytest.warns(UserWarning, match="ZZZZ"):
        prices = load_prices(
            ["AAPL", "ZZZZ"], start="2024-01-01", cache_dir=tmp_path, downloader=fake
        )

    assert list(prices.columns) == ["AAPL"]


def test_no_data_for_all_tickers_raises(tmp_path):
    fake = make_fake_downloader({"AAPL": [100.0, 101.0, 102.0]})

    with pytest.raises(ValueError):
        load_prices(["ZZZZ", "YYYY"], start="2024-01-01", cache_dir=tmp_path, downloader=fake)
