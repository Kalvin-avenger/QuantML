"""Download (or read from cache) prices for the whole universe and sanity-check them."""

from quantml.data import load_prices
from quantml.universe import UNIVERSE


def main() -> None:
    prices = load_prices(UNIVERSE, start="2015-01-01")

    print("shape:", prices.shape)
    print("date range:", prices.index.min().date(), "to", prices.index.max().date())

    print("\nfirst valid date per ticker (latest 10):")
    print(prices.apply(lambda s: s.first_valid_index()).sort_values().tail(10))

    print("\ndropped tickers:", [t for t in UNIVERSE if t not in prices.columns])

    # Largest one-day moves: split artifacts would show up as about -50%, -75% or -90%
    moves = prices.pct_change(fill_method=None).abs()
    top = moves.max().sort_values().tail(8).index
    print("\nlargest one-day moves:")
    print(moves.max()[top].map("{:.1%}".format).to_string())
    print(moves.idxmax()[top].dt.date.to_string())


if __name__ == "__main__":
    main()
