"""Shared constants. Change these here rather than scattering numbers around the code."""

RETAILERS = ["Aldi", "ASDA", "Morrisons", "Sainsburys", "Tesco"]

# A price this much below the trailing reference price counts as "on promotion".
PROMO_THRESHOLD = 0.05

# Trailing window (days) used to estimate the "regular" price. Trailing only, so no lookahead.
REF_WINDOW = 42
REF_QUANTILE = 0.9

# Forecast horizon in days. Models predict the price ratio p[t+k] / p[t] for k = 1..HORIZON.
HORIZON = 14

# "Meaningfully cheaper": the classifier target is "min price in the next HORIZON days
# is at least this fraction below today's price".
CHEAPER_DELTA = 0.05

# Missing scrape days are forward-filled for at most this many days; longer gaps stay missing.
MAX_FFILL_DAYS = 3

# Sequence length (days of history) fed to the transformer.
SEQ_LEN = 56
