PASTE INTO LLM
take tasks from timeline at bottom, one at a time. tackle bit by bit


Here’s the plan. One honest flag first: I couldn’t confirm the size of Wikimedia’s hourly pageview files from here. They’re big, and downloading all-language files for many days could be tens of GB on hackathon wifi. So the data download starts first, small (2 days), and grows overnight.

Project: “Ahead of the Spike”

The question: every Wikipedia page view is served either from fast memory (a cache) or slowly from the main servers. If a cache can hold only the top N articles each hour, can we predict what people will read next hour well enough to beat the standard caching rules?

Headline results to aim for:

Hit rate: the share of next hour’s views served from the cache, for your model against the standard rules, across cache sizes.
Spike catch time: when a page suddenly goes viral (a death, a match, breaking news), how many hours earlier you had it cached than the standard rules.
The time-theme slide, “what is foresight worth?”: if a cache needs k hours’ notice to warm up, how does hit rate fall as k goes from 1 to 6?

The cache doesn’t change what people read, so the backtest on historical data is exact. Simplification to state upfront: decisions are made once per hour, not per request.

Shared interfaces (agree these in the first 30 minutes)
views.parquet: columns hour, page, views (English Wikipedia, desktop and mobile combined). EIE produces it.
A predictor is a function: given an hour, return the predicted views for the next hour for each page. Computing and Physics write these.
backtest.py scores any predictor and writes demo.json for the frontend. Computing owns it.
Roles

2nd year Computing: the models and the backtest (the spine)

First, the backtest harness, before any ML. It takes predictors and cache sizes (e.g. 1k, 10k, 100k articles) and outputs hit rate per hour plus spike catch time.
Baselines:
last hour’s top pages (roughly “least recently used”, the standard rule);
top pages over the last 24 hours (roughly “least frequently used”);
same hour yesterday.
XGBoost on per-page features: views 1–6 and 24 hours ago, growth rate, mobile share, hour of day, weekday. Keep it to a candidate set: pages with meaningful views in the last day.
Overnight: a transformer over each page’s last 48 hourly views predicts the next 1–6 hours. Train it on Colab.

2nd year Physics: the physics of attention

Spike decay model: after a page spikes, attention fades, often roughly like a power law or exponential. Fit the decay to past spikes and use it to forecast how long a viral page stays hot. It’s a parametric predictor that plugs into the same backtest, and a strong feature for XGBoost.
The cache as an optimisation: capacity is limited and swapping pages in costs bandwidth. So choose cache contents to maximise expected hits minus a switching cost. A greedy rule or dynamic programming, against the naive “just take the top N”.
Statistics: confidence intervals on the hit-rate differences (bootstrap over hours), so the claims hold up.

2nd year EIE: data engineering

Now: write a downloader that streams the hourly pageview files and keeps only English desktop (en) and mobile (en.m). Never store the raw files.
Get 2 days into views.parquet by about 02:30, so everyone can develop on real data. Then leave it running overnight to reach 7–10 days. Ten days gives “same hour last week” features.
Data checks: missing hours, odd spikes that look like bots (pages viewed at robotic constant rates). If time allows, Computing’s isolation-forest experience can flag these.
Morning: a small FastAPI backend that serves the replay data to the demo.

2nd year Design Eng: the demo and the story

The demo: replay a day of Wikipedia’s attention.
A live leaderboard of the most-read pages.
Each one marked green if your cache had it in time, red if it missed.
A side-by-side hit-rate counter: your model vs the standard rule.
The spike story: with EIE, find 2–3 dramatic real events in the data. For each, show its views curve with markers for when each method cached it.
Pitch deck and narrative: problem (one sentence), the question, the headline numbers, the spike story, the “what is foresight worth?” curve, honest limits.
Build against fake demo.json tonight, so you aren’t blocked waiting for real results.

<img width="1108" height="790" alt="image" src="https://github.com/user-attachments/assets/6d04a586-a1cb-4bf0-98e5-363f8b33d182" />


Honest limits to put in the pitch
Pages that go from nothing to a million views within one hour can’t be predicted from hourly data. Everyone catches them an hour late. Showing exactly where that limit is will impress the judges more than hiding it.
Hourly decisions simplify real caches, which decide per request.
Article sizes aren’t in this data, so every page counts as the same size.
Stretch goal: Wikimedia has also released anonymised request traces from its caching servers for research. If you find them on Wikitech, they’d let you test per request.

Sources:

Wikimedia pageviews readme
Phabricator: Wikimedia caching trace release (T225538)
