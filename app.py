For your scanner’s objective (1–10 trading day swing trades and options direction), I would choose ATR(14) as the default.

Why ATR(14)?

✅ Faster adaptation to changing volatility

✅ Industry standard (used widely across trading platforms)

✅ Better for breakout and momentum trades

✅ Reacts quickly after earnings, news, or sector rotation

For example:

* A stock like State Bank of India can move from low volatility to high volatility in a few sessions.
* ATR(14) adjusts faster than ATR(20), helping stop-losses reflect current market conditions.

⸻

When ATR(20) is Better

ATR(20) becomes useful when:

* Trading positional swings (2–6 weeks)
* Trading slower-moving large caps
* Trying to reduce stop-loss changes from short-term volatility spikes

The downside is that ATR(20) can react slowly when volatility suddenly expands.

⸻

My Recommendation

Use a hybrid approach:

ATR_PERIOD = 14

for all calculations.

Then display:

Volatility Regime:
LOW
NORMAL
HIGH

based on ATR relative to its historical average.

Example:

RELIANCE
Signal: CALL
Score: 92
ATR(14): 38.5
Volatility: NORMAL
Entry: 1490
SL: 1432
T1: 1567
T2: 1644

⸻

Even Better for Version 2

Add an advanced mode:

Conservative Mode → ATR(20)
Aggressive Mode → ATR(14)

via a Streamlit sidebar toggle.

This allows the same scanner to serve both:

* Fast options traders
* Slower swing traders

without changing the underlying architecture.

⸻

One More Suggestion Before Coding News & Events

I would not let news directly add +20 points.

Instead:

News Sentiment	Adjustment
Strong Positive	+5
Positive	+3
Neutral	0
Negative	-3
Strong Negative	-5

And keep Event Risk separate:

⚠ Earnings in 1 Day
⚠ RBI Policy Tomorrow
⚠ Results This Week

This prevents a stock from becoming a “Strong CALL” purely because of headlines while maintaining the integrity of your technical scoring engine. The technical score should remain the primary driver; news and events should increase or decrease confidence, not override the chart.
