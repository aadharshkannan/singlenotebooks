# V4 Methodology Delta

- Outcome cells preserve exact token mass budgets from live V3 execution with no budget reinterpretation.
- V4 continues to use no Cochran sample sizing and no finite-population correction (FPC).
- Selection remains whole-session maximal packing under exact token budgets.
- Random and MinHash arms remain selected-only estimators in V4.
- MinHash bucket miss is treated as novelty/no-candidate and does not trigger exhaustive scan fallback.
- Embedding arm reports selected-only metrics plus judged+IDW model-assisted estimates.
- IDW uses same-agent k=8 angular neighbors with the configured fallback chain from V4 IDW logic.
- Deterministic expected labels are treated as pseudo-judge outputs after membership freeze.
- V4 makes no design-unbiasedness claim; IDW outputs are model-assisted diagnostics.
- V3 source outcomes did not already perform IDW augmentation.
