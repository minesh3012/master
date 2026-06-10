# How to add AI logging to any strategy in 4 lines

1. Copy AIFeatureBuilder.cs and AITradeLogger.cs into your NT8 Custom folder:
   Documents/NinjaTrader 8/bin/Custom/

2. Add using statement:
   using NTAIShared;

3. Add field:
   private AITradeLogger _logger;

4. Wire it in:
   DataLoaded  → _logger = new AITradeLogger("StrategyName", Instrument.FullName);
   OnBarUpdate → _logger.OnBar(fv, Time[0], "signal", isLong, isShort, ...);
   OnPositionUpdate (open)  → _logger.OnTradeOpen(fv, Time[0], "signal", isLong);
   OnPositionUpdate (close) → _logger.OnTradeClose();
   Terminated  → _logger?.Close();

# Output CSV location
Documents/NinjaTrader 8/bin/Custom/AITraining/
One file per instrument+strategy combination.

# Signal-only indicators
Use _logger.LogSignal(fv, Time[0], "SignalName", direction) instead.
No trade state needed. Phase column will be "Signal".

# Key design decisions
- AIFeatureBuilder is the ONLY place features are computed.
  Never duplicate feature logic in a strategy.
- All features are ATR-normalized — no raw prices reach the model.
- Mechanical rules run whether AI is enabled or not.
  AI logging is additive, never blocking.