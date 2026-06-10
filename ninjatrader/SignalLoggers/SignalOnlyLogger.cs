// Add this to any INDICATOR that generates signals but doesn't trade.
// It logs the signal bar + surrounding context for AI training.
// No position, no entry/exit — just "the indicator fired here".

using NTAIShared;

// In State.DataLoaded:
//   _logger = new AITradeLogger("MyIndicatorName", Instrument.FullName);

// In OnBarUpdate, wherever your signal fires:
//   if (bullSignalFired)
//       _logger.LogSignal(fv, Time[0], "BullSignal", direction: 1);
//   if (bearSignalFired)
//       _logger.LogSignal(fv, Time[0], "BearSignal", direction: -1);

// The CSV will have Phase="Signal" and Offset=1 or -1 for direction.
// In Python training, filter Phase=="Signal" rows to get signal-only training data.