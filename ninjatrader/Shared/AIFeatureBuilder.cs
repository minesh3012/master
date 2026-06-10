using System;
using NinjaTrader.NinjaScript;
 
namespace NTAIShared
{
    public static class AIFeatureBuilder
    {
        public const int FeatureCount = 37;
 
        public static double[] Build(
            ISeries<double> close,
            ISeries<double> open,
            ISeries<double> high,
            ISeries<double> low,
            ISeries<double> volume,
            double ema9,    double ema9_1,  double ema9_2,  double ema9_3,
            double ema21,   double ema50,   double ema200,
            double atr14,
            double bbUpper, double bbLower,
            double adx14,   double mom14,   double roc5,
            double vwap,
            double volSma20,
            double priorHigh,   double priorLow,    double priorClose,
            double sessionHigh, double sessionLow,  double sessionOpen,
            double localHigh10, double localLow10,
            bool inBullOB,  bool inBearOB)
        {
            if (atr14 <= 0) atr14 = 1;
 
            double c     = close[0];
            double o     = open[0];
            double h     = high[0];
            double l     = low[0];
            double range = h - l;
 
            double bodyRatio   = range > 0 ? Math.Abs(c - o) / range        : 0;
            double upperWick   = range > 0 ? (h - Math.Max(o, c)) / range   : 0;
            double lowerWick   = range > 0 ? (Math.Min(o, c) - l) / range   : 0;
            double barDir      = c > o ? 1 : (c < o ? -1 : 0);
            double barRangeAtr = range / atr14;
 
            double distVWAP      = (c - vwap)        / atr14;
            double distEMA9      = (c - ema9)         / atr14;
            double distEMA21     = (c - ema21)        / atr14;
            double distEMA50     = (c - ema50)        / atr14;
            double distEMA200    = (c - ema200)       / atr14;
            double distPDH       = (c - priorHigh)    / atr14;
            double distPDL       = (c - priorLow)     / atr14;
            double distPDC       = (c - priorClose)   / atr14;
            double distSessHigh  = (c - sessionHigh)  / atr14;
            double distSessLow   = (c - sessionLow)   / atr14;
            double distSessOpen  = (c - sessionOpen)  / atr14;
            double distLocHigh10 = (c - localHigh10)  / atr14;
            double distLocLow10  = (c - localLow10)   / atr14;
 
            double fastSlope  = (ema9 - ema9_3) / atr14;
            double slopeAccel = ((ema9 - ema9_1) - (ema9_1 - ema9_2)) / atr14;
            double trendUp    = (ema9 > ema21 && ema21 > ema50) ? 1 : 0;
            double trendDown  = (ema9 < ema21 && ema21 < ema50) ? 1 : 0;
 
            double bbPos      = (bbUpper > bbLower) ? (c - bbLower) / (bbUpper - bbLower) : 0.5;
            double bbWidthAtr = (bbUpper - bbLower) / atr14;
 
            double volNorm = volSma20 > 0 ? volume[0] / volSma20 : 1.0;
            double volDir  = c > o ? volNorm : -volNorm;
 
            double momNorm = mom14 / atr14;
 
            double sweepUp   = (high[0] > localHigh10 && c < localHigh10) ? 1 : 0;
            double sweepDown = (low[0]  < localLow10  && c > localLow10)  ? 1 : 0;
 
            double b1 = (close[1] - open[1]) / atr14;
            double b2 = (close[2] - open[2]) / atr14;
            double b3 = (close[3] - open[3]) / atr14;
 
            return new double[]
            {
                bodyRatio, upperWick, lowerWick, barDir, barRangeAtr,
                distVWAP, distSessHigh, distSessLow, distSessOpen,
                distEMA9, distEMA21, distEMA50, distEMA200,
                distPDH, distPDL, distPDC,
                distLocHigh10, distLocLow10,
                fastSlope, slopeAccel, trendUp, trendDown,
                bbPos, bbWidthAtr,
                volNorm, volDir, adx14,
                momNorm, roc5,
                sweepUp, sweepDown,
                inBullOB ? 1.0 : 0.0,
                inBearOB ? 1.0 : 0.0,
                b1, b2, b3
            };
        }
    }
}