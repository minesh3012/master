using System;
using NinjaTrader.Cbi;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Indicators;
using NTAIShared;
using System.ComponentModel.DataAnnotations;
 
namespace NinjaTrader.NinjaScript.Strategies
{
    public class AI_Strategy_Template : Strategy
    {
        private EMA          _ema9, _ema21, _ema50, _ema200;
        private ATR          _atr14;
        private ADX          _adx14;
        private Bollinger    _bb;
        private Momentum     _mom14;
        private ROC          _roc5;
        private PriorDayOHLC _prior;
 
        private double _cumPV, _cumVol, _vwap;
        private double _entryPrice, _mae, _mfe;
        private int    _barsSinceEntry;
 
        private AITradeLogger _logger;
 
        [NinjaScriptProperty]
        [Display(Name = "Enable AI Logging", Order = 1, GroupName = "AI")]
        public bool EnableAILogging { get; set; } = true;
 
        [NinjaScriptProperty]
        [Display(Name = "Pre bars", Order = 2, GroupName = "AI")]
        public int PreBars { get; set; } = 5;
 
        [NinjaScriptProperty]
        [Display(Name = "Post bars", Order = 3, GroupName = "AI")]
        public int PostBars { get; set; } = 5;
 
        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Name            = "AI_Strategy_Template";
                Calculate       = Calculate.OnBarClose;
                IsExitOnSessionCloseStrategy = true;
                ExitOnSessionCloseSeconds    = 30;
                DefaultQuantity = 1;
            }
            else if (State == State.DataLoaded)
            {
                _ema9   = EMA(9);
                _ema21  = EMA(21);
                _ema50  = EMA(50);
                _ema200 = EMA(200);
                _atr14  = ATR(14);
                _adx14  = ADX(14);
                _bb     = Bollinger(2, 20);
                _mom14  = Momentum(14);
                _roc5   = ROC(5);
                _prior  = PriorDayOHLC();
 
                if (EnableAILogging)
                    _logger = new AITradeLogger(
                        Name, Instrument.FullName, PreBars, PostBars);
            }
            else if (State == State.Terminated)
            {
                _logger?.Close();
            }
        }
 
        protected override void OnBarUpdate()
        {
            if (BarsInProgress != 0 || CurrentBar < 200) return;
 
            if (Bars.IsFirstBarOfSession) { _cumPV = 0; _cumVol = 0; }
            double tp = (High[0] + Low[0] + Close[0]) / 3.0;
            _cumPV  += tp * Volume[0];
            _cumVol += Volume[0];
            _vwap    = _cumVol > 0 ? _cumPV / _cumVol : Close[0];
 
            double atr = _atr14[0];
 
            double sessionHigh = MAX(High, CurrentBar)[0];
            double sessionLow  = MIN(Low,  CurrentBar)[0];
 
            int dayIdx    = Bars.GetBar(Time[0].Date);
            double dayOpen = Opens[0][CurrentBar - dayIdx];
 
            double[] fv = AIFeatureBuilder.Build(
                Close, Open, High, Low, Volume,
                _ema9[0],  _ema9[1],  _ema9[2],  _ema9[3],
                _ema21[0], _ema50[0], _ema200[0],
                atr,
                _bb.Upper[0], _bb.Lower[0],
                _adx14[0], _mom14[0], _roc5[0],
                _vwap,
                SMA(Volume, 20)[0],
                _prior.PriorHigh[0], _prior.PriorLow[0], _prior.PriorClose[0],
                sessionHigh, sessionLow, dayOpen,
                MAX(High, 10)[0], MIN(Low, 10)[0],
                false, false
            );
 
            bool trendUp   = _ema9[0] > _ema21[0] && _ema21[0] > _ema50[0];
            bool trendDown = _ema9[0] < _ema21[0] && _ema21[0] < _ema50[0];
            bool aboveVwap = Close[0] > _vwap;
            bool belowVwap = Close[0] < _vwap;
            bool strongAdx = _adx14[0] > 20;
 
            bool longEntry  = trendUp   && aboveVwap && strongAdx;
            bool shortEntry = trendDown && belowVwap && strongAdx;
            bool longExit   = trendDown || Close[0] < _vwap - atr;
            bool shortExit  = trendUp   || Close[0] > _vwap + atr;
 
            if (Position.MarketPosition != MarketPosition.Flat)
            {
                double unreal = Position.MarketPosition == MarketPosition.Long
                    ? (Close[0] - _entryPrice) / atr
                    : (_entryPrice - Close[0]) / atr;
                _mae = Math.Min(_mae, unreal);
                _mfe = Math.Max(_mfe, unreal);
                _barsSinceEntry++;
            }
 
            double unrealNow = Position.MarketPosition == MarketPosition.Long
                ? (Close[0] - _entryPrice) / atr
                : Position.MarketPosition == MarketPosition.Short
                    ? (_entryPrice - Close[0]) / atr : 0;
 
            _logger?.OnBar(fv, Time[0], "Mechanical",
                Position.MarketPosition == MarketPosition.Long,
                Position.MarketPosition == MarketPosition.Short,
                _barsSinceEntry, unrealNow, _mae, _mfe);
 
            if (Position.MarketPosition == MarketPosition.Flat)
            {
                if (longEntry)  EnterLong(DefaultQuantity,  "Long");
                if (shortEntry) EnterShort(DefaultQuantity, "Short");
            }
            else
            {
                if (Position.MarketPosition == MarketPosition.Long  && longExit)
                    ExitLong("Exit", "Long");
                if (Position.MarketPosition == MarketPosition.Short && shortExit)
                    ExitShort("Exit", "Short");
            }
        }
 
        protected override void OnPositionUpdate(
            Position position, double averagePrice,
            int quantity, MarketPosition marketPosition)
        {
            if (position.Account != Account) return;
 
            if (marketPosition == MarketPosition.Long ||
                marketPosition == MarketPosition.Short)
            {
                _entryPrice     = averagePrice;
                _mae            = 0;
                _mfe            = 0;
                _barsSinceEntry = 0;
 
                double atr = _atr14[0];
                double sessionHigh = MAX(High, CurrentBar)[0];
                double sessionLow  = MIN(Low,  CurrentBar)[0];
                int dayIdx    = Bars.GetBar(Time[0].Date);
                double dayOpen = Opens[0][CurrentBar - dayIdx];
 
                double[] fv = AIFeatureBuilder.Build(
                    Close, Open, High, Low, Volume,
                    _ema9[0],  _ema9[1],  _ema9[2],  _ema9[3],
                    _ema21[0], _ema50[0], _ema200[0],
                    atr,
                    _bb.Upper[0], _bb.Lower[0],
                    _adx14[0], _mom14[0], _roc5[0],
                    _vwap,
                    SMA(Volume, 20)[0],
                    _prior.PriorHigh[0], _prior.PriorLow[0], _prior.PriorClose[0],
                    sessionHigh, sessionLow, dayOpen,
                    MAX(High, 10)[0], MIN(Low, 10)[0],
                    false, false
                );
 
                _logger?.OnTradeOpen(fv, Time[0], "Mechanical",
                    marketPosition == MarketPosition.Long);
            }
            else if (marketPosition == MarketPosition.Flat)
            {
                _logger?.OnTradeClose();
            }
        }
    }
}
 