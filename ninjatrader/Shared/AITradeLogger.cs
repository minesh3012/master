using System;
using System.IO;
using System.Collections.Generic;
 
namespace NTAIShared
{
    public class AITradeLogger
    {
        private readonly string _strategyName;
        private readonly string _instrument;
        private StreamWriter _writer;
 
        private int  _tradeId        = 0;
        private bool _inTrade        = false;
        private int  _barsSinceEntry = 0;
        private int  _postRemaining  = 0;
 
        private readonly int _preBars;
        private readonly int _postBars;
 
        private readonly Queue<PendingRow> _preBuffer;
 
        private struct PendingRow
        {
            public double[] fv;
            public DateTime time;
            public string   signal;
            public int      offset;
        }
 
        public bool Enabled { get; set; } = true;
 
        public AITradeLogger(string strategyName, string instrument,
                             int preBars = 5, int postBars = 5,
                             string outputFolder = null)
        {
            _strategyName = strategyName;
            _instrument   = instrument;
            _preBars      = preBars;
            _postBars     = postBars;
            _preBuffer    = new Queue<PendingRow>(preBars + 1);
 
            string folder = outputFolder
                ?? Path.Combine(NinjaTrader.Core.Globals.UserDataDir, "AITraining");
            Directory.CreateDirectory(folder);
 
            string path = Path.Combine(folder,
                $"{instrument.Replace(" ", "_")}_{strategyName}.csv");
 
            bool isNew = !File.Exists(path) || new FileInfo(path).Length == 0;
            _writer = new StreamWriter(path, append: true);
            if (isNew) WriteHeader();
        }
 
        public void OnBar(
            double[] fv, DateTime time, string signal,
            bool isLong, bool isShort,
            int barsSinceEntry, double unrealAtr, double maeAtr, double mfeAtr)
        {
            if (!Enabled || _writer == null) return;
 
            if (!_inTrade && _postRemaining == 0)
            {
                if (_preBuffer.Count >= _preBars)
                    _preBuffer.Dequeue();
                _preBuffer.Enqueue(new PendingRow
                {
                    fv = fv, time = time, signal = signal,
                    offset = _preBuffer.Count - _preBars
                });
                return;
            }
 
            string phase;
            int offset;
 
            if (_inTrade)
            {
                phase  = "In";
                offset = _barsSinceEntry;
                _barsSinceEntry++;
            }
            else
            {
                phase  = "Post";
                offset = _postBars - _postRemaining + 1;
                _postRemaining--;
            }
 
            WriteLine(fv, time, _tradeId, signal, phase, offset,
                      isLong, isShort, barsSinceEntry, unrealAtr, maeAtr, mfeAtr);
        }
 
        public void OnTradeOpen(
            double[] fv, DateTime time, string signal, bool isLong)
        {
            if (!Enabled || _writer == null) return;
 
            _tradeId++;
            _inTrade        = true;
            _barsSinceEntry = 0;
 
            int i = -_preBuffer.Count;
            foreach (var row in _preBuffer)
            {
                WriteLine(row.fv, row.time, _tradeId, row.signal,
                          "Pre", i, isLong, !isLong, 0, 0, 0, 0);
                i++;
            }
            _preBuffer.Clear();
 
            WriteLine(fv, time, _tradeId, signal, "In", 0,
                      isLong, !isLong, 0, 0, 0, 0);
        }
 
        public void OnTradeClose()
        {
            if (!Enabled) return;
            _inTrade       = false;
            _postRemaining = _postBars;
        }
 
        public void LogSignal(
            double[] fv, DateTime time, string signal, int direction)
        {
            if (!Enabled || _writer == null) return;
            WriteLine(fv, time, 0, signal, "Signal", direction,
                      direction == 1, direction == -1, 0, 0, 0, 0);
        }
 
        public void Close()
        {
            _writer?.Flush();
            _writer?.Close();
            _writer = null;
        }
 
        private void WriteHeader()
        {
            _writer.Write(
                "Instrument,Strategy,TradeId,Signal,Phase,Offset,Time," +
                "IsLong,IsShort,BarsSinceEntry,UnrealAtr,MAE_Atr,MFE_Atr");
 
            for (int i = 0; i < AIFeatureBuilder.FeatureCount; i++)
                _writer.Write($",F{i:D2}");
 
            _writer.WriteLine();
        }
 
        private void WriteLine(
            double[] fv, DateTime time, int tradeId,
            string signal, string phase, int offset,
            bool isLong, bool isShort,
            int barsSince, double unreal, double mae, double mfe)
        {
            var ci = System.Globalization.CultureInfo.InvariantCulture;
 
            _writer.Write(
                $"{_instrument},{_strategyName},{tradeId},{signal}," +
                $"{phase},{offset},{time:o}," +
                $"{(isLong?1:0)},{(isShort?1:0)},{barsSince}," +
                $"{unreal.ToString("F4",ci)},{mae.ToString("F4",ci)},{mfe.ToString("F4",ci)}");
 
            for (int i = 0; i < fv.Length; i++)
                _writer.Write($",{fv[i].ToString("F6", ci)}");
 
            _writer.WriteLine();
        }
    }
}