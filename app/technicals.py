import pandas as pd
import numpy as np
import talib

def talib_get_momentum_indicators_for_one_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a wide set of TA-Lib momentum indicators for one ticker.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: Date, Ticker, Open, High, Low, Close, Volume.

    Returns
    -------
    pd.DataFrame
        New DataFrame with Date, Ticker, and momentum indicator columns.
        All columns are lowercase snake_case for consistency.
    """
    import talib

    # Momentum indicators
    adx   = talib.ADX(df.High.values, df.Low.values, df.Close.values, timeperiod=14)   # Trend strength
    adxr  = talib.ADXR(df.High.values, df.Low.values, df.Close.values, timeperiod=14)  # Smoothed ADX
    apo   = talib.APO(df.Close.values, fastperiod=12, slowperiod=26, matype=0)         # Absolute Price Oscillator
    aroon_up, aroon_down = talib.AROON(df.High.values, df.Low.values, timeperiod=14)   # Aroon up/down (trend detection)
    aroonosc = talib.AROONOSC(df.High.values, df.Low.values, timeperiod=14)            # Aroon oscillator
    bop   = talib.BOP(df.Open.values, df.High.values, df.Low.values, df.Close.values)  # Balance of Power
    cci   = talib.CCI(df.High.values, df.Low.values, df.Close.values, timeperiod=14)   # Commodity Channel Index
    cmo   = talib.CMO(df.Close.values, timeperiod=14)                                  # Chande Momentum Oscillator
    dx    = talib.DX(df.High.values, df.Low.values, df.Close.values, timeperiod=14)    # Directional Movement Index

    # MACD family
    macd, macd_signal, macd_hist = talib.MACD(df.Close.values, fastperiod=12, slowperiod=26, signalperiod=9)
    macd_ext, macd_signal_ext, macd_hist_ext = talib.MACDEXT(df.Close.values, fastperiod=12, fastmatype=0,
                                                             slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
    macd_fix, macd_signal_fix, macd_hist_fix = talib.MACDFIX(df.Close.values, signalperiod=9)

    # Money Flow
    mfi   = talib.MFI(df.High.values, df.Low.values, df.Close.values, df.Volume.values, timeperiod=14)  # Money Flow Index

    # Directional indicators
    minus_di = talib.MINUS_DI(df.High.values, df.Low.values, df.Close.values, timeperiod=14)  # Minus DI
    plus_di  = talib.PLUS_DI(df.High.values, df.Low.values, df.Close.values, timeperiod=14)   # Plus DI
    plus_dm  = talib.PLUS_DM(df.High.values, df.Low.values, timeperiod=14)                    # Plus DM

    # Momentum/Rate-of-Change
    mom   = talib.MOM(df.Close.values, timeperiod=10)         # Momentum
    ppo   = talib.PPO(df.Close.values, fastperiod=12, slowperiod=26, matype=0)  # % Price Oscillator
    roc   = talib.ROC(df.Close.values, timeperiod=10)         # Rate of Change (%)
    rocp  = talib.ROCP(df.Close.values, timeperiod=10)        # ROC (fractional)
    rocr  = talib.ROCR(df.Close.values, timeperiod=10)        # ROC ratio
    rocr100 = talib.ROCR100(df.Close.values, timeperiod=10)   # ROC ratio * 100
    rsi   = talib.RSI(df.Close.values, timeperiod=14)         # Relative Strength Index

    # Stochastics
    stoch_slowk, stoch_slowd = talib.STOCH(df.High.values, df.Low.values, df.Close.values,
                                           fastk_period=5, slowk_period=3, slowk_matype=0,
                                           slowd_period=3, slowd_matype=0)     # Stochastic Oscillator
    stoch_fastk, stoch_fastd = talib.STOCHF(df.High.values, df.Low.values, df.Close.values,
                                            fastk_period=5, fastd_period=3, fastd_matype=0)  # Stochastic Fast
    stochrsi_fastk, stochrsi_fastd = talib.STOCHRSI(df.Close.values, timeperiod=14,
                                                    fastk_period=5, fastd_period=3, fastd_matype=0)  # Stoch RSI

    # Other oscillators
    trix  = talib.TRIX(df.Close.values, timeperiod=30)  # Triple Smoothed EMA ROC
    ultosc = talib.ULTOSC(df.High.values, df.Low.values, df.Close.values,
                          timeperiod1=7, timeperiod2=14, timeperiod3=28)  # Ultimate Oscillator
    willr = talib.WILLR(df.High.values, df.Low.values, df.Close.values, timeperiod=14)  # Williams %R

    return pd.DataFrame({
        "date": df.Date.values,
        "ticker": df.Ticker.values,

        "adx": adx, "adxr": adxr, "apo": apo,
        "aroon_up": aroon_up, "aroon_down": aroon_down, "aroonosc": aroonosc,
        "bop": bop, "cci": cci, "cmo": cmo, "dx": dx,

        "macd": macd, "macd_signal": macd_signal, "macd_hist": macd_hist,
        "macd_ext": macd_ext, "macd_signal_ext": macd_signal_ext, "macd_hist_ext": macd_hist_ext,
        "macd_fix": macd_fix, "macd_signal_fix": macd_signal_fix, "macd_hist_fix": macd_hist_fix,

        "mfi": mfi, "minus_di": minus_di, "plus_di": plus_di, "plus_dm": plus_dm,
        "mom": mom, "ppo": ppo, "roc": roc, "rocp": rocp, "rocr": rocr, "rocr100": rocr100,
        "rsi": rsi,

        "stoch_slowk": stoch_slowk, "stoch_slowd": stoch_slowd,
        "stoch_fastk": stoch_fastk, "stoch_fastd": stoch_fastd,
        "stochrsi_fastk": stochrsi_fastk, "stochrsi_fastd": stochrsi_fastd,

        "trix": trix, "ultosc": ultosc, "willr": willr,
    })


def talib_get_volume_volatility_cycle_price_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute TA-Lib volume, volatility, cycle, and price-transform indicators for one ticker.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: Date, Ticker, Open, High, Low, Close, Volume.

    Returns
    -------
    pd.DataFrame
        Columns: date, ticker, and a set of talib-derived features (snake_case).
    """
    import talib
    import numpy as np

    # --- Inputs (ensure numpy arrays) ---
    high = df.High.values
    low = df.Low.values
    close = df.Close.values
    open_ = df.Open.values
    volume = df.Volume.values if "Volume" in df.columns else np.zeros_like(close)

    # -------- Volume indicators --------
    # AD  Chaikin A/D Line
    ad = talib.AD(high, low, close, volume)
    # ADOSC  Chaikin A/D Oscillator
    adosc = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
    # OBV  On Balance Volume
    obv = talib.OBV(close, volume)

    # -------- Volatility indicators --------
    # ATR  Average True Range
    atr = talib.ATR(high, low, close, timeperiod=14)
    # NATR  Normalized ATR (% of price)
    natr = talib.NATR(high, low, close, timeperiod=14)
    # TRANGE - True Range (optional addition)
    trange = talib.TRANGE(high, low, close)

    # -------- Cycle indicators (Hilbert Transform) --------
    ht_dcperiod = talib.HT_DCPERIOD(close)                         # Dominant Cycle Period
    ht_dcphase = talib.HT_DCPHASE(close)                           # Dominant Cycle Phase
    ht_phasor_inphase, ht_phasor_quadrature = talib.HT_PHASOR(close)  # Phasor components
    ht_sine_sine, ht_sine_leadsine = talib.HT_SINE(close)          # Sine & Lead Sine
    ht_trendmode = talib.HT_TRENDMODE(close)                       # 1=trend, 0=cycle (per TA-Lib)

    # -------- Price transforms --------
    avgprice = talib.AVGPRICE(open_, high, low, close)             # (H+L+O+C)/4
    medprice = talib.MEDPRICE(high, low)                           # (H+L)/2
    typprice = talib.TYPPRICE(high, low, close)                    # (H+L+C)/3
    wclprice = talib.WCLPRICE(high, low, close)                    # (H+L+2C)/4

    # Assemble DataFrame
    volume_volatility_cycle_price_df = pd.DataFrame({
        "date": pd.to_datetime(df.Date.values),
        "ticker": df.Ticker.values,

        # Volume
        "ad": ad,
        "adosc": adosc,
        "obv": obv,

        # Volatility
        "atr": atr,
        "natr": natr,
        "trange": trange,

        # Cycle
        "ht_dcperiod": ht_dcperiod,
        "ht_dcphase": ht_dcphase,
        "ht_phasor_inphase": ht_phasor_inphase,
        "ht_phasor_quadrature": ht_phasor_quadrature,
        "ht_sine_sine": ht_sine_sine,
        "ht_sine_leadsine": ht_sine_leadsine,
        "ht_trendmode": ht_trendmode,

        # Price transforms
        "avgprice": avgprice,
        "medprice": medprice,
        "typprice": typprice,
        "wclprice": wclprice,
    })

    return volume_volatility_cycle_price_df





def talib_get_pattern_recognition_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute TA-Lib candlestick pattern recognition indicators for one ticker.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: Date, Ticker, Open, High, Low, Close, Volume.
    
    Returns
    -------
    pd.DataFrame
        New DataFrame with date, ticker, and pattern recognition indicator columns.
        All columns are lowercase snake_case for consistency.
        Pattern indicators return: 100 = bullish, -100 = bearish, 0 = no pattern
    """
    import talib
    import pandas as pd

    # TA-Lib Pattern Recognition indicators
    # https://github.com/TA-Lib/ta-lib-python/blob/master/docs/func_groups/pattern_recognition.md
    # Nice article about candles (pattern recognition) 
    # https://medium.com/analytics-vidhya/recognizing-over-50-candlestick-patterns-with-python-4f02a1822cb5

    # CDL2CROWS - Two Crows
    talib_cdl2crows = talib.CDL2CROWS(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDL3BLACKCROWS - Three Black Crows
    talib_cdl3blackcrows = talib.CDL3BLACKCROWS(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDL3INSIDE - Three Inside Up/Down
    talib_cdl3inside = talib.CDL3INSIDE(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDL3LINESTRIKE - Three-Line Strike
    talib_cdl3linestrike = talib.CDL3LINESTRIKE(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDL3OUTSIDE - Three Outside Up/Down
    talib_cdl3outside = talib.CDL3OUTSIDE(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDL3STARSINSOUTH - Three Stars In The South
    talib_cdl3starsinsouth = talib.CDL3STARSINSOUTH(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDL3WHITESOLDIERS - Three Advancing White Soldiers
    talib_cdl3whitesoldiers = talib.CDL3WHITESOLDIERS(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLABANDONEDBABY - Abandoned Baby
    talib_cdlabandonedbaby = talib.CDLABANDONEDBABY(
        df.Open.values, df.High.values, df.Low.values, df.Close.values, penetration=0)
    # CDLADVANCEBLOCK - Advance Block
    talib_cdladvanceblock = talib.CDLADVANCEBLOCK(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLBELTHOLD - Belt-hold
    talib_cdlbelthold = talib.CDLBELTHOLD(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLBREAKAWAY - Breakaway
    talib_cdlbreakaway = talib.CDLBREAKAWAY(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLCLOSINGMARUBOZU - Closing Marubozu
    talib_cdlclosingmarubozu = talib.CDLCLOSINGMARUBOZU(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLCONCEALBABYSWALL - Concealing Baby Swallow
    talib_cdlconcealbabyswall = talib.CDLCONCEALBABYSWALL(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLCOUNTERATTACK - Counterattack
    talib_cdlcounterattack = talib.CDLCOUNTERATTACK(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLDARKCLOUDCOVER - Dark Cloud Cover
    talib_cdldarkcloudcover = talib.CDLDARKCLOUDCOVER(
        df.Open.values, df.High.values, df.Low.values, df.Close.values, penetration=0)
    # CDLDOJI - Doji
    talib_cdldoji = talib.CDLDOJI(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLDOJISTAR - Doji Star
    talib_cdldojistar = talib.CDLDOJISTAR(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLDRAGONFLYDOJI - Dragonfly Doji
    talib_cdldragonflydoji = talib.CDLDRAGONFLYDOJI(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLENGULFING - Engulfing Pattern
    talib_cdlengulfing = talib.CDLENGULFING(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)

    # CDLEVENINGDOJISTAR - Evening Doji Star
    talib_cdleveningdojistar = talib.CDLEVENINGDOJISTAR(
        df.Open.values, df.High.values, df.Low.values, df.Close.values, penetration=0)
    # CDLEVENINGSTAR - Evening Star
    talib_cdleveningstar = talib.CDLEVENINGSTAR(
        df.Open.values, df.High.values, df.Low.values, df.Close.values, penetration=0)
    # CDLGAPSIDESIDEWHITE - Up/Down-gap side-by-side white lines
    talib_cdlgapsidesidewhite = talib.CDLGAPSIDESIDEWHITE(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLGRAVESTONEDOJI - Gravestone Doji
    talib_cdlgravestonedoji = talib.CDLGRAVESTONEDOJI(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLHAMMER - Hammer
    talib_cdlhammer = talib.CDLHAMMER(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLHANGINGMAN - Hanging Man
    talib_cdlhangingman = talib.CDLHANGINGMAN(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLHARAMI - Harami Pattern
    talib_cdlharami = talib.CDLHARAMI(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLHARAMICROSS - Harami Cross Pattern
    talib_cdlharamicross = talib.CDLHARAMICROSS(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLHIGHWAVE - High-Wave Candle
    talib_cdlhighwave = talib.CDLHIGHWAVE(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLHIKKAKE - Hikkake Pattern
    talib_cdlhikkake = talib.CDLHIKKAKE(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLHIKKAKEMOD - Modified Hikkake Pattern
    talib_cdlhikkakemod = talib.CDLHIKKAKEMOD(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)

    # CDLHOMINGPIGEON - Homing Pigeon
    talib_cdlhomingpigeon = talib.CDLHOMINGPIGEON(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLIDENTICAL3CROWS - Identical Three Crows
    talib_cdlidentical3crows = talib.CDLIDENTICAL3CROWS(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLINNECK - In-Neck Pattern
    talib_cdlinneck = talib.CDLINNECK(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLINVERTEDHAMMER - Inverted Hammer
    talib_cdlinvertedhammer = talib.CDLINVERTEDHAMMER(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLKICKING - Kicking
    talib_cdlkicking = talib.CDLKICKING(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLKICKINGBYLENGTH - Kicking - bull/bear determined by the longer marubozu
    talib_cdlkickingbylength = talib.CDLKICKINGBYLENGTH(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLLADDERBOTTOM - Ladder Bottom
    talib_cdlladderbottom = talib.CDLLADDERBOTTOM(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLLONGLEGGEDDOJI - Long Legged Doji
    talib_cdllongleggeddoji = talib.CDLLONGLEGGEDDOJI(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLLONGLINE - Long Line Candle
    talib_cdllongline = talib.CDLLONGLINE(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLMARUBOZU - Marubozu
    talib_cdlmarubozu = talib.CDLMARUBOZU(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLMATCHINGLOW - Matching Low
    talib_cdlmatchinglow = talib.CDLMATCHINGLOW(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)

    # CDLMATHOLD - Mat Hold
    talib_cdlmathold = talib.CDLMATHOLD(
        df.Open.values, df.High.values, df.Low.values, df.Close.values, penetration=0)
    # CDLMORNINGDOJISTAR - Morning Doji Star
    talib_cdlmorningdojistar = talib.CDLMORNINGDOJISTAR(
        df.Open.values, df.High.values, df.Low.values, df.Close.values, penetration=0)
    # CDLMORNINGSTAR - Morning Star
    talib_cdlmorningstar = talib.CDLMORNINGSTAR(
        df.Open.values, df.High.values, df.Low.values, df.Close.values, penetration=0)
    # CDLONNECK - On-Neck Pattern
    talib_cdlonneck = talib.CDLONNECK(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLPIERCING - Piercing Pattern
    talib_cdlpiercing = talib.CDLPIERCING(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLRICKSHAWMAN - Rickshaw Man
    talib_cdlrickshawman = talib.CDLRICKSHAWMAN(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLRISEFALL3METHODS - Rising/Falling Three Methods
    talib_cdlrisefall3methods = talib.CDLRISEFALL3METHODS(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLSEPARATINGLINES - Separating Lines
    talib_cdlseparatinglines = talib.CDLSEPARATINGLINES(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLSHOOTINGSTAR - Shooting Star
    talib_cdlshootingstar = talib.CDLSHOOTINGSTAR(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLSHORTLINE - Short Line Candle
    talib_cdlshortline = talib.CDLSHORTLINE(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLSPINNINGTOP - Spinning Top
    talib_cdlspinningtop = talib.CDLSPINNINGTOP(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)

    # CDLSTALLEDPATTERN - Stalled Pattern
    talib_cdlstalledpattern = talib.CDLSTALLEDPATTERN(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLSTICKSANDWICH - Stick Sandwich
    talib_cdlsticksandwich = talib.CDLSTICKSANDWICH(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLTAKURI - Takuri (Dragonfly Doji with very long lower shadow)
    talib_cdltakuri = talib.CDLTAKURI(  # Fixed variable name
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLTASUKIGAP - Tasuki Gap
    talib_cdltasukigap = talib.CDLTASUKIGAP(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLTHRUSTING - Thrusting Pattern
    talib_cdlthrusting = talib.CDLTHRUSTING(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLTRISTAR - Tristar Pattern
    talib_cdltristar = talib.CDLTRISTAR(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLUNIQUE3RIVER - Unique 3 River
    talib_cdlunique3river = talib.CDLUNIQUE3RIVER(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLUPSIDEGAP2CROWS - Upside Gap Two Crows
    talib_cdlupsidegap2crows = talib.CDLUPSIDEGAP2CROWS(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLXSIDEGAP3METHODS - Upside/Downside Gap Three Methods
    talib_cdlxsidegap3methods = talib.CDLXSIDEGAP3METHODS(
        df.Open.values, df.High.values, df.Low.values, df.Close.values)

    pattern_indicators_df = pd.DataFrame({
        'date': pd.to_datetime(df.Date.values),  # Consistent snake_case and datetime conversion
        'ticker': df.Ticker.values,
        
        # TA-Lib Pattern Recognition indicators
        'cdl2crows': talib_cdl2crows,
        'cdl3blackcrows': talib_cdl3blackcrows,  # Fixed variable name
        'cdl3inside': talib_cdl3inside,
        'cdl3linestrike': talib_cdl3linestrike,
        'cdl3outside': talib_cdl3outside,
        'cdl3starsinsouth': talib_cdl3starsinsouth,
        'cdl3whitesoldiers': talib_cdl3whitesoldiers,
        'cdlabandonedbaby': talib_cdlabandonedbaby,
        'cdladvanceblock': talib_cdladvanceblock,  # Fixed column name
        'cdlbelthold': talib_cdlbelthold,
        'cdlbreakaway': talib_cdlbreakaway,
        'cdlclosingmarubozu': talib_cdlclosingmarubozu,
        'cdlconcealbabyswall': talib_cdlconcealbabyswall,
        'cdlcounterattack': talib_cdlcounterattack,
        'cdldarkcloudcover': talib_cdldarkcloudcover,
        'cdldoji': talib_cdldoji,
        'cdldojistar': talib_cdldojistar,
        'cdldragonflydoji': talib_cdldragonflydoji,
        'cdlengulfing': talib_cdlengulfing,
        'cdleveningdojistar': talib_cdleveningdojistar,
        'cdleveningstar': talib_cdleveningstar,
        'cdlgapsidesidewhite': talib_cdlgapsidesidewhite,
        'cdlgravestonedoji': talib_cdlgravestonedoji,
        'cdlhammer': talib_cdlhammer,
        'cdlhangingman': talib_cdlhangingman,
        'cdlharami': talib_cdlharami,
        'cdlharamicross': talib_cdlharamicross,
        'cdlhighwave': talib_cdlhighwave,
        'cdlhikkake': talib_cdlhikkake,
        'cdlhikkakemod': talib_cdlhikkakemod,
        'cdlhomingpigeon': talib_cdlhomingpigeon,
        'cdlidentical3crows': talib_cdlidentical3crows,
        'cdlinneck': talib_cdlinneck,
        'cdlinvertedhammer': talib_cdlinvertedhammer,
        'cdlkicking': talib_cdlkicking,
        'cdlkickingbylength': talib_cdlkickingbylength,
        'cdlladderbottom': talib_cdlladderbottom,
        'cdllongleggeddoji': talib_cdllongleggeddoji,
        'cdllongline': talib_cdllongline,
        'cdlmarubozu': talib_cdlmarubozu,
        'cdlmatchinglow': talib_cdlmatchinglow,
        'cdlmathold': talib_cdlmathold,
        'cdlmorningdojistar': talib_cdlmorningdojistar,
        'cdlmorningstar': talib_cdlmorningstar,
        'cdlonneck': talib_cdlonneck,
        'cdlpiercing': talib_cdlpiercing,
        'cdlrickshawman': talib_cdlrickshawman,
        'cdlrisefall3methods': talib_cdlrisefall3methods,
        'cdlseparatinglines': talib_cdlseparatinglines,
        'cdlshootingstar': talib_cdlshootingstar,
        'cdlshortline': talib_cdlshortline,
        'cdlspinningtop': talib_cdlspinningtop,
        'cdlstalledpattern': talib_cdlstalledpattern,
        'cdlsticksandwich': talib_cdlsticksandwich,
        'cdltakuri': talib_cdltakuri,  # Fixed variable reference
        'cdltasukigap': talib_cdltasukigap,
        'cdlthrusting': talib_cdlthrusting,
        'cdltristar': talib_cdltristar,
        'cdlunique3river': talib_cdlunique3river,
        'cdlupsidegap2crows': talib_cdlupsidegap2crows,
        'cdlxsidegap3methods': talib_cdlxsidegap3methods
    })

    return pattern_indicators_df