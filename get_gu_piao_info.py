"""A股行情抓取与历史入库。

作用:
    抓取最新 A 股行情和必要的历史窗口，计算均线、涨跌幅、量比、
    振幅、换手、阶段高低点等策略特征，并维护 a_stock_analysis 与
    a_stock_analysis_history。它是每日推荐流程的数据入口。

流程:
    先读取基础股票池并并发请求腾讯、东方财富、Sina 等行情源；
    再对行情做重试、兜底、交易日一致性和字段规范化；
    然后计算策略所需的滚动指标和最新快照标记；
    最后写入当前分析表和历史表，并返回本次目标交易日与入库统计。
"""

from collections import Counter
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from datetime import datetime, timedelta
from multiprocessing import Pool
import time
import signal

import akshare as ak
import pandas as pd

import func


LOOKBACK_DAYS = 260
SUPPLEMENT_LOOKBACK_DAYS = 45
SUPPLEMENT_REQUIRED_RECENT_ROWS = 11
MAX_PROCESSES = 10
MAX_FETCH_RETRIES = 2
RETRY_BASE_SLEEP_SECONDS = 1
TENCENT_TIMEOUT = 15
EASTMONEY_TIMEOUT = 8
SINA_TIMEOUT = 8
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 0
PROGRESS_LOG_INTERVAL_SECONDS = 5
SOURCE_FETCH_RETRIES = {
    "tencent": 2,
    "eastmoney": 1,
    "sina": 1,
}
PRIMARY_HISTORY_SOURCES = ("tencent", "eastmoney")
SUPPLEMENT_SOURCE_SPECS = (
    {
        "name": "eastmoney",
        "columns": ("成交额", "换手率"),
        "lookback_days": SUPPLEMENT_LOOKBACK_DAYS,
        "skip_if_primary": True,
    },
    {
        "name": "sina",
        "columns": ("成交额", "换手率"),
        "lookback_days": SUPPLEMENT_LOOKBACK_DAYS,
        "skip_if_primary": True,
    },
)
FETCH_TASK_TIMEOUT_SECONDS = 90
POOL_RESULT_POLL_SECONDS = 0.25
CANONICAL_PRICE_ADJUST = "qfq"
CANONICAL_VOLUME_UNIT = "手"
CANONICAL_AMOUNT_UNIT = "元"
CANONICAL_TURNOVER_UNIT = "百分点"
TENCENT_SHARE_VOLUME_PREFIXES = ("688", "689")
RAW_SHARE_VOLUME_AMOUNT_THRESHOLD = 200000000000
MARKET_DATA_COLUMNS = ["开盘", "收盘", "最高", "最低", "成交量", "成交额", "换手率"]
STRING_NUMERIC_COLUMNS = {
    "latest_price",
    "today_change",
    "change_3d",
    "change_5d",
    "change_10d",
    "change_20d",
    "change_30d",
    "today_vol",
    "vol_avg_3d",
    "vol_avg_5d",
    "vol_avg_10d",
    "vol_avg_20d",
    "today_amp",
    "amp_3d",
    "amp_5d",
    "amp_10d",
    "amp_20d",
    "amp_30d",
    "vr_today",
    "vr_3d",
    "vr_5d",
    "vr_10d",
    "vr_20d",
    "vr_30d",
    "today_amount",
    "amount_avg_5d",
    "amount_avg_10d",
    "turnover_rate",
    "turnover_avg_5d",
    "turnover_avg_10d",
    "volatility_10d",
    "volatility_20d",
    "ma20_slope_5d",
    "ma60_slope_10d",
}
DECIMAL_COLUMNS = {
    "today_open",
    "today_high",
    "today_low",
    "ma5",
    "ma10",
    "ma20",
    "ma60",
    "ma120",
    "high_20d",
    "low_20d",
    "high_60d",
    "low_60d",
    "high_120d",
    "low_120d",
}
HISTORY_COLUMNS = [
    "last_data_date",
    "stock_code",
    "stock_name",
    "latest_price",
    "today_open",
    "today_high",
    "today_low",
    "today_change",
    "change_3d",
    "change_5d",
    "change_10d",
    "change_20d",
    "change_30d",
    "today_vol",
    "today_amount",
    "turnover_rate",
    "vol_avg_3d",
    "vol_avg_5d",
    "vol_avg_10d",
    "vol_avg_20d",
    "amount_avg_5d",
    "amount_avg_10d",
    "turnover_avg_5d",
    "turnover_avg_10d",
    "industry",
    "stock_rank",
    "today_amp",
    "amp_3d",
    "amp_5d",
    "amp_10d",
    "amp_20d",
    "amp_30d",
    "vr_today",
    "vr_3d",
    "vr_5d",
    "vr_10d",
    "vr_20d",
    "vr_30d",
    "is_last_info",
    "ma5",
    "ma10",
    "ma20",
    "ma60",
    "ma120",
    "high_20d",
    "low_20d",
    "high_60d",
    "low_60d",
    "high_120d",
    "low_120d",
    "volatility_10d",
    "volatility_20d",
    "ma20_slope_5d",
    "ma60_slope_10d",
]


def _round_or_none(value, digits=2):
    if pd.isna(value):
        return None
    return round(float(value), digits)


def _to_decimal(value, scale=2):
    if value is None or pd.isna(value):
        return None

    try:
        quant = Decimal("1").scaleb(-scale)
        return Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError, TypeError):
        return None


def _format_numeric_string(value, scale=2):
    decimal_value = _to_decimal(value, scale)
    if decimal_value is None:
        return None

    text = format(decimal_value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _normalize_payload_for_storage(payload):
    normalized_payload = {}

    for key, value in payload.items():
        if key in STRING_NUMERIC_COLUMNS:
            normalized_payload[key] = _format_numeric_string(value, 2)
        elif key in DECIMAL_COLUMNS:
            normalized_payload[key] = _to_decimal(value, 2)
        else:
            normalized_payload[key] = value

    return normalized_payload


def _strict_normalize_update_payload(payload):
    normalized = _normalize_payload_for_storage(payload)

    # 防御式兜底：即便上游改动了字段类型，这里也保证 varchar 数值字段最终是字符串。
    for key in STRING_NUMERIC_COLUMNS:
        value = normalized.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            normalized[key] = _format_numeric_string(value, 2)

    # decimal 字段确保写入 Decimal 或 None。
    for key in DECIMAL_COLUMNS:
        value = normalized.get(key)
        if value is None:
            continue
        if not isinstance(value, Decimal):
            normalized[key] = _to_decimal(value, 2)

    return normalized


def _format_fetch_error(source_name, error):
    return f"{source_name}:{error.__class__.__name__}: {error}"

def _run_with_timeout(seconds, func, *args, **kwargs):
    # 仅在 Unix 主线程可用；当前 worker 进程模型满足这个条件。
    def _timeout_handler(signum, frame):
        raise TimeoutError(f"fetch_timeout_{seconds}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(seconds)
    try:
        return func(*args, **kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def _to_prefixed_symbol(stock_code):
    code = str(stock_code)

    if code.startswith(("600", "601", "603", "605", "688", "689", "900")):
        return f"sh{code}"
    if code.startswith(("000", "001", "002", "003", "300", "301", "200")):
        return f"sz{code}"
    return f"bj{code}"


def _stock_code_text(stock_code):
    return str(stock_code or "").strip()


def _tencent_volume_is_share_unit(stock_code):
    return _stock_code_text(stock_code).startswith(TENCENT_SHARE_VOLUME_PREFIXES)


def _to_trade_date_string(value):
    if pd.isna(value):
        return None
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def _safe_pct_change(close_series, days):
    if len(close_series) <= days:
        return None

    latest_price = close_series.iloc[-1]
    base_price = close_series.iloc[-days - 1]

    if pd.isna(latest_price) or pd.isna(base_price) or base_price == 0:
        return None

    return round((latest_price - base_price) / base_price * 100, 2)


def _safe_previous_average(series, days):
    if len(series) <= days:
        return None

    window = pd.to_numeric(series.iloc[-days - 1:-1], errors="coerce").dropna()
    if len(window) < days:
        return None

    return round(window.mean(), 2)


def _safe_volume_ratio(volume_series, days):
    if len(volume_series) <= days:
        return None

    today_volume = pd.to_numeric(volume_series.iloc[-1], errors="coerce")
    average_volume = _safe_previous_average(volume_series, days)

    if pd.isna(today_volume) or average_volume in (None, 0):
        return None

    return round(today_volume / average_volume, 2)


def _resolve_volume_unit_multiplier(stock_code=None, close_series=None, volume_series=None):
    if close_series is not None and volume_series is not None:
        close_series = pd.to_numeric(close_series, errors="coerce")
        volume_series = pd.to_numeric(volume_series, errors="coerce")
        latest_estimated_amount = (close_series * volume_series * 100).dropna()
        # 标准入库成交量口径是“手”。如果未来某个源漏归一化、直接给“股”，
        # 用极端成交额兜底识别，避免成交额被再放大100倍。
        if not latest_estimated_amount.empty and latest_estimated_amount.iloc[-1] >= RAW_SHARE_VOLUME_AMOUNT_THRESHOLD:
            return 1.0

    return 100.0


def _estimate_amount_series(close_series, volume_series, amount_series=None, stock_code=None):
    close_series = pd.to_numeric(close_series, errors="coerce")
    volume_series = pd.to_numeric(volume_series, errors="coerce")
    volume_unit_multiplier = _resolve_volume_unit_multiplier(
        stock_code=stock_code,
        close_series=close_series,
        volume_series=volume_series,
    )
    estimated_amount = close_series * volume_series * volume_unit_multiplier

    if amount_series is None:
        return estimated_amount

    amount_series = pd.to_numeric(amount_series, errors="coerce")
    return amount_series.where(amount_series.notna(), estimated_amount)


def _safe_pct_change_series(series, periods):
    numeric_series = pd.to_numeric(series, errors="coerce")
    base_series = numeric_series.shift(periods)
    pct_series = (numeric_series - base_series) / base_series * 100
    pct_series = pct_series.where(base_series != 0)
    return pct_series


def _safe_previous_average_series(series, days):
    numeric_series = pd.to_numeric(series, errors="coerce")
    return numeric_series.shift(1).rolling(days, min_periods=days).mean()


def _safe_volume_ratio_series(volume_series, days):
    numeric_volume = pd.to_numeric(volume_series, errors="coerce")
    average_volume = _safe_previous_average_series(numeric_volume, days)
    ratio_series = numeric_volume / average_volume
    ratio_series = ratio_series.where(average_volume != 0)
    return ratio_series


def _prepare_hist_dataframe(raw_df):
    if raw_df.empty or "日期" not in raw_df.columns:
        return pd.DataFrame()
    df = raw_df.copy()
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    df = df[df["日期"].notna()].sort_values("日期").reset_index(drop=True)
    for column in MARKET_DATA_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.dropna(subset=["收盘"]).reset_index(drop=True)


def _should_drop_today_bar(now=None):
    current_dt = now or datetime.now()
    if current_dt.hour < MARKET_CLOSE_HOUR:
        return True
    if current_dt.hour == MARKET_CLOSE_HOUR and current_dt.minute < MARKET_CLOSE_MINUTE:
        return True
    return False


def _strip_incomplete_today_bar(df, now=None):
    if df.empty or "日期" not in df.columns:
        return df

    latest_trade_date = _to_trade_date_string(df.iloc[-1]["日期"])
    if not latest_trade_date:
        return df

    current_dt = now or datetime.now()
    today_text = current_dt.strftime("%Y-%m-%d")
    if latest_trade_date != today_text:
        return df

    if not _should_drop_today_bar(current_dt):
        return df

    return df.iloc[:-1].reset_index(drop=True)


def _resolve_recent_start_date(start_date, end_date, lookback_days):
    try:
        end_dt = datetime.strptime(str(end_date), "%Y%m%d")
        recent_start = (end_dt - timedelta(days=lookback_days)).strftime("%Y%m%d")
    except ValueError:
        return start_date

    return max(str(start_date), recent_start)


def _fetch_hist_from_eastmoney(stock_code, start_date, end_date):
    return _run_with_timeout(
        EASTMONEY_TIMEOUT,
        ak.stock_zh_a_hist,
        symbol=stock_code,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust=CANONICAL_PRICE_ADJUST,
    )


def _fetch_hist_from_tencent_raw(stock_code, start_date, end_date):
    return ak.stock_zh_a_hist_tx(
        symbol=_to_prefixed_symbol(stock_code),
        start_date=start_date,
        end_date=end_date,
        adjust=CANONICAL_PRICE_ADJUST,
        timeout=TENCENT_TIMEOUT,
    )


def _fetch_hist_from_tencent(stock_code, start_date, end_date):
    tx_df = _run_with_timeout(
        TENCENT_TIMEOUT + 3,
        _fetch_hist_from_tencent_raw,
        stock_code,
        start_date,
        end_date,
    )

    if tx_df.empty:
        return tx_df

    rename_map = {
        "date": "日期",
        "open": "开盘",
        "close": "收盘",
        "high": "最高",
        "low": "最低",
        "amount": "成交量",
    }
    prepared_df = tx_df.rename(columns=rename_map)
    if "成交量" in prepared_df.columns:
        prepared_df["成交量"] = pd.to_numeric(prepared_df["成交量"], errors="coerce")
        if _tencent_volume_is_share_unit(stock_code):
            prepared_df["成交量"] = prepared_df["成交量"] / 100.0
    return prepared_df


def _fetch_hist_from_sina(stock_code, start_date, end_date):
    sina_df = _run_with_timeout(
        SINA_TIMEOUT,
        ak.stock_zh_a_daily,
        symbol=_to_prefixed_symbol(stock_code),
        start_date=start_date,
        end_date=end_date,
        adjust=CANONICAL_PRICE_ADJUST,
    )

    if sina_df.empty:
        return sina_df

    rename_map = {
        "date": "日期",
        "open": "开盘",
        "close": "收盘",
        "high": "最高",
        "low": "最低",
        "volume": "成交量",
        "amount": "成交额",
        "turnover": "换手率",
    }
    prepared_df = sina_df.rename(columns=rename_map)
    if "成交量" in prepared_df.columns:
        prepared_df["成交量"] = pd.to_numeric(prepared_df["成交量"], errors="coerce") / 100.0
    if "换手率" in prepared_df.columns:
        prepared_df["换手率"] = pd.to_numeric(prepared_df["换手率"], errors="coerce") * 100.0
    return prepared_df


def _fetch_source_history(source_name, stock_code, start_date, end_date):
    source_fetchers = {
        "tencent": _fetch_hist_from_tencent,
        "eastmoney": _fetch_hist_from_eastmoney,
        "sina": _fetch_hist_from_sina,
    }
    fetcher = source_fetchers.get(source_name)
    if fetcher is None:
        return pd.DataFrame(), [f"{source_name}:unsupported_source"]

    errors = []
    retry_times = SOURCE_FETCH_RETRIES.get(source_name, MAX_FETCH_RETRIES)
    retry_times = max(1, min(retry_times, MAX_FETCH_RETRIES))
    for attempt in range(1, retry_times + 1):
        try:
            raw_df = fetcher(stock_code, start_date, end_date)
            prepared_df = _strip_incomplete_today_bar(_prepare_hist_dataframe(raw_df))
            if not prepared_df.empty:
                return prepared_df, errors

            errors.append(f"{source_name}:empty_history")
        except Exception as error:
            errors.append(_format_fetch_error(source_name, error))

        if attempt < retry_times:
            time.sleep(RETRY_BASE_SLEEP_SECONDS * attempt)

    return pd.DataFrame(), errors


def _merge_supplement_columns(base_df, supplement_df, columns):
    if base_df.empty or supplement_df.empty:
        return base_df

    available_columns = [column for column in columns if column in supplement_df.columns]
    if not available_columns:
        return base_df

    supplement_slice = supplement_df[["日期", *available_columns]].drop_duplicates(subset=["日期"], keep="last").copy()
    merged_df = base_df.merge(supplement_slice, on="日期", how="left", suffixes=("", "__supp"))

    for column in available_columns:
        supplement_column = f"{column}__supp"
        merged_df[column] = merged_df[supplement_column].where(
            merged_df[supplement_column].notna(),
            merged_df[column],
        )
        merged_df = merged_df.drop(columns=[supplement_column])

    return merged_df


def _has_recent_column_coverage(df, columns, required_rows=SUPPLEMENT_REQUIRED_RECENT_ROWS):
    if df.empty:
        return False

    available_columns = [column for column in columns if column in df.columns]
    if len(available_columns) != len(columns):
        return False

    recent_rows = min(len(df), required_rows)
    recent_df = df[available_columns].tail(recent_rows)
    if recent_df.empty:
        return False

    return recent_df.notna().all(axis=0).all()


def _fetch_history_with_fallback(stock_code, start_date, end_date):
    errors = []
    primary_df = pd.DataFrame()
    primary_source = None

    for source_name in PRIMARY_HISTORY_SOURCES:
        source_df, source_errors = _fetch_source_history(source_name, stock_code, start_date, end_date)
        errors.extend(source_errors)
        if source_df.empty:
            continue
        primary_df = source_df
        primary_source = source_name
        break

    if primary_df.empty or primary_source is None:
        return pd.DataFrame(), None, [], errors

    merged_df = primary_df
    supplement_sources = []

    for supplement_spec in SUPPLEMENT_SOURCE_SPECS:
        supplement_name = supplement_spec["name"]
        if supplement_spec.get("skip_if_primary") and supplement_name == primary_source:
            continue

        target_columns = tuple(supplement_spec.get("columns", ()))
        if target_columns and _has_recent_column_coverage(merged_df, target_columns):
            break

        supplement_start_date = _resolve_recent_start_date(
            start_date,
            end_date,
            supplement_spec.get("lookback_days", SUPPLEMENT_LOOKBACK_DAYS),
        )
        supplement_df, supplement_errors = _fetch_source_history(
            supplement_name,
            stock_code,
            supplement_start_date,
            end_date,
        )
        errors.extend(supplement_errors)
        if supplement_df.empty:
            continue

        merged_df = _merge_supplement_columns(merged_df, supplement_df, target_columns)
        supplement_sources.append(supplement_name)

    return merged_df, primary_source, supplement_sources, errors


def _prepare_feature_history(df, stock_code=None):
    effective_df = _strip_incomplete_today_bar(df)
    if effective_df.empty:
        return effective_df.copy()

    effective_df = effective_df.copy()

    open_series = effective_df["开盘"]
    close_series = effective_df["收盘"]
    high_series = effective_df["最高"]
    low_series = effective_df["最低"]
    volume_series = effective_df["成交量"]
    amount_series = _estimate_amount_series(close_series, volume_series, effective_df["成交额"], stock_code=stock_code)
    turnover_series = effective_df["换手率"]
    daily_return_series = _safe_pct_change_series(close_series, 1)

    effective_df["振幅"] = (high_series - low_series) / close_series.shift(1) * 100
    effective_df["成交额_估算后"] = amount_series
    effective_df["换手率_原值"] = turnover_series

    effective_df["ma5"] = close_series.rolling(5, min_periods=5).mean()
    effective_df["ma10"] = close_series.rolling(10, min_periods=10).mean()
    effective_df["ma20"] = close_series.rolling(20, min_periods=20).mean()
    effective_df["ma60"] = close_series.rolling(60, min_periods=60).mean()
    effective_df["ma120"] = close_series.rolling(120, min_periods=120).mean()
    effective_df["high_20d"] = high_series.rolling(20, min_periods=20).max()
    effective_df["low_20d"] = low_series.rolling(20, min_periods=20).min()
    effective_df["high_60d"] = high_series.rolling(60, min_periods=60).max()
    effective_df["low_60d"] = low_series.rolling(60, min_periods=60).min()
    effective_df["high_120d"] = high_series.rolling(120, min_periods=120).max()
    effective_df["low_120d"] = low_series.rolling(120, min_periods=120).min()
    effective_df["volatility_10d"] = daily_return_series.rolling(10, min_periods=10).std()
    effective_df["volatility_20d"] = daily_return_series.rolling(20, min_periods=20).std()
    effective_df["ma20_slope_5d"] = effective_df["ma20"].pct_change(periods=5, fill_method=None) * 100
    effective_df["ma60_slope_10d"] = effective_df["ma60"].pct_change(periods=10, fill_method=None) * 100

    effective_df["today_change"] = _safe_pct_change_series(close_series, 1)
    effective_df["change_3d"] = _safe_pct_change_series(close_series, 3)
    effective_df["change_5d"] = _safe_pct_change_series(close_series, 5)
    effective_df["change_10d"] = _safe_pct_change_series(close_series, 10)
    effective_df["change_20d"] = _safe_pct_change_series(close_series, 20)
    effective_df["change_30d"] = _safe_pct_change_series(close_series, 30)
    effective_df["vol_avg_3d"] = _safe_previous_average_series(volume_series, 3)
    effective_df["vol_avg_5d"] = _safe_previous_average_series(volume_series, 5)
    effective_df["vol_avg_10d"] = _safe_previous_average_series(volume_series, 10)
    effective_df["vol_avg_20d"] = _safe_previous_average_series(volume_series, 20)
    effective_df["amount_avg_5d"] = _safe_previous_average_series(amount_series, 5)
    effective_df["amount_avg_10d"] = _safe_previous_average_series(amount_series, 10)
    effective_df["turnover_avg_5d"] = _safe_previous_average_series(turnover_series, 5)
    effective_df["turnover_avg_10d"] = _safe_previous_average_series(turnover_series, 10)
    effective_df["amp_3d"] = _safe_previous_average_series(effective_df["振幅"], 3)
    effective_df["amp_5d"] = _safe_previous_average_series(effective_df["振幅"], 5)
    effective_df["amp_10d"] = _safe_previous_average_series(effective_df["振幅"], 10)
    effective_df["amp_20d"] = _safe_previous_average_series(effective_df["振幅"], 20)
    effective_df["amp_30d"] = _safe_previous_average_series(effective_df["振幅"], 30)
    effective_df["vr_today"] = _safe_volume_ratio_series(volume_series, 1)
    effective_df["vr_3d"] = _safe_volume_ratio_series(volume_series, 3)
    effective_df["vr_5d"] = _safe_volume_ratio_series(volume_series, 5)
    effective_df["vr_10d"] = _safe_volume_ratio_series(volume_series, 10)
    effective_df["vr_20d"] = _safe_volume_ratio_series(volume_series, 20)
    effective_df["vr_30d"] = _safe_volume_ratio_series(volume_series, 30)
    effective_df["last_data_date"] = effective_df["日期"].apply(_to_trade_date_string)

    return effective_df


def _build_payload_from_feature_row(feature_row):
    last_data_date = _to_trade_date_string(feature_row["日期"])

    payload = {
        "latest_price": _round_or_none(feature_row["收盘"]),
        "today_open": _round_or_none(feature_row["开盘"]),
        "today_high": _round_or_none(feature_row["最高"]),
        "today_low": _round_or_none(feature_row["最低"]),
        "last_data_date": last_data_date,
        "today_change": _round_or_none(feature_row["today_change"]),
        "change_3d": _round_or_none(feature_row["change_3d"]),
        "change_5d": _round_or_none(feature_row["change_5d"]),
        "change_10d": _round_or_none(feature_row["change_10d"]),
        "change_20d": _round_or_none(feature_row["change_20d"]),
        "change_30d": _round_or_none(feature_row["change_30d"]),
        "today_vol": _round_or_none(feature_row["成交量"]),
        "today_amount": _round_or_none(feature_row["成交额_估算后"]),
        "turnover_rate": _round_or_none(feature_row["换手率_原值"]),
        "vol_avg_3d": _round_or_none(feature_row["vol_avg_3d"]),
        "vol_avg_5d": _round_or_none(feature_row["vol_avg_5d"]),
        "vol_avg_10d": _round_or_none(feature_row["vol_avg_10d"]),
        "vol_avg_20d": _round_or_none(feature_row["vol_avg_20d"]),
        "amount_avg_5d": _round_or_none(feature_row["amount_avg_5d"]),
        "amount_avg_10d": _round_or_none(feature_row["amount_avg_10d"]),
        "turnover_avg_5d": _round_or_none(feature_row["turnover_avg_5d"]),
        "turnover_avg_10d": _round_or_none(feature_row["turnover_avg_10d"]),
        "today_amp": _round_or_none(feature_row["振幅"]),
        "amp_3d": _round_or_none(feature_row["amp_3d"]),
        "amp_5d": _round_or_none(feature_row["amp_5d"]),
        "amp_10d": _round_or_none(feature_row["amp_10d"]),
        "amp_20d": _round_or_none(feature_row["amp_20d"]),
        "amp_30d": _round_or_none(feature_row["amp_30d"]),
        "vr_today": _round_or_none(feature_row["vr_today"]),
        "vr_3d": _round_or_none(feature_row["vr_3d"]),
        "vr_5d": _round_or_none(feature_row["vr_5d"]),
        "vr_10d": _round_or_none(feature_row["vr_10d"]),
        "vr_20d": _round_or_none(feature_row["vr_20d"]),
        "vr_30d": _round_or_none(feature_row["vr_30d"]),
        "ma5": _round_or_none(feature_row["ma5"]),
        "ma10": _round_or_none(feature_row["ma10"]),
        "ma20": _round_or_none(feature_row["ma20"]),
        "ma60": _round_or_none(feature_row["ma60"]),
        "ma120": _round_or_none(feature_row["ma120"]),
        "high_20d": _round_or_none(feature_row["high_20d"]),
        "low_20d": _round_or_none(feature_row["low_20d"]),
        "high_60d": _round_or_none(feature_row["high_60d"]),
        "low_60d": _round_or_none(feature_row["low_60d"]),
        "high_120d": _round_or_none(feature_row["high_120d"]),
        "low_120d": _round_or_none(feature_row["low_120d"]),
        "volatility_10d": _round_or_none(feature_row["volatility_10d"]),
        "volatility_20d": _round_or_none(feature_row["volatility_20d"]),
        "ma20_slope_5d": _round_or_none(feature_row["ma20_slope_5d"]),
        "ma60_slope_10d": _round_or_none(feature_row["ma60_slope_10d"]),
    }

    return _strict_normalize_update_payload(payload)


def _build_update_payload(df, stock_code=None):
    feature_df = _prepare_feature_history(df, stock_code=stock_code)
    if feature_df.empty:
        raise ValueError("effective_history_empty")

    latest_row = feature_df.iloc[-1]
    return _build_payload_from_feature_row(latest_row)


def fetch_stock_data(per_code_info):
    stock_code = per_code_info["stock_code"]

    try:
        start_date = (datetime.today() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
        end_date = datetime.today().strftime("%Y%m%d")
        df, primary_source, supplement_sources, source_errors = _fetch_history_with_fallback(stock_code, start_date, end_date)

        if df.empty:
            return {
                "stock_code": stock_code,
                "success": False,
                "last_data_date": None,
                "reason": "all_sources_failed",
                "source": None,
                "primary_source": None,
                "supplement_sources": [],
                "source_errors": source_errors,
                "is_stale": False,
            }

        payload = _strict_normalize_update_payload(_build_update_payload(df, stock_code=stock_code))
        source_label = "+".join([primary_source, *supplement_sources]) if primary_source else None
        if not payload["last_data_date"]:
            return {
                "stock_code": stock_code,
                "success": False,
                "last_data_date": None,
                "reason": "missing_trade_date",
                "source": source_label,
                "primary_source": primary_source,
                "supplement_sources": supplement_sources,
                "source_errors": source_errors,
                "is_stale": False,
            }

        update_result = func.executeUpdate("a_stock_analysis", payload, {"stock_code": stock_code})
        if not update_result.get("updateResult", False):
            return {
                "stock_code": stock_code,
                "success": False,
                "last_data_date": payload["last_data_date"],
                "reason": "db_update_failed",
                "source": source_label,
                "primary_source": primary_source,
                "supplement_sources": supplement_sources,
                "source_errors": source_errors,
                "is_stale": False,
            }

        return {
            "stock_code": stock_code,
            "success": True,
            "last_data_date": payload["last_data_date"],
            "reason": None,
            "source": source_label,
            "primary_source": primary_source,
            "supplement_sources": supplement_sources,
            "source_errors": source_errors,
            "is_stale": False,
        }

    except Exception as error:
        message = f"{stock_code} 抓取失败: {error}"
        print(message)
        func.logInfo(message)
        return {
            "stock_code": stock_code,
            "success": False,
            "last_data_date": None,
            "reason": str(error),
            "source": None,
            "primary_source": None,
            "supplement_sources": [],
            "source_errors": [],
            "is_stale": False,
        }


def _resolve_target_trade_date(results):
    successful_dates = [result["last_data_date"] for result in results if result.get("success") and result.get("last_data_date")]
    if not successful_dates:
        return None

    date_counter = Counter(successful_dates)
    max_count = max(date_counter.values())
    candidate_dates = sorted(date for date, count in date_counter.items() if count == max_count)
    return candidate_dates[-1]


def _refresh_latest_snapshot_flags(target_trade_date):
    func.executeUpdate("a_stock_analysis", {"is_last_info": 0}, {"is_last_info": 1})
    func.executeUpdate("a_stock_analysis", {"is_last_info": 1}, {"last_data_date": target_trade_date})


def _get_latest_snapshot(target_trade_date):
    return pd.DataFrame(
        func.executeSelect("a_stock_analysis", {"is_last_info": 1, "last_data_date": target_trade_date})["resultData"]
    )


def _build_fetch_failure_result(per_code_info, reason):
    return {
        "stock_code": (per_code_info or {}).get("stock_code"),
        "success": False,
        "last_data_date": None,
        "reason": reason,
        "source": None,
        "primary_source": None,
        "supplement_sources": [],
        "source_errors": [reason],
        "is_stale": False,
    }


def _log_fetch_progress(completed_count, total_count, progress_start_ts, last_progress_log_ts, force=False):
    now_ts = time.time()
    should_log = (
        force
        or completed_count == total_count
        or now_ts - last_progress_log_ts >= PROGRESS_LOG_INTERVAL_SECONDS
    )
    if not should_log:
        return last_progress_log_ts

    elapsed_seconds = max(now_ts - progress_start_ts, 1e-6)
    speed = completed_count / elapsed_seconds
    remaining_count = total_count - completed_count
    eta_seconds = int(remaining_count / speed) if speed > 0 else None
    progress_ratio = round(completed_count / total_count * 100, 2) if total_count else 0

    progress_message = (
        f"抓取进度: {completed_count}/{total_count} ({progress_ratio}%), "
        f"剩余: {remaining_count}, 速度: {speed:.2f} 条/秒"
    )
    if eta_seconds is not None:
        progress_message += f", 预计剩余: {eta_seconds} 秒"

    print(progress_message)
    func.logInfo(progress_message)
    return now_ts


def _collect_fetch_results(code_info, pool_size):
    total_count = len(code_info)
    results = []
    progress_start_ts = time.time()
    last_progress_log_ts = progress_start_ts
    worker_count = max(1, min(int(pool_size), total_count))

    def record_result(result):
        nonlocal last_progress_log_ts
        results.append(result)
        last_progress_log_ts = _log_fetch_progress(
            len(results),
            total_count,
            progress_start_ts,
            last_progress_log_ts,
        )

    def run_direct_retry(per_code_info):
        try:
            return fetch_stock_data(per_code_info)
        except Exception as error:
            return _build_fetch_failure_result(
                per_code_info,
                f"direct_retry_error:{error.__class__.__name__}:{error}",
            )

    next_index = 0
    pending = []
    pool = None

    while next_index < total_count or pending:
        if pool is None:
            pool = Pool(processes=worker_count, maxtasksperchild=100)
        pool_finished = False

        try:
            while len(pending) < worker_count and next_index < total_count:
                per_code_info = code_info[next_index]
                next_index += 1
                pending.append(
                    {
                        "item": per_code_info,
                        "result": pool.apply_async(fetch_stock_data, (per_code_info,)),
                        "started_at": time.time(),
                    }
                )

            while pending:
                now_ts = time.time()
                next_pending = []
                timeout_items = []
                retry_items = []

                for task in pending:
                    per_code_info = task["item"]
                    async_result = task["result"]
                    started_at = task["started_at"]
                    if async_result.ready():
                        try:
                            record_result(async_result.get())
                        except Exception as error:
                            record_result(
                                _build_fetch_failure_result(
                                    per_code_info,
                                    f"worker_error:{error.__class__.__name__}:{error}",
                                )
                            )
                        if next_index < total_count:
                            next_item = code_info[next_index]
                            next_index += 1
                            next_pending.append(
                                {
                                    "item": next_item,
                                    "result": pool.apply_async(fetch_stock_data, (next_item,)),
                                    "started_at": time.time(),
                                }
                            )
                    elif now_ts - started_at >= FETCH_TASK_TIMEOUT_SECONDS:
                        record_result(
                            _build_fetch_failure_result(
                                per_code_info,
                                f"worker_timeout_{FETCH_TASK_TIMEOUT_SECONDS}s",
                            )
                        )
                        timeout_items.append(per_code_info)
                    else:
                        next_pending.append(task)

                if timeout_items:
                    retry_items = [task["item"] for task in next_pending]
                    pool.terminate()
                    pool.join()
                    pool_finished = True
                    pool = None
                    pending = []
                    for per_code_info in retry_items:
                        record_result(run_direct_retry(per_code_info))
                    break

                pending = next_pending
                if pending and len(pending) < worker_count:
                    while len(pending) < worker_count and next_index < total_count:
                        per_code_info = code_info[next_index]
                        next_index += 1
                        pending.append(
                            {
                                "item": per_code_info,
                                "result": pool.apply_async(fetch_stock_data, (per_code_info,)),
                                "started_at": time.time(),
                            }
                        )
                if pending:
                    time.sleep(POOL_RESULT_POLL_SECONDS)

            if not pending and next_index >= total_count and not pool_finished:
                pool.close()
                pool.join()
                pool_finished = True
                pool = None
        finally:
            if not pool_finished:
                pool.terminate()
                pool.join()
                pool = None

    _log_fetch_progress(
        len(results),
        total_count,
        progress_start_ts,
        last_progress_log_ts,
        force=True,
    )
    return results


def get_gu_piao_info():
    func.logInfo("开始抓取个股数据")

    code_info = func.executeSelect("a_stock_analysis")["resultData"]
    if not code_info:
        func.logInfo("a_stock_analysis 没有股票基础数据，退出")
        return {"target_trade_date": None, "success_count": 0, "failure_count": 0}


    # for perCodeInfo in code_info:
    #     res = fetch_stock_data(perCodeInfo)
    #     print(res)
    #     exit()

    pool_size = max(1, MAX_PROCESSES)
    results = _collect_fetch_results(code_info, pool_size)

    success_results = [result for result in results if result.get("success")]
    failed_results = [result for result in results if not result.get("success")]
    target_trade_date = _resolve_target_trade_date(results)
    stale_results = [
        result
        for result in success_results
        if target_trade_date
        and result.get("last_data_date")
        and result.get("last_data_date") < target_trade_date
    ]

    summary = {
        "target_trade_date": target_trade_date,
        "universe_count": len(code_info),
        "success_count": len(success_results),
        "failure_count": len(failed_results),
        "failure_reasons": dict(Counter(result.get("reason") or "unknown" for result in failed_results)),
        "source_usage": dict(Counter(result.get("source") or "unknown" for result in success_results)),
        "primary_source_usage": dict(Counter(result.get("primary_source") or "unknown" for result in success_results)),
        "supplement_usage": dict(
            Counter(
                supplement_source
                for result in success_results
                for supplement_source in result.get("supplement_sources", [])
            )
        ),
        "stale_count": len(stale_results),
    }

    if not target_trade_date:
        func.logInfo("本次抓取没有拿到有效交易日，跳过最新快照刷新和历史入库")
        func.logInfo(summary)
        return summary

    _refresh_latest_snapshot_flags(target_trade_date)
    latest_snapshot_df = _get_latest_snapshot(target_trade_date)
    latest_snapshot_count = len(latest_snapshot_df)
    coverage_ratio = round(latest_snapshot_count / len(code_info) * 100, 2) if code_info else 0

    summary["latest_snapshot_count"] = latest_snapshot_count
    summary["coverage_ratio"] = coverage_ratio

    print(
        f"成功抓取 {len(success_results)} 支股票数据，本次最新交易日: {target_trade_date}，"
        f"最新快照覆盖率: {coverage_ratio}%"
    )
    func.logInfo(summary)

    history_summary = history_data_save(target_trade_date, len(code_info))
    summary["history_saved_count"] = history_summary.get("saved_count", 0)

    return summary


def history_data_save(target_trade_date=None, universe_count=None):
    if not target_trade_date:
        func.logInfo("没有目标交易日，跳过历史入库")
        return {"saved_count": 0, "target_trade_date": None}

    df = _get_latest_snapshot(target_trade_date)
    if df.empty:
        message = f"交易日 {target_trade_date} 没有最新快照可以插入历史表。"
        print(message)
        func.logInfo(message)
        return {"saved_count": 0, "target_trade_date": target_trade_date}

    for column in HISTORY_COLUMNS:
        if column not in df.columns:
            df[column] = None

    df_to_insert = df[HISTORY_COLUMNS].drop_duplicates(subset=["last_data_date", "stock_code"]).copy()

    success_count = 0
    for _, row in df_to_insert.iterrows():
        record = _strict_normalize_update_payload(row.to_dict())
        func.executeDelete(
            "a_stock_analysis_history",
            {"last_data_date": record["last_data_date"], "stock_code": record["stock_code"]},
        )
        insert_result = func.executeInsert("a_stock_analysis_history", record)
        if insert_result.get("insertResult", False):
            success_count += 1

    coverage_ratio = round(success_count / universe_count * 100, 2) if universe_count else None
    message = f"成功插入 {success_count} 条记录到历史表，数据日期: {target_trade_date}"
    if coverage_ratio is not None:
        message += f"，覆盖率: {coverage_ratio}%"

    print(message)
    func.logInfo(message)

    return {
        "saved_count": success_count,
        "target_trade_date": target_trade_date,
        "coverage_ratio": coverage_ratio,
    }


if __name__ == "__main__":
    print(get_gu_piao_info())
