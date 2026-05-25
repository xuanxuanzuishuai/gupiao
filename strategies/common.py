import argparse
import json
import math
import os
import pickle
import re
import signal
from pathlib import Path

import pandas as pd
import pymysql

import func
import analysis_gu_piao_risk_overlay as risk_overlay
from analysis_gu_piao_data_print_result import (
    DEFAULT_ENTRY_OFFSET_DAYS,
    DEFAULT_FEE_BPS,
    DEFAULT_LIMIT_PCT,
    DEFAULT_SLIPPAGE_BPS,
    _build_price_paths,
    _summarize_holding_returns,
)


MODEL_VERSION = "adaptive_profile_v7"
SHORT_TERM_MODEL_LABEL = "短线自适应选股模型"
SHORT_TERM_MODEL_DISPLAY = SHORT_TERM_MODEL_LABEL
ADAPTIVE_HEALTH_MODEL_LABEL = "自适应落库健康验证模型"
ADAPTIVE_HEALTH_MODEL_DISPLAY = ADAPTIVE_HEALTH_MODEL_LABEL
LONG_RUNWAY_MODEL_VERSION = "long_runway_profile_v2"
LONG_RUNWAY_MODEL_LABEL = "长跑潜力模型"
LONG_RUNWAY_MODEL_DISPLAY = LONG_RUNWAY_MODEL_LABEL
MODEL_DEFINITION = f"{SHORT_TERM_MODEL_DISPLAY}({MODEL_VERSION})：基于最新数据滚动学习多风格短线特征，按最新市场与个股状态给出推荐和持有计划；{ADAPTIVE_HEALTH_MODEL_DISPLAY}负责落库前实盘信号和walk-forward回测校验；{LONG_RUNWAY_MODEL_DISPLAY}({LONG_RUNWAY_MODEL_VERSION})另行学习长周期潜力、趋势质量与阶段判断。"
MODEL_NATURE = "趋势成长型（非基本面成长）"
RECENCY_HALF_LIFE_DAYS = 21
TOP_WINNER_RATIO = 0.2
LOSER_RATIO = 0.5
TOP_CANDIDATE_COUNT = 30
DAILY_ADAPTIVE_TOP_PICK_COUNT = 3
ADAPTIVE_NOTE_MAX_LENGTH = 1600
ADAPTIVE_MIN_RISK_ADJUSTED_SCORE = 80.0
ADAPTIVE_MAX_RISK_SCORE = 2.0
EXTERNAL_RISK_SCORE_PENALTY_MULTIPLIER = 0.25
ADAPTIVE_PRECISION_STYLES = {"steady_climb"}
ADAPTIVE_PRECISION_TREND_STATES = {"周月同步抬升", "月线抬升，周线整理", "月线抬升，周线回踩"}
SHORT_TERM_LOOKBACK_TRADE_DAYS = 90
MIN_DAILY_ROWS = 50
MIN_STYLE_PROFILE_ROWS = 80
MIN_SNAPSHOT_COVERAGE_RATIO = 85.0
HORIZON_DAYS = (5, 10)
ADAPTIVE_SIGNAL_HEALTH_LOOKBACK_TRADE_DAYS = 60
ADAPTIVE_SIGNAL_HEALTH = {
    "hold_days": 10,
    "min_avg_return": 2.0,
    "min_trade_win_rate": 55.0,
    "min_evaluated_trades": 8,
}
ADAPTIVE_BACKTEST_HEALTH_LOOKBACK_TRADE_DAYS = 120
ADAPTIVE_PERSIST_BACKTEST_TIMEOUT_SECONDS = 45 * 60
ADAPTIVE_BACKTEST_HEALTH = {
    "hold_days": 10,
    "min_evaluated_days": 30,
    "min_avg_top_return": 2.0,
    "min_avg_top_win_rate": 55.0,
    "min_excess_return": 0.5,
}
ADAPTIVE_HEALTH_POLICIES = {
    "confirmed": {
        "label": "实盘确认",
        "confidence_weight": 1.0,
        "max_pick_ratio": 1.0,
        "position_hint": None,
    },
    "bootstrap": {
        "label": "启动期验证",
        "confidence_weight": 0.55,
        "max_pick_ratio": 0.5,
        "position_hint": "启动期小仓验证",
    },
}
LONG_RUNWAY_HORIZONS = (60, 120, 252)
LONG_RUNWAY_HALF_LIFE_DAYS = 90
LONG_RUNWAY_WINNER_RATIO = 0.15
LONG_RUNWAY_LOSER_RATIO = 0.5
LONG_RUNWAY_MIN_DAILY_ROWS = 50
LONG_RUNWAY_REBALANCE_TRADE_DAYS = 20
LONG_RUNWAY_CACHE_SCHEMA_VERSION = 1
LONG_RUNWAY_CACHE_DIR = Path("cache") / "long_runway"
LONG_RUNWAY_CONTEXT_CACHE_PATH = LONG_RUNWAY_CACHE_DIR / "context.pkl"
LONG_RUNWAY_ROLLING_CONTEXT_TRADE_DAYS = 260
LONG_RUNWAY_FORWARD_REFRESH_BUFFER_DAYS = 5
LONG_RUNWAY_HISTORY_QUERY_CHUNK_SIZE = 100000
EXCLUDED_NAME_PREFIXES = ("ST", "*ST", "S*ST", "退")
DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "rootroot",
    "database": "gu_piao",
    "charset": "utf8mb4",
}
SHORT_TERM_FRAME_COLUMNS = [
    "stock_code",
    "stock_name",
    "industry",
    "last_data_date",
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
    "turnover_rate",
    "stock_rank",
    "ma5",
    "ma10",
    "ma20",
    "ma60",
    "high_20d",
    "low_20d",
]
LONG_RUNWAY_EXTRA_COLUMNS = [
    "today_open",
    "today_high",
    "today_low",
    "amount_avg_5d",
    "amount_avg_10d",
    "turnover_avg_5d",
    "turnover_avg_10d",
    "volatility_10d",
    "volatility_20d",
    "ma20_slope_5d",
    "ma60_slope_10d",
    "ma120",
    "high_60d",
    "low_60d",
    "high_120d",
    "low_120d",
]
LONG_RUNWAY_FRAME_COLUMNS = SHORT_TERM_FRAME_COLUMNS + [
    column for column in LONG_RUNWAY_EXTRA_COLUMNS if column not in SHORT_TERM_FRAME_COLUMNS
]
ADAPTIVE_STRATEGY_TYPE = "adaptive_model"
ADAPTIVE_STRATEGY_LABEL = f"{SHORT_TERM_MODEL_LABEL}策略"
LONG_RUNWAY_STRATEGY_TYPE = "long_runway"
LONG_RUNWAY_STRATEGY_LABEL = f"{LONG_RUNWAY_MODEL_LABEL}中长期跟踪"
LEGACY_ADAPTIVE_STRATEGY_TYPES = ("自适应模型策略",)
RECOMMENDATION_TIER_FORMAL = "正式推荐"
RECOMMENDATION_TIER_OBSERVE = "观察候选"
RECOMMENDATION_TIER_RESEARCH = "研究价值"
RECOMMENDATION_TIER_AVOID = "暂不参与"
STYLE_PRIORITY = ("breakout", "steady_climb", "rebound")
STYLE_LABELS = {
    "breakout": "强势突破",
    "steady_climb": "慢涨跟随",
    "rebound": "低位修复",
}

NUMERIC_COLUMNS = [
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
    "turnover_rate",
    "amount_avg_5d",
    "amount_avg_10d",
    "turnover_avg_5d",
    "turnover_avg_10d",
    "stock_rank",
    "ma5",
    "ma10",
    "ma20",
    "ma60",
    "high_20d",
    "low_20d",
    "today_open",
    "today_high",
    "today_low",
    "volatility_10d",
    "volatility_20d",
    "ma20_slope_5d",
    "ma60_slope_10d",
    "ma120",
    "high_60d",
    "low_60d",
    "high_120d",
    "low_120d",
]

FAMILY_FEATURES = {
    "trend": [
        "change_3d",
        "change_5d",
        "change_10d",
        "price_vs_ma20",
        "ma5_vs_ma20",
        "ma20_vs_ma60",
    ],
    "timeframe": [
        "change_20d",
        "change_30d",
        "price_vs_ma60",
        "ma5_vs_ma60",
    ],
    "momentum": [
        "today_change",
        "price_vs_ma5",
        "close_to_20d_high",
    ],
    "volume": [
        "today_vol",
        "volume_vs_avg_5d",
        "volume_vs_avg_10d",
        "vr_today",
        "vr_5d",
    ],
    "rebound": [
        "change_20d",
        "change_30d",
        "close_to_20d_low",
        "today_amp",
        "amp_vs_avg_5d",
    ],
    "industry": [
        "industry_change_5d",
        "industry_change_10d",
        "industry_alpha_5d",
        "industry_alpha_10d",
        "industry_breadth_5d",
        "industry_breadth_10d",
    ],
    "attention": [
        "stock_rank_score",
    ],
}

FEATURE_LABELS = {
    "change_3d": "3日涨幅",
    "change_5d": "5日涨幅",
    "change_10d": "10日涨幅",
    "price_vs_ma20": "价格相对20日均线",
    "price_vs_ma60": "价格相对60日均线",
    "ma5_vs_ma20": "5日均线相对20日均线",
    "ma5_vs_ma60": "5日均线相对60日均线",
    "ma20_vs_ma60": "20日均线相对60日均线",
    "today_change": "今日涨跌幅",
    "price_vs_ma5": "价格相对5日均线",
    "close_to_20d_high": "收盘接近20日高点",
    "today_vol": "今日成交量",
    "volume_vs_avg_5d": "成交量相对5日均量",
    "volume_vs_avg_10d": "成交量相对10日均量",
    "vr_today": "今日量比",
    "vr_5d": "5日量比",
    "change_20d": "20日涨幅",
    "change_30d": "30日涨幅",
    "close_to_20d_low": "收盘接近20日低点",
    "today_amp": "今日振幅",
    "amp_vs_avg_5d": "振幅相对5日均振幅",
    "industry_change_5d": "行业5日平均涨幅",
    "industry_change_10d": "行业10日平均涨幅",
    "industry_alpha_5d": "行业5日相对市场强度",
    "industry_alpha_10d": "行业10日相对市场强度",
    "industry_breadth_5d": "行业5日上涨广度",
    "industry_breadth_10d": "行业10日上涨广度",
    "stock_rank_score": "人气排名得分",
}

FAMILY_LABELS = {
    "trend": "趋势延续",
    "timeframe": "周月共振",
    "momentum": "加速突破",
    "volume": "量能放大",
    "rebound": "低位修复",
    "industry": "行业共振",
    "attention": "人气聚焦",
}

FEATURE_UNIT_PERCENT_POINT = "percent_point"
FEATURE_UNIT_EXCESS_RATIO = "excess_ratio"
FEATURE_UNIT_MULTIPLE_RATIO = "multiple_ratio"
FEATURE_UNIT_POSITION_RATIO = "position_ratio"
FEATURE_UNIT_RAW_VOLUME = "raw_volume"
FEATURE_UNIT_RAW_AMOUNT = "raw_amount"

PERCENT_POINT_FEATURES = {
    "today_change",
    "change_3d",
    "change_5d",
    "change_10d",
    "change_20d",
    "change_30d",
    "today_amp",
    "amp_3d",
    "amp_5d",
    "amp_10d",
    "amp_20d",
    "amp_30d",
    "turnover_rate",
    "volatility_10d",
    "volatility_20d",
    "ma20_slope_5d",
    "ma60_slope_10d",
    "industry_change_5d",
    "industry_change_10d",
    "industry_alpha_5d",
    "industry_alpha_10d",
    "market_change_5d",
    "market_change_10d",
    "industry_ret_20d",
    "industry_ret_60d",
    "industry_ret_120d",
    "industry_alpha_20d",
    "industry_alpha_60d",
    "industry_alpha_120d",
    "industry_breadth_5d",
    "industry_breadth_10d",
    "industry_breadth_20d",
    "industry_breadth_60d",
    "industry_breadth_120d",
    "market_ret_20d",
    "market_ret_60d",
    "market_ret_120d",
    "market_breadth_5d",
    "market_breadth_10d",
    "market_breadth_20d",
    "market_breadth_60d",
    "market_breadth_120d",
    "ret_20d",
    "ret_60d",
    "ret_120d",
    "ret_252d",
}
EXCESS_RATIO_FEATURES = {
    "price_vs_ma5",
    "price_vs_ma20",
    "price_vs_ma60",
    "price_vs_ma120",
    "price_vs_ma240",
    "ma5_vs_ma20",
    "ma5_vs_ma60",
    "ma20_vs_ma60",
    "ma60_vs_ma120",
    "ma120_vs_ma240",
}
MULTIPLE_RATIO_FEATURES = {
    "close_to_20d_high",
    "close_to_20d_low",
    "close_to_60d_high",
    "close_to_60d_low",
    "close_to_120d_high",
    "close_to_120d_low",
    "close_to_240d_high",
    "close_to_240d_low",
    "volume_vs_avg_3d",
    "volume_vs_avg_5d",
    "volume_vs_avg_10d",
    "volume_vs_avg_20d",
    "volume_vs_avg_60d",
    "volume_vs_avg_120d",
    "vr_today",
    "vr_3d",
    "vr_5d",
    "vr_10d",
    "vr_20d",
    "vr_30d",
    "amount_vs_avg_5d",
    "amount_vs_avg_10d",
    "turnover_vs_avg_5d",
    "turnover_vs_avg_10d",
    "amp_vs_avg_3d",
    "amp_vs_avg_5d",
    "amp_vs_avg_10d",
    "amp_vs_avg_20d",
    "volatility_ratio_10_20",
}
POSITION_RATIO_FEATURES = {
    "industry_strength_rank_5d",
    "industry_strength_rank_10d",
    "range_position_120d",
    "range_position_240d",
    "close_strength",
    "upper_shadow_ratio",
    "body_to_range_ratio",
    "stock_rank_score",
}
RAW_VOLUME_FEATURES = {
    "today_vol",
}
RAW_AMOUNT_FEATURES = {
    "today_amount",
}
FEATURE_UNIT_BY_COLUMN = {
    **{column: FEATURE_UNIT_PERCENT_POINT for column in PERCENT_POINT_FEATURES},
    **{column: FEATURE_UNIT_EXCESS_RATIO for column in EXCESS_RATIO_FEATURES},
    **{column: FEATURE_UNIT_MULTIPLE_RATIO for column in MULTIPLE_RATIO_FEATURES},
    **{column: FEATURE_UNIT_POSITION_RATIO for column in POSITION_RATIO_FEATURES},
    **{column: FEATURE_UNIT_RAW_VOLUME for column in RAW_VOLUME_FEATURES},
    **{column: FEATURE_UNIT_RAW_AMOUNT for column in RAW_AMOUNT_FEATURES},
}

STYLE_NOTE_HINTS = {
    "breakout": "周线和月线同步抬升，收盘逼近阶段高点，量能放大",
    "steady_climb": "周线月线逐步抬高，价格围绕20日均线缓慢上行",
    "rebound": "低位区域波动扩张、靠近20日低点、等待拐点修复",
}

LONG_RUNWAY_FAMILY_FEATURES = {
    "trend": [
        "ret_60d",
        "ret_120d",
        "ret_252d",
        "price_vs_ma60",
        "price_vs_ma120",
        "price_vs_ma240",
        "ma60_vs_ma120",
        "ma120_vs_ma240",
    ],
    "acceleration": [
        "ret_20d",
        "today_change",
        "price_vs_ma20",
        "price_vs_ma60",
        "close_to_60d_high",
    ],
    "base": [
        "close_to_120d_high",
        "close_to_240d_high",
        "close_to_120d_low",
        "close_to_240d_low",
        "range_position_120d",
        "range_position_240d",
    ],
    "volume": [
        "today_vol",
        "volume_vs_avg_20d",
        "volume_vs_avg_60d",
        "volume_vs_avg_120d",
        "vr_today",
        "vr_20d",
    ],
    "liquidity": [
        "today_amount",
        "amount_vs_avg_5d",
        "amount_vs_avg_10d",
        "turnover_rate",
        "turnover_vs_avg_5d",
        "turnover_vs_avg_10d",
    ],
    "quality": [
        "volatility_10d",
        "volatility_20d",
        "volatility_ratio_10_20",
        "ma20_slope_5d",
        "ma60_slope_10d",
        "close_strength",
        "upper_shadow_ratio",
        "body_to_range_ratio",
    ],
    "industry": [
        "industry_ret_20d",
        "industry_ret_60d",
        "industry_ret_120d",
        "industry_alpha_20d",
        "industry_alpha_60d",
        "industry_alpha_120d",
        "industry_breadth_20d",
        "industry_breadth_60d",
        "industry_breadth_120d",
    ],
    "attention": [
        "stock_rank_score",
    ],
}

LONG_RUNWAY_FEATURE_LABELS = {
    "ret_20d": "20日涨幅",
    "ret_60d": "60日涨幅",
    "ret_120d": "120日涨幅",
    "ret_252d": "252日涨幅",
    "price_vs_ma20": "价格相对20日均线",
    "price_vs_ma60": "价格相对60日均线",
    "price_vs_ma120": "价格相对120日均线",
    "price_vs_ma240": "价格相对240日均线",
    "ma60_vs_ma120": "60日均线相对120日均线",
    "ma120_vs_ma240": "120日均线相对240日均线",
    "today_change": "今日涨跌幅",
    "close_to_60d_high": "收盘接近60日高点",
    "close_to_120d_high": "收盘接近120日高点",
    "close_to_240d_high": "收盘接近240日高点",
    "close_to_120d_low": "收盘接近120日低点",
    "close_to_240d_low": "收盘接近240日低点",
    "range_position_120d": "120日区间位置",
    "range_position_240d": "240日区间位置",
    "today_vol": "今日成交量",
    "volume_vs_avg_20d": "成交量相对20日均量",
    "volume_vs_avg_60d": "成交量相对60日均量",
    "volume_vs_avg_120d": "成交量相对120日均量",
    "vr_today": "今日量比",
    "vr_20d": "20日量比",
    "today_amount": "今日成交额",
    "amount_vs_avg_5d": "成交额相对5日均额",
    "amount_vs_avg_10d": "成交额相对10日均额",
    "turnover_rate": "今日换手率",
    "turnover_vs_avg_5d": "换手率相对5日均值",
    "turnover_vs_avg_10d": "换手率相对10日均值",
    "volatility_10d": "10日波动率",
    "volatility_20d": "20日波动率",
    "volatility_ratio_10_20": "10日波动率相对20日波动率",
    "ma20_slope_5d": "20日均线5日斜率",
    "ma60_slope_10d": "60日均线10日斜率",
    "close_strength": "收盘强度",
    "upper_shadow_ratio": "上影占比",
    "body_to_range_ratio": "实体占振幅比例",
    "industry_ret_20d": "行业20日平均涨幅",
    "industry_ret_60d": "行业60日平均涨幅",
    "industry_ret_120d": "行业120日平均涨幅",
    "industry_alpha_20d": "行业20日相对市场强度",
    "industry_alpha_60d": "行业60日相对市场强度",
    "industry_alpha_120d": "行业120日相对市场强度",
    "industry_breadth_20d": "行业20日上涨广度",
    "industry_breadth_60d": "行业60日上涨广度",
    "industry_breadth_120d": "行业120日上涨广度",
    "stock_rank_score": "人气排名得分",
}

LONG_RUNWAY_FAMILY_LABELS = {
    "trend": "长周期趋势",
    "acceleration": "爆发动量",
    "base": "长平台基座",
    "volume": "量能确认",
    "liquidity": "流动性确认",
    "quality": "趋势质量",
    "industry": "行业共振",
    "attention": "资金关注",
}

LONG_RUNWAY_FEATURE_COLUMNS = []
for _family_features in LONG_RUNWAY_FAMILY_FEATURES.values():
    for _feature in _family_features:
        if _feature not in LONG_RUNWAY_FEATURE_COLUMNS:
            LONG_RUNWAY_FEATURE_COLUMNS.append(_feature)

LONG_RUNWAY_RESULT_DETAIL_COLUMNS = [
    "ret_20d",
    "ret_60d",
    "ret_120d",
    "ret_252d",
    "ma120",
    "ma240",
    "high_60d",
    "high_120d",
    "high_240d",
    "low_60d",
    "low_120d",
    "low_240d",
    "price_vs_ma60",
    "price_vs_ma120",
    "price_vs_ma240",
    "ma60_vs_ma120",
    "ma120_vs_ma240",
    "close_to_60d_high",
    "close_to_120d_high",
    "close_to_240d_high",
    "close_to_120d_low",
    "close_to_240d_low",
    "range_position_120d",
    "range_position_240d",
    "volume_vs_avg_20d",
    "volume_vs_avg_60d",
    "volume_vs_avg_120d",
    "amount_vs_avg_5d",
    "amount_vs_avg_10d",
    "turnover_vs_avg_5d",
    "turnover_vs_avg_10d",
    "volatility_ratio_10_20",
    "ma20_slope_5d",
    "ma60_slope_10d",
    "close_strength",
    "upper_shadow_ratio",
    "body_to_range_ratio",
    "industry_ret_20d",
    "industry_ret_60d",
    "industry_ret_120d",
    "industry_alpha_20d",
    "industry_alpha_60d",
    "industry_alpha_120d",
    "industry_breadth_20d",
    "industry_breadth_60d",
    "industry_breadth_120d",
    "market_ret_60d",
    "market_breadth_60d",
    "market_ret_120d",
    "market_breadth_120d",
    "historical_max_return_60d",
    "historical_max_return_120d",
    "historical_max_return_252d",
    "historical_max_signal_date_60d",
    "historical_max_signal_date_120d",
    "historical_max_signal_date_252d",
    "historical_max_exit_date_60d",
    "historical_max_exit_date_120d",
    "historical_max_exit_date_252d",
]

MODEL_FEATURE_COLUMNS = []
for _family_features in FAMILY_FEATURES.values():
    for _feature in _family_features:
        if _feature not in MODEL_FEATURE_COLUMNS:
            MODEL_FEATURE_COLUMNS.append(_feature)


def _assert_feature_unit_registry_complete():
    expected_features = set(MODEL_FEATURE_COLUMNS) | set(LONG_RUNWAY_FEATURE_COLUMNS)
    missing = sorted(feature for feature in expected_features if feature not in FEATURE_UNIT_BY_COLUMN)
    if missing:
        raise KeyError(f"模型特征缺少单位登记: {missing}")


def _run_with_wall_timeout(seconds, callback, *args, **kwargs):
    if not seconds:
        return callback(*args, **kwargs)

    def _timeout_handler(signum, frame):
        raise TimeoutError(f"timeout_{int(seconds)}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(int(seconds))
    try:
        return callback(*args, **kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def _emit_runtime_status(message):
    func.logInfo(message)
    print(message, flush=True)


def _normalize_stock_code(value):
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    matched = re.search(r"(\d{6})", text)
    return matched.group(1) if matched else None


def _normalize_stock_code_series(series):
    normalized = series.astype(str).str.extract(r"(\d{6})", expand=False)
    return normalized.where(normalized.notna(), None)


def _ordered_unique(values):
    result = []
    seen = set()

    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)

    return result


def _parse_industry(value):
    if not isinstance(value, str):
        return []
    return _ordered_unique(value.split(","))


def _ensure_columns(df, columns, default_value):
    for column in columns:
        if column not in df.columns:
            df[column] = default_value


def _normalize_scalar(value):
    if isinstance(value, list):
        return value

    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value

    return value


def _db_connection():
    return pymysql.connect(
        host="127.0.0.1",
        user="root",
        password="rootroot",
        database="gu_piao",
        charset="utf8mb4",
    )


def _to_date_text(value):
    if value is None:
        return None

    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None

    return ts.strftime("%Y-%m-%d")


def _round_or_none(value, digits=4):
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def _to_float(value):
    if value is None or pd.isna(value):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _assert_feature_unit(column, expected_unit):
    actual_unit = FEATURE_UNIT_BY_COLUMN.get(column)
    if actual_unit is None:
        raise KeyError(f"未登记特征口径: {column}")
    if actual_unit != expected_unit:
        raise ValueError(f"特征口径不匹配: {column} 是 {actual_unit}, 不能当作 {expected_unit} 使用")


def _feature_series(frame, column, expected_unit):
    _assert_feature_unit(column, expected_unit)
    if column not in frame.columns:
        return pd.Series(pd.NA, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _feature_value(record, column, expected_unit):
    _assert_feature_unit(column, expected_unit)
    return _to_float(record.get(column))


def _safe_ratio(numerator, denominator):
    denominator = denominator.replace(0, pd.NA)
    result = numerator / denominator
    return result.replace([math.inf, -math.inf], pd.NA)


def _weighted_mean(series, weights):
    if series.empty or weights.empty:
        return None

    valid_mask = series.notna() & weights.notna()
    if not valid_mask.any():
        return None

    values = pd.to_numeric(series[valid_mask], errors="coerce")
    valid_weights = pd.to_numeric(weights[valid_mask], errors="coerce")
    valid_mask = values.notna() & valid_weights.notna() & (valid_weights > 0)
    if not valid_mask.any():
        return None

    values = values[valid_mask]
    valid_weights = valid_weights[valid_mask]
    return float((values * valid_weights).sum() / valid_weights.sum())


def _query_frame(sql, params=None, chunksize=None, progress_label=None):
    connection = None
    try:
        connect_kwargs = dict(DB_CONFIG)
        if chunksize:
            connect_kwargs["cursorclass"] = pymysql.cursors.SSCursor
        connection = pymysql.connect(**connect_kwargs)
        with connection.cursor() as cursor:
            cursor.execute(sql, params or [])
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            if not chunksize:
                rows = cursor.fetchall()
                return pd.DataFrame.from_records(rows, columns=columns)

            frames = []
            fetched_rows = 0
            rows = cursor.fetchmany(int(chunksize))
            while rows:
                chunk = pd.DataFrame.from_records(rows, columns=columns)
                frames.append(chunk)
                fetched_rows += len(chunk)
                if progress_label:
                    _emit_runtime_status(f"{progress_label}: 已读取 rows={fetched_rows}")
                rows = cursor.fetchmany(int(chunksize))

        if not frames:
            return pd.DataFrame(columns=columns)
        return pd.concat(frames, ignore_index=True)
    except Exception as error:
        func.logInfo(f"{SHORT_TERM_MODEL_DISPLAY}/{LONG_RUNWAY_MODEL_DISPLAY}数据查询失败: {error}")
        return pd.DataFrame()
    finally:
        if connection:
            connection.close()


def _base_select_sql(table_name, columns=None):
    selected_columns = columns or SHORT_TERM_FRAME_COLUMNS
    quoted_columns = ", ".join(f"`{column}`" for column in selected_columns)
    return f"SELECT {quoted_columns} FROM {table_name}"


def _load_latest_snapshot(columns=None):
    latest_snapshot = _query_frame(f"{_base_select_sql('a_stock_analysis', columns=columns)} WHERE is_last_info=1")

    if latest_snapshot.empty:
        func.logInfo("没有 is_last_info=1 的最新快照，回退到历史表最新交易日")
        latest_snapshot = _query_frame(
            f"""
            SELECT
                {", ".join(f"`{column}`" for column in (columns or SHORT_TERM_FRAME_COLUMNS))}
            FROM a_stock_analysis_history
            WHERE last_data_date = (
                SELECT MAX(last_data_date) FROM a_stock_analysis_history
            )
            """
        )
    if latest_snapshot.empty:
        func.logInfo("历史表最新交易日为空，最后回退到 a_stock_analysis 全表去重分析")
        latest_snapshot = _query_frame(_base_select_sql("a_stock_analysis", columns=columns))

    if latest_snapshot.empty:
        return latest_snapshot

    return _prepare_common_frame(latest_snapshot, dedupe_keys=("stock_code",))


def _assess_snapshot_coverage(latest_snapshot):
    if latest_snapshot.empty:
        return {
            "trade_date": None,
            "trade_date_count": 0,
            "universe_count": 0,
            "coverage_ratio": 0.0,
            "meets_min_coverage": False,
        }

    latest_trade_date = pd.to_datetime(latest_snapshot["last_data_date"], errors="coerce").max()
    latest_trade_date_text = _to_date_text(latest_trade_date)
    trade_date_count = int(len(latest_snapshot))

    universe_count = int(trade_date_count)
    connection = None
    try:
        connection = _db_connection()
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM a_stock_analysis")
            row = cursor.fetchone()
            if row and row[0]:
                universe_count = int(row[0])
    except Exception as error:
        func.logInfo(f"{SHORT_TERM_MODEL_DISPLAY}读取全市场股票池数量失败，回退到快照数量: {error}")
    finally:
        if connection:
            connection.close()

    coverage_ratio = round(trade_date_count / universe_count * 100, 2) if universe_count else 0.0
    return {
        "trade_date": latest_trade_date_text,
        "trade_date_count": trade_date_count,
        "universe_count": universe_count,
        "coverage_ratio": coverage_ratio,
        "meets_min_coverage": coverage_ratio >= MIN_SNAPSHOT_COVERAGE_RATIO,
    }


def _load_history(start_date=None, end_date=None, tail_trade_days=None, columns=None, chunked=False, progress_label=None):
    selected_columns = columns or SHORT_TERM_FRAME_COLUMNS
    params = []
    if start_date is None and tail_trade_days:
        cutoff_date = None
        connection = None
        try:
            connection = _db_connection()
            with connection.cursor() as cursor:
                date_filter = "WHERE last_data_date <= %s" if end_date is not None else ""
                cutoff_params = [end_date] if end_date is not None else []
                cutoff_params.append(max(int(tail_trade_days) - 1, 0))
                cursor.execute(
                    f"""
                    SELECT last_data_date
                    FROM (
                        SELECT DISTINCT last_data_date
                        FROM a_stock_analysis_history
                        {date_filter}
                        ORDER BY last_data_date DESC
                        LIMIT 1 OFFSET %s
                    ) cutoff_trade_day
                    """,
                    cutoff_params,
                )
                row = cursor.fetchone()
                if row and row[0]:
                    cutoff_date = row[0]
        except Exception as error:
            func.logInfo(f"{SHORT_TERM_MODEL_DISPLAY}读取最近交易日cutoff失败，回退到旧逻辑: {error}")
        finally:
            if connection:
                connection.close()

        if cutoff_date is not None:
            sql = f"{_base_select_sql('a_stock_analysis_history', columns=selected_columns)} WHERE last_data_date >= %s"
            params.append(cutoff_date)
            if end_date is not None:
                sql += " AND last_data_date <= %s"
                params.append(end_date)
        else:
            date_filter = "WHERE last_data_date <= %s" if end_date is not None else ""
            sql = f"""
                {_base_select_sql('a_stock_analysis_history', columns=selected_columns)}
                WHERE last_data_date IN (
                    SELECT last_data_date
                    FROM (
                        SELECT DISTINCT last_data_date
                        FROM a_stock_analysis_history
                        {date_filter}
                        ORDER BY last_data_date DESC
                        LIMIT %s
                    ) recent_trade_days
                )
            """
            if end_date is not None:
                params.append(end_date)
            params.append(int(tail_trade_days))
    else:
        sql = f"{_base_select_sql('a_stock_analysis_history', columns=selected_columns)} WHERE 1=1"
        if start_date is not None:
            sql += " AND last_data_date >= %s"
            params.append(start_date)
        if end_date is not None:
            sql += " AND last_data_date <= %s"
            params.append(end_date)

    use_chunked_query = bool(chunked)
    history = _query_frame(
        sql,
        params=params,
        chunksize=LONG_RUNWAY_HISTORY_QUERY_CHUNK_SIZE if use_chunked_query else None,
        progress_label=progress_label if use_chunked_query else None,
    )

    if history.empty:
        return history

    history = _prepare_common_frame(history, dedupe_keys=("last_data_date", "stock_code"))
    if history.empty:
        return history

    return history.reset_index(drop=True)


def _prepare_short_term_history(history):
    if history is None or history.empty:
        return history

    _assert_feature_unit_registry_complete()
    prepared = _build_market_context(history.copy())
    prepared = _build_forward_returns(prepared)
    prepared = _add_percentile_columns(prepared, MODEL_FEATURE_COLUMNS)
    return prepared


def _tail_trade_days_frame(frame, tail_trade_days):
    if frame is None or frame.empty or not tail_trade_days or "last_data_date" not in frame.columns:
        return frame

    trade_dates = sorted(frame["last_data_date"].dropna().unique())
    if len(trade_dates) <= int(tail_trade_days):
        return frame

    cutoff_date = trade_dates[-int(tail_trade_days)]
    return frame[frame["last_data_date"] >= cutoff_date].copy().reset_index(drop=True)


def _load_history_trade_dates(end_date=None):
    sql = "SELECT DISTINCT last_data_date FROM a_stock_analysis_history WHERE 1=1"
    params = []
    if end_date is not None:
        sql += " AND last_data_date <= %s"
        params.append(end_date)
    sql += " ORDER BY last_data_date"
    rows = _query_frame(sql, params=params)
    if rows.empty or "last_data_date" not in rows.columns:
        return []
    return list(pd.to_datetime(rows["last_data_date"], errors="coerce").dropna().sort_values().unique())
























def _load_adaptive_signal_health(reference_date):
    if not reference_date:
        return {}

    settings = ADAPTIVE_SIGNAL_HEALTH
    hold_days = int(settings["hold_days"])
    evaluated_trades = 0
    avg_return = None
    trade_win_rate = None

    connection = None
    try:
        connection = pymysql.connect(**DB_CONFIG)
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT last_data_date
                FROM (
                    SELECT DISTINCT last_data_date
                    FROM a_stock_analysis_history
                    WHERE last_data_date < %s
                    ORDER BY last_data_date DESC
                    LIMIT %s
                ) recent_trade_days
                ORDER BY last_data_date
                """,
                [reference_date, int(max(1, ADAPTIVE_SIGNAL_HEALTH_LOOKBACK_TRADE_DAYS))],
            )
            trade_days = [row[0] for row in cursor.fetchall()]

        if not trade_days:
            return {}

        start_date = str(trade_days[0])
        signals = _query_frame(
            """
            SELECT trade_date, stock_code
            FROM a_stock_strategy_result
            WHERE strategy_type = %s
              AND trade_date >= %s
              AND trade_date < %s
            ORDER BY trade_date, stock_code
            """,
            params=[ADAPTIVE_STRATEGY_TYPE, start_date, reference_date],
        )
    finally:
        if connection:
            connection.close()

    if signals.empty:
        return {
            "enabled": False,
            "hold_days": hold_days,
            "signal_count": 0,
            "signal_days": 0,
            "evaluated_trades": 0,
            "avg_return": None,
            "trade_win_rate": None,
            "failure_reasons": ["insufficient_signal_history"],
        }

    signals["signal_date"] = pd.to_datetime(signals["trade_date"], errors="coerce")
    signals = signals.dropna(subset=["signal_date", "stock_code"]).copy()
    if signals.empty:
        return {
            "enabled": False,
            "hold_days": hold_days,
            "signal_count": 0,
            "signal_days": 0,
            "evaluated_trades": 0,
            "avg_return": None,
            "trade_win_rate": None,
            "failure_reasons": ["insufficient_signal_history"],
        }

    stock_codes = signals["stock_code"].astype(str).unique().tolist()
    placeholders = ",".join(["%s"] * len(stock_codes))
    history = _query_frame(
        f"""
        SELECT last_data_date, stock_code, latest_price, today_change
        FROM a_stock_analysis_history
        WHERE stock_code IN ({placeholders})
          AND last_data_date >= %s
          AND last_data_date < %s
        ORDER BY stock_code, last_data_date
        """,
        params=stock_codes + [start_date, reference_date],
    )

    failure_reasons = []
    if history.empty:
        failure_reasons.append("signal_history_price_empty")
    else:
        history["last_data_date"] = pd.to_datetime(history["last_data_date"], errors="coerce")
        history = history.dropna(subset=["last_data_date", "stock_code"]).copy()
        history = history.sort_values(["stock_code", "last_data_date"])
        history = history.drop_duplicates(subset=["last_data_date", "stock_code"], keep="last")

        if history.empty:
            failure_reasons.append("signal_history_price_empty")
        else:
            price_paths = _build_price_paths(
                history,
                [hold_days],
                DEFAULT_ENTRY_OFFSET_DAYS,
                round_trip_cost_pct=(DEFAULT_FEE_BPS + DEFAULT_SLIPPAGE_BPS) * 2 / 100,
            )
            merge_columns = [
                "last_data_date",
                "stock_code",
                "entry_change",
                f"exit_change_{hold_days}d",
                f"gross_return_{hold_days}d",
                f"return_{hold_days}d",
            ]
            detail = signals.merge(
                price_paths[merge_columns].rename(columns={"last_data_date": "signal_date"}),
                on=["signal_date", "stock_code"],
                how="left",
            )

            return_col = f"return_{hold_days}d"
            gross_return_col = f"gross_return_{hold_days}d"
            exit_change_col = f"exit_change_{hold_days}d"
            entry_abs = pd.to_numeric(detail["entry_change"], errors="coerce").abs()
            exit_abs = pd.to_numeric(detail[exit_change_col], errors="coerce").abs()
            tradable = (entry_abs.isna() | (entry_abs < DEFAULT_LIMIT_PCT)) & (
                exit_abs.isna() | (exit_abs < DEFAULT_LIMIT_PCT)
            )
            filtered_mask = detail[return_col].notna() & (~tradable)
            detail.loc[filtered_mask, [return_col, gross_return_col]] = pd.NA

            valid_details = detail[detail[return_col].notna()].copy()
            holding_summary = _summarize_holding_returns(valid_details, return_col, gross_return_col)
            evaluated_trades = int(holding_summary["evaluated_trades"])
            avg_return = holding_summary["avg_return"]
            trade_win_rate = holding_summary["trade_win_rate"]

    if evaluated_trades < int(settings["min_evaluated_trades"]):
        failure_reasons.append("insufficient_trades")
    if avg_return is None or float(avg_return) < float(settings["min_avg_return"]):
        failure_reasons.append("avg_return_below_threshold")
    if trade_win_rate is None or float(trade_win_rate) < float(settings["min_trade_win_rate"]):
        failure_reasons.append("win_rate_below_threshold")

    return {
        "enabled": not failure_reasons,
        "mode": "realized_signal",
        "hold_days": hold_days,
        "signal_count": int(len(signals)),
        "signal_days": int(signals["signal_date"].nunique()) if not signals.empty else 0,
        "evaluated_trades": evaluated_trades,
        "avg_return": avg_return,
        "trade_win_rate": trade_win_rate,
        "failure_reasons": list(dict.fromkeys(failure_reasons)),
    }


def _evaluate_adaptive_backtest_health(backtest_result):
    settings = ADAPTIVE_BACKTEST_HEALTH
    hold_days = int(settings["hold_days"])
    if not backtest_result or backtest_result.get("success") is False:
        reason = (backtest_result or {}).get("reason") or "backtest_failed"
        return {
            "enabled": False,
            "mode": "walk_forward_backtest",
            "hold_days": hold_days,
            "window_trade_days": int(ADAPTIVE_BACKTEST_HEALTH_LOOKBACK_TRADE_DAYS),
            "evaluated_days": 0,
            "avg_top_return": None,
            "avg_top_win_rate": None,
            "avg_universe_return": None,
            "excess_return": None,
            "failure_reasons": [reason],
        }

    metrics = ((backtest_result or {}).get("backtest") or {}).get(f"{hold_days}d") or {}

    evaluated_days = int(metrics.get("evaluated_days") or 0)
    avg_top_return = metrics.get("avg_top_return")
    avg_top_win_rate = metrics.get("avg_top_win_rate")
    avg_universe_return = metrics.get("avg_universe_return")
    excess_return = None
    if avg_top_return is not None and avg_universe_return is not None:
        excess_return = _round_or_none(float(avg_top_return) - float(avg_universe_return), 4)

    failure_reasons = []
    if evaluated_days < int(settings["min_evaluated_days"]):
        failure_reasons.append("insufficient_backtest_days")
    if avg_top_return is None or float(avg_top_return) < float(settings["min_avg_top_return"]):
        failure_reasons.append("avg_top_return_below_threshold")
    if avg_top_win_rate is None or float(avg_top_win_rate) < float(settings["min_avg_top_win_rate"]):
        failure_reasons.append("top_win_rate_below_threshold")
    if excess_return is None or float(excess_return) < float(settings["min_excess_return"]):
        failure_reasons.append("excess_return_below_threshold")

    return {
        "enabled": not failure_reasons,
        "mode": "walk_forward_backtest",
        "hold_days": hold_days,
        "window_trade_days": int(ADAPTIVE_BACKTEST_HEALTH_LOOKBACK_TRADE_DAYS),
        "evaluated_days": evaluated_days,
        "avg_top_return": _round_or_none(avg_top_return, 4),
        "avg_top_win_rate": _round_or_none(avg_top_win_rate, 2),
        "avg_universe_return": _round_or_none(avg_universe_return, 4),
        "excess_return": excess_return,
        "failure_reasons": failure_reasons,
    }


def _combine_adaptive_health(signal_health, backtest_health):
    signal_health = signal_health or {}
    backtest_health = backtest_health or {}
    signal_threshold = int(ADAPTIVE_SIGNAL_HEALTH["min_evaluated_trades"])
    signal_evaluated = int(signal_health.get("evaluated_trades") or 0)
    signal_available = signal_evaluated >= signal_threshold

    enabled = False
    decision_mode = "rejected"
    failure_reasons = []
    ignored_signal_reasons = []

    if signal_available:
        enabled = bool(signal_health.get("enabled")) and bool(backtest_health.get("enabled"))
        decision_mode = "confirmed" if enabled else "confirmed_rejected"
        if not signal_health.get("enabled"):
            failure_reasons.extend(signal_health.get("failure_reasons") or [])
        if not backtest_health.get("enabled"):
            failure_reasons.extend(backtest_health.get("failure_reasons") or [])
    else:
        enabled = bool(backtest_health.get("enabled"))
        decision_mode = "bootstrap" if enabled else "bootstrap_rejected"
        if not enabled:
            failure_reasons.extend(backtest_health.get("failure_reasons") or [])
        if signal_health and not enabled:
            failure_reasons.extend(signal_health.get("failure_reasons") or [])
        elif signal_health:
            ignored_signal_reasons.extend(signal_health.get("failure_reasons") or [])

    policy = ADAPTIVE_HEALTH_POLICIES.get(decision_mode) or {
        "label": "拒绝落库",
        "confidence_weight": 0.0,
        "max_pick_ratio": 0.0,
        "position_hint": None,
    }

    return {
        "enabled": enabled,
        "mode": decision_mode,
        "mode_label": policy.get("label"),
        "confidence_weight": policy.get("confidence_weight"),
        "max_pick_ratio": policy.get("max_pick_ratio"),
        "position_hint": policy.get("position_hint"),
        "signal_health": signal_health,
        "backtest_health": backtest_health,
        "signal_health_available": signal_available,
        "evaluated_trades": signal_health.get("evaluated_trades"),
        "avg_return": signal_health.get("avg_return"),
        "trade_win_rate": signal_health.get("trade_win_rate"),
        "evaluated_days": backtest_health.get("evaluated_days"),
        "avg_top_return": backtest_health.get("avg_top_return"),
        "avg_top_win_rate": backtest_health.get("avg_top_win_rate"),
        "avg_universe_return": backtest_health.get("avg_universe_return"),
        "excess_return": backtest_health.get("excess_return"),
        "failure_reasons": list(dict.fromkeys(failure_reasons)),
        "ignored_signal_failure_reasons": list(dict.fromkeys(ignored_signal_reasons)),
    }


def _prepare_common_frame(df, dedupe_keys):
    frame = df.copy()
    _ensure_columns(frame, ["stock_code", "stock_name", "industry", "last_data_date"], None)
    _ensure_columns(frame, NUMERIC_COLUMNS, pd.NA)

    frame["stock_code"] = _normalize_stock_code_series(frame["stock_code"])
    frame = frame[frame["stock_code"].notna()].copy()
    if frame.empty:
        return frame

    frame["stock_name"] = frame["stock_name"].fillna("").astype(str).str.strip()
    frame["stock_name"] = frame["stock_name"].replace("", pd.NA).fillna(frame["stock_code"])

    stock_name = frame["stock_name"].fillna("").astype(str).str.strip()
    is_excluded_name = stock_name.str.startswith(EXCLUDED_NAME_PREFIXES)

    frame["last_data_date"] = pd.to_datetime(frame["last_data_date"], errors="coerce")
    frame = frame[frame["last_data_date"].notna()].copy()
    if frame.empty:
        return frame

    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame[~is_excluded_name & frame["latest_price"].gt(0) & frame["today_vol"].gt(0)].copy()
    if frame.empty:
        return frame

    industry_series = frame["industry"].fillna("").astype(str).str.split(",").str[0].str.strip()
    frame["industry_1"] = industry_series.replace("", pd.NA)

    for column in ["price_vs_ma5", "price_vs_ma20", "price_vs_ma60", "ma5_vs_ma20", "ma5_vs_ma60", "ma20_vs_ma60"]:
        frame[column] = pd.NA
    frame["price_vs_ma5"] = _safe_ratio(frame["latest_price"], frame["ma5"]) - 1
    frame["price_vs_ma20"] = _safe_ratio(frame["latest_price"], frame["ma20"]) - 1
    frame["price_vs_ma60"] = _safe_ratio(frame["latest_price"], frame["ma60"]) - 1
    frame["ma5_vs_ma20"] = _safe_ratio(frame["ma5"], frame["ma20"]) - 1
    frame["ma5_vs_ma60"] = _safe_ratio(frame["ma5"], frame["ma60"]) - 1
    frame["ma20_vs_ma60"] = _safe_ratio(frame["ma20"], frame["ma60"]) - 1
    frame["close_to_20d_high"] = _safe_ratio(frame["latest_price"], frame["high_20d"])
    frame["close_to_20d_low"] = _safe_ratio(frame["latest_price"], frame["low_20d"])
    frame["volume_vs_avg_3d"] = _safe_ratio(frame["today_vol"], frame["vol_avg_3d"])
    frame["volume_vs_avg_5d"] = _safe_ratio(frame["today_vol"], frame["vol_avg_5d"])
    frame["volume_vs_avg_10d"] = _safe_ratio(frame["today_vol"], frame["vol_avg_10d"])
    frame["volume_vs_avg_20d"] = _safe_ratio(frame["today_vol"], frame["vol_avg_20d"])
    frame["amount_vs_avg_5d"] = _safe_ratio(frame["today_amount"], frame["amount_avg_5d"])
    frame["amount_vs_avg_10d"] = _safe_ratio(frame["today_amount"], frame["amount_avg_10d"])
    frame["turnover_vs_avg_5d"] = _safe_ratio(frame["turnover_rate"], frame["turnover_avg_5d"])
    frame["turnover_vs_avg_10d"] = _safe_ratio(frame["turnover_rate"], frame["turnover_avg_10d"])
    frame["amp_vs_avg_3d"] = _safe_ratio(frame["today_amp"], frame["amp_3d"])
    frame["amp_vs_avg_5d"] = _safe_ratio(frame["today_amp"], frame["amp_5d"])
    frame["amp_vs_avg_10d"] = _safe_ratio(frame["today_amp"], frame["amp_10d"])
    frame["amp_vs_avg_20d"] = _safe_ratio(frame["today_amp"], frame["amp_20d"])
    frame["volatility_ratio_10_20"] = _safe_ratio(frame["volatility_10d"], frame["volatility_20d"])

    intraday_range = pd.to_numeric(frame["today_high"], errors="coerce") - pd.to_numeric(frame["today_low"], errors="coerce")
    intraday_body = (
        pd.to_numeric(frame["latest_price"], errors="coerce") - pd.to_numeric(frame["today_open"], errors="coerce")
    ).abs()
    intraday_upper_shadow = pd.to_numeric(frame["today_high"], errors="coerce") - pd.concat(
        [
            pd.to_numeric(frame["today_open"], errors="coerce"),
            pd.to_numeric(frame["latest_price"], errors="coerce"),
        ],
        axis=1,
    ).max(axis=1)
    intraday_upper_shadow = intraday_upper_shadow.clip(lower=0)
    valid_range = intraday_range.where(intraday_range > 0)
    frame["close_strength"] = (
        (pd.to_numeric(frame["latest_price"], errors="coerce") - pd.to_numeric(frame["today_low"], errors="coerce"))
        / valid_range
    )
    frame["upper_shadow_ratio"] = intraday_upper_shadow / valid_range
    frame["body_to_range_ratio"] = intraday_body / valid_range

    stock_rank_series = pd.to_numeric(frame["stock_rank"], errors="coerce")
    frame["stock_rank_score"] = 1 / (stock_rank_series + 1)

    derived_columns = [
        "price_vs_ma5",
        "price_vs_ma20",
        "price_vs_ma60",
        "ma5_vs_ma20",
        "ma5_vs_ma60",
        "ma20_vs_ma60",
        "close_to_20d_high",
        "close_to_20d_low",
        "volume_vs_avg_3d",
        "volume_vs_avg_5d",
        "volume_vs_avg_10d",
        "volume_vs_avg_20d",
        "amount_vs_avg_5d",
        "amount_vs_avg_10d",
        "turnover_vs_avg_5d",
        "turnover_vs_avg_10d",
        "amp_vs_avg_3d",
        "amp_vs_avg_5d",
        "amp_vs_avg_10d",
        "amp_vs_avg_20d",
        "volatility_ratio_10_20",
        "close_strength",
        "upper_shadow_ratio",
        "body_to_range_ratio",
        "stock_rank_score",
    ]
    for column in derived_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    sort_columns = list(dedupe_keys)
    if "last_data_date" in frame.columns and "last_data_date" not in sort_columns:
        sort_columns.append("last_data_date")

    frame = frame.sort_values(sort_columns, na_position="last").copy()
    frame = frame.drop_duplicates(subset=list(dedupe_keys), keep="last")
    return frame.reset_index(drop=True)


def _build_market_context(frame):
    prepared = frame.copy()
    prepared["_change_5d_up"] = prepared["change_5d"].gt(0).astype(float) * 100
    prepared["_change_10d_up"] = prepared["change_10d"].gt(0).astype(float) * 100

    market_ctx = (
        prepared.groupby("last_data_date", sort=False)
        .agg(
            market_change_5d=("change_5d", "mean"),
            market_change_10d=("change_10d", "mean"),
            market_breadth_5d=("_change_5d_up", "mean"),
            market_breadth_10d=("_change_10d_up", "mean"),
        )
        .reset_index()
    )

    industry_ctx = (
        prepared.dropna(subset=["industry_1"])
        .groupby(["last_data_date", "industry_1"], sort=False)
        .agg(
            industry_change_5d=("change_5d", "mean"),
            industry_change_10d=("change_10d", "mean"),
            industry_breadth_5d=("_change_5d_up", "mean"),
            industry_breadth_10d=("_change_10d_up", "mean"),
            industry_count=("stock_code", "count"),
        )
        .reset_index()
    )

    if not industry_ctx.empty:
        industry_ctx = industry_ctx.merge(market_ctx, on="last_data_date", how="left")
        industry_ctx["industry_alpha_5d"] = industry_ctx["industry_change_5d"] - industry_ctx["market_change_5d"]
        industry_ctx["industry_alpha_10d"] = industry_ctx["industry_change_10d"] - industry_ctx["market_change_10d"]
        industry_ctx["industry_strength_rank_5d"] = industry_ctx.groupby("last_data_date")["industry_change_5d"].rank(
            method="average",
            pct=True,
            ascending=True,
        )
        industry_ctx["industry_strength_rank_10d"] = industry_ctx.groupby("last_data_date")["industry_change_10d"].rank(
            method="average",
            pct=True,
            ascending=True,
        )

        frame = frame.merge(
            industry_ctx[
                [
                    "last_data_date",
                    "industry_1",
                    "industry_change_5d",
                    "industry_change_10d",
                    "industry_breadth_5d",
                    "industry_breadth_10d",
                    "industry_alpha_5d",
                    "industry_alpha_10d",
                    "industry_strength_rank_5d",
                    "industry_strength_rank_10d",
                ]
            ],
            on=["last_data_date", "industry_1"],
            how="left",
        )
    else:
        for column in [
            "industry_change_5d",
            "industry_change_10d",
            "industry_breadth_5d",
            "industry_breadth_10d",
            "industry_alpha_5d",
            "industry_alpha_10d",
            "industry_strength_rank_5d",
            "industry_strength_rank_10d",
        ]:
            frame[column] = pd.NA

    frame = frame.merge(market_ctx, on="last_data_date", how="left")
    return frame






def _build_forward_returns(frame, horizons=HORIZON_DAYS):
    prepared = frame.sort_values(["stock_code", "last_data_date"]).copy()
    cost_pct = (DEFAULT_FEE_BPS + DEFAULT_SLIPPAGE_BPS) * 2 / 100
    price_columns = ["last_data_date", "stock_code", "latest_price", "today_change"]
    for column in ["today_open", "today_high", "today_low"]:
        if column in prepared.columns:
            price_columns.append(column)
    price_paths = _build_price_paths(
        prepared[price_columns],
        horizons,
        DEFAULT_ENTRY_OFFSET_DAYS,
        round_trip_cost_pct=cost_pct,
    )

    merge_columns = ["last_data_date", "stock_code", "entry_change"]
    for horizon_days in horizons:
        merge_columns.extend(
            [
                f"exit_date_{horizon_days}d",
                f"exit_change_{horizon_days}d",
                f"gross_return_{horizon_days}d",
                f"return_{horizon_days}d",
            ]
        )

    prepared = prepared.merge(
        price_paths[merge_columns],
        on=["last_data_date", "stock_code"],
        how="left",
    )

    for horizon_days in horizons:
        return_col = f"return_{horizon_days}d"
        gross_return_col = f"gross_return_{horizon_days}d"
        exit_change_col = f"exit_change_{horizon_days}d"
        entry_abs = pd.to_numeric(prepared["entry_change"], errors="coerce").abs()
        exit_abs = pd.to_numeric(prepared[exit_change_col], errors="coerce").abs()
        tradable = (entry_abs.isna() | (entry_abs < DEFAULT_LIMIT_PCT)) & (
            exit_abs.isna() | (exit_abs < DEFAULT_LIMIT_PCT)
        )
        prepared.loc[prepared[return_col].notna() & (~tradable), [return_col, gross_return_col]] = pd.NA
        prepared[f"forward_return_{horizon_days}d"] = prepared[return_col]
        prepared[f"forward_gross_return_{horizon_days}d"] = prepared[gross_return_col]
        prepared[f"forward_trade_date_{horizon_days}d"] = prepared[f"exit_date_{horizon_days}d"]

    return prepared


def _add_percentile_columns(frame, columns):
    ranked = frame.copy()
    available_columns = [column for column in columns if column in ranked.columns]
    if not available_columns:
        return ranked

    pct_frame = ranked.groupby("last_data_date", sort=False)[available_columns].rank(
        method="average",
        pct=True,
        ascending=True,
    )
    pct_frame.columns = [f"{column}_pct" for column in available_columns]
    ranked = pd.concat([ranked, pct_frame], axis=1)

    return ranked


def _style_mask(frame, style_name):
    if style_name == "breakout":
        return (
            (_feature_series(frame, "today_change", FEATURE_UNIT_PERCENT_POINT) >= 3.5)
            & (_feature_series(frame, "change_20d", FEATURE_UNIT_PERCENT_POINT) > 0)
            & (_feature_series(frame, "change_30d", FEATURE_UNIT_PERCENT_POINT) >= 0)
            & (_feature_series(frame, "close_to_20d_high", FEATURE_UNIT_MULTIPLE_RATIO) >= 0.95)
            & (_feature_series(frame, "price_vs_ma20", FEATURE_UNIT_EXCESS_RATIO) > 0)
            & (_feature_series(frame, "ma5_vs_ma20", FEATURE_UNIT_EXCESS_RATIO) > 0)
            & (_feature_series(frame, "industry_change_5d", FEATURE_UNIT_PERCENT_POINT) > 0)
        )

    if style_name == "steady_climb":
        return (
            _feature_series(frame, "change_5d", FEATURE_UNIT_PERCENT_POINT).gt(0)
            & _feature_series(frame, "change_20d", FEATURE_UNIT_PERCENT_POINT).gt(0)
            & _feature_series(frame, "price_vs_ma20", FEATURE_UNIT_EXCESS_RATIO).between(-0.03, 0.12)
            & _feature_series(frame, "ma5_vs_ma20", FEATURE_UNIT_EXCESS_RATIO).gt(-0.01)
            & (
                _feature_series(frame, "ma20_vs_ma60", FEATURE_UNIT_EXCESS_RATIO).isna()
                | _feature_series(frame, "ma20_vs_ma60", FEATURE_UNIT_EXCESS_RATIO).gt(-0.02)
            )
            & _feature_series(frame, "close_to_20d_high", FEATURE_UNIT_MULTIPLE_RATIO).between(0.78, 0.98)
            & (_feature_series(frame, "industry_alpha_5d", FEATURE_UNIT_PERCENT_POINT) > 0)
            & (_feature_series(frame, "industry_alpha_10d", FEATURE_UNIT_PERCENT_POINT) > 0)
        )

    if style_name == "rebound":
        return (
            (_feature_series(frame, "change_20d", FEATURE_UNIT_PERCENT_POINT) < 0)
            & (_feature_series(frame, "close_to_20d_low", FEATURE_UNIT_MULTIPLE_RATIO) <= 1.06)
            & (_feature_series(frame, "today_change", FEATURE_UNIT_PERCENT_POINT) > 0)
            & (_feature_series(frame, "amp_vs_avg_5d", FEATURE_UNIT_MULTIPLE_RATIO) > 1.0)
        )

    return pd.Series(False, index=frame.index)


def _resolve_candidate_style(record):
    today_change = _feature_value(record, "today_change", FEATURE_UNIT_PERCENT_POINT)
    change_5d = _feature_value(record, "change_5d", FEATURE_UNIT_PERCENT_POINT)
    change_20d = _feature_value(record, "change_20d", FEATURE_UNIT_PERCENT_POINT)
    change_30d = _feature_value(record, "change_30d", FEATURE_UNIT_PERCENT_POINT)
    close_to_20d_high = _feature_value(record, "close_to_20d_high", FEATURE_UNIT_MULTIPLE_RATIO)
    close_to_20d_low = _feature_value(record, "close_to_20d_low", FEATURE_UNIT_MULTIPLE_RATIO)
    price_vs_ma20 = _feature_value(record, "price_vs_ma20", FEATURE_UNIT_EXCESS_RATIO)
    ma5_vs_ma20 = _feature_value(record, "ma5_vs_ma20", FEATURE_UNIT_EXCESS_RATIO)
    ma20_vs_ma60 = _feature_value(record, "ma20_vs_ma60", FEATURE_UNIT_EXCESS_RATIO)
    industry_change_5d = _feature_value(record, "industry_change_5d", FEATURE_UNIT_PERCENT_POINT)
    industry_alpha_5d = _feature_value(record, "industry_alpha_5d", FEATURE_UNIT_PERCENT_POINT)
    industry_alpha_10d = _feature_value(record, "industry_alpha_10d", FEATURE_UNIT_PERCENT_POINT)
    amp_vs_avg_5d = _feature_value(record, "amp_vs_avg_5d", FEATURE_UNIT_MULTIPLE_RATIO)

    if (
        today_change is not None
        and close_to_20d_high is not None
        and price_vs_ma20 is not None
        and ma5_vs_ma20 is not None
        and industry_change_5d is not None
        and today_change >= 3.5
        and change_20d is not None
        and change_20d > 0
        and (change_30d is None or change_30d >= 0)
        and close_to_20d_high >= 0.95
        and price_vs_ma20 > 0
        and ma5_vs_ma20 > 0
        and (ma20_vs_ma60 is None or ma20_vs_ma60 > 0)
        and industry_change_5d > 0
    ):
        return "breakout"

    if (
        change_5d is not None
        and change_20d is not None
        and price_vs_ma20 is not None
        and ma5_vs_ma20 is not None
        and close_to_20d_high is not None
        and industry_alpha_5d is not None
        and industry_alpha_10d is not None
        and change_5d > 0
        and change_20d > 0
        and price_vs_ma20 >= -0.03
        and price_vs_ma20 <= 0.12
        and ma5_vs_ma20 > -0.01
        and (ma20_vs_ma60 is None or ma20_vs_ma60 > -0.02)
        and 0.78 <= close_to_20d_high <= 0.98
        and industry_alpha_5d > 0
        and industry_alpha_10d > 0
    ):
        return "steady_climb"

    if (
        change_20d is not None
        and close_to_20d_low is not None
        and today_change is not None
        and amp_vs_avg_5d is not None
        and change_20d < 0
        and close_to_20d_low <= 1.06
        and today_change > 0
        and amp_vs_avg_5d > 1.0
    ):
        return "rebound"

    return None


def _record_matches_style(record, style_name):
    if not style_name:
        return False

    resolved_style = _resolve_candidate_style(record)
    return resolved_style == style_name


def _build_style_horizon_profiles(frame, horizon_days):
    style_profiles = {}
    for style_name in STYLE_PRIORITY:
        style_frame = frame[_style_mask(frame, style_name)].copy()
        if style_frame.empty:
            continue
        profile = _build_horizon_profile(style_frame, horizon_days)
        if profile.get("profile_weight", 0) >= MIN_STYLE_PROFILE_ROWS:
            profile["style_name"] = style_name
            profile["style_label"] = STYLE_LABELS.get(style_name, style_name)
            profile["style_note"] = STYLE_NOTE_HINTS.get(style_name)
            style_profiles[style_name] = profile

    return style_profiles


def _build_horizon_profile(
    frame,
    horizon_days,
    family_features=FAMILY_FEATURES,
    feature_labels=FEATURE_LABELS,
    family_labels=FAMILY_LABELS,
    half_life_days=RECENCY_HALF_LIFE_DAYS,
    winner_ratio=TOP_WINNER_RATIO,
    loser_ratio=LOSER_RATIO,
    min_daily_rows=MIN_DAILY_ROWS,
):
    future_col = f"forward_return_{horizon_days}d"
    valid = frame[frame[future_col].notna()].copy()
    if valid.empty:
        return {
            "horizon_days": horizon_days,
            "profile_weight": 0,
            "sample_rows": 0,
            "sample_days": 0,
            "winner_rows": 0,
            "loser_rows": 0,
            "positive_rate": None,
            "families": {},
            "top_features": [],
            "feature_count": 0,
        }

    valid = valid[valid["last_data_date"].notna()].copy()
    if valid.empty:
        return {
            "horizon_days": horizon_days,
            "profile_weight": 0,
            "sample_rows": 0,
            "sample_days": 0,
            "winner_rows": 0,
            "loser_rows": 0,
            "positive_rate": None,
            "families": {},
            "top_features": [],
            "feature_count": 0,
        }

    latest_date = valid["last_data_date"].max()
    lag_days = (latest_date - valid["last_data_date"]).dt.days.clip(lower=0)
    valid["recency_weight"] = (-lag_days / half_life_days).map(math.exp)
    valid["sample_weight"] = valid["recency_weight"] * (1 + valid[future_col].abs().clip(lower=0, upper=20) / 20)
    valid["daily_rank_pct"] = valid.groupby("last_data_date")[future_col].rank(
        method="average",
        pct=True,
        ascending=True,
    )

    valid["daily_rows"] = valid.groupby("last_data_date")["stock_code"].transform("count")
    valid = valid[valid["daily_rows"] >= min_daily_rows].copy()
    if valid.empty:
        return {
            "horizon_days": horizon_days,
            "profile_weight": 0,
            "sample_rows": 0,
            "sample_days": 0,
            "winner_rows": 0,
            "loser_rows": 0,
            "positive_rate": None,
            "families": {},
            "top_features": [],
            "feature_count": 0,
        }

    winners = valid[valid["daily_rank_pct"] >= (1 - winner_ratio)].copy()
    losers = valid[valid["daily_rank_pct"] <= loser_ratio].copy()

    families = {}
    top_feature_rows = []
    feature_count = 0

    for family_name, feature_list in family_features.items():
        family_items = []
        for feature_name in feature_list:
            pct_column = f"{feature_name}_pct"
            if pct_column not in valid.columns:
                continue

            target_pct = _weighted_mean(winners[pct_column], winners["sample_weight"])
            loser_pct = _weighted_mean(losers[pct_column], losers["sample_weight"])
            if target_pct is None or loser_pct is None:
                continue

            lift = target_pct - loser_pct
            importance = abs(lift)
            if importance < 0.01:
                continue

            item = {
                "feature": feature_name,
                "label": feature_labels.get(feature_name, feature_name),
                "pct_column": pct_column,
                "target_pct": _round_or_none(target_pct, 4),
                "loser_pct": _round_or_none(loser_pct, 4),
                "lift": _round_or_none(lift, 4),
                "importance": _round_or_none(importance, 4),
            }
            family_items.append(item)
            top_feature_rows.append(
                {
                    "family": family_name,
                    "feature": feature_name,
                    "label": item["label"],
                    "pct_column": pct_column,
                    "target_pct": item["target_pct"],
                    "loser_pct": item["loser_pct"],
                    "lift": item["lift"],
                    "importance": item["importance"],
                }
            )
            feature_count += 1

        family_items.sort(key=lambda item: item["importance"], reverse=True)
        family_importance = sum(float(item["importance"]) for item in family_items)
        families[family_name] = {
            "importance": _round_or_none(family_importance, 4),
            "items": family_items,
        }

    total_family_importance = sum(float(data["importance"] or 0) for data in families.values())
    for family_name, family_data in families.items():
        family_importance = float(family_data["importance"] or 0)
        family_data["normalized_importance"] = (
            round(family_importance / total_family_importance, 4) if total_family_importance > 0 else 0
        )

    top_feature_rows.sort(key=lambda item: item["importance"], reverse=True)
    top_feature_rows = top_feature_rows[:10]

    return {
        "horizon_days": horizon_days,
        "profile_weight": len(valid),
        "sample_rows": int(len(valid)),
        "sample_days": int(valid["last_data_date"].nunique()),
        "winner_rows": int(len(winners)),
        "loser_rows": int(len(losers)),
        "positive_rate": _round_or_none((valid[future_col] > 0).mean() * 100, 2),
        "families": families,
        "top_features": top_feature_rows,
        "feature_count": feature_count,
    }


def _summarize_style_profiles(style_horizon_profiles):
    style_summaries = []

    for style_name in STYLE_PRIORITY:
        style_horizon_subset = {}
        for horizon_days, style_profiles in style_horizon_profiles.items():
            profile = style_profiles.get(style_name)
            if profile:
                style_horizon_subset[horizon_days] = profile

        if not style_horizon_subset:
            continue

        top_families, top_features = _summarize_families(style_horizon_subset)
        sample_rows = sum(int(profile.get("sample_rows") or 0) for profile in style_horizon_subset.values())
        sample_days = max(int(profile.get("sample_days") or 0) for profile in style_horizon_subset.values())

        style_summaries.append(
            {
                "style": style_name,
                "label": STYLE_LABELS.get(style_name, style_name),
                "note": STYLE_NOTE_HINTS.get(style_name),
                "sample_rows": sample_rows,
                "sample_days": sample_days,
                "top_families": top_families[:3],
                "top_features": top_features[:5],
                "horizons": {
                    f"{horizon_days}d": {
                        "sample_rows": profile["sample_rows"],
                        "sample_days": profile["sample_days"],
                        "winner_rows": profile["winner_rows"],
                        "loser_rows": profile["loser_rows"],
                        "positive_rate": profile["positive_rate"],
                        "top_features": profile["top_features"][:5],
                    }
                    for horizon_days, profile in style_horizon_subset.items()
                },
            }
        )

    return style_summaries


def _score_one_profile(record, profile):
    if not profile or profile.get("profile_weight", 0) <= 0:
        return {
            "score": None,
            "family_scores": {},
            "family_details": {},
            "top_family": None,
            "top_family_score": None,
            "coverage": 0,
        }

    family_scores = {}
    family_details = {}
    seen_features = set()

    weighted_score_sum = 0.0
    weighted_family_sum = 0.0

    for family_name, family_data in profile["families"].items():
        family_weight = float(family_data.get("normalized_importance") or 0)
        items = family_data.get("items", [])
        if family_weight <= 0 or not items:
            continue

        feature_score_sum = 0.0
        feature_weight_sum = 0.0
        feature_details = []

        for item in items:
            pct_column = item["pct_column"]
            if pct_column in seen_features:
                continue

            value = record.get(pct_column)
            if value is None or pd.isna(value):
                continue

            seen_features.add(pct_column)
            target_pct = float(item["target_pct"])
            fit = max(0.0, 1.0 - abs(float(value) - target_pct) / 0.5)
            importance = float(item["importance"])
            contribution = fit * importance
            feature_score_sum += contribution
            feature_weight_sum += importance
            feature_details.append(
                {
                    "feature": item["feature"],
                    "label": item["label"],
                    "candidate_pct": _round_or_none(value, 4),
                    "target_pct": item["target_pct"],
                    "fit": _round_or_none(fit, 4),
                    "importance": item["importance"],
                    "contribution": _round_or_none(contribution, 4),
                }
            )

        if feature_weight_sum <= 0:
            continue

        family_score = feature_score_sum / feature_weight_sum
        family_scores[family_name] = round(family_score * 100, 2)
        family_details[family_name] = sorted(
            feature_details,
            key=lambda item: item["contribution"] or 0,
            reverse=True,
        )[:3]

        weighted_score_sum += family_score * family_weight
        weighted_family_sum += family_weight

    raw_score = weighted_score_sum / weighted_family_sum if weighted_family_sum > 0 else None
    coverage = round(len(seen_features) / profile["feature_count"], 4) if profile["feature_count"] > 0 else 0
    final_score = None
    if raw_score is not None:
        final_score = max(0.0, min(1.0, raw_score * (0.6 + 0.4 * coverage))) * 100

    top_family = None
    top_family_score = None
    if family_scores:
        top_family = max(family_scores.items(), key=lambda item: item[1])[0]
        top_family_score = family_scores[top_family]

    return {
        "score": _round_or_none(final_score, 2),
        "family_scores": family_scores,
        "family_details": family_details,
        "top_family": top_family,
        "top_family_score": top_family_score,
        "coverage": coverage,
    }


def _build_trend_state(record):
    change_5d = _to_float(record.get("change_5d"))
    change_20d = _to_float(record.get("change_20d"))
    change_30d = _to_float(record.get("change_30d"))
    price_vs_ma20 = _to_float(record.get("price_vs_ma20"))
    price_vs_ma60 = _to_float(record.get("price_vs_ma60"))
    ma5_vs_ma20 = _to_float(record.get("ma5_vs_ma20"))
    ma5_vs_ma60 = _to_float(record.get("ma5_vs_ma60"))
    ma20_vs_ma60 = _to_float(record.get("ma20_vs_ma60"))
    close_to_20d_high = _to_float(record.get("close_to_20d_high"))
    close_to_20d_low = _to_float(record.get("close_to_20d_low"))

    score = 0
    week_up = change_5d is not None and change_5d > 0
    month_up = change_20d is not None and change_20d > 0
    month_strong = change_30d is not None and change_30d > 0
    above_ma20 = price_vs_ma20 is not None and price_vs_ma20 > 0
    above_ma60 = price_vs_ma60 is not None and price_vs_ma60 > 0
    ma20_up = ma20_vs_ma60 is not None and ma20_vs_ma60 > 0
    ma5_up = ma5_vs_ma20 is not None and ma5_vs_ma20 > 0
    ma5_vs_60_up = ma5_vs_ma60 is not None and ma5_vs_ma60 > 0

    for condition in [week_up, month_up, month_strong, above_ma20, above_ma60, ma20_up, ma5_up, ma5_vs_60_up]:
        if condition:
            score += 1

    if score >= 6 and week_up and month_up and (ma20_up or above_ma60):
        state = "周月同步抬升"
    elif month_up and week_up and (price_vs_ma20 is None or price_vs_ma20 >= -0.03):
        state = "月线抬升，周线回踩"
    elif month_up and not week_up:
        state = "月线抬升，周线整理"
    elif week_up and not month_up:
        state = "周线反弹，月线待确认"
    elif change_20d is not None and change_20d < 0 and (price_vs_ma20 is None or price_vs_ma20 < 0):
        state = "弱势修复"
    else:
        state = "趋势震荡"

    detail_bits = []
    if change_5d is not None:
        detail_bits.append(f"周线{_round_or_none(change_5d, 2)}%")
    if change_20d is not None:
        detail_bits.append(f"月线{_round_or_none(change_20d, 2)}%")
    if change_30d is not None:
        detail_bits.append(f"30日{_round_or_none(change_30d, 2)}%")
    if price_vs_ma20 is not None:
        detail_bits.append(f"价格相对20日均线{_round_or_none(price_vs_ma20 * 100, 2)}%")
    if ma20_vs_ma60 is not None:
        detail_bits.append(f"20日均线相对60日均线{_round_or_none(ma20_vs_ma60 * 100, 2)}%")
    if close_to_20d_high is not None:
        if close_to_20d_high >= 1:
            detail_bits.append(f"收盘较20日高点高{_round_or_none((close_to_20d_high - 1) * 100, 2)}%")
        else:
            detail_bits.append(f"收盘较20日高点低{_round_or_none((1 - close_to_20d_high) * 100, 2)}%")
    if close_to_20d_low is not None:
        if close_to_20d_low >= 1:
            detail_bits.append(f"收盘较20日低点高{_round_or_none((close_to_20d_low - 1) * 100, 2)}%")
        else:
            detail_bits.append(f"收盘较20日低点低{_round_or_none((1 - close_to_20d_low) * 100, 2)}%")

    return {
        "state": state,
        "score": score,
        "summary": "，".join(detail_bits) if detail_bits else "趋势特征不足",
    }


def _build_reason(profile_result, horizon_days, style_label=None, trend_state=None):
    top_family = profile_result.get("top_family")
    if not top_family:
        base_reason = "特征覆盖不足，暂时没有稳定的历史画像可对齐。"
        if trend_state:
            base_reason = f"{trend_state}，{base_reason}"
        if style_label:
            return f"{style_label}风格，{base_reason}"
        return base_reason

    family_details = profile_result.get("family_details", {}).get(top_family, [])
    if not family_details:
        base_reason = "特征覆盖不足，暂时没有稳定的历史画像可对齐。"
        if trend_state:
            base_reason = f"{trend_state}，{base_reason}"
        if style_label:
            return f"{style_label}风格，{base_reason}"
        return base_reason

    feature_bits = []
    for item in family_details[:3]:
        label = item.get("label") or item.get("feature")
        candidate_pct = item.get("candidate_pct")
        if not label:
            continue
        if candidate_pct is not None and not pd.isna(candidate_pct):
            feature_bits.append(f"{label}({int(round(float(candidate_pct) * 100))}分位)")
        else:
            feature_bits.append(label)

    if not feature_bits:
        base_reason = "特征覆盖不足，暂时没有稳定的历史画像可对齐。"
        if trend_state:
            base_reason = f"{trend_state}，{base_reason}"
        if style_label:
            return f"{style_label}风格，{base_reason}"
        return base_reason

    score = profile_result.get("score")
    if score is not None and not pd.isna(score):
        base_reason = f"{horizon_days}日视角，当前更贴近{'、'.join(feature_bits)}这组特征，匹配度约{score}分。"
    else:
        base_reason = f"{horizon_days}日视角，当前更贴近{'、'.join(feature_bits)}这组特征。"

    if trend_state:
        base_reason = f"{trend_state}，{base_reason}"

    if style_label:
        return f"{style_label}风格，{base_reason}"
    return base_reason


def _nearest_support_below(price, levels):
    if price is None or pd.isna(price):
        return None

    valid_levels = []
    for level in levels:
        if level is None or pd.isna(level):
            continue
        if float(level) < float(price):
            valid_levels.append(float(level))

    if not valid_levels:
        return None

    return max(valid_levels)


def _build_trade_plan(record, market_env, style_label=None, style_name=None, trend_state=None, adaptive_score=None):
    latest_price = _to_float(record.get("latest_price"))
    ma20 = _to_float(record.get("ma20"))
    ma60 = _to_float(record.get("ma60"))
    low_20d = _to_float(record.get("low_20d"))
    high_20d = _to_float(record.get("high_20d"))
    price_vs_ma20 = _to_float(record.get("price_vs_ma20"))
    change_5d = _to_float(record.get("change_5d"))
    change_20d = _to_float(record.get("change_20d"))
    change_30d = _to_float(record.get("change_30d"))
    today_change = _to_float(record.get("today_change"))
    risk_labels = record.get("risk_labels") or "无明显"
    no_buy_condition = record.get("no_buy_condition") or "承接不足则放弃"

    style_name = style_name or "generic"
    style_label = style_label or STYLE_LABELS.get(style_name)
    trend_state = trend_state or "趋势震荡"
    market_env = market_env or "未知"
    weak_market = market_env in ("弱势", "偏弱")

    action = "观望"
    hold_low, hold_high = 2, 6
    exp_low, exp_high = 1.0, 5.0
    stop_price = None
    stop_rule = "趋势未确认，先等更明确的方向。"
    reenter_rule = "重新站上20日均线后再考虑。"
    position_hint = "轻仓"

    if style_name == "breakout":
        action = "顺势跟随" if market_env in ("强势", "偏强") else "小仓观察"
        hold_low, hold_high = 3, 8
        exp_low, exp_high = 3.0, 10.0
        support = _nearest_support_below(latest_price, [ma20, ma60, low_20d])
        stop_price = _round_or_none((support * 0.99) if support else (latest_price * 0.95 if latest_price else None), 2)
        stop_rule = "跌破20日均线或放量滞涨时先撤，若失守更近的支撑位则退出。"
        reenter_rule = "重新站上20日均线并保持2天以上，再考虑继续跟随。"
        position_hint = "轻仓" if weak_market else "中等仓位"
        if trend_state in ("周月同步抬升", "月线抬升，周线回踩"):
            hold_low += 2
            hold_high += 3
            exp_low += 1.0
            exp_high += 3.0

    elif style_name == "steady_climb":
        action = "谨慎跟随" if weak_market else "持有跟随"
        hold_low, hold_high = 5, 15
        exp_low, exp_high = 4.0, 12.0
        support = _nearest_support_below(latest_price, [ma20, ma60, low_20d])
        stop_price = _round_or_none((support * 0.985) if support else (latest_price * 0.94 if latest_price else None), 2)
        stop_rule = "跌破20日均线后两日不收回，或月线抬升被破坏时撤。"
        reenter_rule = "周线和月线继续抬高后，再做加仓判断。"
        position_hint = "轻仓" if weak_market else "中等仓位"
        if trend_state == "周月同步抬升":
            hold_high += 4
            exp_high += 3.0
        elif trend_state == "月线抬升，周线回踩":
            hold_high += 2
            exp_high += 2.0

    elif style_name == "rebound":
        action = "试仓观察" if market_env in ("强势", "偏强", "震荡") else "轻仓观察"
        hold_low, hold_high = 1, 4
        exp_low, exp_high = 2.0, 6.0
        stop_price = _round_or_none((low_20d * 0.995) if low_20d else (latest_price * 0.95 if latest_price else None), 2)
        stop_rule = "跌破20日低点或修复位转弱时撤。"
        reenter_rule = "只有重新站回修复位并继续放量，才考虑延长持有。"
        position_hint = "小仓试探"
        if trend_state == "弱势修复":
            exp_high += 1.0

    else:
        if adaptive_score is not None and adaptive_score >= 75:
            action = "观察后跟随"
        stop_price = _round_or_none((latest_price * 0.95) if latest_price else None, 2)
        position_hint = "低仓位"

    if adaptive_score is not None:
        if adaptive_score >= 85:
            hold_high += 2
            exp_high += 2.0
            if style_name in ("steady_climb", "breakout") and market_env in ("强势", "偏强"):
                action = "重点跟随" if style_name == "steady_climb" else "顺势跟随"
                position_hint = "中高仓位"
        elif adaptive_score < 70:
            hold_high = max(hold_low, hold_high - 2)
            exp_high = max(exp_low + 0.5, exp_high - 2.0)
            if action in ("顺势跟随", "持有跟随"):
                action = "观望"
                position_hint = "轻仓"

    if weak_market and style_name in ("steady_climb", "breakout"):
        hold_high = min(hold_high, hold_low + 8)
        exp_high = min(exp_high, exp_low + 7.0)
        position_hint = "轻仓"
        if action in ("持有跟随", "重点跟随", "顺势跟随"):
            action = "谨慎跟随"

    hold_low = max(1, int(round(hold_low)))
    hold_high = max(hold_low, int(round(hold_high)))
    expected_low = _round_or_none(exp_low, 1)
    expected_high = _round_or_none(exp_high, 1)
    hold_period = f"{hold_low}-{hold_high}个交易日" if hold_low != hold_high else f"{hold_low}个交易日"
    expected_return = f"{expected_low}%~{expected_high}%"

    trend_piece = trend_state
    if change_5d is not None and change_20d is not None:
        trend_piece = f"{trend_state}（周线{_round_or_none(change_5d, 2)}%，月线{_round_or_none(change_20d, 2)}%）"
    elif today_change is not None:
        trend_piece = f"{trend_state}（今日{_round_or_none(today_change, 2)}%）"

    buy_entry_strategy = (
        f"买入口径:开盘不秒买，平开或小高开先看15分钟承接；盘中优先等回踩不破{stop_price}附近防守区，"
        f"重新放量走强后分批；承接确认要求高开不超过3%或高开后主动回踩，低开后能快速收回，"
        f"否则放弃当日买点。卖出纪律:跌破{stop_price}且不能收回先降风险，"
        f"到{expected_return}区间遇到放量滞涨或长上影分批兑现。"
    )
    if style_label:
        conclusion = (
            f"{style_label}/{trend_piece};入场:次日回踩承接确认再跟;"
            f"防守:{stop_price};止盈:{expected_return};持有:{hold_period};"
            f"仓位:{position_hint};风险:{risk_labels};放弃:{no_buy_condition};{buy_entry_strategy}"
        )
    else:
        conclusion = (
            f"{trend_piece};动作:{action};入场:承接确认;"
            f"防守:{stop_price};止盈:{expected_return};持有:{hold_period};"
            f"仓位:{position_hint};风险:{risk_labels};放弃:{no_buy_condition};{buy_entry_strategy}"
        )

    return {
        "action": action,
        "hold_period": hold_period,
        "expected_return": expected_return,
        "stop_price": stop_price,
        "stop_rule": stop_rule,
        "reenter_rule": reenter_rule,
        "position_hint": position_hint,
        "conclusion": conclusion,
        "entry_strategy_note": buy_entry_strategy,
        "risk_labels": risk_labels,
        "no_buy_condition": no_buy_condition,
    }
















def _score_short_term_risk(record, style_name=None):
    style_name = style_name or "generic"
    change_10d = _feature_value(record, "change_10d", FEATURE_UNIT_PERCENT_POINT) or 0.0
    change_20d = _feature_value(record, "change_20d", FEATURE_UNIT_PERCENT_POINT) or 0.0
    change_30d = _feature_value(record, "change_30d", FEATURE_UNIT_PERCENT_POINT) or 0.0
    today_change = _feature_value(record, "today_change", FEATURE_UNIT_PERCENT_POINT) or 0.0
    today_amp = _feature_value(record, "today_amp", FEATURE_UNIT_PERCENT_POINT) or 0.0
    amp_20d = _feature_value(record, "amp_20d", FEATURE_UNIT_PERCENT_POINT) or 0.0
    amp_30d = _feature_value(record, "amp_30d", FEATURE_UNIT_PERCENT_POINT) or 0.0
    vr_today = _feature_value(record, "vr_today", FEATURE_UNIT_MULTIPLE_RATIO) or 0.0
    vr_5d = _feature_value(record, "vr_5d", FEATURE_UNIT_MULTIPLE_RATIO) or 0.0
    volatility_ratio = _feature_value(record, "volatility_ratio_10_20", FEATURE_UNIT_MULTIPLE_RATIO) or 1.0
    upper_shadow_ratio = _feature_value(record, "upper_shadow_ratio", FEATURE_UNIT_POSITION_RATIO) or 0.0
    today_amount = _feature_value(record, "today_amount", FEATURE_UNIT_RAW_AMOUNT) or 0.0
    turnover_rate = _feature_value(record, "turnover_rate", FEATURE_UNIT_PERCENT_POINT) or 0.0
    turnover_vs_avg_5d = _feature_value(record, "turnover_vs_avg_5d", FEATURE_UNIT_MULTIPLE_RATIO) or 0.0

    risk = 0.0
    labels = []

    if style_name != "rebound":
        overheat = max(change_10d - 22.0, 0.0) * 0.16 + max(change_20d - 35.0, 0.0) * 0.10 + max(change_30d - 55.0, 0.0) * 0.08
        if overheat > 0:
            risk += overheat
            labels.append("涨幅过热")
    else:
        if change_20d > 15:
            risk += 3.0
            labels.append("修复不低")

    amp_base = amp_20d or amp_30d
    if amp_base > 0 and today_amp / amp_base >= 1.35:
        risk += min((today_amp / amp_base - 1.35) * 4.0, 4.0)
        labels.append("波动放大")
    if volatility_ratio >= 1.18:
        risk += min((volatility_ratio - 1.18) * 8.0, 4.0)
        labels.append("短波动升温")
    if upper_shadow_ratio >= 0.28:
        risk += min((upper_shadow_ratio - 0.28) * 8.0, 3.0)
        labels.append("上影压力")
    if style_name != "rebound" and change_30d >= 45 and vr_today < max(1.15, vr_5d * 0.95):
        risk += 2.5
        labels.append("量能未确认")
    if today_change >= 8.5:
        risk += 2.0
        labels.append("当日过急")
    if today_amount and today_amount < 80000000:
        risk += 1.5
        labels.append("流动性一般")
    if turnover_vs_avg_5d >= 1.8 and turnover_rate >= 8:
        risk += 1.5
        labels.append("换手过热")

    labels = list(dict.fromkeys(labels))
    no_buy_condition = "高开冲高但量能不足不追"
    if labels:
        if "上影压力" in labels:
            no_buy_condition = "不能快速收复上影区则放弃"
        elif "涨幅过热" in labels:
            no_buy_condition = "高开超过3%且不回踩承接则放弃"
        elif "流动性一般" in labels:
            no_buy_condition = "成交额未放大不参与"

    return {
        "risk_score": _round_or_none(min(max(risk, 0.0), 30.0), 2),
        "risk_labels": "、".join(labels[:3]) if labels else "无明显",
        "no_buy_condition": no_buy_condition,
    }


def _score_candidates(snapshot, horizon_profiles, style_horizon_profiles=None, apply_precision_filter=True):
    if snapshot.empty:
        return pd.DataFrame()

    style_horizon_profiles = style_horizon_profiles or {}
    records = []
    profile_weights = {
        horizon: float(profile.get("profile_weight") or 0)
        for horizon, profile in horizon_profiles.items()
    }
    total_profile_weight = sum(profile_weights.values())
    if total_profile_weight <= 0:
        total_profile_weight = float(len(profile_weights)) or 1.0
        profile_weights = {horizon: 1.0 for horizon in horizon_profiles}

    for row in snapshot.itertuples(index=False):
        record = row._asdict()
        trend_profile = _build_trend_state(record)
        hard_style = _resolve_candidate_style(record)
        result = {
            "stock_code": _normalize_scalar(record.get("stock_code")),
            "stock_name": _normalize_scalar(record.get("stock_name")),
            "industry_1": _normalize_scalar(record.get("industry_1")),
            "latest_price": _normalize_scalar(record.get("latest_price")),
            "today_change": _normalize_scalar(record.get("today_change")),
            "today_amp": _normalize_scalar(record.get("today_amp")),
            "amp_30d": _normalize_scalar(record.get("amp_30d")),
            "vr_today": _normalize_scalar(record.get("vr_today")),
            "vr_30d": _normalize_scalar(record.get("vr_30d")),
            "stock_rank": _normalize_scalar(record.get("stock_rank")),
            "today_amount": _normalize_scalar(record.get("today_amount")),
            "turnover_rate": _normalize_scalar(record.get("turnover_rate")),
            "ma20": _normalize_scalar(record.get("ma20")),
            "ma60": _normalize_scalar(record.get("ma60")),
            "low_20d": _normalize_scalar(record.get("low_20d")),
            "high_20d": _normalize_scalar(record.get("high_20d")),
            "change_5d": _normalize_scalar(record.get("change_5d")),
            "change_20d": _normalize_scalar(record.get("change_20d")),
            "change_30d": _normalize_scalar(record.get("change_30d")),
            "price_vs_ma20": _normalize_scalar(record.get("price_vs_ma20")),
            "style": None,
            "style_label": None,
            "hard_style": hard_style,
            "hard_style_label": STYLE_LABELS.get(hard_style) if hard_style else None,
            "style_gate_passed": bool(hard_style),
            "trend_state": trend_profile["state"],
            "trend_score": trend_profile["score"],
            "trend_detail": trend_profile["summary"],
            "market_change_5d": _normalize_scalar(record.get("market_change_5d")),
            "market_breadth_5d": _normalize_scalar(record.get("market_breadth_5d")),
            "score_5d": None,
            "score_10d": None,
            "adaptive_score": None,
            "risk_score": None,
            "risk_adjusted_score": None,
            "risk_labels": None,
            "no_buy_condition": None,
            "dominant_horizon": None,
            "dominant_family": None,
            "dominant_family_score": None,
            "dominant_style": None,
            "dominant_style_label": None,
            "coverage": None,
            "family_scores": {},
            "reason": None,
        }
        for detail_column in LONG_RUNWAY_RESULT_DETAIL_COLUMNS:
            if detail_column not in result:
                result[detail_column] = _normalize_scalar(record.get(detail_column))

        horizon_scores = {}
        horizon_details = {}
        horizon_family_scores = {}
        horizon_coverages = {}
        horizon_sources = {}

        for horizon_days, profile in horizon_profiles.items():
            generic_result = _score_one_profile(record, profile)
            selected_source = "generic"
            selected_result = generic_result

            best_style_name = None
            best_style_result = None
            for style_name in STYLE_PRIORITY:
                if not _record_matches_style(record, style_name):
                    continue
                style_profile = style_horizon_profiles.get(horizon_days, {}).get(style_name)
                if not style_profile:
                    continue

                style_result = _score_one_profile(record, style_profile)
                style_score = style_result["score"]
                if style_score is None:
                    continue

                if best_style_result is None or style_score > best_style_result["score"]:
                    best_style_name = style_name
                    best_style_result = style_result

            if best_style_result is not None:
                selected_source = best_style_name
                selected_result = best_style_result

            score = selected_result["score"]
            result[f"score_{horizon_days}d"] = score
            horizon_scores[horizon_days] = score
            horizon_details[horizon_days] = selected_result
            horizon_family_scores[horizon_days] = selected_result.get("family_scores", {})
            horizon_coverages[horizon_days] = selected_result.get("coverage", 0)
            horizon_sources[horizon_days] = selected_source
            result[f"source_{horizon_days}d"] = selected_source

        valid_horizon_scores = [
            (horizon_days, score, profile_weights.get(horizon_days, 1.0))
            for horizon_days, score in horizon_scores.items()
            if score is not None
        ]
        if valid_horizon_scores:
            weighted_sum = sum(score * weight for _, score, weight in valid_horizon_scores)
            weight_sum = sum(weight for _, _, weight in valid_horizon_scores)
            result["adaptive_score"] = _round_or_none(weighted_sum / weight_sum, 2)

            dominant_horizon = max(valid_horizon_scores, key=lambda item: item[1])[0]
            result["dominant_horizon"] = dominant_horizon
            result["dominant_family"] = horizon_details[dominant_horizon].get("top_family")
            result["dominant_family_score"] = horizon_details[dominant_horizon].get("top_family_score")
            result["coverage"] = _round_or_none(horizon_coverages.get(dominant_horizon), 4)
            result["family_scores"] = horizon_family_scores.get(dominant_horizon, {})
            dominant_source = horizon_sources.get(dominant_horizon)
            dominant_style_label = STYLE_LABELS.get(dominant_source) if dominant_source != "generic" else None
            result["style"] = dominant_source if dominant_source != "generic" else None
            result["style_label"] = dominant_style_label
            result["dominant_style"] = dominant_source if dominant_source != "generic" else None
            result["dominant_style_label"] = dominant_style_label
            result["reason"] = _build_reason(
                horizon_details[dominant_horizon],
                dominant_horizon,
                style_label=dominant_style_label,
                trend_state=trend_profile["state"],
            )
        else:
            result["coverage"] = 0
            result["reason"] = "历史样本不足，当前无法稳定评分。"

        risk_overlay = _score_short_term_risk(record, result.get("style"))
        result.update(risk_overlay)
        if result["adaptive_score"] is not None:
            result["risk_adjusted_score"] = _round_or_none(
                max(0.0, float(result["adaptive_score"]) - float(result["risk_score"] or 0.0)),
                2,
            )

        records.append(result)

    candidate_df = pd.DataFrame(records)
    if candidate_df.empty:
        return candidate_df

    if apply_precision_filter and style_horizon_profiles and ADAPTIVE_PRECISION_STYLES:
        candidate_df = candidate_df[candidate_df["style"].isin(ADAPTIVE_PRECISION_STYLES)].copy()
        if candidate_df.empty:
            return candidate_df

    if apply_precision_filter and style_horizon_profiles and ADAPTIVE_PRECISION_TREND_STATES:
        candidate_df = candidate_df[candidate_df["trend_state"].isin(ADAPTIVE_PRECISION_TREND_STATES)].copy()
        if candidate_df.empty:
            return candidate_df

    if apply_precision_filter:
        score_floor = ADAPTIVE_MIN_RISK_ADJUSTED_SCORE
        candidate_df = candidate_df[
            pd.to_numeric(candidate_df["risk_adjusted_score"], errors="coerce").fillna(0) >= score_floor
        ].copy()
        candidate_df = candidate_df[
            pd.to_numeric(candidate_df["risk_score"], errors="coerce").fillna(99) <= ADAPTIVE_MAX_RISK_SCORE
        ].copy()
        if candidate_df.empty:
            return candidate_df

    sort_columns = ["risk_adjusted_score", "adaptive_score"]
    for horizon_days in sorted(horizon_profiles.keys()):
        score_column = f"score_{horizon_days}d"
        if score_column in candidate_df.columns:
            sort_columns.append(score_column)
    if "stock_code" in candidate_df.columns:
        sort_columns.append("stock_code")

    ascending = [False] * (len(sort_columns) - 1) + [True] if len(sort_columns) > 1 else [False]
    candidate_df = candidate_df.sort_values(
        sort_columns,
        ascending=ascending,
        na_position="last",
    ).reset_index(drop=True)
    return candidate_df


def _merge_risk_label_text(primary, overlay):
    items = []
    for text in [primary, overlay]:
        if not text:
            continue
        for item in str(text).replace("，", "、").split("、"):
            item = item.strip()
            if item and item not in {"无明显", "无明显特殊池风险"}:
                items.append(item)
    items = list(dict.fromkeys(items))
    return "、".join(items[:5]) if items else "无明显"


def _apply_candidate_risk_overlay(
    candidate_df,
    history=None,
    trade_date=None,
    overlay_frame=None,
    include_external=False,
    filter_blocked=True,
    filter_downgraded=False,
    score_penalty_multiplier=1.0,
):
    if candidate_df.empty:
        return candidate_df

    enriched = risk_overlay.apply_risk_overlay_to_candidates(
        candidate_df,
        history=history,
        trade_date=trade_date,
        overlay_frame=overlay_frame,
        include_external=include_external,
        score_column="risk_adjusted_score",
        filter_blocked=filter_blocked,
        filter_downgraded=filter_downgraded,
        score_penalty_multiplier=score_penalty_multiplier,
    )
    if enriched.empty:
        return enriched

    enriched["risk_labels"] = enriched.apply(
        lambda row: _merge_risk_label_text(row.get("risk_labels"), row.get("risk_overlay_labels")),
        axis=1,
    )
    enriched["no_buy_condition"] = enriched.apply(
        lambda row: row.get("risk_overlay_action")
        if bool(row.get("risk_overlay_block_formal"))
        else row.get("no_buy_condition"),
        axis=1,
    )

    if "risk_adjusted_score" in enriched.columns:
        enriched = enriched[
            pd.to_numeric(enriched["risk_adjusted_score"], errors="coerce").fillna(0)
            >= ADAPTIVE_MIN_RISK_ADJUSTED_SCORE
        ].copy()
    if enriched.empty:
        return enriched

    sort_columns = ["risk_adjusted_score", "adaptive_score"]
    for horizon_days in HORIZON_DAYS:
        score_column = f"score_{horizon_days}d"
        if score_column in enriched.columns:
            sort_columns.append(score_column)
    if "stock_code" in enriched.columns:
        sort_columns.append("stock_code")
    ascending = [False] * (len(sort_columns) - 1) + [True] if len(sort_columns) > 1 else [False]
    return enriched.sort_values(sort_columns, ascending=ascending, na_position="last").reset_index(drop=True)


def _compose_model_note(profile_summary):
    style_summaries = profile_summary.get("top_styles", [])
    top_features = profile_summary.get("top_features", [])
    top_families = profile_summary.get("top_families", [])

    feature_text = "、".join(item["label"] for item in top_features[:5] if item.get("label"))
    family_text = "、".join(item["label"] for item in top_families[:3] if item.get("label"))
    style_text = "、".join(item["label"] for item in style_summaries[:3] if item.get("label"))

    if style_text and feature_text:
        if family_text:
            return f"历史上更容易走强的票已分化成 {style_text} 等风格；更敏感的特征更偏向 {feature_text}；若概括成结构，大致是 {family_text}。"
        return f"历史上更容易走强的票已分化成 {style_text} 等风格；更敏感的特征更偏向 {feature_text}。"

    if feature_text:
        if family_text:
            return f"历史上更容易走强的票，更敏感的特征更偏向 {feature_text}；若概括成风格，大致是 {family_text}。"
        return f"历史上更容易走强的票，更敏感的特征更偏向 {feature_text}。"
    if style_text:
        return f"历史上更容易走强的票已分化成 {style_text} 等风格。"
    if family_text:
        return f"历史上更容易走强的票，核心更偏向 {family_text}。"
    return "历史样本仍偏少，暂时只能给出弱模型摘要。"


def _backtest_horizon_profiles(history, top_candidate_count=TOP_CANDIDATE_COUNT, eval_step=1):
    trade_dates = sorted(history["last_data_date"].dropna().unique())
    normalized_eval_step = max(1, int(eval_step or 1))
    eval_trade_dates = trade_dates[::normalized_eval_step]
    overlay_frame = risk_overlay.build_special_pool_overlay(history, all_dates=True)
    metrics = {}

    for horizon_days in HORIZON_DAYS:
        metrics[horizon_days] = {
            "evaluated_days": 0,
            "universe_return_sum": 0.0,
            "universe_win_sum": 0.0,
            "universe_count": 0,
            "top_return_sum": 0.0,
            "top_win_sum": 0.0,
            "top_count": 0,
            "top_style_counts": {},
            "top_trend_state_counts": {},
            "style_return_stats": {},
            "market_regime_stats": {},
        }

    for eval_date in eval_trade_dates:
        snapshot = history[history["last_data_date"] == eval_date].copy()
        if snapshot.empty or len(snapshot) < MIN_DAILY_ROWS:
            continue
        eval_date_text = _to_date_text(eval_date)
        market_regime = risk_overlay.classify_market_regime(
            pd.to_numeric(snapshot.get("market_change_5d"), errors="coerce").mean(),
            pd.to_numeric(snapshot.get("market_breadth_5d"), errors="coerce").mean(),
        )

        eval_horizon_profiles = {}
        eval_style_profiles = {}
        for horizon_days in HORIZON_DAYS:
            horizon_mask = (
                history[f"forward_trade_date_{horizon_days}d"].notna()
                & (history[f"forward_trade_date_{horizon_days}d"] <= eval_date)
            )
            horizon_frame = history[horizon_mask].copy()
            eval_horizon_profiles[horizon_days] = _build_horizon_profile(horizon_frame, horizon_days)
            eval_style_profiles[horizon_days] = _build_style_horizon_profiles(horizon_frame, horizon_days)

        scored = _score_candidates(snapshot, eval_horizon_profiles, eval_style_profiles)
        scored = _apply_candidate_risk_overlay(
            scored,
            history=history,
            trade_date=eval_date_text,
            overlay_frame=overlay_frame,
            filter_blocked=True,
        )
        if scored.empty:
            continue

        for horizon_days in HORIZON_DAYS:
            future_col = f"forward_return_{horizon_days}d"
            if future_col not in snapshot.columns:
                continue

            universe_returns = pd.to_numeric(snapshot[future_col], errors="coerce").dropna()
            top_stock_codes = scored.head(int(top_candidate_count))["stock_code"].dropna().tolist()
            top_returns = pd.to_numeric(
                snapshot[snapshot["stock_code"].isin(top_stock_codes)][future_col],
                errors="coerce",
            ).dropna()
            if universe_returns.empty or top_returns.empty:
                continue

            top_subset = scored.head(int(top_candidate_count)).merge(
                snapshot[["stock_code", future_col]],
                on="stock_code",
                how="left",
            )

            metrics[horizon_days]["evaluated_days"] += 1
            metrics[horizon_days]["universe_return_sum"] += float(universe_returns.mean()) if not universe_returns.empty else 0.0
            metrics[horizon_days]["universe_win_sum"] += float((universe_returns > 0).mean()) if not universe_returns.empty else 0.0
            metrics[horizon_days]["universe_count"] += 1
            metrics[horizon_days]["top_return_sum"] += float(top_returns.mean()) if not top_returns.empty else 0.0
            metrics[horizon_days]["top_win_sum"] += float((top_returns > 0).mean()) if not top_returns.empty else 0.0
            metrics[horizon_days]["top_count"] += 1

            regime_stat = metrics[horizon_days]["market_regime_stats"].setdefault(
                market_regime,
                {
                    "label": risk_overlay.market_regime_label(market_regime),
                    "evaluated_days": 0,
                    "universe_return_sum": 0.0,
                    "universe_win_sum": 0.0,
                    "top_return_sum": 0.0,
                    "top_win_sum": 0.0,
                    "top_count": 0,
                },
            )
            regime_stat["evaluated_days"] += 1
            regime_stat["universe_return_sum"] += float(universe_returns.mean()) if not universe_returns.empty else 0.0
            regime_stat["universe_win_sum"] += float((universe_returns > 0).mean()) if not universe_returns.empty else 0.0
            regime_stat["top_return_sum"] += float(top_returns.mean()) if not top_returns.empty else 0.0
            regime_stat["top_win_sum"] += float((top_returns > 0).mean()) if not top_returns.empty else 0.0
            regime_stat["top_count"] += 1

            top_scored = scored.head(int(top_candidate_count))
            for style_name, count in top_scored["style"].fillna("generic").value_counts().items():
                metrics[horizon_days]["top_style_counts"][style_name] = metrics[horizon_days]["top_style_counts"].get(style_name, 0) + int(count)

            for trend_state, count in top_scored["trend_state"].fillna("未知").value_counts().items():
                metrics[horizon_days]["top_trend_state_counts"][trend_state] = metrics[horizon_days]["top_trend_state_counts"].get(trend_state, 0) + int(count)

            for style_name, group in top_subset.groupby(top_subset["style"].fillna("generic")):
                returns = pd.to_numeric(group[future_col], errors="coerce").dropna()
                if returns.empty:
                    continue
                style_stat = metrics[horizon_days]["style_return_stats"].setdefault(
                    style_name,
                    {"count": 0, "return_sum": 0.0, "win_sum": 0.0},
                )
                style_stat["count"] += int(len(returns))
                style_stat["return_sum"] += float(returns.sum())
                style_stat["win_sum"] += int((returns > 0).sum())

    result = {}
    for horizon_days, data in metrics.items():
        days = max(data["evaluated_days"], 1)
        result[horizon_days] = {
            "evaluated_days": data["evaluated_days"],
            "avg_universe_return": _round_or_none(data["universe_return_sum"] / days, 4) if data["evaluated_days"] else None,
            "avg_universe_win_rate": _round_or_none(data["universe_win_sum"] / days * 100, 2) if data["evaluated_days"] else None,
            "avg_top_return": _round_or_none(data["top_return_sum"] / days, 4) if data["evaluated_days"] else None,
            "avg_top_win_rate": _round_or_none(data["top_win_sum"] / days * 100, 2) if data["evaluated_days"] else None,
            "top_style_counts": dict(sorted(data["top_style_counts"].items(), key=lambda item: item[1], reverse=True)),
            "top_trend_state_counts": dict(sorted(data["top_trend_state_counts"].items(), key=lambda item: item[1], reverse=True)),
            "style_return_stats": {
                style_name: {
                    "count": style_stat["count"],
                    "avg_return": _round_or_none(style_stat["return_sum"] / style_stat["count"], 4) if style_stat["count"] else None,
                    "win_rate": _round_or_none(style_stat["win_sum"] / style_stat["count"] * 100, 2) if style_stat["count"] else None,
                }
                for style_name, style_stat in sorted(
                    data["style_return_stats"].items(),
                    key=lambda item: item[1]["return_sum"] / item[1]["count"] if item[1]["count"] else -999,
                    reverse=True,
                )
            },
            "market_regime_stats": {
                regime: {
                    "label": stat.get("label"),
                    "evaluated_days": stat["evaluated_days"],
                    "avg_universe_return": _round_or_none(stat["universe_return_sum"] / stat["evaluated_days"], 4)
                    if stat["evaluated_days"]
                    else None,
                    "avg_universe_win_rate": _round_or_none(stat["universe_win_sum"] / stat["evaluated_days"] * 100, 2)
                    if stat["evaluated_days"]
                    else None,
                    "avg_top_return": _round_or_none(stat["top_return_sum"] / stat["top_count"], 4)
                    if stat["top_count"]
                    else None,
                    "avg_top_win_rate": _round_or_none(stat["top_win_sum"] / stat["top_count"] * 100, 2)
                    if stat["top_count"]
                    else None,
                }
                for regime, stat in sorted(data["market_regime_stats"].items())
            },
        }

    return result


def _enrich_candidate_trade_plans(candidate_df, market_env):
    if candidate_df.empty:
        return candidate_df

    enriched_rows = []
    for row in candidate_df.to_dict("records"):
        trade_plan = _build_trade_plan(
            row,
            market_env,
            style_label=row.get("style_label") or row.get("dominant_style_label"),
            style_name=row.get("style") or row.get("dominant_style"),
            trend_state=row.get("trend_state"),
            adaptive_score=row.get("adaptive_score"),
        )
        row.update(trade_plan)
        row["recommendation_tier"] = RECOMMENDATION_TIER_OBSERVE
        row["recommendation_tier_reason"] = "个股已通过风格硬门槛、趋势结构和风险调整分，等待系统级健康验证后才可转为正式推荐。"
        enriched_rows.append(row)

    return pd.DataFrame(enriched_rows)


def _adaptive_health_policy(health_snapshot):
    mode = (health_snapshot or {}).get("mode")
    return ADAPTIVE_HEALTH_POLICIES.get(mode) or {
        "label": mode or "未知校验",
        "confidence_weight": 0.0,
        "max_pick_ratio": 0.0,
        "position_hint": None,
    }


def _replace_position_in_conclusion(conclusion, position_hint):
    if not conclusion or not position_hint:
        return conclusion
    text = str(conclusion)
    if "仓位:" in text:
        return re.sub(r"仓位:[^;；]*", f"仓位:{position_hint}", text, count=1)
    return f"{text};仓位:{position_hint}"


def _apply_adaptive_health_policy_to_candidates(candidate_df, health_snapshot):
    if candidate_df is None or candidate_df.empty:
        return candidate_df

    policy = _adaptive_health_policy(health_snapshot)
    confidence_weight = float(policy.get("confidence_weight") or 0.0)
    adjusted = candidate_df.copy()
    score_base = pd.to_numeric(
        adjusted.get("risk_adjusted_score", adjusted.get("adaptive_score")),
        errors="coerce",
    ).fillna(0)
    adjusted["health_adjusted_score"] = (score_base * confidence_weight).round(4)
    adjusted["health_confidence_weight"] = confidence_weight
    adjusted["health_mode"] = (health_snapshot or {}).get("mode")
    adjusted["health_mode_label"] = policy.get("label")

    position_hint = policy.get("position_hint")
    if position_hint:
        if "position_hint" in adjusted.columns:
            adjusted["position_hint_before_health"] = adjusted["position_hint"]
        adjusted["position_hint"] = position_hint
        if "conclusion" in adjusted.columns:
            adjusted["conclusion"] = adjusted["conclusion"].apply(
                lambda value: _replace_position_in_conclusion(value, position_hint)
            )

    sort_columns = ["health_adjusted_score", "risk_adjusted_score", "adaptive_score"]
    for horizon_days in HORIZON_DAYS:
        score_column = f"score_{horizon_days}d"
        if score_column in adjusted.columns:
            sort_columns.append(score_column)
    if "stock_code" in adjusted.columns:
        sort_columns.append("stock_code")
    sort_columns = [column for column in sort_columns if column in adjusted.columns]
    ascending = [False] * (len(sort_columns) - 1) + [True] if len(sort_columns) > 1 else [False]
    return adjusted.sort_values(sort_columns, ascending=ascending, na_position="last").reset_index(drop=True)


def _effective_adaptive_max_picks(max_picks, health_snapshot):
    base = int(max(1, max_picks or DAILY_ADAPTIVE_TOP_PICK_COUNT))
    policy = _adaptive_health_policy(health_snapshot)
    ratio = float(policy.get("max_pick_ratio") or 0.0)
    if ratio <= 0:
        return 0
    return max(1, min(base, int(math.ceil(base * ratio))))


def _summarize_industries(candidate_df):
    if candidate_df.empty or "industry_1" not in candidate_df.columns:
        return []

    industry_df = candidate_df.dropna(subset=["industry_1"]).copy()
    if industry_df.empty:
        return []

    summary_df = (
        industry_df.groupby("industry_1")
        .agg(
            stock_count=("stock_code", "count"),
            avg_score=("adaptive_score", "mean"),
            top_score=("adaptive_score", "max"),
            avg_score_5d=("score_5d", "mean"),
            avg_score_10d=("score_10d", "mean"),
        )
        .reset_index()
        .sort_values(["avg_score", "stock_count"], ascending=[False, False], na_position="last")
    )

    return summary_df.head(10).to_dict("records")


def _summarize_families(horizon_profiles):
    family_totals = {}
    feature_totals = {}

    for horizon_days, profile in horizon_profiles.items():
        horizon_weight = float(profile.get("profile_weight") or 0)
        if horizon_weight <= 0:
            continue

        for family_name, family_data in profile.get("families", {}).items():
            family_importance = float(family_data.get("importance") or 0)
            family_totals[family_name] = family_totals.get(family_name, 0) + family_importance * horizon_weight

            for item in family_data.get("items", []):
                key = (family_name, item["feature"])
                feature_totals.setdefault(
                    key,
                    {
                        "family": family_name,
                        "feature": item["feature"],
                        "label": item["label"],
                        "importance": 0.0,
                        "target_sum": 0.0,
                        "weight_sum": 0.0,
                    },
                )
                feature_totals[key]["importance"] += float(item["importance"]) * horizon_weight
                feature_totals[key]["target_sum"] += float(item["target_pct"]) * float(item["importance"]) * horizon_weight
                feature_totals[key]["weight_sum"] += float(item["importance"]) * horizon_weight

    total_family_importance = sum(family_totals.values())
    top_families = []
    for family_name, importance in sorted(family_totals.items(), key=lambda item: item[1], reverse=True):
        top_families.append(
            {
                "family": family_name,
                "label": FAMILY_LABELS.get(family_name, family_name),
                "importance": _round_or_none(importance, 4),
                "normalized_importance": _round_or_none(importance / total_family_importance, 4)
                if total_family_importance > 0
                else 0,
            }
        )

    top_features = []
    for item in sorted(feature_totals.values(), key=lambda data: data["importance"], reverse=True):
        weight_sum = item["weight_sum"]
        target_pct = item["target_sum"] / weight_sum if weight_sum > 0 else None
        top_features.append(
            {
                "family": item["family"],
                "feature": item["feature"],
                "label": item["label"],
                "importance": _round_or_none(item["importance"], 4),
                "target_pct": _round_or_none(target_pct, 4),
            }
        )

    return top_families, top_features[:10]


def _build_adaptive_validation_note(health_snapshot):
    if not health_snapshot:
        return ""

    mode = health_snapshot.get("mode")
    signal_health = health_snapshot.get("signal_health") or {}
    backtest_health = health_snapshot.get("backtest_health") or {}

    signal_window = int(ADAPTIVE_SIGNAL_HEALTH_LOOKBACK_TRADE_DAYS)
    backtest_window = int(ADAPTIVE_BACKTEST_HEALTH_LOOKBACK_TRADE_DAYS)

    if mode == "confirmed":
        parts = [
            "验证:实盘确认通过",
            (
                f"{signal_window}日实盘{signal_health.get('hold_days') or ADAPTIVE_SIGNAL_HEALTH['hold_days']}日"
                f"{signal_health.get('avg_return')}%/{signal_health.get('trade_win_rate')}%"
            ),
            (
                f"{backtest_window}日walk-forward{backtest_health.get('hold_days') or ADAPTIVE_BACKTEST_HEALTH['hold_days']}日"
                f"{backtest_health.get('avg_top_return')}%/超额{backtest_health.get('excess_return')}%"
            ),
        ]
        return "；".join(parts) + "。"

    if mode == "bootstrap":
        parts = [
            "验证:启动期通过",
            f"按{ADAPTIVE_HEALTH_MODEL_DISPLAY}降权",
            "仓位仅小仓验证",
            f"{signal_window}日实盘样本未满{ADAPTIVE_SIGNAL_HEALTH['min_evaluated_trades']}笔",
            (
                f"先按{backtest_window}日walk-forward"
                f"{backtest_health.get('hold_days') or ADAPTIVE_BACKTEST_HEALTH['hold_days']}日"
                f"{backtest_health.get('avg_top_return')}%/超额{backtest_health.get('excess_return')}%"
            ),
        ]
        return "；".join(parts) + "。"

    if mode == "confirmed_rejected":
        return "验证:双重校验未通过。"

    if mode == "bootstrap_rejected":
        return "验证:启动期校验未通过。"

    return "验证:未形成有效校验结论。"


def _build_adaptive_strategy_record(candidate, trade_date, rank=None, total=None, health_snapshot=None):
    conclusion = candidate.get("conclusion") or candidate.get("reason") or ""
    if conclusion and len(conclusion) > ADAPTIVE_NOTE_MAX_LENGTH:
        conclusion = conclusion[:ADAPTIVE_NOTE_MAX_LENGTH]

    selection_note = ""
    if rank:
        health_mode = (health_snapshot or {}).get("mode")
        if int(rank) == 1:
            if health_mode == "bootstrap":
                selection_note = "启动期推荐首选（小仓验证）。"
            elif health_mode == "confirmed":
                selection_note = "实盘确认推荐首选。"
            else:
                selection_note = "正式推荐首选。"
        else:
            total_text = f"/{int(total)}" if total else ""
            if health_mode == "bootstrap":
                selection_note = f"启动期推荐候选{int(rank)}{total_text}（小仓验证）。"
            elif health_mode == "confirmed":
                selection_note = f"实盘确认推荐候选{int(rank)}{total_text}。"
            else:
                selection_note = f"正式推荐候选{int(rank)}{total_text}。"

    validation_note = _build_adaptive_validation_note(health_snapshot)
    strategy_note = conclusion or f"{SHORT_TERM_MODEL_DISPLAY}候选。"
    if validation_note:
        strategy_note = f"{strategy_note} {validation_note}".strip()
    if selection_note:
        strategy_note = f"{selection_note}{strategy_note}"
    if len(strategy_note) > ADAPTIVE_NOTE_MAX_LENGTH:
        strategy_note = strategy_note[: ADAPTIVE_NOTE_MAX_LENGTH - 3] + "..."

    return {
        "trade_date": trade_date,
        "strategy_type": ADAPTIVE_STRATEGY_TYPE,
        "stock_code": _normalize_scalar(candidate.get("stock_code")),
        "stock_name": _normalize_scalar(candidate.get("stock_name")),
        "today_change": _normalize_scalar(candidate.get("today_change")),
        "industry": _normalize_scalar(candidate.get("industry_1")) or "",
        "change_30d": _normalize_scalar(candidate.get("change_30d")),
        "vr_today": _normalize_scalar(candidate.get("vr_today")),
        "vr_30d": _normalize_scalar(candidate.get("vr_30d")),
        "today_amp": _normalize_scalar(candidate.get("today_amp")),
        "amp_30d": _normalize_scalar(candidate.get("amp_30d")),
        "stock_rank": _normalize_scalar(candidate.get("stock_rank")),
        "today_amount": _normalize_scalar(candidate.get("today_amount")),
        "turnover_rate": _normalize_scalar(candidate.get("turnover_rate")),
        "strategy_note": strategy_note,
    }


def _clear_adaptive_strategy_rows(trade_date):
    if not trade_date:
        return
    strategy_types = [ADAPTIVE_STRATEGY_TYPE, *LEGACY_ADAPTIVE_STRATEGY_TYPES]
    for strategy_type in strategy_types:
        func.executeDelete("a_stock_strategy_result", {"trade_date": trade_date, "strategy_type": strategy_type})






































