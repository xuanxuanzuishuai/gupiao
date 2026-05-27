"""自适应选股公共基础层。

作用:
    集中放置模型常量、字段定义、数据库读取、历史样本准备、特征衍生、
    快照覆盖率检查、长跑缓存和通用工具函数。它不代表某一个策略，
    而是 short_term、long_runway、scoring、workflow 共同依赖的底座。

流程:
    先从 a_stock_analysis / a_stock_analysis_history 读取快照和历史；
    再标准化股票代码、行业、数值字段和日期；
    然后补齐市场/行业上下文、远期收益、百分位特征和风险覆盖辅助结果；
    上层策略拿到准备好的 DataFrame 后再进入画像训练和候选评分。
"""

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
import analysis_gu_piao_adaptive_risk_overlay_model as risk_overlay
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
SHORT_TERM_ADAPTIVE_BACKTEST_HEALTH_LOOKBACK_TRADE_DAYS = 90
LONG_RUNWAY_BACKTEST_HEALTH_LOOKBACK_TRADE_DAYS = 120
ADAPTIVE_BACKTEST_HEALTH_LOOKBACK_TRADE_DAYS = SHORT_TERM_ADAPTIVE_BACKTEST_HEALTH_LOOKBACK_TRADE_DAYS
ADAPTIVE_PERSIST_BACKTEST_TIMEOUT_SECONDS = 45 * 60
ADAPTIVE_BACKTEST_HEALTH = {
    "hold_days": 10,
    "min_evaluated_days": 30,
    "min_avg_top_return": 2.0,
    "min_avg_top_win_rate": 54.5,
    "min_excess_return": 0.5,
}
LONG_RUNWAY_BACKTEST_HEALTH = {
    "hold_days": 120,
    "min_evaluated_days": 5,
    "min_avg_top_return": 8.0,
    "min_avg_top_win_rate": 50.0,
    "min_excess_return": 2.0,
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
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LONG_RUNWAY_CACHE_DIR = PROJECT_ROOT / "cache" / "long_runway"
LONG_RUNWAY_CONTEXT_CACHE_PATH = LONG_RUNWAY_CACHE_DIR / "context.pkl"
ADAPTIVE_BACKTEST_CACHE_SCHEMA_VERSION = 2
ADAPTIVE_BACKTEST_CACHE_DIR = PROJECT_ROOT / "cache" / "adaptive_backtest"
ADAPTIVE_DAILY_DECISION_CACHE_SCHEMA_VERSION = 1
ADAPTIVE_DAILY_DECISION_CACHE_DIR = PROJECT_ROOT / "cache" / "adaptive_daily_decision"
ADAPTIVE_RISK_CONTEXT_TRADE_DAYS = int(getattr(risk_overlay, "DEFAULT_HISTORY_TAIL_DAYS", 260))
ADAPTIVE_DAILY_DECISION_CONTEXT_TRADE_DAYS = max(
    int(SHORT_TERM_LOOKBACK_TRADE_DAYS),
    int(ADAPTIVE_RISK_CONTEXT_TRADE_DAYS),
)
ADAPTIVE_BACKTEST_CONTEXT_TRADE_DAYS = (
    int(SHORT_TERM_ADAPTIVE_BACKTEST_HEALTH_LOOKBACK_TRADE_DAYS)
    + int(ADAPTIVE_DAILY_DECISION_CONTEXT_TRADE_DAYS)
)
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
ADAPTIVE_BACKTEST_HEALTH_PROFILES = {
    ADAPTIVE_STRATEGY_TYPE: {
        "label": SHORT_TERM_MODEL_DISPLAY,
        "lookback_trade_days": SHORT_TERM_ADAPTIVE_BACKTEST_HEALTH_LOOKBACK_TRADE_DAYS,
        "settings": ADAPTIVE_BACKTEST_HEALTH,
    },
    LONG_RUNWAY_STRATEGY_TYPE: {
        "label": LONG_RUNWAY_MODEL_DISPLAY,
        "lookback_trade_days": LONG_RUNWAY_BACKTEST_HEALTH_LOOKBACK_TRADE_DAYS,
        "settings": LONG_RUNWAY_BACKTEST_HEALTH,
    },
}
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


def _point_in_time_history_slice(history, as_of_date, tail_trade_days=None):
    if history is None or history.empty or "last_data_date" not in history.columns:
        return pd.DataFrame()

    as_of_ts = pd.to_datetime(as_of_date, errors="coerce")
    if pd.isna(as_of_ts):
        return pd.DataFrame()

    date_series = pd.to_datetime(history["last_data_date"], errors="coerce")
    frame = history.loc[date_series.notna() & date_series.le(as_of_ts)].copy()
    frame["last_data_date"] = date_series.loc[frame.index]
    if frame.empty:
        return frame

    if tail_trade_days:
        trade_dates = sorted(frame["last_data_date"].dropna().unique())
        if len(trade_dates) > int(tail_trade_days):
            cutoff_date = trade_dates[-int(tail_trade_days)]
            frame = frame[frame["last_data_date"] >= cutoff_date].copy()

    return frame.reset_index(drop=True)


def _adaptive_daily_decision_checksum(context_frame, as_of_date=None):
    # 日级缓存的文件名已经按模型版本、交易日、topN 和 schema 隔离。
    # 这里不再对大窗口做逐行哈希，否则“校验缓存”会比重新计算还慢。
    return 0


def _adaptive_daily_decision_cache_signature(context_frame, as_of_date, top_candidate_count, include_external=False):
    if include_external:
        return None
    if context_frame is None or context_frame.empty or "last_data_date" not in context_frame.columns:
        return None

    date_series = pd.to_datetime(context_frame["last_data_date"], errors="coerce").dropna()
    if date_series.empty:
        return None

    as_of_text = _to_date_text(as_of_date)
    if not as_of_text:
        return None

    snapshot_rows = int((date_series == pd.to_datetime(as_of_text)).sum())
    return {
        "schema_version": ADAPTIVE_DAILY_DECISION_CACHE_SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "as_of_date": as_of_text,
        "context_start_date": _to_date_text(date_series.min()),
        "context_end_date": _to_date_text(date_series.max()),
        "context_trade_days": int(date_series.nunique()),
        "context_rows": int(len(context_frame)),
        "snapshot_rows": snapshot_rows,
        "training_trade_days": int(SHORT_TERM_LOOKBACK_TRADE_DAYS),
        "risk_context_trade_days": int(ADAPTIVE_RISK_CONTEXT_TRADE_DAYS),
        "top_candidate_count": int(top_candidate_count or TOP_CANDIDATE_COUNT),
        "include_external": bool(include_external),
        "checksum": _adaptive_daily_decision_checksum(context_frame, as_of_date=as_of_text),
    }


def _adaptive_daily_decision_cache_path_for_key(as_of_date, top_candidate_count, include_external=False):
    if include_external:
        return None
    as_of_text = _to_date_text(as_of_date)
    if not as_of_text:
        return None
    normalized_top_count = int(top_candidate_count or TOP_CANDIDATE_COUNT)
    filename = (
        f"{MODEL_VERSION}_{as_of_text}_"
        f"top{normalized_top_count}_"
        f"schema{ADAPTIVE_DAILY_DECISION_CACHE_SCHEMA_VERSION}.pkl"
    )
    return ADAPTIVE_DAILY_DECISION_CACHE_DIR / filename


def _load_adaptive_daily_decision_cache(context_frame, as_of_date, top_candidate_count, include_external=False):
    cache_path = _adaptive_daily_decision_cache_path_for_key(
        as_of_date,
        top_candidate_count,
        include_external=include_external,
    )
    if not cache_path or not cache_path.exists():
        return None
    signature = _adaptive_daily_decision_cache_signature(
        context_frame,
        as_of_date,
        top_candidate_count,
        include_external=include_external,
    )
    if not signature:
        return None
    try:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as error:
        func.logInfo(f"{SHORT_TERM_MODEL_DISPLAY}日级决策缓存读取失败: {error}")
        return None
    if (payload or {}).get("signature") != signature:
        return None
    result = (payload or {}).get("result")
    if result:
        func.logInfo(f"{SHORT_TERM_MODEL_DISPLAY}日级决策缓存命中: {cache_path}")
    return result


def _save_adaptive_daily_decision_cache(context_frame, as_of_date, top_candidate_count, result, include_external=False):
    signature = _adaptive_daily_decision_cache_signature(
        context_frame,
        as_of_date,
        top_candidate_count,
        include_external=include_external,
    )
    cache_path = _adaptive_daily_decision_cache_path_for_key(
        as_of_date,
        top_candidate_count,
        include_external=include_external,
    )
    if not cache_path or not signature:
        return
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as handle:
            pickle.dump({"signature": signature, "result": result}, handle)
        func.logInfo(f"{SHORT_TERM_MODEL_DISPLAY}日级决策缓存已写入: {cache_path}")
    except Exception as error:
        func.logInfo(f"{SHORT_TERM_MODEL_DISPLAY}日级决策缓存写入失败: {error}")


def _adaptive_backtest_cache_signature(history, top_candidate_count, eval_step, eval_window_trade_days=None):
    if history is None or history.empty or "last_data_date" not in history.columns:
        return None

    date_series = pd.to_datetime(history["last_data_date"], errors="coerce").dropna()
    if date_series.empty:
        return None

    start_date = _to_date_text(date_series.min())
    end_date = _to_date_text(date_series.max())
    normalized_eval_step = max(1, int(eval_step or 1))
    normalized_top_count = int(top_candidate_count or TOP_CANDIDATE_COUNT)
    checksum_columns = [
        column
        for column in [
            "last_data_date",
            "stock_code",
            "latest_price",
            "today_change",
            "forward_return_5d",
            "forward_return_10d",
            "risk_score",
            "risk_adjusted_score",
        ]
        if column in history.columns
    ]
    checksum_frame = history[checksum_columns].copy()
    checksum = int(pd.util.hash_pandas_object(checksum_frame, index=False).sum()) & 0xFFFFFFFFFFFFFFFF
    return {
        "schema_version": ADAPTIVE_BACKTEST_CACHE_SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "start_date": start_date,
        "end_date": end_date,
        "trade_days": int(date_series.nunique()),
        "row_count": int(len(history)),
        "top_candidate_count": normalized_top_count,
        "eval_step": normalized_eval_step,
        "eval_window_trade_days": int(eval_window_trade_days or 0),
        "checksum": checksum,
    }


def _adaptive_backtest_cache_path(signature):
    if not signature:
        return None
    filename = (
        f"{signature['model_version']}_{signature['start_date']}_{signature['end_date']}_"
        f"{signature['trade_days']}d_{signature['row_count']}r_"
        f"top{signature['top_candidate_count']}_step{signature['eval_step']}_"
        f"win{signature['eval_window_trade_days']}_"
        f"{signature['checksum']:016x}.pkl"
    )
    return ADAPTIVE_BACKTEST_CACHE_DIR / filename


def _load_adaptive_backtest_cache(history, top_candidate_count, eval_step, eval_window_trade_days=None):
    signature = _adaptive_backtest_cache_signature(
        history,
        top_candidate_count,
        eval_step,
        eval_window_trade_days=eval_window_trade_days,
    )
    cache_path = _adaptive_backtest_cache_path(signature)
    if not cache_path or not cache_path.exists():
        return None
    try:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as error:
        func.logInfo(f"{SHORT_TERM_MODEL_DISPLAY}walk-forward缓存读取失败: {error}")
        return None
    if (payload or {}).get("signature") != signature:
        return None
    result = (payload or {}).get("result")
    if result:
        func.logInfo(f"{SHORT_TERM_MODEL_DISPLAY}walk-forward缓存命中: {cache_path}")
    return result


def _save_adaptive_backtest_cache(history, top_candidate_count, eval_step, result, eval_window_trade_days=None):
    signature = _adaptive_backtest_cache_signature(
        history,
        top_candidate_count,
        eval_step,
        eval_window_trade_days=eval_window_trade_days,
    )
    cache_path = _adaptive_backtest_cache_path(signature)
    if not cache_path:
        return
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as handle:
            pickle.dump({"signature": signature, "result": result}, handle)
        func.logInfo(f"{SHORT_TERM_MODEL_DISPLAY}walk-forward缓存已写入: {cache_path}")
    except Exception as error:
        func.logInfo(f"{SHORT_TERM_MODEL_DISPLAY}walk-forward缓存写入失败: {error}")


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


def _prepare_long_runway_history(history):
    _emit_runtime_status(f"{LONG_RUNWAY_MODEL_DISPLAY}缓存: 长周期上下文构建开始, rows={len(history)}")
    runway_history = _build_long_runway_context(history)
    _emit_runtime_status(f"{LONG_RUNWAY_MODEL_DISPLAY}缓存: 长周期上下文构建完成, rows={len(runway_history)}")
    runway_history = _build_long_runway_forward_returns(runway_history, horizons=LONG_RUNWAY_HORIZONS)
    _emit_runtime_status(f"{LONG_RUNWAY_MODEL_DISPLAY}缓存: 远期收益构建完成, rows={len(runway_history)}")
    runway_history = _add_percentile_columns(runway_history, LONG_RUNWAY_FEATURE_COLUMNS)
    _emit_runtime_status(f"{LONG_RUNWAY_MODEL_DISPLAY}缓存: 横截面分位构建完成, rows={len(runway_history)}")
    return runway_history.reset_index(drop=True)


def _long_runway_cache_config():
    return {
        "schema_version": LONG_RUNWAY_CACHE_SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "long_runway_model_version": LONG_RUNWAY_MODEL_VERSION,
        "horizons": tuple(LONG_RUNWAY_HORIZONS),
        "half_life_days": LONG_RUNWAY_HALF_LIFE_DAYS,
        "winner_ratio": LONG_RUNWAY_WINNER_RATIO,
        "loser_ratio": LONG_RUNWAY_LOSER_RATIO,
        "min_daily_rows": LONG_RUNWAY_MIN_DAILY_ROWS,
        "feature_columns": tuple(LONG_RUNWAY_FEATURE_COLUMNS),
    }


def _long_runway_cache_metadata(runway_history):
    latest_trade_date = None
    sample_start = None
    sample_end = None
    trade_days = 0
    row_count = 0
    if runway_history is not None and not runway_history.empty:
        dates = pd.to_datetime(runway_history["last_data_date"], errors="coerce").dropna()
        if not dates.empty:
            sample_start = dates.min().strftime("%Y-%m-%d")
            sample_end = dates.max().strftime("%Y-%m-%d")
            latest_trade_date = sample_end
            trade_days = int(dates.nunique())
        row_count = int(len(runway_history))
    return {
        **_long_runway_cache_config(),
        "latest_trade_date": latest_trade_date,
        "sample_start": sample_start,
        "sample_end": sample_end,
        "trade_days": trade_days,
        "row_count": row_count,
        "cache_path": str(LONG_RUNWAY_CONTEXT_CACHE_PATH),
    }


def _long_runway_cache_is_compatible(metadata):
    metadata = metadata or {}
    expected = _long_runway_cache_config()
    for key, expected_value in expected.items():
        actual_value = metadata.get(key)
        if isinstance(expected_value, tuple):
            actual_value = tuple(actual_value or ())
        if actual_value != expected_value:
            return False
    return True


def _load_long_runway_cache_payload():
    path = LONG_RUNWAY_CONTEXT_CACHE_PATH
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as error:
        _emit_runtime_status(f"{LONG_RUNWAY_MODEL_DISPLAY}缓存读取失败，将全量重建: {error}")
        return None

    if not _long_runway_cache_is_compatible((payload or {}).get("metadata")):
        _emit_runtime_status(f"{LONG_RUNWAY_MODEL_DISPLAY}缓存版本或模型口径变化，将全量重建")
        return None
    runway_history = (payload or {}).get("runway_history")
    if runway_history is None or runway_history.empty:
        _emit_runtime_status(f"{LONG_RUNWAY_MODEL_DISPLAY}缓存为空，将全量重建")
        return None
    return payload


def _save_long_runway_cache(runway_history, horizon_profiles=None):
    if runway_history is None or runway_history.empty:
        return None

    LONG_RUNWAY_CONTEXT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": _long_runway_cache_metadata(runway_history),
        "runway_history": runway_history.reset_index(drop=True),
        "horizon_profiles": horizon_profiles,
    }
    temp_path = LONG_RUNWAY_CONTEXT_CACHE_PATH.with_suffix(".tmp")
    with temp_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp_path, LONG_RUNWAY_CONTEXT_CACHE_PATH)
    return payload["metadata"]


def _merge_runway_history_cache(cached_history, new_context):
    frames = []
    if cached_history is not None and not cached_history.empty:
        frames.append(cached_history)
    if new_context is not None and not new_context.empty:
        frames.append(new_context)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged = _prepare_common_frame(merged, dedupe_keys=("last_data_date", "stock_code"))
    return merged.sort_values(["last_data_date", "stock_code"]).reset_index(drop=True)


def _refresh_long_runway_forward_returns(frame, affected_start_date):
    if frame is None or frame.empty or affected_start_date is None:
        return frame

    prepared = frame.copy()
    prepared["last_data_date"] = pd.to_datetime(prepared["last_data_date"], errors="coerce")
    affected_start = pd.to_datetime(affected_start_date, errors="coerce")
    if pd.isna(affected_start):
        return prepared

    price_columns = ["last_data_date", "stock_code", "latest_price", "today_change"]
    for column in ["today_open", "today_high", "today_low"]:
        if column in prepared.columns:
            price_columns.append(column)

    tail_prices = prepared[prepared["last_data_date"] >= affected_start][price_columns].copy()
    if tail_prices.empty:
        return prepared

    refreshed = _build_long_runway_forward_returns(tail_prices, horizons=LONG_RUNWAY_HORIZONS)
    update_columns = ["entry_change"]
    for horizon_days in LONG_RUNWAY_HORIZONS:
        update_columns.extend(
            [
                f"exit_date_{horizon_days}d",
                f"exit_change_{horizon_days}d",
                f"gross_return_{horizon_days}d",
                f"return_{horizon_days}d",
                f"forward_return_{horizon_days}d",
                f"forward_gross_return_{horizon_days}d",
                f"forward_trade_date_{horizon_days}d",
            ]
        )
    update_columns = [column for column in update_columns if column in refreshed.columns]

    keys = ["last_data_date", "stock_code"]
    base = prepared.set_index(keys)
    updates = refreshed.set_index(keys)[update_columns]
    for column in update_columns:
        if column not in base.columns:
            base[column] = pd.NA
    common_index = base.index.intersection(updates.index)
    if len(common_index) > 0:
        base.loc[common_index, update_columns] = updates.loc[common_index, update_columns]
    return base.reset_index().sort_values(["last_data_date", "stock_code"]).reset_index(drop=True)


def _attach_long_runway_historical_memory(runway_history, snapshot, as_of_date):
    if runway_history is None or runway_history.empty or snapshot is None or snapshot.empty:
        return snapshot

    as_of = pd.to_datetime(as_of_date, errors="coerce")
    if pd.isna(as_of):
        return snapshot

    memories = []
    for horizon_days in LONG_RUNWAY_HORIZONS:
        return_col = f"forward_return_{horizon_days}d"
        exit_col = f"forward_trade_date_{horizon_days}d"
        if return_col not in runway_history.columns or exit_col not in runway_history.columns:
            continue

        history_slice = runway_history[["stock_code", "last_data_date", exit_col, return_col]].copy()
        history_slice[return_col] = pd.to_numeric(history_slice[return_col], errors="coerce")
        history_slice[exit_col] = pd.to_datetime(history_slice[exit_col], errors="coerce")
        history_slice["last_data_date"] = pd.to_datetime(history_slice["last_data_date"], errors="coerce")
        history_slice = history_slice[
            history_slice[return_col].notna()
            & history_slice[exit_col].notna()
            & (history_slice[exit_col] <= as_of)
        ].copy()
        if history_slice.empty:
            continue

        idx = history_slice.groupby("stock_code")[return_col].idxmax()
        memory = history_slice.loc[idx, ["stock_code", "last_data_date", exit_col, return_col]].copy()
        memory = memory.rename(
            columns={
                "last_data_date": f"historical_max_signal_date_{horizon_days}d",
                exit_col: f"historical_max_exit_date_{horizon_days}d",
                return_col: f"historical_max_return_{horizon_days}d",
            }
        )
        memories.append(memory)

    if not memories:
        return snapshot

    enriched = snapshot.copy()
    for memory in memories:
        stale_columns = [column for column in memory.columns if column != "stock_code" and column in enriched.columns]
        if stale_columns:
            enriched = enriched.drop(columns=stale_columns)
        enriched = enriched.merge(memory, on="stock_code", how="left")
    return enriched


def _build_long_runway_history_full_cache(end_date=None):
    history = _load_history(
        end_date=end_date,
        columns=LONG_RUNWAY_FRAME_COLUMNS,
        chunked=True,
        progress_label=f"{LONG_RUNWAY_MODEL_DISPLAY}历史读取",
    )
    if history.empty:
        return history, {"cache_mode": "full_rebuild_failed", "cache_path": str(LONG_RUNWAY_CONTEXT_CACHE_PATH)}

    _emit_runtime_status(f"{LONG_RUNWAY_MODEL_DISPLAY}缓存: 全量构建开始, end_date={_to_date_text(end_date) or 'latest'}")
    runway_history = _prepare_long_runway_history(history)
    metadata = _save_long_runway_cache(runway_history, horizon_profiles=None)
    _emit_runtime_status(
        f"{LONG_RUNWAY_MODEL_DISPLAY}缓存: 全量构建完成, "
        f"latest_trade_date={(metadata or {}).get('latest_trade_date')}, "
        f"trade_days={(metadata or {}).get('trade_days')}, rows={(metadata or {}).get('row_count')}"
    )
    return runway_history, {"cache_mode": "full_rebuild", "cache_path": str(LONG_RUNWAY_CONTEXT_CACHE_PATH)}


def _load_or_update_long_runway_history_cache(end_date=None, rebuild_cache=False):
    requested_end = _to_date_text(end_date)
    if rebuild_cache:
        return _build_long_runway_history_full_cache(end_date=end_date)

    payload = _load_long_runway_cache_payload()
    if payload is None:
        return _build_long_runway_history_full_cache(end_date=end_date)

    cached_history = payload["runway_history"].copy()
    cached_history["last_data_date"] = pd.to_datetime(cached_history["last_data_date"], errors="coerce")
    cache_end = _to_date_text((payload.get("metadata") or {}).get("latest_trade_date"))
    if not requested_end:
        trade_dates = _load_history_trade_dates()
        requested_end = pd.to_datetime(trade_dates[-1]).strftime("%Y-%m-%d") if trade_dates else cache_end

    if requested_end == cache_end:
        _emit_runtime_status(
            f"{LONG_RUNWAY_MODEL_DISPLAY}缓存: 命中, trade_date={cache_end}, "
            f"rows={len(cached_history)}, path={LONG_RUNWAY_CONTEXT_CACHE_PATH}"
        )
        return cached_history, {
            "cache_mode": "hit",
            "cache_path": str(LONG_RUNWAY_CONTEXT_CACHE_PATH),
            "horizon_profiles": payload.get("horizon_profiles"),
        }

    if cache_end and requested_end and pd.to_datetime(requested_end) < pd.to_datetime(cache_end):
        _emit_runtime_status(
            f"{LONG_RUNWAY_MODEL_DISPLAY}缓存: 请求日期早于缓存日期，为避免未来函数，按目标日期全量重建: "
            f"requested={requested_end}, cache={cache_end}"
        )
        history = _load_history(
            end_date=requested_end,
            columns=LONG_RUNWAY_FRAME_COLUMNS,
            chunked=True,
            progress_label=f"{LONG_RUNWAY_MODEL_DISPLAY}历史读取",
        )
        if history.empty:
            return history, {"cache_mode": "historical_rebuild_failed", "cache_path": str(LONG_RUNWAY_CONTEXT_CACHE_PATH)}
        return _prepare_long_runway_history(history), {
            "cache_mode": "historical_rebuild_no_save",
            "cache_path": str(LONG_RUNWAY_CONTEXT_CACHE_PATH),
        }

    trade_dates = _load_history_trade_dates(end_date=requested_end)
    trade_date_texts = [pd.to_datetime(date).strftime("%Y-%m-%d") for date in trade_dates]
    if not trade_date_texts or cache_end not in trade_date_texts:
        _emit_runtime_status(f"{LONG_RUNWAY_MODEL_DISPLAY}缓存: 交易日断点缺失，将全量重建")
        return _build_long_runway_history_full_cache(end_date=requested_end)

    cache_end_index = trade_date_texts.index(cache_end)
    new_trade_dates = trade_date_texts[cache_end_index + 1 :]
    if not new_trade_dates:
        return cached_history, {
            "cache_mode": "hit",
            "cache_path": str(LONG_RUNWAY_CONTEXT_CACHE_PATH),
            "horizon_profiles": payload.get("horizon_profiles"),
        }

    raw_start_index = max(0, cache_end_index - LONG_RUNWAY_ROLLING_CONTEXT_TRADE_DAYS)
    raw_start_date = trade_date_texts[raw_start_index]
    _emit_runtime_status(
        f"{LONG_RUNWAY_MODEL_DISPLAY}缓存: 增量刷新开始, cache_end={cache_end}, "
        f"target={requested_end}, new_trade_days={len(new_trade_dates)}, raw_start={raw_start_date}"
    )
    tail_history = _load_history(
        start_date=raw_start_date,
        end_date=requested_end,
        columns=LONG_RUNWAY_FRAME_COLUMNS,
        chunked=True,
        progress_label=f"{LONG_RUNWAY_MODEL_DISPLAY}增量历史读取",
    )
    if tail_history.empty:
        _emit_runtime_status(f"{LONG_RUNWAY_MODEL_DISPLAY}缓存: 增量数据为空，沿用旧缓存")
        return cached_history, {
            "cache_mode": "incremental_empty",
            "cache_path": str(LONG_RUNWAY_CONTEXT_CACHE_PATH),
            "horizon_profiles": payload.get("horizon_profiles"),
        }

    tail_context = _prepare_long_runway_history(tail_history)
    first_new_date = pd.to_datetime(new_trade_dates[0])
    new_context = tail_context[tail_context["last_data_date"] >= first_new_date].copy()
    merged = _merge_runway_history_cache(cached_history, new_context)

    affected_start_offset = max(LONG_RUNWAY_HORIZONS) + DEFAULT_ENTRY_OFFSET_DAYS + len(new_trade_dates) + LONG_RUNWAY_FORWARD_REFRESH_BUFFER_DAYS
    affected_start_index = max(0, cache_end_index - affected_start_offset)
    affected_start_date = trade_date_texts[affected_start_index]
    merged = _refresh_long_runway_forward_returns(merged, affected_start_date)
    metadata = _save_long_runway_cache(merged, horizon_profiles=None)
    _emit_runtime_status(
        f"{LONG_RUNWAY_MODEL_DISPLAY}缓存: 增量刷新完成, "
        f"latest_trade_date={(metadata or {}).get('latest_trade_date')}, "
        f"trade_days={(metadata or {}).get('trade_days')}, rows={(metadata or {}).get('row_count')}, "
        f"affected_start={affected_start_date}"
    )
    return merged, {"cache_mode": "incremental", "cache_path": str(LONG_RUNWAY_CONTEXT_CACHE_PATH)}


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


def _adaptive_backtest_health_profile(strategy_type=None):
    strategy_type = strategy_type or ADAPTIVE_STRATEGY_TYPE
    return (
        ADAPTIVE_BACKTEST_HEALTH_PROFILES.get(strategy_type)
        or ADAPTIVE_BACKTEST_HEALTH_PROFILES[ADAPTIVE_STRATEGY_TYPE]
    )


def _adaptive_backtest_health_settings(strategy_type=None):
    return dict(_adaptive_backtest_health_profile(strategy_type).get("settings") or ADAPTIVE_BACKTEST_HEALTH)


def _adaptive_backtest_health_lookback_trade_days(strategy_type=None):
    profile = _adaptive_backtest_health_profile(strategy_type)
    return int(profile.get("lookback_trade_days") or ADAPTIVE_BACKTEST_HEALTH_LOOKBACK_TRADE_DAYS)


def _adaptive_backtest_context_trade_days(strategy_type=None):
    return (
        int(_adaptive_backtest_health_lookback_trade_days(strategy_type))
        + int(ADAPTIVE_DAILY_DECISION_CONTEXT_TRADE_DAYS)
    )


def _evaluate_adaptive_backtest_health(backtest_result, strategy_type=None):
    profile = _adaptive_backtest_health_profile(strategy_type)
    settings = _adaptive_backtest_health_settings(strategy_type)
    hold_days = int(settings["hold_days"])
    window_trade_days = _adaptive_backtest_health_lookback_trade_days(strategy_type)
    if not backtest_result or backtest_result.get("success") is False:
        reason = (backtest_result or {}).get("reason") or "backtest_failed"
        return {
            "enabled": False,
            "mode": "walk_forward_backtest",
            "strategy_type": strategy_type or ADAPTIVE_STRATEGY_TYPE,
            "strategy_label": profile.get("label"),
            "hold_days": hold_days,
            "window_trade_days": window_trade_days,
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
        "strategy_type": strategy_type or ADAPTIVE_STRATEGY_TYPE,
        "strategy_label": profile.get("label"),
        "hold_days": hold_days,
        "window_trade_days": window_trade_days,
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
    _ensure_columns(frame, NUMERIC_COLUMNS, math.nan)

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


def _build_long_runway_context(frame):
    _assert_feature_unit_registry_complete()
    prepared = frame.sort_values(["stock_code", "last_data_date"]).copy()
    grouped = prepared.groupby("stock_code")

    for horizon_days in (20, 60, 120, 252):
        prepared[f"ret_{horizon_days}d"] = grouped["latest_price"].pct_change(horizon_days) * 100

    prepared["ma120"] = (
        grouped["latest_price"].rolling(120, min_periods=120).mean().reset_index(level=0, drop=True)
    )
    prepared["ma240"] = (
        grouped["latest_price"].rolling(240, min_periods=240).mean().reset_index(level=0, drop=True)
    )

    for horizon_days in (60, 120, 240):
        price_rolling = grouped["latest_price"].rolling(horizon_days, min_periods=horizon_days)
        prepared[f"high_{horizon_days}d"] = (
            price_rolling.max().reset_index(level=0, drop=True)
        )
        prepared[f"low_{horizon_days}d"] = (
            price_rolling.min().reset_index(level=0, drop=True)
        )
        volume_rolling = grouped["today_vol"].rolling(horizon_days, min_periods=horizon_days).mean()
        prepared[f"volume_avg_{horizon_days}d"] = (
            volume_rolling.reset_index(level=0, drop=True).groupby(prepared["stock_code"]).shift(1)
        )

    prepared["price_vs_ma20"] = _safe_ratio(prepared["latest_price"], prepared["ma20"]) - 1
    prepared["price_vs_ma60"] = _safe_ratio(prepared["latest_price"], prepared["ma60"]) - 1
    prepared["price_vs_ma120"] = _safe_ratio(prepared["latest_price"], prepared["ma120"]) - 1
    prepared["price_vs_ma240"] = _safe_ratio(prepared["latest_price"], prepared["ma240"]) - 1
    prepared["ma60_vs_ma120"] = _safe_ratio(prepared["ma60"], prepared["ma120"]) - 1
    prepared["ma120_vs_ma240"] = _safe_ratio(prepared["ma120"], prepared["ma240"]) - 1

    prepared["close_to_60d_high"] = _safe_ratio(prepared["latest_price"], prepared["high_60d"])
    prepared["close_to_120d_high"] = _safe_ratio(prepared["latest_price"], prepared["high_120d"])
    prepared["close_to_240d_high"] = _safe_ratio(prepared["latest_price"], prepared["high_240d"])
    prepared["close_to_60d_low"] = _safe_ratio(prepared["latest_price"], prepared["low_60d"])
    prepared["close_to_120d_low"] = _safe_ratio(prepared["latest_price"], prepared["low_120d"])
    prepared["close_to_240d_low"] = _safe_ratio(prepared["latest_price"], prepared["low_240d"])

    prepared["volume_vs_avg_20d"] = _safe_ratio(prepared["today_vol"], prepared["vol_avg_20d"])
    prepared["volume_vs_avg_60d"] = _safe_ratio(prepared["today_vol"], prepared["volume_avg_60d"])
    prepared["volume_vs_avg_120d"] = _safe_ratio(prepared["today_vol"], prepared["volume_avg_120d"])

    prepared["range_position_120d"] = (
        (prepared["latest_price"] - prepared["low_120d"]) / (prepared["high_120d"] - prepared["low_120d"])
    )
    prepared["range_position_240d"] = (
        (prepared["latest_price"] - prepared["low_240d"]) / (prepared["high_240d"] - prepared["low_240d"])
    )

    market_ctx = (
        prepared.groupby("last_data_date")
        .agg(
            market_ret_20d=("ret_20d", "mean"),
            market_ret_60d=("ret_60d", "mean"),
            market_ret_120d=("ret_120d", "mean"),
            market_breadth_20d=("ret_20d", lambda s: (s > 0).mean() * 100),
            market_breadth_60d=("ret_60d", lambda s: (s > 0).mean() * 100),
            market_breadth_120d=("ret_120d", lambda s: (s > 0).mean() * 100),
        )
        .reset_index()
    )

    industry_ctx = (
        prepared.dropna(subset=["industry_1"])
        .groupby(["last_data_date", "industry_1"])
        .agg(
            industry_ret_20d=("ret_20d", "mean"),
            industry_ret_60d=("ret_60d", "mean"),
            industry_ret_120d=("ret_120d", "mean"),
            industry_breadth_20d=("ret_20d", lambda s: (s > 0).mean() * 100),
            industry_breadth_60d=("ret_60d", lambda s: (s > 0).mean() * 100),
            industry_breadth_120d=("ret_120d", lambda s: (s > 0).mean() * 100),
            industry_count=("stock_code", "count"),
        )
        .reset_index()
    )

    if not industry_ctx.empty:
        industry_ctx = industry_ctx.merge(market_ctx, on="last_data_date", how="left")
        industry_ctx["industry_alpha_20d"] = industry_ctx["industry_ret_20d"] - industry_ctx["market_ret_20d"]
        industry_ctx["industry_alpha_60d"] = industry_ctx["industry_ret_60d"] - industry_ctx["market_ret_60d"]
        industry_ctx["industry_alpha_120d"] = industry_ctx["industry_ret_120d"] - industry_ctx["market_ret_120d"]
        industry_ctx["industry_strength_rank_20d"] = industry_ctx.groupby("last_data_date")["industry_ret_20d"].rank(
            method="average",
            pct=True,
            ascending=True,
        )
        industry_ctx["industry_strength_rank_60d"] = industry_ctx.groupby("last_data_date")["industry_ret_60d"].rank(
            method="average",
            pct=True,
            ascending=True,
        )
        industry_ctx["industry_strength_rank_120d"] = industry_ctx.groupby("last_data_date")["industry_ret_120d"].rank(
            method="average",
            pct=True,
            ascending=True,
        )

        prepared = prepared.merge(
            industry_ctx[
                [
                    "last_data_date",
                    "industry_1",
                    "industry_ret_20d",
                    "industry_ret_60d",
                    "industry_ret_120d",
                    "industry_breadth_20d",
                    "industry_breadth_60d",
                    "industry_breadth_120d",
                    "industry_alpha_20d",
                    "industry_alpha_60d",
                    "industry_alpha_120d",
                    "industry_strength_rank_20d",
                    "industry_strength_rank_60d",
                    "industry_strength_rank_120d",
                ]
            ],
            on=["last_data_date", "industry_1"],
            how="left",
        )
    else:
        for column in [
            "industry_ret_20d",
            "industry_ret_60d",
            "industry_ret_120d",
            "industry_breadth_20d",
            "industry_breadth_60d",
            "industry_breadth_120d",
            "industry_alpha_20d",
            "industry_alpha_60d",
            "industry_alpha_120d",
            "industry_strength_rank_20d",
            "industry_strength_rank_60d",
            "industry_strength_rank_120d",
        ]:
            prepared[column] = pd.NA

    prepared = prepared.merge(market_ctx, on="last_data_date", how="left")
    prepared["stock_rank_score"] = 1 / (pd.to_numeric(prepared["stock_rank"], errors="coerce") + 1)

    derived_columns = [
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
        "price_vs_ma20",
        "price_vs_ma60",
        "price_vs_ma120",
        "price_vs_ma240",
        "ma60_vs_ma120",
        "ma120_vs_ma240",
        "close_to_60d_high",
        "close_to_120d_high",
        "close_to_240d_high",
        "close_to_60d_low",
        "close_to_120d_low",
        "close_to_240d_low",
        "volume_avg_60d",
        "volume_avg_120d",
        "volume_vs_avg_20d",
        "volume_vs_avg_60d",
        "volume_vs_avg_120d",
        "range_position_120d",
        "range_position_240d",
        "industry_ret_20d",
        "industry_ret_60d",
        "industry_ret_120d",
        "industry_breadth_20d",
        "industry_breadth_60d",
        "industry_breadth_120d",
        "industry_alpha_20d",
        "industry_alpha_60d",
        "industry_alpha_120d",
        "industry_strength_rank_20d",
        "industry_strength_rank_60d",
        "industry_strength_rank_120d",
        "stock_rank_score",
    ]
    for column in derived_columns:
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    prepared["range_position_120d"] = prepared["range_position_120d"].replace([math.inf, -math.inf], pd.NA)
    prepared["range_position_240d"] = prepared["range_position_240d"].replace([math.inf, -math.inf], pd.NA)

    return prepared


def _build_long_runway_forward_returns(frame, horizons=LONG_RUNWAY_HORIZONS):
    if frame is None or frame.empty:
        return frame

    prepared = frame.sort_values(["stock_code", "last_data_date"]).reset_index(drop=True).copy()
    cost_pct = (DEFAULT_FEE_BPS + DEFAULT_SLIPPAGE_BPS) * 2 / 100
    prepared["latest_price"] = pd.to_numeric(prepared["latest_price"], errors="coerce")
    prepared["today_change"] = pd.to_numeric(prepared["today_change"], errors="coerce")
    grouped = prepared.groupby("stock_code", sort=False)

    entry_offset = int(DEFAULT_ENTRY_OFFSET_DAYS)
    entry_close = grouped["latest_price"].shift(-entry_offset)
    prepared["entry_change"] = grouped["today_change"].shift(-entry_offset)

    for horizon_days in horizons:
        exit_offset = entry_offset + int(horizon_days)
        exit_close = grouped["latest_price"].shift(-exit_offset)
        exit_change_col = f"exit_change_{horizon_days}d"
        gross_return_col = f"gross_return_{horizon_days}d"
        return_col = f"return_{horizon_days}d"

        prepared[f"exit_date_{horizon_days}d"] = grouped["last_data_date"].shift(-exit_offset)
        prepared[exit_change_col] = grouped["today_change"].shift(-exit_offset)
        prepared[gross_return_col] = (exit_close - entry_close) / entry_close * 100
        prepared[return_col] = prepared[gross_return_col] - cost_pct

        entry_abs = pd.to_numeric(prepared["entry_change"], errors="coerce").abs()
        exit_abs = pd.to_numeric(prepared[exit_change_col], errors="coerce").abs()
        tradable = (entry_abs.isna() | (entry_abs < DEFAULT_LIMIT_PCT)) & (
            exit_abs.isna() | (exit_abs < DEFAULT_LIMIT_PCT)
        )
        blocked_mask = prepared[return_col].notna() & (~tradable)
        prepared.loc[blocked_mask, [return_col, gross_return_col]] = pd.NA
        prepared[f"forward_return_{horizon_days}d"] = prepared[return_col]
        prepared[f"forward_gross_return_{horizon_days}d"] = prepared[gross_return_col]
        prepared[f"forward_trade_date_{horizon_days}d"] = prepared[f"exit_date_{horizon_days}d"]

    return prepared


def _build_forward_returns(frame, horizons=HORIZON_DAYS):
    if frame is None or frame.empty:
        return frame

    prepared = frame.sort_values(["stock_code", "last_data_date"]).reset_index(drop=True).copy()
    cost_pct = (DEFAULT_FEE_BPS + DEFAULT_SLIPPAGE_BPS) * 2 / 100
    prepared["latest_price"] = pd.to_numeric(prepared["latest_price"], errors="coerce")
    prepared["today_change"] = pd.to_numeric(prepared["today_change"], errors="coerce")
    grouped = prepared.groupby("stock_code", sort=False)

    entry_offset = int(DEFAULT_ENTRY_OFFSET_DAYS)
    entry_close = grouped["latest_price"].shift(-entry_offset)
    entry_close = entry_close.where(entry_close > 0)
    prepared["entry_change"] = grouped["today_change"].shift(-entry_offset)
    entry_abs = pd.to_numeric(prepared["entry_change"], errors="coerce").abs()

    for horizon_days in horizons:
        exit_offset = entry_offset + int(horizon_days)
        exit_close = grouped["latest_price"].shift(-exit_offset)
        exit_change_col = f"exit_change_{horizon_days}d"
        gross_return_col = f"gross_return_{horizon_days}d"
        return_col = f"return_{horizon_days}d"

        prepared[f"exit_date_{horizon_days}d"] = grouped["last_data_date"].shift(-exit_offset)
        prepared[exit_change_col] = grouped["today_change"].shift(-exit_offset)
        prepared[gross_return_col] = (exit_close - entry_close) / entry_close * 100
        prepared[return_col] = prepared[gross_return_col] - cost_pct

        exit_abs = pd.to_numeric(prepared[exit_change_col], errors="coerce").abs()
        tradable = (entry_abs.isna() | (entry_abs < DEFAULT_LIMIT_PCT)) & (
            exit_abs.isna() | (exit_abs < DEFAULT_LIMIT_PCT)
        )
        prepared.loc[prepared[return_col].notna() & (~tradable), [return_col, gross_return_col]] = pd.NA
        prepared[gross_return_col] = prepared[gross_return_col].replace([math.inf, -math.inf], pd.NA)
        prepared[return_col] = prepared[return_col].replace([math.inf, -math.inf], pd.NA)
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


__all__ = [name for name in globals() if not name.startswith("__")]
