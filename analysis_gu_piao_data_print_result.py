import math
from datetime import datetime

import pandas as pd
import pymysql

import func
import analysis_gu_piao_risk_overlay as risk_overlay


def _emit_runtime_status(message):
    func.logInfo(message)
    print(message, flush=True)


def _normalize_trade_date_text(value):
    if value is None:
        return None

    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return str(value)

    return ts.strftime("%Y-%m-%d")


NUMERIC_COLUMNS = [
    "latest_price",
    "today_open",
    "today_high",
    "today_low",
    "high_20d",
    "low_20d",
    "high_60d",
    "low_60d",
    "high_120d",
    "low_120d",
    "ma5",
    "ma10",
    "ma20",
    "ma60",
    "ma120",
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
    "volatility_10d",
    "volatility_20d",
    "ma20_slope_5d",
    "ma60_slope_10d",
    "stock_rank",
]

INDUSTRY_MEAN_COLUMNS = ["change_3d", "change_5d", "change_10d", "change_30d"]
DETAIL_COLUMNS = [
    "signal_date",
    "strategy_type",
    "strategy_note",
    "stock_code",
    "stock_name",
    "industry_1",
    "market_env",
    "market_score",
    "signal_close",
    "today_change",
    "change_5d",
    "change_10d",
    "change_30d",
    "vr_today",
    "today_amp",
    "market_regime_segment",
    "market_regime_label",
    "special_pool_label",
    "risk_overlay_score",
    "risk_overlay_labels",
]
TEMPORAL_FEATURE_COLUMNS = [
    "prev_latest_price",
    "prev_high_20d",
    "prev_low_20d",
    "prev_today_vol",
    "prev_vr_today",
    "prev_vr_5d",
    "prev2_high_20d",
    "prev2_low_20d",
    "close_vs_prev_high_ratio",
    "prev_close_vs_prev2_high_ratio",
    "range_position_prev_20d",
    "prev_range_position_prev_20d",
    "breakout_buffer_pct",
    "trend_extension_ma20_ratio",
    "trend_extension_ma5_ratio",
    "volume_confirmation_score",
]
TOP_RATIO = 0.2
PRECISE_TOP_RATIO = 0.1
DEFAULT_HOLDING_DAYS = (1, 3, 5, 9, 10, 14, 16)
DEFAULT_ENTRY_OFFSET_DAYS = 1
DEFAULT_FEE_BPS = 3
DEFAULT_SLIPPAGE_BPS = 5
DEFAULT_LIMIT_PCT = 9.8
WEAK_RS_STRATEGY_TYPE = "weak_rs_follow"
BACKTEST_ENTRY_MODE_DEFINITIONS = {
    "next_close": {
        "label": "次日收盘",
        "return_template": "return_{hold_days}d",
        "gross_template": "gross_return_{hold_days}d",
        "description": "沿用原始口径，信号后第1个交易日收盘买入，用于和历史结果保持可比。",
    },
    "next_open": {
        "label": "次日开盘",
        "return_template": "return_next_open_{hold_days}d",
        "gross_template": "gross_return_next_open_{hold_days}d",
        "description": "按信号后第1个交易日开盘价买入，更接近可执行的集合竞价后入场。",
    },
    "next_avg": {
        "label": "次日均价",
        "return_template": "return_next_avg_{hold_days}d",
        "gross_template": "gross_return_next_avg_{hold_days}d",
        "description": "用次日OHLC均价近似日内平均成交成本，降低单一开收盘价格偏差。",
    },
    "acceptance": {
        "label": "承接确认",
        "return_template": "return_acceptance_{hold_days}d",
        "gross_template": "gross_return_acceptance_{hold_days}d",
        "description": "只有次日未出现无承接追高、收盘位置不弱时才视为成交，用次日均价近似等待确认后的成本。",
    },
}
EXCLUDED_NAME_PREFIXES = ("ST", "*ST", "S*ST", "退")
RETIRED_STRATEGIES = {"short_loose", "short_strict", "ma_cross", "steady_climb"}
MANAGED_DAILY_STRATEGY_TYPES = (
    "momentum_follow",
    WEAK_RS_STRATEGY_TYPE,
    "short_term_bet",
    "strong_breakout",
    "down_reversal",
    "ma_cross",
    "steady_climb",
    "short_loose",
    "short_strict",
)
STRATEGY_PRIORITY_BONUS = {
    "momentum_follow": 5.0,
    WEAK_RS_STRATEGY_TYPE: 3.0,
    "short_term_bet": 5.5,
    "strong_breakout": 2.5,
    "down_reversal": 2.0,
}
GLOBAL_PICK_LIMIT = {
    "strong": 3,
    "neutral": 2,
    "weak": 1,
    "unknown": 2,
}
GLOBAL_INDUSTRY_LIMIT = {
    "strong": 1,
    "neutral": 1,
    "weak": 1,
    "unknown": 1,
}
NOTE_MAX_LENGTH = 1600
MAX_OVERHEAT_SCORE = {
    "momentum_follow": 6.5,
    WEAK_RS_STRATEGY_TYPE: 4.8,
    "short_term_bet": 7.0,
    "strong_breakout": 6.5,
    "down_reversal": 4.5,
}
MIN_PRECISION_SCORE = {
    "strong": 24.0,
    "neutral": 22.0,
    "weak": 18.0,
    "unknown": 22.0,
}
MIN_RECOMMENDATION_COVERAGE_RATIO = 85.0
STRATEGY_HEALTH_LOOKBACK_TRADE_DAYS = 40
STRATEGY_HEALTH_SETTINGS = {
    "momentum_follow": {
        "hold_days": 10,
        "min_avg_return": 3.0,
        "min_trade_win_rate": 55.0,
        "min_evaluated_trades": 12,
    },
    WEAK_RS_STRATEGY_TYPE: {
        "hold_days": 10,
        "entry_mode": "acceptance",
        "min_avg_return": 1.5,
        "min_trade_win_rate": 52.0,
        "min_evaluated_trades": 6,
    },
    "short_term_bet": {
        "hold_days": 10,
        "min_avg_return": 2.0,
        "min_trade_win_rate": 55.0,
        "min_evaluated_trades": 4,
    },
    "strong_breakout": {
        "hold_days": 10,
        "min_avg_return": 2.0,
        "min_trade_win_rate": 50.0,
        "min_evaluated_trades": 5,
    },
    "down_reversal": {
        "hold_days": 10,
        "min_avg_return": 1.0,
        "min_trade_win_rate": 50.0,
        "min_evaluated_trades": 4,
    },
}
GENERATED_HEALTH_STRATEGY_TYPES = {WEAK_RS_STRATEGY_TYPE}
STRATEGY_NOTES = {
    "short_strict": "弱市里短期回撤后放量长阳反抽，但历史样本太少，已退出日常主推荐池。",
    "short_loose": "弱市里超跌后的放量修复，但历史回测偏弱，已退出主策略池，仅保留为观察模板。",
    "strong_breakout": "强势市场中，优先看前高确认后的放量突破，最近回测更偏 14-16 个交易日兑现。",
    "down_reversal": "弱势市场中，中期下跌后在低位区域出现反转，最近回测更像 16 个交易日左右的慢兑现机会。",
    "short_term_bet": "强势市场中，当日强势拉升叠加量能放大与均线多头，最近回测更像 14-16 个交易日的进攻信号。",
    "momentum_follow": "强势市场中，多周期涨幅与行业共振，最近回测更偏 16 个交易日左右跟随。",
    WEAK_RS_STRATEGY_TYPE: "弱势市场中，只观察相对大盘和行业更抗跌、均线仍抬升、低波动且不过热的趋势票，定位是小仓验证，不等同强势趋势跟随。",
    "ma_cross": "强势或震荡偏强市场里的均线贴近金叉，但近期稳定性不足，已退出日常主推荐池。",
    "steady_climb": "缓慢抬升、均线多头、行业相对市场更强，但当前样本偏少，已退出日常主推荐池。",
}
STRATEGY_DISPLAY_NAMES = {
    "momentum_follow": "趋势跟随",
    WEAK_RS_STRATEGY_TYPE: "弱势抗跌趋势",
    "short_term_bet": "强势进攻",
    "strong_breakout": "强突破",
    "down_reversal": "低位反转",
    "ma_cross": "均线金叉",
    "steady_climb": "稳步慢涨",
    "short_loose": "超跌修复",
    "short_strict": "严格反抽",
}
STRATEGY_ALLOWED_MARKET_REGIMES = {
    "momentum_follow": {"strong"},
    WEAK_RS_STRATEGY_TYPE: {"weak"},
    "short_term_bet": {"strong"},
    "strong_breakout": {"strong"},
    "down_reversal": {"weak"},
    "ma_cross": {"strong", "neutral"},
    "short_loose": {"weak"},
    "short_strict": {"weak"},
}
STRATEGY_HEALTH_REASON_LABELS = {
    "insufficient_trades": "实盘样本不足",
    "avg_return_below_threshold": "平均收益未达标",
    "win_rate_below_threshold": "胜率未达标",
}
STRATEGY_EXECUTION_PROFILES = {
    "momentum_follow": {
        "hold_days": 10,
        "max_stop_pct": 7.2,
        "min_rr": 1.45,
        "base_target_pct": 9.5,
        "action": "趋势跟随",
        "entry_hint": "次日优先看平开到小高开承接，不追连续加速。",
    },
    WEAK_RS_STRATEGY_TYPE: {
        "hold_days": 10,
        "max_stop_pct": 6.8,
        "min_rr": 1.2,
        "base_target_pct": 6.8,
        "action": "弱势抗跌趋势",
        "entry_hint": "只做小仓验证，次日必须相对大盘抗跌且分时承接稳定。",
    },
    "short_term_bet": {
        "hold_days": 10,
        "max_stop_pct": 8.2,
        "min_rr": 1.35,
        "base_target_pct": 10.5,
        "action": "强势进攻",
        "entry_hint": "高开过大先等回踩，避免抢情绪顶。",
    },
    "strong_breakout": {
        "hold_days": 10,
        "max_stop_pct": 7.0,
        "min_rr": 1.4,
        "base_target_pct": 8.5,
        "action": "突破跟随",
        "entry_hint": "不追过大高开，优先看突破位附近承接。",
    },
    "down_reversal": {
        "hold_days": 10,
        "max_stop_pct": 8.5,
        "min_rr": 1.15,
        "base_target_pct": 6.5,
        "action": "低位试错",
        "entry_hint": "只适合轻仓试错，次日转弱要快撤。",
    },
}


def _ordered_unique(values):
    result = []
    seen = set()

    for value in values:
        item = value.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)

    return result


def _format_strategy_name(strategy_type):
    display_name = STRATEGY_DISPLAY_NAMES.get(strategy_type)
    if display_name:
        return f"{display_name}({strategy_type})"
    return str(strategy_type)


def _format_strategy_names(strategy_types):
    names = [_format_strategy_name(strategy_type) for strategy_type in _ordered_unique(strategy_types)]
    return "、".join(names) if names else "无"


def _health_failure_reason_text(health_info):
    reasons = (health_info or {}).get("failure_reasons") or []
    labels = [
        STRATEGY_HEALTH_REASON_LABELS.get(reason, reason)
        for reason in reasons
    ]
    return "、".join(labels) if labels else "健康度未达标"


def _allowed_daily_strategies_for_market(market_regime):
    return [
        strategy_type
        for strategy_type, allowed_regimes in STRATEGY_ALLOWED_MARKET_REGIMES.items()
        if strategy_type not in RETIRED_STRATEGIES and market_regime in allowed_regimes
    ]


def _count_final_strategy_candidates(strategies):
    total = 0
    for strategy_df in (strategies or {}).values():
        if strategy_df is None or strategy_df.empty or "stock_code" not in strategy_df.columns:
            continue
        total += int(strategy_df["stock_code"].dropna().nunique())
    return total


def _build_no_daily_recommendation_message(market_env, market_regime, strategies, strategy_health):
    final_candidate_count = _count_final_strategy_candidates(strategies)
    enabled_strategies = [
        strategy_type
        for strategy_type, health_info in (strategy_health or {}).items()
        if health_info.get("enabled", True)
    ]
    market_allowed = _allowed_daily_strategies_for_market(market_regime)
    market_allowed_enabled = [
        strategy_type
        for strategy_type in enabled_strategies
        if strategy_type in market_allowed
    ]
    market_blocked_enabled = [
        strategy_type
        for strategy_type in enabled_strategies
        if strategy_type not in market_allowed
    ]
    market_allowed_disabled = [
        strategy_type
        for strategy_type in market_allowed
        if not (strategy_health.get(strategy_type) or {}).get("enabled", True)
    ]

    parts = [f"固定策略无正式推荐: 市场={market_env}"]
    if market_allowed:
        parts.append(f"当前市场只放行{_format_strategy_names(market_allowed)}")
    else:
        parts.append("当前市场没有启用的固定正式策略")

    if market_allowed_enabled:
        parts.append(
            f"{_format_strategy_names(market_allowed_enabled)}已通过健康度，但形态精筛/风险覆盖后没有可落库候选"
        )

    for strategy_type in market_allowed_disabled:
        parts.append(
            f"{_format_strategy_name(strategy_type)}健康度未达标"
            f"({_health_failure_reason_text(strategy_health.get(strategy_type))})"
        )

    if market_blocked_enabled:
        parts.append(
            f"健康达标但不适合当前市场的策略={_format_strategy_names(market_blocked_enabled)}"
        )

    parts.append(f"最终可落库候选={final_candidate_count}")
    return "；".join(parts) + "。"


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


def _top_pick_count(total_count, ratio=TOP_RATIO):
    return max(1, math.ceil(total_count * ratio))


def _round_or_none(value, digits=4):
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def _estimate_amount_from_price_volume(price_series, volume_series, stock_code_series=None):
    price_series = pd.to_numeric(price_series, errors="coerce")
    volume_series = pd.to_numeric(volume_series, errors="coerce")
    if stock_code_series is None:
        stock_code_series = pd.Series("", index=price_series.index, dtype="object")
    stock_code_series = stock_code_series.fillna("").astype(str).str.strip()

    volume_unit_multiplier = pd.Series(100.0, index=price_series.index, dtype="float64")
    volume_unit_multiplier = volume_unit_multiplier.mask(
        stock_code_series.str.startswith(("688", "689")),
        1.0,
    )
    estimated_amount_if_hand = price_series * volume_series * 100
    volume_unit_multiplier = volume_unit_multiplier.mask(
        estimated_amount_if_hand >= 200000000000,
        1.0,
    )
    return price_series * volume_series * volume_unit_multiplier


def _get_strategy_note(strategy_type):
    return STRATEGY_NOTES.get(strategy_type, "基于量价与均线结构识别的候选形态。")


def _safe_float(value):
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _clip_value(value, lower=None, upper=None):
    if value is None or pd.isna(value):
        return None
    result = float(value)
    if lower is not None:
        result = max(lower, result)
    if upper is not None:
        result = min(upper, result)
    return result


def _choose_stop_price(latest_price, candidates, fallback_ratio=0.93):
    valid_candidates = []
    for candidate in candidates:
        candidate_value = _safe_float(candidate)
        if candidate_value is None or candidate_value <= 0:
            continue
        if latest_price and candidate_value >= latest_price:
            continue
        valid_candidates.append(candidate_value)

    if valid_candidates:
        return max(valid_candidates)
    if latest_price:
        return latest_price * fallback_ratio
    return None


def _resolve_stop_price(row, strategy_type):
    latest_price = _safe_float(row.get("latest_price"))
    if latest_price is None or latest_price <= 0:
        return None

    if strategy_type == "strong_breakout":
        candidates = [
            _safe_float(row.get("prev_high_20d")) * 0.99 if _safe_float(row.get("prev_high_20d")) else None,
            _safe_float(row.get("ma20")) * 0.99 if _safe_float(row.get("ma20")) else None,
            _safe_float(row.get("today_low")) * 0.995 if _safe_float(row.get("today_low")) else None,
        ]
        return _choose_stop_price(latest_price, candidates, fallback_ratio=0.94)

    if strategy_type == "short_term_bet":
        candidates = [
            _safe_float(row.get("ma10")) * 0.985 if _safe_float(row.get("ma10")) else None,
            _safe_float(row.get("ma20")) * 0.98 if _safe_float(row.get("ma20")) else None,
            _safe_float(row.get("today_low")) * 0.995 if _safe_float(row.get("today_low")) else None,
        ]
        return _choose_stop_price(latest_price, candidates, fallback_ratio=0.93)

    if strategy_type == "momentum_follow":
        candidates = [
            _safe_float(row.get("ma10")) * 0.99 if _safe_float(row.get("ma10")) else None,
            _safe_float(row.get("ma20")) * 0.985 if _safe_float(row.get("ma20")) else None,
            _safe_float(row.get("today_low")) * 0.995 if _safe_float(row.get("today_low")) else None,
        ]
        return _choose_stop_price(latest_price, candidates, fallback_ratio=0.94)

    if strategy_type == WEAK_RS_STRATEGY_TYPE:
        candidates = [
            _safe_float(row.get("ma20")) * 0.99 if _safe_float(row.get("ma20")) else None,
            _safe_float(row.get("ma10")) * 0.985 if _safe_float(row.get("ma10")) else None,
            _safe_float(row.get("today_low")) * 0.995 if _safe_float(row.get("today_low")) else None,
        ]
        return _choose_stop_price(latest_price, candidates, fallback_ratio=0.945)

    if strategy_type == "down_reversal":
        candidates = [
            _safe_float(row.get("today_low")) * 0.992 if _safe_float(row.get("today_low")) else None,
            _safe_float(row.get("low_20d")) * 0.99 if _safe_float(row.get("low_20d")) else None,
            _safe_float(row.get("ma5")) * 0.99 if _safe_float(row.get("ma5")) else None,
        ]
        return _choose_stop_price(latest_price, candidates, fallback_ratio=0.92)

    return _choose_stop_price(latest_price, [], fallback_ratio=0.93)


def _resolve_target_pct(row, strategy_type):
    profile = STRATEGY_EXECUTION_PROFILES.get(strategy_type, {})
    base_target_pct = float(profile.get("base_target_pct", 8.0))
    today_change = _safe_float(row.get("today_change")) or 0.0
    change_10d = _safe_float(row.get("change_10d")) or 0.0
    change_20d = _safe_float(row.get("change_20d")) or 0.0
    breakout_buffer_pct = _safe_float(row.get("breakout_buffer_pct")) or 0.0
    ind_alpha_change_10d = _safe_float(row.get("ind_alpha_change_10d")) or 0.0

    if strategy_type == "strong_breakout":
        return _clip_value(base_target_pct + min(breakout_buffer_pct, 3.0) * 0.5 + min(ind_alpha_change_10d, 5.0) * 0.4, 6.0, 14.0)
    if strategy_type == "short_term_bet":
        return _clip_value(base_target_pct + max(today_change - 4.0, 0.0) * 0.6 + min(ind_alpha_change_10d, 5.0) * 0.25, 7.0, 15.0)
    if strategy_type == "momentum_follow":
        return _clip_value(base_target_pct + min(change_10d, 20.0) * 0.18 + min(ind_alpha_change_10d, 5.0) * 0.35, 6.0, 13.0)
    if strategy_type == WEAK_RS_STRATEGY_TYPE:
        return _clip_value(base_target_pct + min(change_10d, 12.0) * 0.10 + min(ind_alpha_change_10d, 5.0) * 0.25, 4.5, 10.5)
    if strategy_type == "down_reversal":
        return _clip_value(base_target_pct + max(-change_20d, 0.0) * 0.08 + today_change * 0.2, 4.0, 10.0)
    return _clip_value(base_target_pct, 4.0, 12.0)


def _resolve_position_hint(strategy_type, market_regime, execution_rr, stop_loss_pct, precision_score):
    execution_rr = _safe_float(execution_rr) or 0.0
    stop_loss_pct = _safe_float(stop_loss_pct) or 99.0
    precision_score = _safe_float(precision_score) or 0.0

    if market_regime == "weak" and strategy_type != "down_reversal":
        return "低"
    if strategy_type == "down_reversal":
        if execution_rr >= 1.4 and stop_loss_pct <= 7.5:
            return "中"
        return "低"
    if execution_rr >= 1.9 and stop_loss_pct <= 6.0 and precision_score >= 24:
        return "高"
    if execution_rr >= 1.45 and stop_loss_pct <= 8.0 and precision_score >= 18:
        return "中"
    return "低"


def _format_price(value):
    numeric = _safe_float(value)
    if numeric is None or numeric <= 0:
        return "--"
    return f"{numeric:.2f}".rstrip("0").rstrip(".")


def _format_pct(value):
    numeric = _safe_float(value)
    if numeric is None:
        return "--"
    return f"{numeric:.1f}"


def _resolve_overheat_score(row, strategy_type):
    if strategy_type == "down_reversal":
        return 0.0

    change_10d = _safe_float(row.get("change_10d")) or 0.0
    change_20d = _safe_float(row.get("change_20d")) or 0.0
    change_30d = _safe_float(row.get("change_30d")) or 0.0
    today_amp = _safe_float(row.get("today_amp")) or 0.0
    amp_20d = _safe_float(row.get("amp_20d")) or 0.0
    turnover_rate = _safe_float(row.get("turnover_rate")) or 0.0
    turnover_avg_5d = _safe_float(row.get("turnover_avg_5d")) or 0.0
    volatility_ratio = _safe_float(row.get("volatility_10_20_ratio")) or 1.0
    upper_shadow_ratio = _safe_float(row.get("upper_shadow_ratio")) or 0.0
    vr_today = _safe_float(row.get("vr_today")) or 0.0

    score = 0.0
    score += max(change_10d - 18.0, 0.0) * 0.10
    score += max(change_20d - 28.0, 0.0) * 0.08
    score += max(change_30d - 45.0, 0.0) * 0.07
    if amp_20d > 0:
        score += max(today_amp / amp_20d - 1.2, 0.0) * 2.0
    score += max(volatility_ratio - 1.12, 0.0) * 4.0
    if turnover_avg_5d > 0:
        score += max(turnover_rate / turnover_avg_5d - 1.5, 0.0) * 1.5
    score += max(upper_shadow_ratio - 0.25, 0.0) * 5.0
    if change_30d >= 60 and vr_today < 1.15:
        score += 1.5
    return round(max(0.0, score), 2)


def _build_risk_labels(row, strategy_type):
    labels = []
    change_30d = _safe_float(row.get("change_30d")) or 0.0
    change_20d = _safe_float(row.get("change_20d")) or 0.0
    today_amp = _safe_float(row.get("today_amp")) or 0.0
    amp_20d = _safe_float(row.get("amp_20d")) or 0.0
    volatility_ratio = _safe_float(row.get("volatility_10_20_ratio")) or 1.0
    upper_shadow_ratio = _safe_float(row.get("upper_shadow_ratio")) or 0.0
    vr_today = _safe_float(row.get("vr_today")) or 0.0
    today_amount = _safe_float(row.get("today_amount")) or 0.0
    execution_rr = _safe_float(row.get("execution_rr")) or 0.0
    stop_loss_pct = _safe_float(row.get("stop_loss_pct")) or 0.0
    overheat_score = _safe_float(row.get("overheat_score")) or 0.0

    if change_30d >= 60 or change_20d >= 40:
        labels.append("涨幅偏高")
    if amp_20d > 0 and today_amp / amp_20d >= 1.35:
        labels.append("波动放大")
    if volatility_ratio >= 1.18:
        labels.append("短波动升温")
    if upper_shadow_ratio >= 0.28:
        labels.append("上影压力")
    if strategy_type != "down_reversal" and change_30d >= 45 and vr_today < 1.15:
        labels.append("量能未确认")
    if today_amount and today_amount < 80000000:
        labels.append("流动性一般")
    if execution_rr and execution_rr < 1.5:
        labels.append("赔率一般")
    if stop_loss_pct >= 8:
        labels.append("止损偏宽")
    if overheat_score >= MAX_OVERHEAT_SCORE.get(strategy_type, 6.5):
        labels.append("过热临界")

    return "、".join(labels[:3]) if labels else "无明显"


def _resolve_entry_trigger(row, strategy_type):
    entry_hint = row.get("entry_hint") or STRATEGY_EXECUTION_PROFILES.get(strategy_type, {}).get("entry_hint") or ""
    if strategy_type == "strong_breakout":
        return "次日不追高,回踩突破位承接再跟"
    if strategy_type == "short_term_bet":
        return "高开过大等回踩,量能不缩再试"
    if strategy_type == "momentum_follow":
        return "平开小高开优先,不追连续加速"
    if strategy_type == WEAK_RS_STRATEGY_TYPE:
        return "只低吸承接确认,弱于大盘不试"
    if strategy_type == "down_reversal":
        return "轻仓试错,低点不破且收回均线"
    return entry_hint or "等待承接确认"


def _resolve_skip_condition(row, strategy_type):
    stop_price = _format_price(row.get("stop_price"))
    if strategy_type == "down_reversal":
        return f"跌破{stop_price}或反抽无量放弃"
    if strategy_type == WEAK_RS_STRATEGY_TYPE:
        return f"跌破{stop_price}或相对大盘不再抗跌放弃"
    if (_safe_float(row.get("overheat_score")) or 0.0) >= MAX_OVERHEAT_SCORE.get(strategy_type, 6.5):
        return "高开冲量不足不追"
    return f"失守{stop_price}或放量滞涨放弃"


def _compose_row_strategy_note(row, strategy_type):
    action = row.get("action_hint") or STRATEGY_EXECUTION_PROFILES.get(strategy_type, {}).get("action") or "观察"
    tier_label = "小仓验证" if strategy_type == WEAK_RS_STRATEGY_TYPE else "正式推荐"
    hold_days = int(_safe_float(row.get("suggested_hold_days")) or STRATEGY_EXECUTION_PROFILES.get(strategy_type, {}).get("hold_days", 10))
    stop_price = _format_price(row.get("stop_price"))
    stop_loss_pct = _format_pct(row.get("stop_loss_pct"))
    target_return_pct = _format_pct(row.get("target_return_pct"))
    execution_rr = _format_pct(row.get("execution_rr"))
    position_hint = row.get("position_hint") or "低"
    risk_labels = row.get("risk_labels") or _build_risk_labels(row, strategy_type)
    entry_trigger = _resolve_entry_trigger(row, strategy_type)
    skip_condition = _resolve_skip_condition(row, strategy_type)
    latest_price = _safe_float(row.get("latest_price"))
    high_open_price = _format_price(latest_price * 1.03 if latest_price else None)
    weak_open_price = _format_price(latest_price * 0.98 if latest_price else None)

    logic_map = {
        "momentum_follow": "趋势跟随票只做强势结构中的低追高胜率买点，核心看多周期上涨、行业相对强、均线多头和量能温和确认。",
        WEAK_RS_STRATEGY_TYPE: "弱势抗跌趋势票不是追涨，核心是大盘偏弱时个股仍保持多周期抬升、行业相对强、低波动和不过热，只能作为小仓验证。",
        "short_term_bet": "强势进攻票只适合市场强、资金承接强、当日不是极端高潮的环境，核心是用小止损博取短线惯性延续。",
        "strong_breakout": "突破票要求突破位有效、收盘靠近高位、上影线不能太长，买点不在追最高价，而在突破后回踩承接。",
        "down_reversal": "低位修复票本质是弱市里的试错，不赌反转一步到位，只看低位止跌后的轻仓修复。",
    }
    opening_rule = (
        f"不建议开盘秒买；若平开至小高开且15分钟不破关键支撑，可按{position_hint}仓位的半仓先试。"
        if strategy_type != "down_reversal"
        else "不抢开盘；低开不破昨日低位并快速收回均线，才允许轻仓试错。"
    )
    intraday_rule = (
        f"优先等回踩承接，价格不破防守区{stop_price}，分时重新放量走强后再分批，避免一次性追高。"
    )
    acceptance_rule = (
        f"若高开超过3%（约高于{high_open_price}）但不回踩，或低开超过2%（约低于{weak_open_price}）后不能快速收回，则不算有效买点。"
    )
    sell_rule = (
        f"防守价{stop_price}，对应风险约{stop_loss_pct}%；跌破且不能收回先退出。"
        f"若达到{target_return_pct}%附近但放量滞涨、上影线加长或市场转弱，分批止盈。"
    )
    note = (
        f"层级:{tier_label}；策略:{action}；金融逻辑:{logic_map.get(strategy_type, _get_strategy_note(strategy_type))}"
        f"；买入口径:1)开盘:{opening_rule} 2)盘中均价:{intraday_rule} "
        f"3)承接确认:{entry_trigger}，{acceptance_rule}"
        f"；持有计划:观察{hold_days}个交易日，目标{target_return_pct}%/RR{execution_rr}；"
        f"风控:{sell_rule}；仓位:{position_hint}；主要风险:{risk_labels}；放弃条件:{skip_condition}。"
    )
    return note[:NOTE_MAX_LENGTH]


def _enrich_strategy_execution(strategy_df, strategy_type, market_regime):
    if strategy_df.empty:
        return strategy_df.copy()

    profile = STRATEGY_EXECUTION_PROFILES.get(strategy_type)
    if not profile:
        return strategy_df.copy()

    enriched = strategy_df.copy()

    def _plan_for_row(row):
        latest_price = _safe_float(row.get("latest_price"))
        stop_price = _resolve_stop_price(row, strategy_type)
        stop_loss_pct = None
        if latest_price and stop_price and latest_price > 0 and stop_price > 0:
            stop_loss_pct = (latest_price - stop_price) / latest_price * 100
        target_return_pct = _resolve_target_pct(row, strategy_type)
        execution_rr = None
        if stop_loss_pct and stop_loss_pct > 0 and target_return_pct is not None:
            execution_rr = target_return_pct / stop_loss_pct

        precision_score = _safe_float(row.get("precision_score")) or _safe_float(row.get("strategy_score")) or 0.0
        overheat_score = _resolve_overheat_score(row, strategy_type)
        position_hint = _resolve_position_hint(strategy_type, market_regime, execution_rr, stop_loss_pct, precision_score)
        stop_quality = 0.0
        if stop_loss_pct is not None:
            stop_quality = max(profile["max_stop_pct"] - stop_loss_pct, -3.0)
        execution_score = (
            (_safe_float(execution_rr) or 0.0) * 3.5
            + max(stop_quality, 0.0) * 0.6
            + (_safe_float(row.get("close_position_in_range")) or 0.0) * 1.8
            - max((_safe_float(stop_loss_pct) or 0.0) - profile["max_stop_pct"], 0.0) * 2.2
            - overheat_score * 0.8
        )
        entry_hint = profile.get("entry_hint")
        if market_regime == "weak" and strategy_type not in {"down_reversal", WEAK_RS_STRATEGY_TYPE}:
            entry_hint = "环境偏弱，除非次日承接非常稳，否则宁可放弃。"

        note_row = row.copy()
        note_row["action_hint"] = profile.get("action")
        note_row["suggested_hold_days"] = profile.get("hold_days")
        note_row["stop_price"] = stop_price
        note_row["stop_loss_pct"] = stop_loss_pct
        note_row["target_return_pct"] = target_return_pct
        note_row["execution_rr"] = execution_rr
        note_row["position_hint"] = position_hint
        note_row["entry_hint"] = entry_hint
        note_row["overheat_score"] = overheat_score
        note_row["execution_score"] = execution_score
        note_row["execution_ok"] = (
            (stop_loss_pct is not None and stop_loss_pct <= profile["max_stop_pct"])
            and (execution_rr is not None and execution_rr >= profile["min_rr"])
            and (overheat_score <= MAX_OVERHEAT_SCORE.get(strategy_type, 6.5))
        )
        note_row["risk_labels"] = _build_risk_labels(note_row, strategy_type)
        note_row["strategy_note"] = _compose_row_strategy_note(note_row, strategy_type)
        return note_row

    enriched = enriched.apply(_plan_for_row, axis=1)
    enriched = enriched[enriched["execution_ok"].fillna(False)].copy()
    return enriched


def _resolve_strategy_note_column():
    return "strategy_note"


def _keep_top_ratio(df, group_col, score_col, ratio=TOP_RATIO):
    if df.empty:
        return df.copy()

    ranked = df.copy()
    ranked["industry_count"] = ranked.groupby(group_col)["stock_code"].transform("count")
    ranked["industry_rank"] = ranked.groupby(group_col)[score_col].rank(method="first", ascending=False)
    ranked["industry_pick_limit"] = ranked["industry_count"].apply(lambda count: _top_pick_count(count, ratio))
    return ranked[ranked["industry_rank"] <= ranked["industry_pick_limit"]].copy()


def _keep_best_candidates(df, group_col, score_col, ratio=PRECISE_TOP_RATIO, max_count=None, min_score=None):
    if df.empty:
        return df.copy()

    selected = _keep_top_ratio(df, group_col, score_col, ratio=ratio)

    if min_score is not None:
        selected = selected[pd.to_numeric(selected[score_col], errors="coerce") >= float(min_score)].copy()

    if selected.empty:
        return selected

    sort_columns = [score_col, "today_change", "change_10d", "stock_code"]
    ascending = [False, False, False, True]
    selected = selected.sort_values(sort_columns, ascending=ascending, na_position="last")
    selected = selected.drop_duplicates(subset=["stock_code"], keep="first")

    if max_count is not None:
        selected = selected.head(int(max_count))

    return selected.copy()


def _apply_precision_budget(strategies, market_regime):
    ranked_frames = []

    for strategy_type, strategy_df in strategies.items():
        if strategy_df.empty:
            continue

        ranked = strategy_df.copy()
        ranked["strategy_type"] = strategy_type
        base_score = pd.to_numeric(ranked.get("strategy_score"), errors="coerce").fillna(0)
        execution_score = pd.to_numeric(ranked.get("execution_score"), errors="coerce").fillna(0)
        overheat_score = pd.to_numeric(ranked.get("overheat_score"), errors="coerce").fillna(0)
        ranked["precision_score"] = (
            base_score
            + execution_score
            + STRATEGY_PRIORITY_BONUS.get(strategy_type, 0.0)
            - overheat_score * 1.2
        )
        ranked_frames.append(ranked)

    if not ranked_frames:
        return strategies

    combined = pd.concat(ranked_frames, ignore_index=True, sort=False)
    combined = combined[combined["stock_code"].notna()].copy()
    if combined.empty:
        return {name: frame.iloc[0:0].copy() for name, frame in strategies.items()}

    combined = combined.sort_values(
        ["precision_score", "execution_score", "strategy_score", "today_change", "change_10d", "stock_code"],
        ascending=[False, False, False, False, False, True],
        na_position="last",
    )
    combined = combined.drop_duplicates(subset=["stock_code"], keep="first")

    industry_limit = GLOBAL_INDUSTRY_LIMIT.get(market_regime, GLOBAL_INDUSTRY_LIMIT["unknown"])
    if "industry_1" in combined.columns:
        combined["global_industry_rank"] = combined.groupby("industry_1").cumcount() + 1
        combined = combined[combined["global_industry_rank"] <= industry_limit].copy()

    min_precision = MIN_PRECISION_SCORE.get(market_regime, MIN_PRECISION_SCORE["unknown"])
    combined = combined[pd.to_numeric(combined["precision_score"], errors="coerce") >= min_precision].copy()

    global_limit = GLOBAL_PICK_LIMIT.get(market_regime, GLOBAL_PICK_LIMIT["unknown"])
    combined = combined.head(global_limit).copy()

    if combined.empty:
        return {name: frame.iloc[0:0].copy() for name, frame in strategies.items()}

    refined = {}
    for strategy_type, strategy_df in strategies.items():
        refined[strategy_type] = combined[combined["strategy_type"] == strategy_type].copy()
        if refined[strategy_type].empty:
            refined[strategy_type] = strategy_df.iloc[0:0].copy()

    return refined


def _apply_risk_overlay_to_strategy_frame(
    strategy_df,
    trade_date,
    overlay_frame=None,
    history=None,
    include_external=False,
):
    if strategy_df.empty:
        return strategy_df.copy()

    enriched = risk_overlay.apply_risk_overlay_to_candidates(
        strategy_df,
        history=history,
        trade_date=trade_date,
        overlay_frame=overlay_frame,
        include_external=include_external,
        score_column="precision_score" if "precision_score" in strategy_df.columns else None,
        filter_blocked=True,
        filter_downgraded=True,
        score_penalty_multiplier=0.85,
    )
    if enriched.empty:
        return enriched

    if "strategy_note" not in enriched.columns:
        enriched["strategy_note"] = ""

    enriched["strategy_note"] = enriched.apply(
        lambda row: risk_overlay.append_overlay_note(
            row.get("strategy_note") or _get_strategy_note(row.get("strategy_type")),
            row,
            max_length=NOTE_MAX_LENGTH,
        ),
        axis=1,
    )
    return enriched


def _apply_risk_overlay_to_strategies(strategies, trade_date, overlay_frame=None, history=None, include_external=False):
    if not strategies:
        return strategies

    filtered = {}
    for strategy_type, strategy_df in strategies.items():
        filtered[strategy_type] = _apply_risk_overlay_to_strategy_frame(
            strategy_df,
            trade_date,
            overlay_frame=overlay_frame,
            history=history,
            include_external=include_external,
        )
    return filtered


def _resolve_market_regime(market_score, market_breadth):
    return risk_overlay.classify_market_regime(market_score, market_breadth)


def _resolve_trade_date(snapshot_df):
    if snapshot_df.empty or "last_data_date" not in snapshot_df.columns:
        return str(datetime.now().date())

    trade_dates = pd.to_datetime(snapshot_df["last_data_date"], errors="coerce").dropna()
    if trade_dates.empty:
        return str(datetime.now().date())

    return str(trade_dates.dt.date.max())


def _filter_snapshot_by_trade_date(snapshot_df, trade_date):
    if snapshot_df.empty or not trade_date or "last_data_date" not in snapshot_df.columns:
        return snapshot_df.copy()

    trade_date_text = _normalize_trade_date_text(trade_date)
    trade_dates = pd.to_datetime(snapshot_df["last_data_date"], errors="coerce")
    return snapshot_df[trade_dates.dt.strftime("%Y-%m-%d") == trade_date_text].copy()


def _assess_snapshot_quality(snapshot_df, trade_date):
    if snapshot_df.empty:
        return {
            "trade_date_count": 0,
            "universe_count": 0,
            "coverage_ratio": 0.0,
            "meets_min_coverage": False,
        }

    trade_dates = pd.to_datetime(snapshot_df["last_data_date"], errors="coerce")
    trade_date_count = int((trade_dates.dt.strftime("%Y-%m-%d") == str(trade_date)).sum())
    universe_count = int(len(snapshot_df))
    connection = None
    try:
        connection = pymysql.connect(
            host="127.0.0.1",
            user="root",
            password="rootroot",
            database="gu_piao",
            charset="utf8mb4",
        )
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM a_stock_analysis")
            db_universe_count = cursor.fetchone()[0]
            if db_universe_count:
                universe_count = int(db_universe_count)
    except Exception as error:
        func.logInfo(f"加载全市场股票池数量失败，回退到快照数量: {error}")
    finally:
        if connection:
            connection.close()

    coverage_ratio = round(trade_date_count / universe_count * 100, 2) if universe_count else 0.0

    return {
        "trade_date_count": trade_date_count,
        "universe_count": universe_count,
        "coverage_ratio": coverage_ratio,
        "meets_min_coverage": coverage_ratio >= MIN_RECOMMENDATION_COVERAGE_RATIO,
    }


def _apply_basic_trade_filters(df):
    if df.empty:
        return df

    filtered = df.copy()
    _ensure_columns(filtered, ["stock_name", "latest_price", "today_vol"], "")

    stock_name = filtered["stock_name"].fillna("").astype(str).str.strip()
    is_excluded_name = stock_name.str.startswith(EXCLUDED_NAME_PREFIXES)
    has_valid_price = pd.to_numeric(filtered["latest_price"], errors="coerce") > 0
    has_valid_volume = pd.to_numeric(filtered["today_vol"], errors="coerce") > 0

    return filtered[~is_excluded_name & has_valid_price & has_valid_volume].copy()


def _load_snapshot():
    latest_snapshot = pd.DataFrame(func.executeSelect("a_stock_analysis", {"is_last_info": 1})["resultData"])

    if latest_snapshot.empty:
        func.logInfo("没有 is_last_info=1 的最新快照，回退到全表去重分析")
        latest_snapshot = pd.DataFrame(func.executeSelect("a_stock_analysis")["resultData"])

    if latest_snapshot.empty:
        return latest_snapshot

    _ensure_columns(latest_snapshot, ["stock_code", "last_data_date"], None)
    latest_snapshot["last_data_date"] = pd.to_datetime(latest_snapshot["last_data_date"], errors="coerce")
    latest_snapshot["has_trade_date"] = latest_snapshot["last_data_date"].notna().astype(int)
    latest_snapshot = latest_snapshot.sort_values(
        ["stock_code", "has_trade_date", "last_data_date"],
        na_position="first",
    )
    latest_snapshot = latest_snapshot.drop_duplicates(subset=["stock_code"], keep="last")
    latest_snapshot = latest_snapshot.drop(columns=["has_trade_date"])
    return latest_snapshot.reset_index(drop=True)


def _query_frame(connection, sql, params=None):
    with connection.cursor() as cursor:
        cursor.execute(sql, params or [])
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
    return pd.DataFrame.from_records(rows, columns=columns)


def _load_history(start_date=None, end_date=None):
    connection = None
    try:
        connection = pymysql.connect(
            host="127.0.0.1",
            user="root",
            password="rootroot",
            database="gu_piao",
            charset="utf8mb4",
        )
        sql = "SELECT * FROM a_stock_analysis_history WHERE 1=1"
        params = []
        if start_date is not None:
            sql += " AND last_data_date >= %s"
            params.append(str(start_date))
        if end_date is not None:
            sql += " AND last_data_date <= %s"
            params.append(str(end_date))

        history = _query_frame(connection, sql, params=params)
    except Exception as error:
        func.logInfo(f"按日期加载历史表失败，回退到全表读取: {error}")
        history = pd.DataFrame(func.executeSelect("a_stock_analysis_history")["resultData"])
    finally:
        if connection:
            connection.close()

    if history.empty:
        return history

    _ensure_columns(history, ["stock_code", "last_data_date"], None)
    history["last_data_date"] = pd.to_datetime(history["last_data_date"], errors="coerce")
    history = history[history["last_data_date"].notna()].copy()
    history = history.sort_values(["stock_code", "last_data_date"])
    history = history.drop_duplicates(subset=["last_data_date", "stock_code"], keep="last")
    return history.reset_index(drop=True)


def _load_recent_history_window(reference_date, lookback_trade_days=3):
    if not reference_date:
        return pd.DataFrame()

    connection = None
    try:
        connection = pymysql.connect(
            host="127.0.0.1",
            user="root",
            password="rootroot",
            database="gu_piao",
            charset="utf8mb4",
        )
        sql = """
            SELECT *
            FROM a_stock_analysis_history
            WHERE last_data_date IN (
                SELECT last_data_date
                FROM (
                    SELECT DISTINCT last_data_date
                    FROM a_stock_analysis_history
                    WHERE last_data_date <= %s
                    ORDER BY last_data_date DESC
                    LIMIT %s
                ) recent_trade_days
            )
        """
        with connection.cursor() as cursor:
            cursor.execute(sql, [reference_date, int(max(1, lookback_trade_days))])
            records = cursor.fetchall()
            field_names = [desc[0] for desc in cursor.description]

        history = pd.DataFrame([dict(zip(field_names, row)) for row in records])
        if history.empty:
            return history

        _ensure_columns(history, ["stock_code", "last_data_date"], None)
        history["last_data_date"] = pd.to_datetime(history["last_data_date"], errors="coerce")
        history = history[history["last_data_date"].notna()].copy()
        history = history.sort_values(["stock_code", "last_data_date"])
        history = history.drop_duplicates(subset=["last_data_date", "stock_code"], keep="last")
        return history.reset_index(drop=True)
    except Exception as error:
        func.logInfo(f"加载最近历史窗口失败: {error}")
        return pd.DataFrame()
    finally:
        if connection:
            connection.close()


def _load_generated_health_signals(connection, start_date, end_date):
    if not GENERATED_HEALTH_STRATEGY_TYPES:
        return pd.DataFrame(columns=["trade_date", "strategy_type", "stock_code", "signal_date"])

    try:
        history = _query_frame(
            connection,
            """
            SELECT *
            FROM a_stock_analysis_history
            WHERE last_data_date >= %s
              AND last_data_date <= %s
            ORDER BY stock_code, last_data_date
            """,
            params=[start_date, str(end_date)],
        )
    except Exception as error:
        func.logInfo(f"生成策略健康度样本失败，跳过生成信号: {error}")
        return pd.DataFrame(columns=["trade_date", "strategy_type", "stock_code", "signal_date"])

    if history.empty:
        return pd.DataFrame(columns=["trade_date", "strategy_type", "stock_code", "signal_date"])

    _ensure_columns(history, ["stock_code", "last_data_date"], None)
    history["last_data_date"] = pd.to_datetime(history["last_data_date"], errors="coerce")
    history = history.dropna(subset=["last_data_date", "stock_code"]).copy()
    history = history.sort_values(["stock_code", "last_data_date"])
    history = history.drop_duplicates(subset=["last_data_date", "stock_code"], keep="last")
    if history.empty:
        return pd.DataFrame(columns=["trade_date", "strategy_type", "stock_code", "signal_date"])

    detail = _collect_history_signals(history, apply_risk_overlay=True)
    if detail.empty:
        return pd.DataFrame(columns=["trade_date", "strategy_type", "stock_code", "signal_date"])

    generated = detail[detail["strategy_type"].isin(GENERATED_HEALTH_STRATEGY_TYPES)].copy()
    if generated.empty:
        return pd.DataFrame(columns=["trade_date", "strategy_type", "stock_code", "signal_date"])

    generated["signal_date"] = pd.to_datetime(generated["signal_date"], errors="coerce")
    generated = generated.dropna(subset=["signal_date", "strategy_type", "stock_code"]).copy()
    generated["trade_date"] = generated["signal_date"].dt.strftime("%Y-%m-%d")
    generated = generated[["trade_date", "strategy_type", "stock_code", "signal_date"]].drop_duplicates()
    func.logInfo(
        f"生成策略健康度样本: strategies={sorted(GENERATED_HEALTH_STRATEGY_TYPES)}, "
        f"signal_count={len(generated)}"
    )
    return generated


def _resolve_health_return_columns(settings):
    hold_days = int(settings["hold_days"])
    entry_mode = settings.get("entry_mode") or "next_close"
    mode_config = BACKTEST_ENTRY_MODE_DEFINITIONS.get(entry_mode, BACKTEST_ENTRY_MODE_DEFINITIONS["next_close"])
    return_col = mode_config["return_template"].format(hold_days=hold_days)
    gross_return_col = mode_config["gross_template"].format(hold_days=hold_days)
    return return_col, gross_return_col, entry_mode, mode_config.get("label")


def _load_strategy_health(reference_date):
    if not reference_date:
        return {}

    connection = None
    try:
        connection = pymysql.connect(
            host="127.0.0.1",
            user="root",
            password="rootroot",
            database="gu_piao",
            charset="utf8mb4",
        )

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
                [reference_date, int(max(1, STRATEGY_HEALTH_LOOKBACK_TRADE_DAYS))],
            )
            trade_days = [row[0] for row in cursor.fetchall()]

        if not trade_days:
            return {}

        start_date = str(trade_days[0])
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT trade_date, strategy_type, stock_code
                FROM a_stock_strategy_result
                WHERE trade_date >= %s AND trade_date < %s
                ORDER BY trade_date, strategy_type, stock_code
                """,
                [start_date, reference_date],
            )
            signal_records = cursor.fetchall()
            signal_fields = [desc[0] for desc in cursor.description]

        signals = pd.DataFrame([dict(zip(signal_fields, row)) for row in signal_records], columns=signal_fields)
        if signals.empty:
            signals = pd.DataFrame(columns=["trade_date", "strategy_type", "stock_code", "signal_date"])
        else:
            signals["signal_date"] = pd.to_datetime(signals["trade_date"], errors="coerce")
            signals = signals.dropna(subset=["signal_date", "stock_code", "strategy_type"]).copy()

        generated_signals = _load_generated_health_signals(connection, start_date, trade_days[-1])
        if not generated_signals.empty:
            signals = pd.concat([signals, generated_signals], ignore_index=True, sort=False)
            signals = signals.drop_duplicates(subset=["signal_date", "strategy_type", "stock_code"], keep="first")

        if signals.empty:
            return {}

        stock_codes = signals["stock_code"].astype(str).unique().tolist()
        placeholders = ",".join(["%s"] * len(stock_codes))
        history_sql = f"""
            SELECT last_data_date, stock_code, latest_price, today_change
            FROM a_stock_analysis_history
            WHERE stock_code IN ({placeholders})
              AND last_data_date >= %s
              AND last_data_date < %s
            ORDER BY stock_code, last_data_date
        """
        history = _query_frame(connection, history_sql, params=stock_codes + [start_date, reference_date])
    except Exception as error:
        func.logInfo(f"加载策略健康度失败，跳过健康过滤: {error}")
        return {}
    finally:
        if connection:
            connection.close()

    if history.empty:
        return {}

    history["last_data_date"] = pd.to_datetime(history["last_data_date"], errors="coerce")
    history = history.dropna(subset=["last_data_date", "stock_code"]).copy()
    history = history.sort_values(["stock_code", "last_data_date"])
    history = history.drop_duplicates(subset=["last_data_date", "stock_code"], keep="last")
    if history.empty:
        return {}

    required_holding_days = sorted(
        {int(settings["hold_days"]) for settings in STRATEGY_HEALTH_SETTINGS.values() if int(settings["hold_days"]) > 0}
    )
    if not required_holding_days:
        return {}

    price_paths = _build_price_paths(
        history,
        required_holding_days,
        DEFAULT_ENTRY_OFFSET_DAYS,
        round_trip_cost_pct=(DEFAULT_FEE_BPS + DEFAULT_SLIPPAGE_BPS) * 2 / 100,
    )
    merge_columns = ["last_data_date", "stock_code", "entry_change"]
    health_return_columns = set()
    for hold_days in required_holding_days:
        merge_columns.extend(
            [
                f"exit_change_{hold_days}d",
                f"gross_return_{hold_days}d",
                f"return_{hold_days}d",
            ]
        )
    for settings in STRATEGY_HEALTH_SETTINGS.values():
        return_col, gross_return_col, _, _ = _resolve_health_return_columns(settings)
        health_return_columns.update([return_col, gross_return_col])
    merge_columns.extend(sorted(column for column in health_return_columns if column not in merge_columns))
    merge_columns = [column for column in merge_columns if column in price_paths.columns]

    detail = signals.merge(
        price_paths[merge_columns].rename(columns={"last_data_date": "signal_date"}),
        on=["signal_date", "stock_code"],
        how="left",
    )

    health_return_pairs = set()
    for settings in STRATEGY_HEALTH_SETTINGS.values():
        return_col, gross_return_col, _, _ = _resolve_health_return_columns(settings)
        if return_col in detail.columns and gross_return_col in detail.columns:
            health_return_pairs.add((return_col, gross_return_col))

    for hold_days in required_holding_days:
        exit_change_col = f"exit_change_{hold_days}d"

        entry_abs = pd.to_numeric(detail["entry_change"], errors="coerce").abs()
        exit_abs = pd.to_numeric(detail[exit_change_col], errors="coerce").abs()
        tradable = (entry_abs.isna() | (entry_abs < DEFAULT_LIMIT_PCT)) & (
            exit_abs.isna() | (exit_abs < DEFAULT_LIMIT_PCT)
        )
        for return_col, gross_return_col in health_return_pairs:
            if not return_col.endswith(f"_{hold_days}d"):
                continue
            filtered_mask = detail[return_col].notna() & (~tradable)
            detail.loc[filtered_mask, [return_col, gross_return_col]] = pd.NA

    health_snapshot = {}
    for strategy_type, settings in STRATEGY_HEALTH_SETTINGS.items():
        hold_days = int(settings["hold_days"])
        return_col, gross_return_col, entry_mode, entry_mode_label = _resolve_health_return_columns(settings)
        strategy_details = detail[detail["strategy_type"] == strategy_type].copy()
        valid_details = strategy_details[strategy_details[return_col].notna()].copy()
        holding_summary = _summarize_holding_returns(valid_details, return_col, gross_return_col)

        evaluated_trades = int(holding_summary["evaluated_trades"])
        avg_return = holding_summary["avg_return"]
        trade_win_rate = holding_summary["trade_win_rate"]

        failure_reasons = []
        if evaluated_trades < int(settings["min_evaluated_trades"]):
            failure_reasons.append("insufficient_trades")
        if avg_return is None or avg_return < float(settings["min_avg_return"]):
            failure_reasons.append("avg_return_below_threshold")
        if trade_win_rate is None or trade_win_rate < float(settings["min_trade_win_rate"]):
            failure_reasons.append("win_rate_below_threshold")

        health_snapshot[strategy_type] = {
            "enabled": not failure_reasons,
            "hold_days": hold_days,
            "entry_mode": entry_mode,
            "entry_mode_label": entry_mode_label,
            "signal_count": int(len(strategy_details)),
            "signal_days": int(strategy_details["signal_date"].nunique()) if not strategy_details.empty else 0,
            "evaluated_trades": evaluated_trades,
            "avg_return": avg_return,
            "trade_win_rate": trade_win_rate,
            "failure_reasons": failure_reasons,
        }

    return health_snapshot


def _apply_strategy_health_gate(strategies, strategy_health):
    if not strategy_health:
        return strategies

    gated_strategies = {}
    for strategy_type, strategy_df in strategies.items():
        health_info = strategy_health.get(strategy_type)
        if health_info and not health_info.get("enabled", True):
            func.logInfo(
                f"策略健康度未达标，暂停推荐: {strategy_type}, "
                f"avg_return={health_info.get('avg_return')}, "
                f"win_rate={health_info.get('trade_win_rate')}, "
                f"evaluated_trades={health_info.get('evaluated_trades')}, "
                f"reasons={health_info.get('failure_reasons')}"
            )
            gated_strategies[strategy_type] = strategy_df.iloc[0:0].copy()
            continue

        gated_strategies[strategy_type] = strategy_df

    return gated_strategies


def _prepare_snapshot(df):
    if df.empty:
        return df

    prepared = df.copy()
    _ensure_columns(prepared, ["industry", "stock_code", "stock_name"], "")
    _ensure_columns(prepared, NUMERIC_COLUMNS, pd.NA)

    for column in NUMERIC_COLUMNS:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    prepared = _apply_basic_trade_filters(prepared)
    if prepared.empty:
        return prepared

    prepared["industry_list"] = prepared["industry"].apply(_parse_industry)
    prepared["industry_1"] = prepared["industry_list"].apply(lambda values: values[0] if values else None)
    prepared = prepared[prepared["industry_1"].notna()].copy()

    if prepared.empty:
        return prepared

    # 腾讯历史源稳定性更好，但通常缺少成交额；这里先用量价估算补齐，
    # 让策略在日常运行中可以真正使用“金额确认”，而不是因为字段缺失直接放行。
    prepared["today_amount"] = prepared["today_amount"].where(
        prepared["today_amount"].notna(),
        _estimate_amount_from_price_volume(
            prepared["latest_price"],
            prepared["today_vol"],
            prepared["stock_code"],
        ),
    )
    prepared["amount_avg_5d"] = prepared["amount_avg_5d"].where(
        prepared["amount_avg_5d"].notna(),
        _estimate_amount_from_price_volume(
            prepared["ma5"],
            prepared["vol_avg_5d"],
            prepared["stock_code"],
        ),
    )
    prepared["amount_avg_10d"] = prepared["amount_avg_10d"].where(
        prepared["amount_avg_10d"].notna(),
        _estimate_amount_from_price_volume(
            prepared["ma10"],
            prepared["vol_avg_10d"],
            prepared["stock_code"],
        ),
    )

    for column in INDUSTRY_MEAN_COLUMNS:
        prepared[f"ind_{column}"] = prepared.groupby("industry_1")[column].transform("mean")

    market_means = {
        column: pd.to_numeric(prepared[column], errors="coerce").mean()
        for column in INDUSTRY_MEAN_COLUMNS
    }
    for column, market_mean in market_means.items():
        prepared[f"market_{column}"] = market_mean
        prepared[f"ind_alpha_{column}"] = prepared[f"ind_{column}"] - market_mean

    intraday_range = (prepared["today_high"] - prepared["today_low"]).replace(0, pd.NA)
    body_high = prepared[["today_open", "latest_price"]].max(axis=1)
    body_low = prepared[["today_open", "latest_price"]].min(axis=1)

    prepared["breakout_close_ratio"] = prepared["latest_price"] / prepared["high_20d"].replace(0, pd.NA)
    prepared["close_vs_high_60d_ratio"] = prepared["latest_price"] / prepared["high_60d"].replace(0, pd.NA)
    prepared["close_vs_high_120d_ratio"] = prepared["latest_price"] / prepared["high_120d"].replace(0, pd.NA)
    prepared["ma5_ma20_gap_ratio"] = (
        (prepared["ma5"] - prepared["ma20"]) / prepared["ma20"].replace(0, pd.NA)
    )
    prepared["close_position_in_range"] = (
        (prepared["latest_price"] - prepared["today_low"]) / intraday_range
    )
    prepared["upper_shadow_ratio"] = (prepared["today_high"] - body_high) / intraday_range
    prepared["lower_shadow_ratio"] = (body_low - prepared["today_low"]) / intraday_range
    prepared["real_body_ratio"] = (prepared["latest_price"] - prepared["today_open"]).abs() / intraday_range
    prepared["amount_confirmation_score"] = (
        (prepared["today_amount"] >= prepared["amount_avg_5d"]).fillna(False).astype(int)
        + (prepared["today_amount"] >= prepared["amount_avg_10d"]).fillna(False).astype(int)
    )
    prepared["turnover_confirmation_score"] = (
        (prepared["turnover_rate"] >= prepared["turnover_avg_5d"]).fillna(False).astype(int)
        + (prepared["turnover_rate"] >= prepared["turnover_avg_10d"]).fillna(False).astype(int)
    )
    prepared["has_amount_context"] = prepared[["today_amount", "amount_avg_5d", "amount_avg_10d"]].notna().sum(axis=1) >= 2
    prepared["has_turnover_context"] = (
        prepared[["turnover_rate", "turnover_avg_5d", "turnover_avg_10d"]].notna().sum(axis=1) >= 2
    )
    prepared["activity_confirmation_score"] = (
        prepared["amount_confirmation_score"]
        + prepared["turnover_confirmation_score"]
        + (prepared["vr_today"] >= 1.2).fillna(False).astype(int)
    )
    prepared["volatility_10_20_ratio"] = (
        prepared["volatility_10d"] / prepared["volatility_20d"].replace(0, pd.NA)
    )

    return prepared.reset_index(drop=True)


def _build_temporal_feature_frame(history):
    if history.empty:
        return pd.DataFrame(columns=["stock_code", "last_data_date", *TEMPORAL_FEATURE_COLUMNS])

    feature_columns = [
        "stock_code",
        "last_data_date",
        "latest_price",
        "today_vol",
        "vol_avg_5d",
        "vr_today",
        "vr_5d",
        "ma5",
        "ma20",
        "high_20d",
        "low_20d",
    ]
    feature_df = history[feature_columns].copy()
    feature_df["last_data_date"] = pd.to_datetime(feature_df["last_data_date"], errors="coerce")
    feature_df = feature_df.dropna(subset=["stock_code", "last_data_date"]).copy()

    numeric_columns = [
        "latest_price",
        "today_vol",
        "vol_avg_5d",
        "vr_today",
        "vr_5d",
        "ma5",
        "ma20",
        "high_20d",
        "low_20d",
    ]
    for column in numeric_columns:
        feature_df[column] = pd.to_numeric(feature_df[column], errors="coerce")

    feature_df = feature_df.sort_values(["stock_code", "last_data_date"]).reset_index(drop=True)
    grouped = feature_df.groupby("stock_code", sort=False)

    for column in ("latest_price", "today_vol", "vr_today", "vr_5d", "high_20d", "low_20d"):
        feature_df[f"prev_{column}"] = grouped[column].shift(1)

    feature_df["prev2_high_20d"] = grouped["high_20d"].shift(2)
    feature_df["prev2_low_20d"] = grouped["low_20d"].shift(2)

    prev_range = (feature_df["prev_high_20d"] - feature_df["prev_low_20d"]).replace(0, pd.NA)
    prev2_range = (feature_df["prev2_high_20d"] - feature_df["prev2_low_20d"]).replace(0, pd.NA)

    feature_df["close_vs_prev_high_ratio"] = (
        feature_df["latest_price"] / feature_df["prev_high_20d"].replace(0, pd.NA)
    )
    feature_df["prev_close_vs_prev2_high_ratio"] = (
        feature_df["prev_latest_price"] / feature_df["prev2_high_20d"].replace(0, pd.NA)
    )
    feature_df["range_position_prev_20d"] = (
        (feature_df["latest_price"] - feature_df["prev_low_20d"]) / prev_range
    )
    feature_df["prev_range_position_prev_20d"] = (
        (feature_df["prev_latest_price"] - feature_df["prev2_low_20d"]) / prev2_range
    )
    feature_df["breakout_buffer_pct"] = (feature_df["close_vs_prev_high_ratio"] - 1) * 100
    feature_df["trend_extension_ma20_ratio"] = (
        feature_df["latest_price"] / feature_df["ma20"].replace(0, pd.NA)
    )
    feature_df["trend_extension_ma5_ratio"] = (
        feature_df["latest_price"] / feature_df["ma5"].replace(0, pd.NA)
    )

    feature_df["volume_confirmation_score"] = (
        (feature_df["vr_today"] >= 1.5).fillna(False).astype(int)
        + (feature_df["today_vol"] > feature_df["vol_avg_5d"]).fillna(False).astype(int)
        + (feature_df["vr_today"] >= feature_df["vr_5d"]).fillna(False).astype(int)
    )

    return feature_df[["stock_code", "last_data_date", *TEMPORAL_FEATURE_COLUMNS]].copy()


def _merge_temporal_features(prepared, temporal_features):
    if prepared.empty:
        return prepared

    if temporal_features.empty:
        enriched = prepared.copy()
        _ensure_columns(enriched, TEMPORAL_FEATURE_COLUMNS, float("nan"))
        return enriched

    merged = prepared.merge(
        temporal_features,
        on=["stock_code", "last_data_date"],
        how="left",
    )
    _ensure_columns(merged, TEMPORAL_FEATURE_COLUMNS, float("nan"))
    return merged


def _apply_market_gate(strategy_df, market_regime, allowed_regimes):
    if market_regime in allowed_regimes:
        return strategy_df
    return strategy_df.iloc[0:0].copy()


def _build_short_term_strategies(df, market_regime):
    strategies = {}
    breakout_ready = (
        (df["close_vs_prev_high_ratio"] >= 1.003)
        & (
            (df["prev_close_vs_prev2_high_ratio"] >= 0.995)
            | (df["close_vs_prev_high_ratio"] >= 1.01)
        )
    )
    breakout_range_ready = df["range_position_prev_20d"] >= 0.85
    strong_volume_ready = df["volume_confirmation_score"] >= 2
    activity_strict_ready = (
        (df["has_turnover_context"] & (df["activity_confirmation_score"] >= 3) & (df["amount_confirmation_score"] >= 1))
        | (~df["has_turnover_context"] & (df["amount_confirmation_score"] >= 1) & (df["vr_today"] >= 1.2))
    )
    close_strong = df["close_position_in_range"] >= 0.68
    close_very_strong = df["close_position_in_range"] >= 0.75
    close_ok = df["close_position_in_range"] >= 0.55
    upper_shadow_tight = df["upper_shadow_ratio"] <= 0.35
    upper_shadow_very_tight = df["upper_shadow_ratio"] <= 0.28
    upper_shadow_loose = df["upper_shadow_ratio"] <= 0.45
    real_body_ready = df["real_body_ratio"] >= 0.3
    real_body_strong = df["real_body_ratio"] >= 0.4
    liquid_amount_strict = (
        (df["today_amount"] >= 200000000)
        | (df["amount_avg_5d"] >= 150000000)
    )
    mid_term_leader = df["close_vs_high_60d_ratio"] >= 0.975
    trend_slope_ready = (df["ma20_slope_5d"] > 0.1) & (df["ma60_slope_10d"] >= 0)

    if "short_strict" in RETIRED_STRATEGIES:
        strategies["short_strict"] = df.iloc[0:0].copy()
    else:
        strategies["short_strict"] = df[
            (df["change_3d"] < 0)
            & (df["change_5d"] < 0)
            & (df["today_change"] > 2)
            & (df["vr_today"] > 2)
            & close_ok
            & upper_shadow_loose
            & (df["today_amp"] > df["amp_3d"])
            & (df["close_vs_high_60d_ratio"] <= 0.9)
            & (df["ind_change_5d"] > -0.5)
        ].copy()
        strategies["short_strict"] = _apply_market_gate(strategies["short_strict"], market_regime, {"weak"})

    if "short_loose" not in RETIRED_STRATEGIES:
        strategies["short_loose"] = df[
            (df["change_3d"] < -2)
            & (df["today_change"] > 2)
            & (df["vr_today"] > 1.5)
            & (df["today_vol"] > df["vol_avg_5d"])
            & close_ok
            & upper_shadow_loose
            & (df["today_amp"] > df["amp_3d"])
            & (df["latest_price"] > df["ma5"])
            & (df["close_vs_high_60d_ratio"] <= 0.9)
            & (df["change_30d"] < 40)
            & (df["ind_change_5d"] > -1)
        ].copy()
        strategies["short_loose"] = _apply_market_gate(strategies["short_loose"], market_regime, {"weak"})
    else:
        strategies["short_loose"] = df.iloc[0:0].copy()

    # 这里优先使用“昨日前高”做突破判断，避免把包含当日最高价的 high_20d 当成突破确认。
    strong_breakout = df[
        (df["today_change"] > 2)
        & (df["today_amp"] >= df["amp_10d"])
        & breakout_ready
        & breakout_range_ready
        & strong_volume_ready
        & activity_strict_ready
        & df["trend_extension_ma20_ratio"].between(1.0, 1.12)
        & (df["breakout_buffer_pct"] >= 0.25)
        & (df["breakout_buffer_pct"] <= 5.5)
        & close_very_strong
        & upper_shadow_very_tight
        & real_body_strong
        & liquid_amount_strict
        & mid_term_leader
        & (df["close_vs_high_120d_ratio"] >= 0.92)
        & trend_slope_ready
        & (df["volatility_10_20_ratio"] <= 1.2)
        & (df["ma5"] > df["ma10"])
        & (df["ma10"] > df["ma20"])
        & (df["change_5d"] > 0)
        & (df["change_10d"] > 0)
        & (df["market_change_3d"] > 0)
        & (df["ind_change_5d"] > 0)
        & (df["ind_alpha_change_3d"] > 0)
        & (df["ind_alpha_change_5d"] > 0)
        & (df["ind_alpha_change_10d"] > 0)
    ].copy()
    strong_breakout["strategy_score"] = (
        strong_breakout["breakout_buffer_pct"].clip(lower=0, upper=4).fillna(0) * 1.5
        + strong_breakout["volume_confirmation_score"].fillna(0) * 2.0
        + strong_breakout["activity_confirmation_score"].fillna(0) * 1.6
        + (strong_breakout["close_position_in_range"].fillna(0) * 8.0)
        + ((1 - strong_breakout["upper_shadow_ratio"]).clip(lower=0, upper=1).fillna(0) * 2.5)
        + strong_breakout["ind_alpha_change_10d"].clip(lower=0, upper=12).fillna(0) * 0.6
        + ((strong_breakout["close_vs_high_120d_ratio"] - 0.9).clip(lower=0).fillna(0) * 35)
        + strong_breakout["ma20_slope_5d"].clip(lower=0, upper=2).fillna(0) * 3.0
        - ((strong_breakout["volatility_10_20_ratio"] - 1).clip(lower=0).fillna(0) * 4.0)
    )
    strategies["strong_breakout"] = _keep_best_candidates(
        _apply_market_gate(strong_breakout, market_regime, {"strong"}),
        "industry_1",
        "strategy_score",
        ratio=0.08,
        max_count=5,
        min_score=18,
    )

    core = (
        (df["change_5d"] < -4)
        & (df["change_20d"] < -10)
        & (df["today_change"] > 1.5)
        & (df["vr_today"] > 1.2)
    )
    position = df["latest_price"] <= df["low_20d"] * 1.02
    extra = (
        ((df["today_vol"] > df["vol_avg_5d"]).fillna(False)).astype(int)
        + ((df["today_amp"] > df["amp_5d"]).fillna(False)).astype(int)
        + ((df["latest_price"] > df["ma5"]).fillna(False)).astype(int)
    )
    strategies["down_reversal"] = df[
        core
        & position
        & (extra >= 2)
        & (df["close_position_in_range"] >= 0.55)
        & (df["upper_shadow_ratio"] <= 0.45)
        & (df["close_vs_high_60d_ratio"] <= 0.88)
        & (df["ind_alpha_change_5d"] > -2)
        & (df["ind_alpha_change_10d"] > -3)
    ].copy()
    strategies["down_reversal"] = _apply_market_gate(strategies["down_reversal"], market_regime, {"weak"})

    short_term_bet = df[
        (df["today_change"] >= 4)
        & (df["today_change"] <= 8.5)
        & (df["today_amp"] >= df["amp_10d"])
        & (df["vr_today"] >= 1.5)
        & activity_strict_ready
        & close_very_strong
        & upper_shadow_tight
        & real_body_strong
        & liquid_amount_strict
        & (df["close_vs_high_60d_ratio"] >= 0.94)
        & (df["close_vs_high_120d_ratio"] >= 0.86)
        & df["trend_extension_ma20_ratio"].between(1.0, 1.14)
        & (df["volatility_10_20_ratio"] <= 1.35)
        & (df["change_3d"] > 0)
        & (df["change_5d"] > 0)
        & (df["change_10d"] > 0)
        & (df["change_10d"] <= 22)
        & (df["change_5d"] > df["ind_change_5d"])
        & (df["ma5"] > df["ma10"])
        & (df["ma10"] > df["ma20"])
        & (df["ma20"] > df["ma60"])
        & (df["market_change_3d"] > 0)
        & (df["ind_alpha_change_3d"] > 0)
        & (df["ind_alpha_change_5d"] > 0)
        & (df["ind_alpha_change_10d"] > 0)
    ].copy()
    short_term_bet["strategy_score"] = (
        short_term_bet["today_change"].clip(lower=0, upper=8.5).fillna(0) * 0.9
        + short_term_bet["activity_confirmation_score"].fillna(0) * 1.8
        + short_term_bet["volume_confirmation_score"].fillna(0) * 1.2
        + short_term_bet["close_position_in_range"].fillna(0) * 7.0
        + short_term_bet["real_body_ratio"].fillna(0) * 4.0
        + short_term_bet["ind_alpha_change_10d"].clip(lower=0, upper=12).fillna(0) * 0.5
        + ((short_term_bet["close_vs_high_60d_ratio"] - 0.9).clip(lower=0).fillna(0) * 18)
        - (short_term_bet["upper_shadow_ratio"].clip(lower=0).fillna(0) * 4.0)
        - ((short_term_bet["volatility_10_20_ratio"] - 1).clip(lower=0).fillna(0) * 3.0)
    )
    strategies["short_term_bet"] = _keep_best_candidates(
        _apply_market_gate(short_term_bet, market_regime, {"strong"}),
        "industry_1",
        "strategy_score",
        ratio=0.08,
        max_count=4,
        min_score=16,
    )

    momentum_follow = df[
        (df["change_5d"] >= 4)
        & (df["change_10d"] >= 8)
        & (df["change_20d"] >= 0)
        & (df["vr_5d"] >= 1.0)
        & (df["change_3d"] > 0)
        & (df["today_change"] >= 1.2)
        & (df["today_change"] <= 7.5)
        & (df["vr_today"] >= 1.0)
        & activity_strict_ready
        & (df["close_vs_high_60d_ratio"] >= 0.94)
        & (df["close_vs_high_120d_ratio"] >= 0.92)
        & trend_slope_ready
        & (df["volatility_10_20_ratio"] <= 1.12)
        & close_strong
        & upper_shadow_tight
        & (df["change_5d"] > df["ind_change_5d"])
        & (df["change_10d"] > df["ind_change_10d"])
        & (df["ma5"] > df["ma10"])
        & (df["ma10"] > df["ma20"])
        & (df["ma20"] > df["ma60"])
        & (df["market_change_3d"] > 0)
        & (df["ind_alpha_change_3d"] > 0)
        & (df["ind_alpha_change_5d"] > 0)
        & (df["ind_alpha_change_10d"] > 0)
    ].copy()
    momentum_follow["strategy_score"] = (
        momentum_follow["change_10d"].clip(lower=0, upper=25).fillna(0) * 0.55
        + momentum_follow["ind_alpha_change_10d"].clip(lower=0, upper=12).fillna(0) * 0.9
        + momentum_follow["activity_confirmation_score"].fillna(0) * 1.4
        + momentum_follow["ma20_slope_5d"].clip(lower=0, upper=2).fillna(0) * 3.0
        + momentum_follow["close_position_in_range"].fillna(0) * 5.0
        + ((momentum_follow["close_vs_high_120d_ratio"] - 0.9).clip(lower=0).fillna(0) * 28)
        + ((1 - momentum_follow["upper_shadow_ratio"]).clip(lower=0, upper=1).fillna(0) * 2.5)
        - ((momentum_follow["volatility_10_20_ratio"] - 1).clip(lower=0).fillna(0) * 4.0)
    )
    strategies["momentum_follow"] = _keep_best_candidates(
        _apply_market_gate(momentum_follow, market_regime, {"strong"}),
        "industry_1",
        "strategy_score",
        ratio=0.08,
        max_count=5,
        min_score=18,
    )

    liquid_amount_defensive = (
        (df["today_amount"] >= 120000000)
        | (df["amount_avg_5d"] >= 100000000)
    )
    weak_relative_strength = df[
        (df["change_5d"] >= 0.5)
        & (df["change_10d"] >= 2.0)
        & (df["change_20d"] >= -2.0)
        & (df["change_10d"] <= 16.0)
        & (df["change_30d"] <= 35.0)
        & (df["today_change"].between(-1.0, 4.0))
        & (df["vr_today"].between(0.75, 1.8))
        & (df["vr_5d"].between(0.8, 1.8))
        & liquid_amount_defensive
        & (df["latest_price"] >= df["ma20"])
        & (df["latest_price"] <= df["ma20"] * 1.08)
        & (df["ma5"] > df["ma10"])
        & (df["ma10"] >= df["ma20"])
        & (df["ma20_slope_5d"] > 0.08)
        & (df["ma60_slope_10d"] >= -0.05)
        & (df["close_vs_high_60d_ratio"].between(0.88, 1.01))
        & (df["close_vs_high_120d_ratio"] >= 0.84)
        & (df["close_position_in_range"] >= 0.55)
        & (df["upper_shadow_ratio"] <= 0.32)
        & (df["volatility_10_20_ratio"] <= 1.08)
        & (df["today_amp"] <= df["amp_5d"] * 1.35)
        & (df["change_5d"] > df["market_change_5d"] + 2.5)
        & (df["change_10d"] > df["market_change_10d"] + 3.0)
        & (df["ind_change_5d"] > df["market_change_5d"])
        & (df["ind_alpha_change_5d"] > 1.5)
        & (df["ind_alpha_change_10d"] > 2.0)
        & (
            (df["has_turnover_context"] & (df["turnover_confirmation_score"] >= 1))
            | (~df["has_turnover_context"] & (df["amount_confirmation_score"] >= 1))
        )
    ].copy()
    weak_relative_strength["strategy_score"] = (
        weak_relative_strength["change_5d"].clip(lower=0, upper=8).fillna(0) * 0.55
        + weak_relative_strength["change_10d"].clip(lower=0, upper=16).fillna(0) * 0.35
        + weak_relative_strength["ind_alpha_change_5d"].clip(lower=0, upper=8).fillna(0) * 1.0
        + weak_relative_strength["ind_alpha_change_10d"].clip(lower=0, upper=12).fillna(0) * 0.75
        + weak_relative_strength["ma20_slope_5d"].clip(lower=0, upper=1.5).fillna(0) * 3.0
        + weak_relative_strength["close_position_in_range"].fillna(0) * 4.0
        + ((weak_relative_strength["close_vs_high_60d_ratio"] - 0.86).clip(lower=0, upper=0.14).fillna(0) * 35)
        + ((1.08 - weak_relative_strength["volatility_10_20_ratio"]).clip(lower=0, upper=0.16).fillna(0) * 6.0)
        - ((weak_relative_strength["latest_price"] / weak_relative_strength["ma20"].replace(0, pd.NA) - 1.04).clip(lower=0).fillna(0) * 45)
        - ((weak_relative_strength["vr_today"] - 1.4).clip(lower=0).fillna(0) * 2.0)
    )
    strategies[WEAK_RS_STRATEGY_TYPE] = _keep_best_candidates(
        _apply_market_gate(weak_relative_strength, market_regime, {"weak"}),
        "industry_1",
        "strategy_score",
        ratio=0.08,
        max_count=3,
        min_score=15,
    )

    if "ma_cross" in RETIRED_STRATEGIES:
        strategies["ma_cross"] = df.iloc[0:0].copy()
    else:
        # 当前快照只有单日特征，所以这里不用 shift(1) 做伪时间序列判断。
        ma_cross = df[
            (df["ma5"] > df["ma20"])
            & (df["ma5_ma20_gap_ratio"].between(0, 0.01))
            & (df["today_change"].between(2.2, 6.0))
            & (df["vr_today"].between(1.4, 3.8))
            & activity_strict_ready
            & (df["close_vs_high_60d_ratio"].between(0.88, 0.99))
            & (df["ma20_slope_5d"] > 0.18)
            & (df["ma60_slope_10d"] >= 0)
            & (df["volatility_10_20_ratio"] <= 1.2)
            & close_strong
            & upper_shadow_tight
            & real_body_ready
            & (df["ma10"] > df["ma20"])
            & (df["ma20"] > df["ma60"])
            & (df["change_5d"] > 0)
            & (df["market_change_3d"] > 0)
            & (df["ind_change_5d"] > 0)
            & (df["ind_alpha_change_3d"] > 0)
            & (df["ind_alpha_change_5d"] > 0)
        ].copy()
        ma_cross["strategy_score"] = (
            ((0.012 - ma_cross["ma5_ma20_gap_ratio"].abs()).clip(lower=0).fillna(0) / 0.012 * 5.0)
            + ma_cross["activity_confirmation_score"].fillna(0) * 1.4
            + ma_cross["ma20_slope_5d"].clip(lower=0, upper=2).fillna(0) * 4.0
            + ma_cross["close_position_in_range"].fillna(0) * 5.5
            + ma_cross["real_body_ratio"].fillna(0) * 2.5
            + ma_cross["ind_alpha_change_5d"].clip(lower=0, upper=8).fillna(0) * 0.8
            - ((ma_cross["volatility_10_20_ratio"] - 1).clip(lower=0).fillna(0) * 3.0)
        )
        strategies["ma_cross"] = _keep_best_candidates(
            _apply_market_gate(ma_cross, market_regime, {"strong", "neutral"}),
            "industry_1",
            "strategy_score",
            ratio=0.08,
            max_count=5,
            min_score=12,
        )

    down_reversal = strategies["down_reversal"].copy()
    if not down_reversal.empty:
        down_reversal = down_reversal[
            (down_reversal["today_change"] >= 2)
            & (down_reversal["vr_today"] >= 1.5)
            & (down_reversal["close_position_in_range"] >= 0.65)
            & (down_reversal["upper_shadow_ratio"] <= 0.35)
            & (down_reversal["real_body_ratio"] >= 0.25)
            & (
                (down_reversal["has_turnover_context"] & (down_reversal["activity_confirmation_score"] >= 2))
                | (~down_reversal["has_turnover_context"] & (down_reversal["amount_confirmation_score"] >= 1))
            )
        ].copy()
        down_reversal["strategy_score"] = (
            down_reversal["today_change"].clip(lower=0, upper=8).fillna(0) * 0.8
            + down_reversal["vr_today"].clip(lower=0, upper=4).fillna(0) * 1.2
            + down_reversal["close_position_in_range"].fillna(0) * 6.0
            + down_reversal["real_body_ratio"].fillna(0) * 3.0
            + (-down_reversal["change_20d"]).clip(lower=0, upper=25).fillna(0) * 0.2
            + down_reversal["activity_confirmation_score"].fillna(0) * 1.0
        )
        down_reversal = _keep_best_candidates(
            down_reversal,
            "industry_1",
            "strategy_score",
            ratio=0.1,
            max_count=2,
            min_score=12,
        )
    strategies["down_reversal"] = down_reversal

    return strategies


def _build_long_term_strategy(df):
    if "steady_climb" in RETIRED_STRATEGIES:
        return df.iloc[0:0].copy()

    df_long = df[
        df["change_20d"].notna()
        & df["vr_5d"].notna()
        & df["amp_5d"].notna()
        & df["ind_change_5d"].notna()
        & df["ind_change_10d"].notna()
        & df["ind_alpha_change_5d"].notna()
        & df["ind_alpha_change_10d"].notna()
    ].copy()

    if df_long.empty:
        return df_long

    slow_climb = df_long[
        (df_long["change_5d"] > 0)
        & (df_long["change_10d"] > 0)
        & (df_long["change_20d"] > 0)
        & (df_long["latest_price"] >= df_long["ma20"])
        & (df_long["latest_price"] <= df_long["ma20"] * 1.08)
        & (df_long["ma5"] > df_long["ma10"])
        & (df_long["ma10"] > df_long["ma20"])
        & (df_long["ma20"] > df_long["ma60"])
        & (df_long["ma5_ma20_gap_ratio"].between(0, 0.08))
        & (df_long["breakout_close_ratio"].between(0.84, 0.96))
        & (df_long["close_vs_high_60d_ratio"].between(0.82, 0.95))
        & (df_long["today_change"].between(0.0, 4.5))
        & (df_long["vr_today"].between(0.7, 1.9))
        & (df_long["vr_5d"] >= 0.8)
        & (
            (df_long["has_turnover_context"] & (df_long["turnover_confirmation_score"] >= 1))
            | (~df_long["has_turnover_context"] & (df_long["amount_confirmation_score"] >= 1))
        )
        & (
            (df_long["has_turnover_context"] & (df_long["turnover_rate"] <= df_long["turnover_avg_10d"] * 1.6))
            | (
                ~df_long["has_turnover_context"]
                & (
                    (df_long["today_amount"] <= df_long["amount_avg_10d"] * 1.9)
                    | df_long["amount_avg_10d"].isna()
                )
            )
        )
        & (df_long["volatility_10_20_ratio"] <= 1.05)
        & (df_long["ma20_slope_5d"] > 0.1)
        & (df_long["ma60_slope_10d"] >= 0)
        & (df_long["close_position_in_range"] >= 0.45)
        & (df_long["today_amp"] <= df_long["amp_5d"] * 1.4)
        & (df_long["ind_change_5d"] > 0)
        & (df_long["ind_change_10d"] > 0)
        & (df_long["ind_alpha_change_5d"] > 0)
        & (df_long["ind_alpha_change_10d"] > 0)
    ].copy()

    slow_climb["strategy_score"] = (
        slow_climb["change_10d"].clip(lower=0, upper=18).fillna(0) * 0.45
        + slow_climb["ma20_slope_5d"].clip(lower=0, upper=2).fillna(0) * 3.0
        + slow_climb["activity_confirmation_score"].fillna(0) * 1.1
        + slow_climb["close_position_in_range"].fillna(0) * 4.0
        - ((slow_climb["volatility_10_20_ratio"] - 1).clip(lower=0).fillna(0) * 4.0)
    )
    return _keep_best_candidates(slow_climb, "industry_1", "strategy_score", ratio=0.08, max_count=4, min_score=10)


def _build_strategy_bundle(df):
    market_score = pd.to_numeric(df["change_5d"], errors="coerce").mean()
    market_breadth = pd.to_numeric(df["change_5d"], errors="coerce").gt(0).mean() * 100
    market_regime = _resolve_market_regime(market_score, market_breadth)
    market_env = risk_overlay.market_env_label(market_regime)

    strategies = _build_short_term_strategies(df, market_regime)
    strategies["steady_climb"] = _build_long_term_strategy(df)
    strategies = {
        strategy_type: _enrich_strategy_execution(strategy_df, strategy_type, market_regime)
        for strategy_type, strategy_df in strategies.items()
    }
    strategies = _apply_precision_budget(strategies, market_regime)

    return {
        "market_env": market_env,
        "market_score": market_score,
        "market_breadth": market_breadth,
        "market_regime": market_regime,
        "strategies": strategies,
    }


def _row_to_record(row, trade_date, strategy_type, note_column=None):
    industry_list = getattr(row, "industry_list", [])

    record = {
        "trade_date": trade_date,
        "strategy_type": strategy_type,
        "stock_code": _normalize_scalar(getattr(row, "stock_code", None)),
        "stock_name": _normalize_scalar(getattr(row, "stock_name", None)),
        "today_change": _normalize_scalar(getattr(row, "today_change", None)),
        "today_amount": _normalize_scalar(getattr(row, "today_amount", None)),
        "turnover_rate": _normalize_scalar(getattr(row, "turnover_rate", None)),
        "change_30d": _normalize_scalar(getattr(row, "change_30d", None)),
        "vr_today": _normalize_scalar(getattr(row, "vr_today", None)),
        "vr_30d": _normalize_scalar(getattr(row, "vr_30d", None)),
        "today_amp": _normalize_scalar(getattr(row, "today_amp", None)),
        "amp_30d": _normalize_scalar(getattr(row, "amp_30d", None)),
        "stock_rank": _normalize_scalar(getattr(row, "stock_rank", None)),
        "industry": ",".join(industry_list) if industry_list else "",
    }
    if note_column:
        record[note_column] = _normalize_scalar(
            getattr(row, note_column, None) or getattr(row, "strategy_note", None) or _get_strategy_note(strategy_type)
        )
    return record


def _clear_strategy_result_bucket(trade_date, strategy_types):
    if not trade_date:
        return

    for strategy_type in strategy_types:
        func.executeDelete("a_stock_strategy_result", {"trade_date": trade_date, "strategy_type": strategy_type})


def clear_daily_strategy_results(trade_date):
    _clear_strategy_result_bucket(trade_date, MANAGED_DAILY_STRATEGY_TYPES)


def _save_strategies(strategies, trade_date):
    strategy_counts = {}
    note_column = _resolve_strategy_note_column()
    if note_column:
        func.logInfo(f"策略备注将写入字段: {note_column}")
    else:
        func.logInfo("策略结果表未发现备注字段(strategy_note/strategy_remark/remark/note)，仅在汇总中输出备注")

    # 先按交易日清空当前日常策略桶，避免重复插入或保留已下线策略的旧结果。
    _clear_strategy_result_bucket(trade_date, MANAGED_DAILY_STRATEGY_TYPES)

    for strategy_type, strategy_df in strategies.items():
        func.logInfo(f"策略 {strategy_type} 形态备注: {_get_strategy_note(strategy_type)}")

        if strategy_df.empty:
            strategy_counts[strategy_type] = 0
            func.logInfo(f"策略 {strategy_type} 无数据，跳过保存")
            continue

        sort_columns = ["today_change", "stock_code"]
        ascending = [False, True]
        if "precision_score" in strategy_df.columns:
            sort_columns = ["precision_score", "execution_score", "strategy_score", "today_change", "stock_code"]
            ascending = [False, False, False, False, True]
        elif "strategy_score" in strategy_df.columns:
            sort_columns = ["strategy_score", "today_change", "stock_code"]
            ascending = [False, False, True]

        unique_df = (
            strategy_df.sort_values(sort_columns, ascending=ascending, na_position="last")
            .drop_duplicates(subset=["stock_code"])
            .copy()
        )
        unique_df = unique_df[unique_df["stock_code"].notna()].copy()
        if unique_df.empty:
            strategy_counts[strategy_type] = 0
            func.logInfo(f"策略 {strategy_type} 去重后无有效 stock_code，跳过保存")
            continue

        strategy_counts[strategy_type] = len(unique_df)

        total = len(unique_df)
        for index, row in enumerate(unique_df.itertuples(index=False), start=1):
            record = _row_to_record(row, trade_date, strategy_type, note_column=note_column)
            func.executeInsert("a_stock_strategy_result", record)

            if index % 50 == 0 or index == total:
                func.logInfo(f"策略 {strategy_type} 已保存 {index}/{total} 条")

    return trade_date, strategy_counts


def _collect_history_signals(history, overlay_frame=None, apply_risk_overlay=True):
    signal_frames = []
    temporal_features = _build_temporal_feature_frame(history)
    if apply_risk_overlay and overlay_frame is None:
        overlay_frame = risk_overlay.build_special_pool_overlay(history, all_dates=True)

    for trade_date, day_df in history.groupby("last_data_date", sort=True):
        prepared = _prepare_snapshot(day_df)
        if prepared.empty:
            continue
        prepared = _merge_temporal_features(prepared, temporal_features)

        strategy_bundle = _build_strategy_bundle(prepared)
        market_env = strategy_bundle["market_env"]
        market_score = strategy_bundle["market_score"]

        for strategy_type, strategy_df in strategy_bundle["strategies"].items():
            if strategy_df.empty:
                continue

            unique_df = strategy_df.drop_duplicates(subset=["stock_code"]).copy()
            unique_df["signal_date"] = trade_date
            unique_df["strategy_type"] = strategy_type
            if "strategy_note" not in unique_df.columns:
                unique_df["strategy_note"] = _get_strategy_note(strategy_type)
            else:
                unique_df["strategy_note"] = unique_df["strategy_note"].fillna(_get_strategy_note(strategy_type))
            if apply_risk_overlay:
                unique_df = _apply_risk_overlay_to_strategy_frame(
                    unique_df,
                    str(pd.to_datetime(trade_date).date()),
                    overlay_frame=overlay_frame,
                    history=history,
                )
                if unique_df.empty:
                    continue
            unique_df["market_env"] = market_env
            unique_df["market_score"] = market_score
            if "market_regime_segment" not in unique_df.columns:
                unique_df["market_regime_segment"] = strategy_bundle["market_regime"]
            if "market_regime_label" not in unique_df.columns:
                unique_df["market_regime_label"] = risk_overlay.market_regime_label(strategy_bundle["market_regime"])
            for overlay_column, default_value in [
                ("special_pool_label", risk_overlay.SPECIAL_POOL_LABELS["normal"]),
                ("risk_overlay_score", 0.0),
                ("risk_overlay_labels", "无明显特殊池风险"),
            ]:
                if overlay_column not in unique_df.columns:
                    unique_df[overlay_column] = default_value
            unique_df["signal_close"] = unique_df["latest_price"]
            signal_frames.append(unique_df[DETAIL_COLUMNS].copy())

    if not signal_frames:
        return pd.DataFrame(columns=DETAIL_COLUMNS)

    return pd.concat(signal_frames, ignore_index=True)


def _build_price_paths(history, holding_days, entry_offset_days, round_trip_cost_pct=0.0):
    price_columns = ["last_data_date", "stock_code", "latest_price", "today_change"]
    for column in ["today_open", "today_high", "today_low"]:
        if column in history.columns:
            price_columns.append(column)

    price_history = history[price_columns].copy()
    for column in ["latest_price", "today_change", "today_open", "today_high", "today_low"]:
        if column not in price_history.columns:
            price_history[column] = pd.NA
        price_history[column] = pd.to_numeric(price_history[column], errors="coerce")

    price_history["today_open"] = price_history["today_open"].fillna(price_history["latest_price"])
    price_history["today_high"] = price_history["today_high"].fillna(
        price_history[["today_open", "latest_price"]].max(axis=1)
    )
    price_history["today_low"] = price_history["today_low"].fillna(
        price_history[["today_open", "latest_price"]].min(axis=1)
    )
    price_history = price_history.dropna(subset=["last_data_date", "stock_code", "latest_price"])
    price_history = price_history[price_history["latest_price"] > 0].copy()
    price_history = price_history.sort_values(["stock_code", "last_data_date"]).reset_index(drop=True)

    grouped = price_history.groupby("stock_code")
    price_history["entry_date"] = grouped["last_data_date"].shift(-entry_offset_days)
    price_history["entry_close"] = grouped["latest_price"].shift(-entry_offset_days)
    price_history["entry_open"] = grouped["today_open"].shift(-entry_offset_days)
    price_history["entry_high"] = grouped["today_high"].shift(-entry_offset_days)
    price_history["entry_low"] = grouped["today_low"].shift(-entry_offset_days)
    price_history["entry_change"] = grouped["today_change"].shift(-entry_offset_days)
    price_history["entry_avg_price"] = (
        price_history[["entry_open", "entry_high", "entry_low", "entry_close"]].mean(axis=1)
    )
    price_history["entry_gap_pct"] = (
        (price_history["entry_open"] - price_history["latest_price"]) / price_history["latest_price"] * 100
    )
    entry_range = (price_history["entry_high"] - price_history["entry_low"]).where(
        price_history["entry_high"] > price_history["entry_low"]
    )
    price_history["entry_close_position"] = (
        (price_history["entry_close"] - price_history["entry_low"]) / entry_range
    )
    price_history["entry_close_from_open_pct"] = (
        (price_history["entry_close"] - price_history["entry_open"]) / price_history["entry_open"] * 100
    )
    high_open_has_pullback = (
        (price_history["entry_gap_pct"] <= 3.0)
        | (price_history["entry_low"] <= price_history["latest_price"] * 1.015)
    )
    price_history["entry_acceptance_ok"] = (
        price_history["entry_open"].gt(0)
        & price_history["entry_close"].gt(0)
        & high_open_has_pullback.fillna(False)
        & price_history["entry_close_position"].fillna(0).ge(0.45)
        & price_history["entry_close_from_open_pct"].fillna(-99).ge(-2.5)
    )
    price_history["entry_acceptance_price"] = price_history["entry_avg_price"].where(
        price_history["entry_acceptance_ok"]
    )

    for hold_days in holding_days:
        exit_offset = entry_offset_days + hold_days
        price_history[f"exit_date_{hold_days}d"] = grouped["last_data_date"].shift(-exit_offset)
        price_history[f"exit_close_{hold_days}d"] = grouped["latest_price"].shift(-exit_offset)
        price_history[f"exit_change_{hold_days}d"] = grouped["today_change"].shift(-exit_offset)
        price_history[f"gross_return_{hold_days}d"] = (
            (price_history[f"exit_close_{hold_days}d"] - price_history["entry_close"])
            / price_history["entry_close"]
            * 100
        )
        price_history[f"return_{hold_days}d"] = price_history[f"gross_return_{hold_days}d"] - round_trip_cost_pct
        price_history[f"gross_return_next_open_{hold_days}d"] = (
            (price_history[f"exit_close_{hold_days}d"] - price_history["entry_open"])
            / price_history["entry_open"]
            * 100
        )
        price_history[f"return_next_open_{hold_days}d"] = (
            price_history[f"gross_return_next_open_{hold_days}d"] - round_trip_cost_pct
        )
        price_history[f"gross_return_next_avg_{hold_days}d"] = (
            (price_history[f"exit_close_{hold_days}d"] - price_history["entry_avg_price"])
            / price_history["entry_avg_price"]
            * 100
        )
        price_history[f"return_next_avg_{hold_days}d"] = (
            price_history[f"gross_return_next_avg_{hold_days}d"] - round_trip_cost_pct
        )
        price_history[f"gross_return_acceptance_{hold_days}d"] = (
            (price_history[f"exit_close_{hold_days}d"] - price_history["entry_acceptance_price"])
            / price_history["entry_acceptance_price"]
            * 100
        )
        price_history[f"return_acceptance_{hold_days}d"] = (
            price_history[f"gross_return_acceptance_{hold_days}d"] - round_trip_cost_pct
        )

    price_history = price_history.replace([math.inf, -math.inf], pd.NA)
    return price_history


def _build_empty_holding_stats():
    return {
        "evaluated_trades": 0,
        "trade_win_rate": None,
        "avg_return": None,
        "avg_gross_return": None,
        "avg_cost_impact": None,
        "median_return": None,
        "return_std": None,
        "best_return": None,
        "worst_return": None,
        "daily_observations": 0,
        "daily_win_rate": None,
        "avg_daily_return": None,
    }


def _summarize_holding_returns(valid_details, return_col, gross_return_col=None):
    if valid_details.empty:
        return _build_empty_holding_stats()

    returns = valid_details[return_col]
    daily_returns = valid_details.groupby("signal_date")[return_col].mean()
    avg_gross_return = None
    avg_cost_impact = None
    if gross_return_col and gross_return_col in valid_details.columns:
        avg_gross_return = _round_or_none(valid_details[gross_return_col].mean(), 4)
        avg_cost_impact = _round_or_none(valid_details[gross_return_col].mean() - returns.mean(), 4)

    return {
        "evaluated_trades": int(len(valid_details)),
        "trade_win_rate": _round_or_none((returns > 0).mean() * 100, 2),
        "avg_return": _round_or_none(returns.mean(), 4),
        "avg_gross_return": avg_gross_return,
        "avg_cost_impact": avg_cost_impact,
        "median_return": _round_or_none(returns.median(), 4),
        "return_std": _round_or_none(returns.std(), 4),
        "best_return": _round_or_none(returns.max(), 4),
        "worst_return": _round_or_none(returns.min(), 4),
        "daily_observations": int(len(daily_returns)),
        "daily_win_rate": _round_or_none((daily_returns > 0).mean() * 100, 2),
        "avg_daily_return": _round_or_none(daily_returns.mean(), 4),
    }


def _summarize_backtest(details, holding_days):
    if details.empty:
        return {}

    summary = {}

    for strategy_type, strategy_details in details.groupby("strategy_type"):
        strategy_summary = {
            "signal_count": int(len(strategy_details)),
            "signal_days": int(strategy_details["signal_date"].nunique()),
            "avg_signals_per_day": _round_or_none(
                len(strategy_details) / max(strategy_details["signal_date"].nunique(), 1), 4
            ),
        }

        for hold_days in holding_days:
            return_col = f"return_{hold_days}d"
            gross_return_col = f"gross_return_{hold_days}d"
            valid_details = strategy_details[strategy_details[return_col].notna()].copy()
            strategy_summary[f"hold_{hold_days}d"] = _summarize_holding_returns(
                valid_details, return_col, gross_return_col
            )

        mode_summary = {}
        for mode_name, mode_config in BACKTEST_ENTRY_MODE_DEFINITIONS.items():
            mode_holds = {
                "label": mode_config["label"],
                "description": mode_config["description"],
            }
            for hold_days in holding_days:
                return_col = mode_config["return_template"].format(hold_days=hold_days)
                gross_return_col = mode_config["gross_template"].format(hold_days=hold_days)
                if return_col not in strategy_details.columns:
                    continue
                valid_details = strategy_details[strategy_details[return_col].notna()].copy()
                mode_holds[f"hold_{hold_days}d"] = _summarize_holding_returns(
                    valid_details,
                    return_col,
                    gross_return_col if gross_return_col in strategy_details.columns else None,
                )
            mode_summary[mode_name] = mode_holds
        strategy_summary["entry_mode_summary"] = mode_summary

        summary[strategy_type] = strategy_summary

    return summary


def _summarize_market_regime_backtest(details, holding_days):
    if details.empty or "market_regime_segment" not in details.columns:
        return {}

    summary = {}
    for regime, regime_details in details.groupby("market_regime_segment"):
        if regime_details.empty:
            continue
        label_values = regime_details.get("market_regime_label")
        label = None
        if label_values is not None and label_values.notna().any():
            label = label_values.dropna().iloc[0]
        regime_summary = {
            "label": label or risk_overlay.market_regime_label(regime),
            "signal_count": int(len(regime_details)),
            "signal_days": int(regime_details["signal_date"].nunique()),
            "strategy_counts": regime_details["strategy_type"].fillna("unknown").value_counts().to_dict(),
        }
        for hold_days in holding_days:
            return_col = f"return_{hold_days}d"
            gross_return_col = f"gross_return_{hold_days}d"
            if return_col not in regime_details.columns:
                continue
            valid_details = regime_details[regime_details[return_col].notna()].copy()
            regime_summary[f"hold_{hold_days}d"] = _summarize_holding_returns(
                valid_details,
                return_col,
                gross_return_col if gross_return_col in regime_details.columns else None,
            )

        entry_mode_summary = {}
        for mode_name, mode_config in BACKTEST_ENTRY_MODE_DEFINITIONS.items():
            mode_holds = {
                "label": mode_config["label"],
                "description": mode_config["description"],
            }
            for hold_days in holding_days:
                return_col = mode_config["return_template"].format(hold_days=hold_days)
                gross_return_col = mode_config["gross_template"].format(hold_days=hold_days)
                if return_col not in regime_details.columns:
                    continue
                valid_details = regime_details[regime_details[return_col].notna()].copy()
                mode_holds[f"hold_{hold_days}d"] = _summarize_holding_returns(
                    valid_details,
                    return_col,
                    gross_return_col if gross_return_col in regime_details.columns else None,
                )
            entry_mode_summary[mode_name] = mode_holds
        regime_summary["entry_mode_summary"] = entry_mode_summary
        summary[regime or "unknown"] = regime_summary

    return summary


def analysis_gu_piao_data(run_backtest=False, backtest_kwargs=None, expected_trade_date=None):
    func.logInfo("开始分析个股数据")
    print("开始分析个股数据")

    snapshot = _load_snapshot()
    if snapshot.empty:
        _emit_runtime_status("固定策略推荐跳过: reason=snapshot_empty")
        result = {
            "market_env": "未知",
            "market_regime": "unknown",
            "market_score": None,
            "strategy_counts": {},
            "strategy_notes": {},
        }
        if run_backtest:
            result["backtest_summary"] = {}
        return result

    trade_date = _resolve_trade_date(snapshot)
    expected_trade_date_text = _normalize_trade_date_text(expected_trade_date)
    trade_date_text = _normalize_trade_date_text(trade_date)
    if expected_trade_date_text and trade_date_text != expected_trade_date_text:
        clear_daily_strategy_results(expected_trade_date_text)
        _emit_runtime_status(
            f"固定策略推荐跳过: reason=snapshot_trade_date_mismatch, "
            f"expected={expected_trade_date_text}, actual={trade_date_text}"
        )
        result = {
            "trade_date": trade_date_text,
            "expected_trade_date": expected_trade_date_text,
            "market_env": "未知",
            "market_regime": "unknown",
            "market_score": None,
            "strategy_counts": {},
            "strategy_notes": {},
            "skipped_reason": "snapshot_trade_date_mismatch",
        }
        if run_backtest:
            result["backtest_summary"] = {}
        return result

    snapshot = _filter_snapshot_by_trade_date(snapshot, trade_date)
    _clear_strategy_result_bucket(trade_date, MANAGED_DAILY_STRATEGY_TYPES)
    snapshot_quality = _assess_snapshot_quality(snapshot, trade_date)
    if not snapshot_quality["meets_min_coverage"]:
        _emit_runtime_status(
            f"最新快照覆盖不足，跳过推荐: trade_date={trade_date}, "
            f"coverage={snapshot_quality['coverage_ratio']}%, "
            f"trade_date_count={snapshot_quality['trade_date_count']}, "
            f"universe_count={snapshot_quality['universe_count']}"
        )
        result = {
            "trade_date": str(trade_date),
            "market_env": "未知",
            "market_regime": "unknown",
            "market_score": None,
            "market_breadth": None,
            "strategy_counts": {},
            "strategy_notes": {},
            "skipped_reason": "snapshot_coverage_below_threshold",
            "snapshot_coverage_ratio": snapshot_quality["coverage_ratio"],
            "snapshot_trade_date_count": snapshot_quality["trade_date_count"],
            "snapshot_universe_count": snapshot_quality["universe_count"],
        }
        if run_backtest:
            result["backtest_summary"] = {}
        return result

    history = _load_recent_history_window(trade_date, lookback_trade_days=3)
    temporal_features = _build_temporal_feature_frame(history)
    df = _prepare_snapshot(snapshot)
    if df.empty:
        _emit_runtime_status("固定策略推荐跳过: reason=industry_info_empty")
        result = {
            "market_env": "未知",
            "market_regime": "unknown",
            "market_score": None,
            "strategy_counts": {},
            "strategy_notes": {},
        }
        if run_backtest:
            result["backtest_summary"] = {}
        return result
    df = _merge_temporal_features(df, temporal_features)

    strategy_bundle = _build_strategy_bundle(df)
    strategy_health = _load_strategy_health(trade_date)
    if not strategy_health:
        _emit_runtime_status(
            f"策略健康度快照缺失，跳过推荐落库: trade_date={trade_date}, "
            "reason=strategy_health_unavailable"
        )
        result = {
            "trade_date": str(trade_date),
            "market_env": strategy_bundle["market_env"],
            "market_regime": strategy_bundle["market_regime"],
            "market_score": _round_or_none(strategy_bundle["market_score"], 4),
            "market_breadth": _round_or_none(strategy_bundle["market_breadth"], 2),
            "snapshot_coverage_ratio": snapshot_quality["coverage_ratio"],
            "strategy_counts": {},
            "strategy_health": {},
            "strategy_notes": {},
            "skipped_reason": "strategy_health_unavailable",
        }
        if run_backtest:
            backtest_result = backtest_strategy_candidates(**(backtest_kwargs or {}))
            result["backtest_summary"] = backtest_result["summary"]
        return result

    strategy_bundle["strategies"] = _apply_strategy_health_gate(strategy_bundle["strategies"], strategy_health)
    overlay_frame = risk_overlay.build_special_pool_overlay(None, trade_date=trade_date)
    overlay_summary = risk_overlay.summarize_overlay(overlay_frame)
    strategy_bundle["strategies"] = _apply_risk_overlay_to_strategies(
        strategy_bundle["strategies"],
        trade_date,
        overlay_frame=overlay_frame,
        include_external=True,
    )
    market_score = strategy_bundle["market_score"]
    market_breadth = strategy_bundle["market_breadth"]
    market_env = strategy_bundle["market_env"]
    market_regime = strategy_bundle["market_regime"]

    if market_regime == "strong":
        func.logInfo("市场处于强势区间,趋势跟随和强突破更有持续性")
    elif market_regime == "neutral":
        func.logInfo("市场处于震荡区间,更适合保守观察,不要把所有突破都当成有效趋势")
    elif market_regime == "weak":
        func.logInfo("市场处于弱势区间,进攻型信号需要大幅收缩,优先看防守和低位反转")
    else:
        func.logInfo("市场环境无法稳定识别,本次推荐偏保守")

    trade_date, strategy_counts = _save_strategies(strategy_bundle["strategies"], trade_date)
    active_strategy_counts = {name: count for name, count in strategy_counts.items() if count > 0}

    summary = {
        "trade_date": str(trade_date),
        "market_env": market_env,
        "market_regime": market_regime,
        "market_score": _round_or_none(market_score, 4),
        "market_breadth": _round_or_none(market_breadth, 2),
        "snapshot_coverage_ratio": snapshot_quality["coverage_ratio"],
        "strategy_counts": active_strategy_counts,
        "recommendation_tiers": {
            "formal_recommendation": int(sum(active_strategy_counts.values())),
            "observation_candidate": 0,
            "research_value": 0,
            "note": "每日固定策略只写入正式推荐；观察候选和研究价值由自适应模型/单股分析承接。",
        },
        "risk_overlay": overlay_summary,
        "strategy_health": strategy_health,
        "strategy_notes": {
            name: _get_strategy_note(name)
            for name, count in active_strategy_counts.items()
            if count > 0
        },
    }

    if run_backtest:
        backtest_result = backtest_strategy_candidates(**(backtest_kwargs or {}))
        summary["backtest_summary"] = backtest_result["summary"]

    formal_recommendation_count = summary["recommendation_tiers"]["formal_recommendation"]
    if formal_recommendation_count == 0:
        _emit_runtime_status(
            _build_no_daily_recommendation_message(
                market_env,
                market_regime,
                strategy_bundle["strategies"],
                strategy_health,
            )
        )

    _emit_runtime_status(
        f"固定策略推荐结果: trade_date={trade_date}, "
        f"market_env={market_env}, market_regime={market_regime}, "
        f"formal_recommendation={formal_recommendation_count}, "
        f"strategy_counts={active_strategy_counts}, "
        f"risk_blocked={overlay_summary.get('blocked_formal')}, "
        f"risk_downgraded={overlay_summary.get('downgraded')}"
    )
    func.logInfo(summary)
    func.logInfo("分析个股数据完毕")
    print("分析个股数据完毕")
    return summary


def backtest_strategy_candidates(
    start_date=None,
    end_date=None,
    holding_days=DEFAULT_HOLDING_DAYS,
    entry_offset_days=DEFAULT_ENTRY_OFFSET_DAYS,
    fee_bps=DEFAULT_FEE_BPS,
    slippage_bps=DEFAULT_SLIPPAGE_BPS,
    apply_limit_filter=True,
    limit_pct=DEFAULT_LIMIT_PCT,
):
    func.logInfo("开始回测个股策略")

    normalized_holding_days = tuple(sorted({int(day) for day in holding_days if int(day) > 0}))
    if not normalized_holding_days:
        raise ValueError("holding_days 至少需要一个大于 0 的持有周期")

    if entry_offset_days < 1:
        raise ValueError("entry_offset_days 需要大于等于 1，避免使用当日收盘数据产生未来函数")

    fee_bps = float(fee_bps)
    slippage_bps = float(slippage_bps)
    if fee_bps < 0 or slippage_bps < 0:
        raise ValueError("fee_bps 和 slippage_bps 不能为负数")
    limit_pct = float(limit_pct)
    if limit_pct <= 0:
        raise ValueError("limit_pct 需要大于 0")

    round_trip_cost_pct = (fee_bps + slippage_bps) * 2 / 100

    history = _load_history(start_date=start_date, end_date=end_date)
    if history.empty:
        func.logInfo("历史表没有可回测数据，退出")
        return {"summary": {}, "detail": pd.DataFrame()}

    overlay_frame = risk_overlay.build_special_pool_overlay(history, all_dates=True)
    signals = _collect_history_signals(history, overlay_frame=overlay_frame, apply_risk_overlay=True)
    if signals.empty:
        func.logInfo("历史回测期间没有策略信号，退出")
        return {"summary": {}, "detail": signals}

    price_paths = _build_price_paths(
        history,
        normalized_holding_days,
        entry_offset_days,
        round_trip_cost_pct=round_trip_cost_pct,
    )
    merge_columns = [
        "last_data_date",
        "stock_code",
        "entry_date",
        "entry_close",
        "entry_open",
        "entry_avg_price",
        "entry_acceptance_price",
        "entry_acceptance_ok",
        "entry_gap_pct",
        "entry_close_position",
        "entry_close_from_open_pct",
        "entry_change",
    ]

    for hold_days in normalized_holding_days:
        merge_columns.extend(
            [
                f"exit_date_{hold_days}d",
                f"exit_close_{hold_days}d",
                f"exit_change_{hold_days}d",
                f"gross_return_{hold_days}d",
                f"return_{hold_days}d",
                f"gross_return_next_open_{hold_days}d",
                f"return_next_open_{hold_days}d",
                f"gross_return_next_avg_{hold_days}d",
                f"return_next_avg_{hold_days}d",
                f"gross_return_acceptance_{hold_days}d",
                f"return_acceptance_{hold_days}d",
            ]
        )
    merge_columns = [column for column in merge_columns if column in price_paths.columns]

    detail = signals.merge(
        price_paths[merge_columns].rename(columns={"last_data_date": "signal_date"}),
        on=["signal_date", "stock_code"],
        how="left",
    )
    detail = detail.sort_values(["signal_date", "strategy_type", "stock_code"]).reset_index(drop=True)
    filtered_by_limit_count = 0

    if apply_limit_filter:
        for hold_days in normalized_holding_days:
            return_col = f"return_{hold_days}d"
            gross_return_col = f"gross_return_{hold_days}d"
            exit_change_col = f"exit_change_{hold_days}d"

            entry_abs = detail["entry_change"].abs()
            exit_abs = detail[exit_change_col].abs()
            entry_tradable = entry_abs.isna() | (entry_abs < limit_pct)
            exit_tradable = exit_abs.isna() | (exit_abs < limit_pct)
            tradable = entry_tradable & exit_tradable

            mode_columns = [(return_col, gross_return_col)]
            for mode_config in BACKTEST_ENTRY_MODE_DEFINITIONS.values():
                mode_return_col = mode_config["return_template"].format(hold_days=hold_days)
                mode_gross_col = mode_config["gross_template"].format(hold_days=hold_days)
                if mode_return_col in detail.columns and (mode_return_col, mode_gross_col) not in mode_columns:
                    mode_columns.append((mode_return_col, mode_gross_col))

            for mode_return_col, mode_gross_col in mode_columns:
                filtered_mask = detail[mode_return_col].notna() & (~tradable)
                filtered_by_limit_count += int(filtered_mask.sum())
                columns_to_null = [mode_return_col]
                if mode_gross_col in detail.columns:
                    columns_to_null.append(mode_gross_col)
                detail.loc[filtered_mask, columns_to_null] = pd.NA

    summary = {
        "sample_start": str(history["last_data_date"].min().date()),
        "sample_end": str(history["last_data_date"].max().date()),
        "trade_days": int(history["last_data_date"].nunique()),
        "signal_count": int(len(detail)),
        "strategy_count": int(detail["strategy_type"].nunique()),
        "entry_offset_days": int(entry_offset_days),
        "holding_days": list(normalized_holding_days),
        "fee_bps": fee_bps,
        "slippage_bps": slippage_bps,
        "round_trip_cost_pct": _round_or_none(round_trip_cost_pct, 4),
        "apply_limit_filter": bool(apply_limit_filter),
        "limit_pct": limit_pct,
        "filtered_by_limit_count": int(filtered_by_limit_count),
        "entry_modes": {
            mode_name: {
                "label": mode_config["label"],
                "description": mode_config["description"],
            }
            for mode_name, mode_config in BACKTEST_ENTRY_MODE_DEFINITIONS.items()
        },
        "risk_overlay_summary": risk_overlay.summarize_overlay(overlay_frame),
        "assumption": (
            "hold_1d means the signal is generated on day t, entry is simulated on t+1 by the selected entry mode, "
            "and exit uses the close after the holding window; acceptance mode skips trades without next-day support confirmation"
        ),
        "strategy_summary": _summarize_backtest(detail, normalized_holding_days),
        "market_regime_summary": _summarize_market_regime_backtest(detail, normalized_holding_days),
        "strategy_notes": {
            name: _get_strategy_note(name)
            for name in sorted(detail["strategy_type"].dropna().unique().tolist())
        },
    }

    func.logInfo(summary)
    func.logInfo("回测个股策略完毕")
    return {"summary": summary, "detail": detail}


if __name__ == "__main__":
    analysis_gu_piao_data()
