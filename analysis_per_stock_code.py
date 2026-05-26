"""单只股票综合分析入口。

作用:
    面向指定股票代码输出更细的单票诊断，把实时行情、历史位置、
    自适应短线画像、长跑跟踪、风险覆盖、持仓建议和解释文本整合到
    一份结构化结果里。它适合人工查看某一只股票为什么能买、该不该买、
    或为什么需要回避。

流程:
    先确定分析日期并加载目标股票近期历史窗口；
    再复用自适应模型构建画像和评分，同时补充单票下跌修复、持仓完整性
    等专项诊断；
    然后调用自适应风险覆盖入口叠加特殊池和事件风险；
    最后汇总为控制台输出或 JSON 结果，供人工复盘和决策参考。
"""

import argparse
import contextlib
import datetime as dt
import io
import json
import math
import re
import sys
import time

import pandas as pd

import analysis_gu_piao_history_adaptive_model as adaptive
import analysis_gu_piao_adaptive_risk_overlay_model as risk_overlay


HOLDING_INTACT_TREND_STATES = {"周月同步抬升", "月线抬升，周线整理", "月线抬升，周线回踩"}
HOLDING_COST_MAX_LOSS_PCT = 6.0
SMALL_VALIDATION_STRATEGY_TYPES = {"weak_rs_follow"}
LONG_RUNWAY_STRATEGY_TYPES = {getattr(adaptive, "LONG_RUNWAY_STRATEGY_TYPE", "long_runway")}
RECENT_SAVED_STRATEGY_LOOKBACK_DAYS = 30
REALTIME_SOURCE_NAME = "sina_hq"
REALTIME_SOURCE_URL = "https://hq.sinajs.cn/list={symbols}"
REALTIME_FALLBACK_SOURCE_NAME = "tencent_quote"
REALTIME_FALLBACK_SOURCE_URL = "https://qt.gtimg.cn/q={symbol}"
REALTIME_PREV_CLOSE_TOLERANCE_PCT = 0.5
ADAPTIVE_DROP_MIN_HISTORY_ROWS = 30
ADAPTIVE_DROP_MIN_EVENT_ROWS = 6
ADAPTIVE_DROP_MIN_GROUP_ROWS = 3
ADAPTIVE_DROP_TRIGGER_PERCENTILE = 40.0
ADAPTIVE_DROP_SHARP_PERCENTILE = 25.0
ADAPTIVE_DROP_EXTREME_PERCENTILE = 10.0


def _build_status_emitter(enabled=False):
    started_at = time.perf_counter()
    last_at = started_at

    def emit(message):
        nonlocal last_at
        if not enabled:
            return
        now = time.perf_counter()
        print(
            f"[个股分析] {message} "
            f"(elapsed={now - started_at:.1f}s, step={now - last_at:.1f}s)",
            file=sys.stderr,
            flush=True,
        )
        last_at = now

    return emit


def _round(value, digits=2):
    if value is None or pd.isna(value):
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _date_text(value):
    return adaptive._to_date_text(value)


def _resolve_market_env(snapshot):
    market_change_5d = _round(pd.to_numeric(snapshot.get("market_change_5d"), errors="coerce").mean(), 4)
    market_breadth_5d = _round(pd.to_numeric(snapshot.get("market_breadth_5d"), errors="coerce").mean(), 2)
    market_regime = risk_overlay.classify_market_regime(market_change_5d, market_breadth_5d)
    market_env = risk_overlay.market_env_label(market_regime)

    return {
        "market_env": market_env,
        "market_regime": market_regime,
        "market_change_5d": market_change_5d,
        "market_breadth_5d": market_breadth_5d,
    }


def _load_history_window(analysis_date=None, lookback_trade_days=adaptive.SHORT_TERM_LOOKBACK_TRADE_DAYS):
    lookback_trade_days = max(20, int(lookback_trade_days or adaptive.SHORT_TERM_LOOKBACK_TRADE_DAYS))

    if analysis_date:
        trade_days = adaptive._query_frame(
            """
            SELECT last_data_date
            FROM (
                SELECT DISTINCT last_data_date
                FROM a_stock_analysis_history
                WHERE last_data_date <= %s
                ORDER BY last_data_date DESC
                LIMIT %s
            ) recent_trade_days
            ORDER BY last_data_date
            """,
            params=[analysis_date, lookback_trade_days],
        )
        if trade_days.empty:
            return pd.DataFrame()
        start_date = _date_text(trade_days["last_data_date"].iloc[0])
        return adaptive._load_history(
            start_date=start_date,
            end_date=analysis_date,
            columns=adaptive.LONG_RUNWAY_FRAME_COLUMNS,
        )

    return adaptive._load_history(
        tail_trade_days=lookback_trade_days,
        columns=adaptive.LONG_RUNWAY_FRAME_COLUMNS,
    )


def _build_profiles(prepared_history):
    horizon_profiles = {}
    style_horizon_profiles = {}
    for horizon_days in adaptive.HORIZON_DAYS:
        horizon_profiles[horizon_days] = adaptive._build_horizon_profile(prepared_history, horizon_days)
        style_horizon_profiles[horizon_days] = adaptive._build_style_horizon_profiles(prepared_history, horizon_days)
    return horizon_profiles, style_horizon_profiles


def _numeric(frame, column):
    if column not in frame.columns:
        return pd.Series(pd.NA, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def _number(value):
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ratio_pct_series(numerator, denominator):
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce").replace(0, pd.NA)
    return (numerator / denominator - 1) * 100


def _percentile_rank_lower(series, value):
    value = _number(value)
    if value is None:
        return None
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float((values <= value).mean() * 100)


def _avg_return_stats(frame, return_col="forward_return_5d"):
    if frame is None or frame.empty or return_col not in frame.columns:
        return {"count": 0, "avg_return": None, "win_rate": None}
    returns = pd.to_numeric(frame[return_col], errors="coerce").dropna()
    if returns.empty:
        return {"count": 0, "avg_return": None, "win_rate": None}
    return {
        "count": int(len(returns)),
        "avg_return": _round(returns.mean(), 2),
        "win_rate": _round((returns > 0).mean() * 100, 2),
    }


def _build_adaptive_drop_event_stats(events):
    if events is None or events.empty:
        return {
            "event_count": 0,
            "evaluated_count": 0,
            "recovered": _avg_return_stats(pd.DataFrame()),
            "unrecovered": _avg_return_stats(pd.DataFrame()),
            "all": _avg_return_stats(pd.DataFrame()),
            "has_support": False,
        }

    if "forward_return_5d" not in events.columns:
        evaluated = events.iloc[0:0].copy()
    else:
        evaluated = events[pd.to_numeric(events["forward_return_5d"], errors="coerce").notna()].copy()
    recovered_flag = (
        evaluated["recovered_key_line"].fillna(False)
        if "recovered_key_line" in evaluated.columns
        else pd.Series(False, index=evaluated.index)
    )
    recovered = evaluated[recovered_flag].copy()
    unrecovered = evaluated[~recovered_flag].copy()
    return {
        "event_count": int(len(events)),
        "evaluated_count": int(len(evaluated)),
        "recovered": _avg_return_stats(recovered),
        "unrecovered": _avg_return_stats(unrecovered),
        "all": _avg_return_stats(evaluated),
        "has_support": bool(len(evaluated) >= ADAPTIVE_DROP_MIN_EVENT_ROWS),
    }


def _build_adaptive_drop_profile(target, prepared, trade_date):
    if prepared is None or prepared.empty:
        return {"enabled": False, "reason": "history_empty"}

    stock_code = adaptive._normalize_stock_code(target.get("stock_code"))
    history = prepared[prepared["stock_code"] == stock_code].copy()
    if history.empty:
        return {"enabled": False, "reason": "stock_history_empty"}

    trade_date_text = _date_text(trade_date)
    history["last_data_date_text"] = history["last_data_date"].apply(_date_text)
    if trade_date_text:
        history = history[history["last_data_date_text"] <= trade_date_text].copy()
    history = history.sort_values("last_data_date").reset_index(drop=True)
    if len(history) < ADAPTIVE_DROP_MIN_HISTORY_ROWS:
        return {
            "enabled": False,
            "reason": "insufficient_history",
            "sample_rows": int(len(history)),
            "min_rows": ADAPTIVE_DROP_MIN_HISTORY_ROWS,
        }

    history["prev_close"] = pd.to_numeric(history["latest_price"], errors="coerce").shift(1)
    history["prior_low"] = pd.to_numeric(history["today_low"], errors="coerce").shift(1)
    history["low_from_prev_pct"] = _ratio_pct_series(history["today_low"], history["prev_close"])
    close_change = pd.to_numeric(history.get("today_change"), errors="coerce")
    history["close_from_prev_pct"] = close_change.where(
        close_change.notna(),
        _ratio_pct_series(history["latest_price"], history["prev_close"]),
    )
    history["recovered_prior_low"] = pd.to_numeric(history["latest_price"], errors="coerce") >= pd.to_numeric(
        history["prior_low"], errors="coerce"
    )
    history["recovered_ma5"] = pd.to_numeric(history["latest_price"], errors="coerce") >= pd.to_numeric(
        history["ma5"], errors="coerce"
    )
    history["recovered_key_line"] = history["recovered_prior_low"].fillna(False) & history["recovered_ma5"].fillna(False)

    low_series = pd.to_numeric(history["low_from_prev_pct"], errors="coerce").dropna()
    if len(low_series) < ADAPTIVE_DROP_MIN_HISTORY_ROWS:
        return {
            "enabled": False,
            "reason": "insufficient_low_samples",
            "sample_rows": int(len(low_series)),
            "min_rows": ADAPTIVE_DROP_MIN_HISTORY_ROWS,
        }

    q10 = float(low_series.quantile(ADAPTIVE_DROP_EXTREME_PERCENTILE / 100))
    q25 = float(low_series.quantile(ADAPTIVE_DROP_SHARP_PERCENTILE / 100))
    q40 = float(low_series.quantile(ADAPTIVE_DROP_TRIGGER_PERCENTILE / 100))
    event_pool = history[pd.to_numeric(history["low_from_prev_pct"], errors="coerce") <= q40].copy()
    sharp_pool = history[pd.to_numeric(history["low_from_prev_pct"], errors="coerce") <= q25].copy()
    extreme_pool = history[pd.to_numeric(history["low_from_prev_pct"], errors="coerce") <= q10].copy()

    return {
        "enabled": True,
        "sample_rows": int(len(low_series)),
        "sample_start": history["last_data_date_text"].dropna().min(),
        "sample_end": history["last_data_date_text"].dropna().max(),
        "trigger_percentile": ADAPTIVE_DROP_TRIGGER_PERCENTILE,
        "sharp_percentile": ADAPTIVE_DROP_SHARP_PERCENTILE,
        "extreme_percentile": ADAPTIVE_DROP_EXTREME_PERCENTILE,
        "trigger_low_pct": _round(q40, 2),
        "sharp_low_pct": _round(q25, 2),
        "extreme_low_pct": _round(q10, 2),
        "low_samples": [float(value) for value in low_series.tolist()],
        "event_stats": {
            "trigger": _build_adaptive_drop_event_stats(event_pool),
            "sharp": _build_adaptive_drop_event_stats(sharp_pool),
            "extreme": _build_adaptive_drop_event_stats(extreme_pool),
        },
    }


def _score_single_record(record, horizon_profiles, style_horizon_profiles):
    trend_profile = adaptive._build_trend_state(record)
    hard_style = adaptive._resolve_candidate_style(record)
    result = dict(record)
    result.update(
        {
            "style": None,
            "style_label": None,
            "hard_style": hard_style,
            "hard_style_label": adaptive.STYLE_LABELS.get(hard_style) if hard_style else None,
            "style_gate_passed": bool(hard_style),
            "trend_state": trend_profile["state"],
            "trend_score": trend_profile["score"],
            "trend_detail": trend_profile["summary"],
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
    )

    profile_weights = {
        horizon: float(profile.get("profile_weight") or 0)
        for horizon, profile in horizon_profiles.items()
    }
    total_profile_weight = sum(profile_weights.values())
    if total_profile_weight <= 0:
        profile_weights = {horizon: 1.0 for horizon in horizon_profiles}

    horizon_scores = {}
    horizon_details = {}
    horizon_family_scores = {}
    horizon_coverages = {}
    horizon_sources = {}

    for horizon_days, profile in horizon_profiles.items():
        selected_source = "generic"
        selected_result = adaptive._score_one_profile(record, profile)

        best_style_name = None
        best_style_result = None
        for style_name in adaptive.STYLE_PRIORITY:
            if not adaptive._record_matches_style(record, style_name):
                continue
            style_profile = style_horizon_profiles.get(horizon_days, {}).get(style_name)
            if not style_profile:
                continue
            style_result = adaptive._score_one_profile(record, style_profile)
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
        result[f"source_{horizon_days}d"] = selected_source
        horizon_scores[horizon_days] = score
        horizon_details[horizon_days] = selected_result
        horizon_family_scores[horizon_days] = selected_result.get("family_scores", {})
        horizon_coverages[horizon_days] = selected_result.get("coverage", 0)
        horizon_sources[horizon_days] = selected_source

    valid_horizon_scores = [
        (horizon_days, score, profile_weights.get(horizon_days, 1.0))
        for horizon_days, score in horizon_scores.items()
        if score is not None
    ]
    if valid_horizon_scores:
        weighted_sum = sum(score * weight for _, score, weight in valid_horizon_scores)
        weight_sum = sum(weight for _, _, weight in valid_horizon_scores)
        result["adaptive_score"] = adaptive._round_or_none(weighted_sum / weight_sum, 2)

        dominant_horizon = max(valid_horizon_scores, key=lambda item: item[1])[0]
        dominant_source = horizon_sources.get(dominant_horizon)
        dominant_style_label = adaptive.STYLE_LABELS.get(dominant_source) if dominant_source != "generic" else None
        result["dominant_horizon"] = dominant_horizon
        result["dominant_family"] = horizon_details[dominant_horizon].get("top_family")
        result["dominant_family_score"] = horizon_details[dominant_horizon].get("top_family_score")
        result["coverage"] = adaptive._round_or_none(horizon_coverages.get(dominant_horizon), 4)
        result["family_scores"] = horizon_family_scores.get(dominant_horizon, {})
        result["style"] = dominant_source if dominant_source != "generic" else None
        result["style_label"] = dominant_style_label
        result["dominant_style"] = dominant_source if dominant_source != "generic" else None
        result["dominant_style_label"] = dominant_style_label
        result["reason"] = adaptive._build_reason(
            horizon_details[dominant_horizon],
            dominant_horizon,
            style_label=dominant_style_label,
            trend_state=trend_profile["state"],
        )
    else:
        result["coverage"] = 0
        result["reason"] = "历史样本不足，当前无法稳定评分。"

    risk_overlay = adaptive._score_short_term_risk(record, result.get("style"))
    result.update(risk_overlay)
    if result["adaptive_score"] is not None:
        result["risk_adjusted_score"] = adaptive._round_or_none(
            max(0.0, float(result["adaptive_score"]) - float(result["risk_score"] or 0.0)),
            2,
        )
    return result


def _style_history_mask(frame, style_name):
    if not style_name:
        return pd.Series(True, index=frame.index)
    try:
        mask = adaptive._style_mask(frame, style_name)
        return mask.fillna(False)
    except (KeyError, TypeError):
        return pd.Series(True, index=frame.index)


def _trend_history_mask(frame, trend_state):
    if trend_state in ("周月同步抬升", "月线抬升，周线回踩"):
        return (_numeric(frame, "change_5d") > 0) & (_numeric(frame, "change_20d") > 0)
    if trend_state == "月线抬升，周线整理":
        return (_numeric(frame, "change_20d") > 0) & (_numeric(frame, "price_vs_ma20") >= -0.05)
    if trend_state == "周线反弹，月线待确认":
        return (_numeric(frame, "change_5d") > 0) & (_numeric(frame, "change_20d") <= 0)
    if trend_state == "弱势修复":
        return (_numeric(frame, "change_20d") < 0) & (_numeric(frame, "price_vs_ma20") <= 0)
    return pd.Series(True, index=frame.index)


def _similarity_frame(prepared_history, target, max_rows=240):
    latest_date = pd.to_datetime(target.get("last_data_date"), errors="coerce")
    history = prepared_history.copy()
    if not pd.isna(latest_date):
        history = history[pd.to_datetime(history["last_data_date"], errors="coerce") < latest_date].copy()
    if history.empty:
        return history

    style_mask = _style_history_mask(history, target.get("style"))
    trend_mask = _trend_history_mask(history, target.get("trend_state"))
    base = history[style_mask & trend_mask].copy()
    if base.empty:
        base = history[style_mask].copy()
    if base.empty:
        base = history.copy()

    features = [
        ("change_5d", 8.0),
        ("change_20d", 18.0),
        ("change_30d", 28.0),
        ("price_vs_ma20", 0.08),
        ("price_vs_ma60", 0.12),
        ("ma20_vs_ma60", 0.08),
        ("close_to_20d_high", 0.14),
        ("close_to_20d_low", 0.18),
        ("today_amp", 4.5),
        ("vr_today", 0.9),
        ("turnover_rate", 7.0),
    ]

    distance_sum = pd.Series(0.0, index=base.index)
    feature_count = 0
    for column, scale in features:
        target_value = _number(target.get(column))
        if target_value is None or column not in base.columns:
            continue
        values = _numeric(base, column)
        distance = ((values - target_value).abs() / scale).clip(upper=3.0)
        distance_sum = distance_sum.add(distance.fillna(3.0), fill_value=3.0)
        feature_count += 1

    if feature_count == 0:
        base["similarity_score"] = 0.0
        return base.head(int(max_rows)).copy()

    base["similarity_score"] = (100 * (-distance_sum / feature_count).map(math.exp)).round(2)
    base = base.sort_values(["similarity_score", "last_data_date"], ascending=[False, False])
    return base.head(int(max_rows)).copy()


def _return_stats(frame, horizon_days):
    return_col = f"forward_return_{horizon_days}d"
    if frame.empty or return_col not in frame.columns:
        return {
            "horizon_days": horizon_days,
            "sample_count": 0,
            "avg_return": None,
            "median_return": None,
            "win_rate": None,
            "worst_return": None,
            "best_return": None,
            "profit_factor": None,
            "confidence_interval": None,
        }

    returns = pd.to_numeric(frame[return_col], errors="coerce").dropna()
    if returns.empty:
        return {
            "horizon_days": horizon_days,
            "sample_count": 0,
            "avg_return": None,
            "median_return": None,
            "win_rate": None,
            "worst_return": None,
            "best_return": None,
            "profit_factor": None,
            "confidence_interval": None,
        }

    positive_sum = float(returns[returns > 0].sum())
    negative_sum = abs(float(returns[returns < 0].sum()))
    profit_factor = None
    if negative_sum > 0:
        profit_factor = positive_sum / negative_sum
    elif positive_sum > 0:
        profit_factor = 99.0

    sample_count = int(len(returns))
    avg_return = float(returns.mean())
    std = float(returns.std(ddof=1)) if sample_count > 1 else 0.0
    margin = 1.96 * std / math.sqrt(sample_count) if sample_count > 1 else 0.0

    return {
        "horizon_days": horizon_days,
        "sample_count": sample_count,
        "avg_return": _round(avg_return, 4),
        "median_return": _round(float(returns.median()), 4),
        "win_rate": _round(float((returns > 0).mean() * 100), 2),
        "worst_return": _round(float(returns.min()), 4),
        "best_return": _round(float(returns.max()), 4),
        "profit_factor": _round(profit_factor, 2),
        "confidence_interval": [_round(avg_return - margin, 4), _round(avg_return + margin, 4)],
    }


def _grade_evidence(stats_5d, stats_10d):
    primary = stats_10d if int(stats_10d.get("sample_count") or 0) >= 30 else stats_5d
    count = int(primary.get("sample_count") or 0)
    avg_return = _number(primary.get("avg_return"))
    win_rate = _number(primary.get("win_rate"))
    profit_factor = _number(primary.get("profit_factor"))
    worst_return = _number(primary.get("worst_return"))

    if count >= 60 and avg_return is not None and avg_return >= 2 and win_rate is not None and win_rate >= 55 and profit_factor is not None and profit_factor >= 1.3:
        grade = "强"
    elif count >= 30 and avg_return is not None and avg_return > 0 and win_rate is not None and win_rate >= 52 and (profit_factor is None or profit_factor >= 1.05):
        grade = "中"
    elif count >= 15 and avg_return is not None and avg_return > 0 and win_rate is not None and win_rate >= 50:
        grade = "弱"
    else:
        grade = "不足"

    warnings = []
    if count < 30:
        warnings.append("相似样本偏少")
    if worst_return is not None and worst_return <= -10:
        warnings.append(f"历史最差样本回撤{_round(worst_return, 2)}%")
    if win_rate is not None and win_rate < 52:
        warnings.append("胜率不够稳定")
    if avg_return is not None and avg_return <= 0:
        warnings.append("平均收益未转正")

    return {
        "grade": grade,
        "primary_horizon": primary.get("horizon_days"),
        "warnings": warnings,
    }


def _build_evidence(prepared_history, target):
    similar = _similarity_frame(prepared_history, target)
    style_name = target.get("style")
    trend_state = target.get("trend_state")
    style_mask = _style_history_mask(prepared_history, style_name)
    trend_mask = _trend_history_mask(prepared_history, trend_state)
    broad = prepared_history[style_mask & trend_mask].copy()

    similar_stats = {
        "5d": _return_stats(similar, 5),
        "10d": _return_stats(similar, 10),
    }
    broad_stats = {
        "5d": _return_stats(broad, 5),
        "10d": _return_stats(broad, 10),
    }
    grade = _grade_evidence(similar_stats["5d"], similar_stats["10d"])
    return {
        "similar_sample": {
            "sample_size": int(len(similar)),
            "avg_similarity": _round(similar["similarity_score"].mean(), 2) if "similarity_score" in similar.columns and not similar.empty else None,
            "stats": similar_stats,
        },
        "style_trend_sample": {
            "sample_size": int(len(broad)),
            "style": style_name,
            "style_label": adaptive.STYLE_LABELS.get(style_name) if style_name else "全部风格",
            "trend_state": trend_state,
            "filter_label": "同风格同趋势样本" if style_name else "同趋势样本",
            "stats": broad_stats,
        },
        "grade": grade,
    }


def _safe_pct_rank(series, value, ascending=True):
    value = _number(value)
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if value is None or numeric.empty:
        return None
    if ascending:
        return _round(float((numeric <= value).mean() * 100), 2)
    return _round(float((numeric >= value).mean() * 100), 2)


def _build_industry_view(snapshot, target):
    industry = target.get("industry_1")
    if not industry or "industry_1" not in snapshot.columns:
        return {
            "industry": industry,
            "sample_count": 0,
            "strength_grade": "未知",
            "summary": "行业数据不足，无法判断行业共振。",
        }

    group = snapshot[snapshot["industry_1"] == industry].copy()
    if group.empty:
        return {
            "industry": industry,
            "sample_count": 0,
            "strength_grade": "未知",
            "summary": "行业样本为空，无法判断行业共振。",
        }

    industry_change_5d = _round(_numeric(group, "change_5d").mean(), 4)
    industry_change_20d = _round(_numeric(group, "change_20d").mean(), 4)
    industry_breadth_5d = _round(float((_numeric(group, "change_5d") > 0).mean() * 100), 2)
    industry_breadth_20d = _round(float((_numeric(group, "change_20d") > 0).mean() * 100), 2)
    market_change_5d = _round(_numeric(snapshot, "change_5d").mean(), 4)
    market_change_20d = _round(_numeric(snapshot, "change_20d").mean(), 4)
    alpha_5d = _round((industry_change_5d or 0) - (market_change_5d or 0), 4)
    alpha_20d = _round((industry_change_20d or 0) - (market_change_20d or 0), 4)

    stock_industry_rank_20d = _safe_pct_rank(group["change_20d"], target.get("change_20d"), ascending=True)
    stock_industry_rank_5d = _safe_pct_rank(group["change_5d"], target.get("change_5d"), ascending=True)

    score = 0
    if alpha_5d is not None and alpha_5d > 0:
        score += 1
    if alpha_20d is not None and alpha_20d > 0:
        score += 1
    if industry_breadth_5d is not None and industry_breadth_5d >= 50:
        score += 1
    if stock_industry_rank_20d is not None and stock_industry_rank_20d >= 60:
        score += 1

    if score >= 3:
        strength_grade = "强共振"
    elif score == 2:
        strength_grade = "中性偏强"
    elif score == 1:
        strength_grade = "偏弱"
    else:
        strength_grade = "弱共振"

    summary = (
        f"{industry}近5日均值{industry_change_5d}%，20日均值{industry_change_20d}%，"
        f"相对市场5日{alpha_5d}%，20日{alpha_20d}%；"
        f"个股20日涨幅处行业{stock_industry_rank_20d}分位。"
    )

    return {
        "industry": industry,
        "sample_count": int(len(group)),
        "strength_grade": strength_grade,
        "industry_change_5d": industry_change_5d,
        "industry_change_20d": industry_change_20d,
        "industry_breadth_5d": industry_breadth_5d,
        "industry_breadth_20d": industry_breadth_20d,
        "market_change_5d": market_change_5d,
        "market_change_20d": market_change_20d,
        "industry_alpha_5d": alpha_5d,
        "industry_alpha_20d": alpha_20d,
        "stock_industry_rank_5d": stock_industry_rank_5d,
        "stock_industry_rank_20d": stock_industry_rank_20d,
        "summary": summary,
    }


def _build_position_view(target, trade_plan, cost_price=None, holding_stop_view=None):
    latest_price = _number(target.get("latest_price"))
    stop_price = _number((holding_stop_view or {}).get("effective_stop_price"))
    if stop_price is None:
        stop_price = _number(trade_plan.get("stop_price"))
    cost = _number(cost_price)

    if cost is None or latest_price is None:
        return {
            "has_cost": False,
            "cost_price": cost,
            "latest_price": latest_price,
            "pnl_pct": None,
            "advice": "未提供持仓成本，按无仓和通用持仓纪律处理。",
        }

    pnl_pct = (latest_price / cost - 1) * 100 if cost > 0 else None
    stop_loss_from_cost = (stop_price / cost - 1) * 100 if cost and stop_price else None
    distance_to_stop = (latest_price / stop_price - 1) * 100 if latest_price and stop_price else None

    if stop_price is not None and latest_price <= stop_price:
        advice = f"现价已不高于防守价{_round(stop_price, 2)}，持仓优先降风险，不做补仓。"
    elif pnl_pct is not None and pnl_pct >= 12:
        advice = "已有较明显浮盈，优先移动止盈；若放量滞涨或跌破短期均线，分批兑现。"
    elif pnl_pct is not None and pnl_pct >= 4:
        advice = "已有浮盈，可继续按计划持有；跌破防守价或盈利回吐过半时降低仓位。"
    elif pnl_pct is not None and pnl_pct <= -6:
        advice = "已有较大浮亏，不建议补仓摊低；只按防守价和反弹质量决定去留。"
    else:
        advice = "持仓盈亏不极端，按防守价管理；未触发新买点前不加仓。"

    return {
        "has_cost": True,
        "cost_price": _round(cost, 2),
        "latest_price": _round(latest_price, 2),
        "pnl_pct": _round(pnl_pct, 2),
        "stop_price": _round(stop_price, 2),
        "dynamic_stop_price": _round(trade_plan.get("stop_price"), 2),
        "effective_stop_price": _round(stop_price, 2),
        "stop_loss_from_cost_pct": _round(stop_loss_from_cost, 2),
        "distance_to_stop_pct": _round(distance_to_stop, 2),
        "advice": advice,
    }


def _format_share_count(value):
    number = _number(value)
    if number is None:
        return None
    return int(max(0, number))


def _round_trade_lot(value):
    shares = _format_share_count(value)
    if shares is None or shares <= 0:
        return None
    if shares < 100:
        return shares
    return int(shares // 100 * 100)


def _price_range_text(low, high):
    low_text = _price_text(low)
    high_text = _price_text(high)
    if low_text == "--" and high_text == "--":
        return "--"
    if low_text == high_text:
        return low_text
    return f"{low_text}-{high_text}"


def _nearest_level_above(price, levels):
    price = _number(price)
    if price is None:
        return None
    valid_levels = []
    for level in levels:
        number = _number(level)
        if number is not None and number > price:
            valid_levels.append(number)
    return min(valid_levels) if valid_levels else None


def _nearest_level_below(price, levels):
    price = _number(price)
    if price is None:
        return None
    valid_levels = []
    for level in levels:
        number = _number(level)
        if number is not None and number < price:
            valid_levels.append(number)
    return max(valid_levels) if valid_levels else None


def _collect_price_levels(level_items, latest_price=None, side=None):
    latest_price = _number(latest_price)
    grouped = {}
    for label, raw_price in level_items:
        price = _number(raw_price)
        if price is None or price <= 0:
            continue
        if latest_price is not None:
            if side == "above" and price <= latest_price:
                continue
            if side == "below" and price >= latest_price:
                continue
        key = _round(price, 2)
        if key is None:
            continue
        grouped.setdefault(key, {"price": key, "labels": []})
        if label and label not in grouped[key]["labels"]:
            grouped[key]["labels"].append(label)
    return sorted(grouped.values(), key=lambda item: item["price"])


def _level_basis_text(level):
    if not level:
        return "--"
    labels = "/".join(level.get("labels") or [])
    price = _price_text(level.get("price"))
    return f"{labels}{price}" if labels else price


def _price_zone_from_levels(levels, max_width_pct=4.0):
    if not levels:
        return "--"
    low = _number(levels[0].get("price"))
    high = low
    if len(levels) >= 2:
        second = _number(levels[1].get("price"))
        if low and second and (second / low - 1) * 100 <= max_width_pct:
            high = second
    return _price_range_text(low, high)


def _build_attack_defense_lines(
    candidate,
    trade_plan,
    holding_stop,
    recommendation_tier,
    formal_candidate,
    market,
    scorecard,
    professional_view,
    position_view,
):
    latest_price = _number(candidate.get("latest_price"))
    ma5 = _number(candidate.get("ma5"))
    ma10 = _number(candidate.get("ma10"))
    ma20 = _number(candidate.get("ma20"))
    ma60 = _number(candidate.get("ma60"))
    today_high = _number(candidate.get("today_high"))
    today_low = _number(candidate.get("today_low"))
    high_20d = _number(candidate.get("high_20d"))
    low_20d = _number(candidate.get("low_20d"))
    high_60d = _number(candidate.get("high_60d"))
    effective_stop = _number((holding_stop or {}).get("effective_stop_price"))
    dynamic_stop = _number((holding_stop or {}).get("dynamic_stop_price"))
    if effective_stop is None:
        effective_stop = _number(trade_plan.get("stop_price"))

    technical_supports = _collect_price_levels(
        [
            ("日内低点", today_low),
            ("20日低点", low_20d),
            ("20日均线", ma20),
            ("60日均线", ma60),
        ],
        latest_price,
        side="below",
    )
    technical_resistances = _collect_price_levels(
        [
            ("5日均线", ma5),
            ("10日均线", ma10),
            ("20日均线", ma20),
            ("日内高点", today_high),
            ("20日高点", high_20d),
            ("60日高点", high_60d),
        ],
        latest_price,
        side="above",
    )

    technical_defense = technical_supports[-1] if technical_supports else None
    technical_attack_zone = _price_zone_from_levels(technical_resistances[:2])
    if technical_attack_zone == "--" and high_20d is not None and latest_price is not None and high_20d <= latest_price:
        technical_attack_zone = _price_text(high_20d)

    if latest_price is not None and ma20 is not None and latest_price < ma20:
        technical_state = "弱势修复，先看能否收复均线"
    elif latest_price is not None and ma20 is not None and ma60 is not None and latest_price > ma20 and ma20 >= ma60:
        technical_state = "趋势仍在均线结构上方"
    else:
        technical_state = "区间震荡，按支撑压力处理"

    if technical_defense:
        technical_defense_text = _price_text(technical_defense.get("price"))
    else:
        technical_defense_text = "--"
    technical_basis = []
    if technical_defense:
        technical_basis.append(f"最近下方支撑:{_level_basis_text(technical_defense)}")
    if technical_resistances:
        technical_basis.append(
            "上方压力:" + "、".join(_level_basis_text(item) for item in technical_resistances[:3])
        )
    if ma20 is not None:
        technical_basis.append(f"20日均线:{_price_text(ma20)}")

    below_ma20 = bool(latest_price is not None and ma20 is not None and latest_price < ma20)
    stop_broken = bool(latest_price is not None and effective_stop is not None and latest_price <= effective_stop)
    professional_attack_candidates = []
    professional_attack_zone = "--"
    if stop_broken:
        reclaim_levels = [level for level in [effective_stop, ma20 if below_ma20 else None, ma10, ma5] if level is not None]
        if reclaim_levels:
            professional_attack_zone = _price_text(max(reclaim_levels))
        professional_state = "风控优先，收复前不谈进攻"
        professional_action = "先站回专业进攻线并解除弱势/风险信号；未确认前，反弹主要用于降风险，不按进攻处理。"
    elif below_ma20:
        reclaim_levels = [level for level in [effective_stop, ma20] if level is not None]
        if reclaim_levels:
            professional_attack_zone = _price_text(max(reclaim_levels))
        professional_state = "修复确认前不加仓"
        professional_action = "收盘重新站上专业进攻线且量能承接改善后，才从持有观察转为重新评估。"
    elif formal_candidate:
        professional_attack_candidates = [
            ("日内高点", today_high),
            ("20日高点", high_20d),
            ("60日高点", high_60d),
        ]
        professional_state = "模型允许进攻，但仍等承接"
        professional_action = "进攻线放量站上且仍为正式推荐时，才考虑按计划加仓或跟随。"
    else:
        professional_attack_candidates = [
            ("20日均线", ma20),
            ("20日高点", high_20d),
            ("60日高点", high_60d),
        ]
        professional_state = "专业侧不支持主动进攻"
        professional_action = "未恢复正式推荐或高质量观察前，进攻线只作为重新评估线，不作为直接买点。"

    if professional_attack_zone == "--":
        professional_attack_levels = _collect_price_levels(
            professional_attack_candidates,
            latest_price,
            side="above",
        )
        professional_attack_zone = _price_zone_from_levels(professional_attack_levels[:2], max_width_pct=5.0)

    professional_basis = [
        f"推荐层级:{recommendation_tier or '--'}",
        f"交易质量评级:{(scorecard or {}).get('rating') or '--'}({(scorecard or {}).get('score') if (scorecard or {}).get('score') is not None else '--'}分)",
        f"证据等级:{(professional_view or {}).get('evidence_grade') or '--'}",
        f"市场:{(market or {}).get('market_env') or '--'}",
    ]
    if holding_stop:
        professional_basis.append(
            f"防守来源:{holding_stop.get('source') or '--'}，当日策略防守{_price_text(dynamic_stop)}"
        )
        if holding_stop.get("cost_stop_price") is not None:
            professional_basis.append(f"成本风控线:{holding_stop.get('cost_stop_price')}")
    if below_ma20 and ma20 is not None:
        professional_basis.append(f"专业进攻参考20日均线:{_price_text(ma20)}")
    if (position_view or {}).get("has_cost"):
        professional_basis.append(f"浮盈亏:{(position_view or {}).get('pnl_pct')}%")

    return {
        "technical": {
            "state": technical_state,
            "defense_line": _round(technical_defense.get("price"), 2) if technical_defense else None,
            "defense_text": technical_defense_text,
            "attack_zone": technical_attack_zone,
            "basis": technical_basis,
            "action": "跌破技术防守且收不回，说明短线结构继续走弱；放量站上技术进攻线，才算从反弹进入修复。",
        },
        "professional": {
            "state": professional_state,
            "defense_line": _round(effective_stop, 2),
            "defense_text": _price_text(effective_stop),
            "attack_zone": professional_attack_zone,
            "basis": professional_basis,
            "action": professional_action,
        },
    }


def _build_holding_t_strategy(
    candidate,
    trade_plan,
    recommendation_tier,
    formal_candidate,
    position_view,
    intraday_view,
    liquidity_view,
    market,
    shares=None,
    available_shares=None,
):
    latest_price = _number(candidate.get("latest_price"))
    if latest_price is None or latest_price <= 0:
        return {
            "enabled": False,
            "feasibility": "不建议",
            "direction": "不做T",
            "reason": "缺少有效收盘价，无法制定盘后做T预案。",
        }

    total_shares = _format_share_count(shares)
    explicit_available = available_shares is not None
    sellable_shares = _format_share_count(available_shares)
    sellable_shares_known = sellable_shares is not None
    if sellable_shares is None and total_shares is not None:
        sellable_shares = total_shares
        sellable_shares_known = True

    has_position = bool((total_shares and total_shares > 0) or (position_view or {}).get("has_cost"))
    has_sellable = bool((sellable_shares and sellable_shares > 0) or (has_position and not sellable_shares_known))

    cost_price = _number((position_view or {}).get("cost_price"))
    pnl_pct = _number((position_view or {}).get("pnl_pct"))
    stop_price = _number(trade_plan.get("stop_price"))
    today_high = _number(candidate.get("today_high"))
    today_low = _number(candidate.get("today_low"))
    high_20d = _number(candidate.get("high_20d"))
    low_20d = _number(candidate.get("low_20d"))
    high_60d = _number(candidate.get("high_60d"))
    ma20 = _number(candidate.get("ma20"))
    ma60 = _number(candidate.get("ma60"))
    today_amp = _number(candidate.get("today_amp")) or 0.0
    amp_20d = _number(candidate.get("amp_20d")) or 0.0
    amp_30d = _number(candidate.get("amp_30d")) or 0.0
    risk_adjusted_score = _number(candidate.get("risk_adjusted_score"))
    risk_score = _number(candidate.get("risk_score"))
    overlay_block = bool(candidate.get("risk_overlay_block_formal"))
    overlay_downgrade = bool(candidate.get("risk_overlay_downgrade"))
    market_env = (market or {}).get("market_env") or "未知"
    tape_grade = (intraday_view or {}).get("tape_grade")
    close_position_pct = _number((intraday_view or {}).get("close_position_pct"))
    upper_shadow_pct = _number((intraday_view or {}).get("upper_shadow_pct")) or 0.0
    liquidity_grade = (liquidity_view or {}).get("grade") or "未知"
    amount = _number((liquidity_view or {}).get("amount"))
    turnover = _number((liquidity_view or {}).get("turnover_rate"))

    amp_base = max(today_amp, amp_20d, amp_30d, 2.0)
    t_edge_pct = max(1.2, min(4.0, amp_base * 0.35))
    pullback_pct = max(1.0, min(3.0, amp_base * 0.25))
    stop_distance_pct = (latest_price / stop_price - 1) * 100 if stop_price and stop_price > 0 else None

    support = _nearest_level_below(
        latest_price,
        [stop_price, ma20, ma60, today_low, low_20d],
    )
    resistance = _nearest_level_above(
        latest_price,
        [today_high, high_20d, high_60d],
    )

    sell_low = latest_price * (1 + t_edge_pct / 100)
    sell_high = latest_price * (1 + (t_edge_pct + max(0.8, t_edge_pct * 0.45)) / 100)
    if resistance is not None:
        sell_low = max(sell_low, resistance * 0.985)
        sell_high = max(sell_low, resistance * 1.01)

    buy_high = latest_price * (1 - pullback_pct / 100)
    if support is not None:
        buy_high = max(buy_high, support * 1.005)
    buy_high = min(buy_high, latest_price * 0.995)
    buy_low = buy_high * 0.985
    if support is not None:
        buy_low = max(buy_low, support * 0.99)
    if stop_price is not None:
        buy_low = max(buy_low, stop_price)
    if buy_low > buy_high:
        buy_low = buy_high

    feasible_space = amp_base >= 3.0 and (sell_low / latest_price - buy_high / latest_price) * 100 >= 1.0
    liquid_enough = (
        liquidity_grade != "流动性有瑕疵"
        and (amount is None or amount >= 80000000)
        and (turnover is None or turnover >= 0.8)
    )
    trend_quality = (
        formal_candidate
        or recommendation_tier == adaptive.RECOMMENDATION_TIER_OBSERVE
        or (risk_adjusted_score is not None and risk_adjusted_score >= 72 and (risk_score or 99) <= 4)
    )
    weak_tape = (
        tape_grade in ("冲高回落", "收盘偏弱")
        or upper_shadow_pct >= 35
        or (close_position_pct is not None and close_position_pct <= 35)
    )
    strong_tape = (
        tape_grade == "承接较强"
        or (close_position_pct is not None and close_position_pct >= 70 and upper_shadow_pct < 30)
    )
    cost_cushion = pnl_pct is not None and pnl_pct >= 4
    deep_loss = pnl_pct is not None and pnl_pct <= -6
    near_stop = stop_distance_pct is not None and stop_distance_pct <= 3
    reverse_t_allowed = (
        formal_candidate
        and strong_tape
        and market_env in ("强势", "偏强")
        and risk_adjusted_score is not None
        and risk_adjusted_score >= 82
        and risk_score is not None
        and risk_score <= 2
        and not overlay_block
        and not overlay_downgrade
        and not near_stop
        and liquid_enough
        and feasible_space
    )

    reasons = []
    warnings = []
    if not has_position:
        reasons.append("未提供持仓成本或持仓股数，无法确认是否有底仓可做T")
    if not explicit_available and total_shares is not None:
        warnings.append("未提供可卖股数，T仓数量按总持仓估算，实盘以券商可卖为准")
    if has_position and total_shares is None and sellable_shares is None:
        warnings.append("未提供持仓股数和可卖股数，仅输出T仓比例和价位，不计算具体股数")
    if sellable_shares_known and not has_sellable:
        warnings.append("未提供或没有可卖底仓，普通A股不能用当日新买仓做日内T")
    if not feasible_space:
        warnings.append("按日线波动估算，次日可覆盖成本的T空间不足")
    if not liquid_enough:
        warnings.append("流动性不足，做T容易被价差和滑点吞掉")
    if overlay_block:
        warnings.append("风险覆盖层拦截正式买点，不能用做T变相加仓")
    elif overlay_downgrade:
        warnings.append("事件覆盖提示，T仓比例必须压低")
    if near_stop:
        warnings.append("现价距离防守价过近，优先控制回撤")

    direction = "不做T"
    feasibility = "不建议"
    action = "不做T，只按防守价管理原仓。"

    if not has_position:
        feasibility = "信息不足"
        direction = "仅观察"
        action = "缺少持仓信息，只能给盘后观察价位；不输出实盘T指令。"
    elif overlay_block or (deep_loss and not trend_quality) or near_stop:
        feasibility = "谨慎"
        direction = "只减不回补"
        action = "若次日冲高到卖出区，只降低风险，不急于买回；重新站稳防守区后再评估。"
    elif sellable_shares_known and not has_sellable:
        feasibility = "不建议"
        direction = "不做T"
        action = "没有确认可卖底仓，普通A股不能执行日内回转。"
    elif not feasible_space or not liquid_enough:
        feasibility = "不建议"
        direction = "不做T"
        action = "波动或流动性不足，盘后只保留支撑/压力观察，不主动做T。"
    elif weak_tape or cost_cushion or recommendation_tier in (adaptive.RECOMMENDATION_TIER_RESEARCH, adaptive.RECOMMENDATION_TIER_AVOID):
        feasibility = "谨慎" if recommendation_tier in (adaptive.RECOMMENDATION_TIER_RESEARCH, adaptive.RECOMMENDATION_TIER_AVOID) else "可做"
        direction = "正T"
        action = "先等次日冲高到卖出区减一部分，只有回踩买回区且承接恢复时才买回。"
    elif reverse_t_allowed:
        feasibility = "可做"
        direction = "反T"
        action = "只在回踩买回区先小买，随后冲高到卖出区卖出同等旧仓，保持总仓位不放大。"
    elif trend_quality:
        feasibility = "谨慎"
        direction = "正T"
        action = "优先冲高减、回踩承接确认再买回；若没有回踩，不强行补回。"
    else:
        feasibility = "不建议"
        direction = "不做T"
        action = "模型和证据没有形成足够优势，持仓只按防守价处理。"

    base_ratio = 0.0
    if direction in ("正T", "反T"):
        base_ratio = 0.25 if feasibility == "可做" else 0.15
        if overlay_downgrade or market_env in ("偏弱", "未知"):
            base_ratio = min(base_ratio, 0.12)
        if direction == "反T":
            base_ratio = min(base_ratio, 0.1)
        if deep_loss:
            base_ratio = min(base_ratio, 0.1)
    elif direction == "只减不回补":
        base_ratio = 0.15 if feasibility == "谨慎" else 0.1

    basis_shares = sellable_shares if sellable_shares is not None else total_shares
    planned_shares = _round_trade_lot(basis_shares * base_ratio) if basis_shares and base_ratio > 0 else None
    if planned_shares is not None and sellable_shares is not None:
        planned_shares = min(planned_shares, sellable_shares)
    if planned_shares is not None and planned_shares < 100:
        warnings.append("计划T仓不足100股，买回会受100股整数倍约束，实际意义有限")

    confidence_points = 0
    if feasible_space:
        confidence_points += 1
    if liquid_enough:
        confidence_points += 1
    if trend_quality:
        confidence_points += 1
    if has_sellable:
        confidence_points += 1
    if overlay_block or near_stop:
        confidence_points -= 1
    confidence = "高" if confidence_points >= 4 and feasibility == "可做" else "中" if confidence_points >= 2 else "低"

    day_rule = (
        "这是收盘后基于日线OHLC生成的次日预案；库里没有分钟线，不能判断次日先到卖点还是先到买点。"
    )
    close_rule = "卖出后没有回踩到买回区，不强行买回，接受降仓；买入后不能冲高卖出旧仓，则收盘前检查是否放大了净仓位。"
    if direction == "只减不回补":
        close_rule = "冲高减仓后不急于买回；只有后续重新站稳支撑并解除风险信号，再重新评估。"
    elif direction in ("不做T", "仅观察"):
        close_rule = "不执行T，只执行防守价和仓位纪律。"

    return {
        "enabled": direction in ("正T", "反T", "只减不回补"),
        "feasibility": feasibility,
        "direction": direction,
        "confidence": confidence,
        "action": action,
        "reason": "；".join(reasons) if reasons else action,
        "total_shares": total_shares,
        "available_shares": sellable_shares,
        "available_shares_explicit": explicit_available,
        "planned_t_ratio_pct": _round(base_ratio * 100, 1) if base_ratio > 0 else 0,
        "planned_t_shares": planned_shares,
        "sell_zone": {
            "low": _round(sell_low, 2),
            "high": _round(sell_high, 2),
            "text": _price_range_text(sell_low, sell_high),
        },
        "buyback_zone": {
            "low": _round(buy_low, 2),
            "high": _round(buy_high, 2),
            "text": _price_range_text(buy_low, buy_high),
        },
        "support_price": _round(support, 2),
        "resistance_price": _round(resistance, 2),
        "stop_price": _round(stop_price, 2),
        "cost_price": _round(cost_price, 2),
        "pnl_pct": _round(pnl_pct, 2),
        "t_edge_pct": _round(t_edge_pct, 2),
        "pullback_pct": _round(pullback_pct, 2),
        "risk_checks": {
            "has_position": has_position,
            "has_sellable_shares": has_sellable,
            "sellable_shares_known": sellable_shares_known,
            "feasible_space": feasible_space,
            "liquid_enough": liquid_enough,
            "trend_quality": trend_quality,
            "weak_tape": weak_tape,
            "strong_tape": strong_tape,
            "overlay_block": overlay_block,
            "overlay_downgrade": overlay_downgrade,
            "near_stop": near_stop,
        },
        "warnings": list(dict.fromkeys(warnings)),
        "abandon_rules": [
            f"跌破防守价{_price_text(stop_price)}且不能快速收回，不做买回，只先降风险。",
            "冲高没有量能或留下更长上影，只减不追。",
            "回踩到买回区但不能重新站上分时均线/开盘价，不买回。",
            "风险公告、特殊股票池或市场继续偏弱时，T仓比例继续下调。",
        ],
        "close_rule": close_rule,
        "day_rule": day_rule,
    }


def _build_execution_scenarios(candidate, trade_plan, formal_candidate, a_share_profile=None, intraday_view=None):
    latest_price = _number(candidate.get("latest_price"))
    stop_price = _number(trade_plan.get("stop_price"))
    high_open_price = _round(latest_price * 1.03, 2) if latest_price else None
    weak_open_price = _round(latest_price * 0.98, 2) if latest_price else None
    stop_text = _round(stop_price, 2)
    board = (a_share_profile or {}).get("board") or "A股"
    limit_pct = _number((a_share_profile or {}).get("limit_pct")) or 10.0
    limit_up_price = (intraday_view or {}).get("limit_up_price")
    tape_grade = (intraday_view or {}).get("tape_grade")

    if formal_candidate:
        normal = "平开或小幅高开，先等回踩不破关键支撑且重新放量，再按计划小到中等仓位参与。"
        high_open = f"若高开超过3%（约高于{high_open_price}），但不回踩承接，不追，放弃当日买点。"
    else:
        normal = "未进入正式候选，平开也不主动买入，只观察是否重新满足模型门槛。"
        high_open = f"若高开超过3%（约高于{high_open_price}），更不追；没有正式信号只看不做。"

    return {
        "a_share_rule": f"{board}涨跌幅约{limit_pct:g}%，涨停参考价{limit_up_price or '--'}；A股T+1，买入当天不能按日内止损卖出。",
        "normal_open": normal,
        "high_open": high_open,
        "weak_open": f"若低开接近或超过2%（约低于{weak_open_price}），先观察是否快速收回；不能收回则不买。",
        "stop_break": f"若跌破防守价{stop_text}且不能快速收回，已有仓位减仓或退出，无仓不参与。",
        "take_profit": f"到达止盈区间{trade_plan.get('expected_return')}后，若放量滞涨或上影线变长，优先分批兑现。",
        "intraday_condition": f"当前日内状态为{tape_grade or '未知'}，次日必须观察集合竞价后能否守住关键支撑并恢复承接。",
    }


def _price_text(value):
    rounded = _round(value, 2)
    return "--" if rounded is None else str(rounded)


def _build_buy_entry_strategy(
    candidate,
    trade_plan,
    recommendation_tier,
    formal_candidate=False,
    a_share_profile=None,
    intraday_view=None,
    liquidity_view=None,
    market=None,
):
    latest_price = _number(candidate.get("latest_price"))
    stop_price = _number(trade_plan.get("stop_price"))
    ma5 = _number(candidate.get("ma5"))
    ma10 = _number(candidate.get("ma10"))
    ma20 = _number(candidate.get("ma20"))
    ma60 = _number(candidate.get("ma60"))
    low_20d = _number(candidate.get("low_20d"))
    high_20d = _number(candidate.get("high_20d"))
    high_60d = _number(candidate.get("high_60d"))
    today_low = _number(candidate.get("today_low"))
    today_high = _number(candidate.get("today_high"))
    today_amp = _number(candidate.get("today_amp")) or 0.0
    amp_20d = _number(candidate.get("amp_20d")) or 0.0
    amp_30d = _number(candidate.get("amp_30d")) or 0.0
    high_open_price = latest_price * 1.03 if latest_price else None
    weak_open_price = latest_price * 0.98 if latest_price else None
    support_candidates = [value for value in [ma20, ma60, low_20d, stop_price] if value and latest_price and value < latest_price]
    support_price = max(support_candidates) if support_candidates else stop_price
    near_support = _nearest_level_below(latest_price, [today_low, ma5, ma10, ma20, stop_price])
    pressure_price = _nearest_level_above(latest_price, [today_high, high_20d, high_60d])
    tape_grade = (intraday_view or {}).get("tape_grade") or "未知"
    market_env = (market or {}).get("market_env") or "未知"
    liquidity_grade = (liquidity_view or {}).get("grade") or "未知"
    tier = recommendation_tier or adaptive.RECOMMENDATION_TIER_AVOID

    amp_base = max(today_amp, amp_20d, amp_30d, 2.0)
    pullback_pct = max(1.0, min(3.0, amp_base * 0.25))
    buy_watch_high = latest_price * (1 - pullback_pct / 100) if latest_price else None
    if near_support is not None:
        buy_watch_high = max(buy_watch_high or 0, near_support * 1.005)
    if latest_price is not None and buy_watch_high is not None:
        buy_watch_high = min(buy_watch_high, latest_price * 0.995)
    buy_watch_low = buy_watch_high * 0.985 if buy_watch_high is not None else None
    if near_support is not None and buy_watch_low is not None:
        buy_watch_low = max(buy_watch_low, near_support * 0.99)
    if stop_price is not None and buy_watch_low is not None:
        buy_watch_low = max(buy_watch_low, stop_price)
    if buy_watch_low is not None and buy_watch_high is not None and buy_watch_low > buy_watch_high:
        buy_watch_low = buy_watch_high

    confirm_low = support_price
    if stop_price is not None and confirm_low is not None:
        confirm_low = max(confirm_low, stop_price)
    confirm_high = buy_watch_low
    if confirm_low is not None and confirm_high is not None and confirm_low > confirm_high:
        confirm_high = confirm_low

    if formal_candidate:
        buy_zone_status = "可执行，但只做回踩承接"
        buy_zone_action = "优先等观察区回踩不破并重新站回分时均线/开盘价；直接高开越过不追线不追。"
    elif tier == adaptive.RECOMMENDATION_TIER_OBSERVE:
        buy_zone_status = "观察候选，只能极小仓试错"
        buy_zone_action = "只有回踩承接明显转强，才允许观察仓；不满足确认条件不买。"
    elif tier == adaptive.RECOMMENDATION_TIER_RESEARCH:
        buy_zone_status = "研究跟踪，不给主动买入"
        buy_zone_action = "区间只用于观察承接质量，未转为正式推荐前不作为买入指令。"
    else:
        buy_zone_status = "不建议买入"
        buy_zone_action = "当前只记录支撑压力和放弃线，等待模型重新给出可执行优势。"

    suggested_buy_zone = {
        "status": buy_zone_status,
        "watch_zone": {
            "low": _round(buy_watch_low, 2),
            "high": _round(buy_watch_high, 2),
            "text": _price_range_text(buy_watch_low, buy_watch_high),
        },
        "support_confirm_zone": {
            "low": _round(confirm_low, 2),
            "high": _round(confirm_high, 2),
            "text": _price_range_text(confirm_low, confirm_high),
        },
        "invalid_price": _round(stop_price, 2),
        "no_chase_price": _round(high_open_price, 2),
        "pressure_price": _round(pressure_price, 2),
        "action": buy_zone_action,
        "basis": [
            f"观察区按近端支撑{_price_text(near_support)}和近20/30日波动回撤{_round(pullback_pct, 2)}%估算。",
            f"强支撑确认区参考关键支撑{_price_text(support_price)}和策略防守{_price_text(stop_price)}。",
            f"高于{_price_text(high_open_price)}约等于高开3%，没有回踩承接不追。",
            f"上方压力参考{_price_text(pressure_price)}，冲高不过或放量滞涨要放弃追买。",
        ],
    }

    if formal_candidate:
        overall = "首选买法是承接确认，不建议开盘秒买；若开盘和盘中都满足条件，可以分两笔执行。"
    elif tier == adaptive.RECOMMENDATION_TIER_OBSERVE:
        overall = "当前只算观察候选，不能按正式买点处理；只有次日承接明显转强，才允许极小仓试错。"
    elif tier == adaptive.RECOMMENDATION_TIER_RESEARCH:
        overall = "当前是研究跟踪，不是买入策略；所有买入口径都以观察为主。"
    else:
        overall = "当前不具备买入条件，买入口径只用于说明放弃规则。"

    open_status = "可小仓试探" if formal_candidate and market_env not in ("偏弱", "未知") else "不作为首选"
    if tier in (adaptive.RECOMMENDATION_TIER_RESEARCH, adaptive.RECOMMENDATION_TIER_AVOID):
        open_status = "不建议"
    elif tier == adaptive.RECOMMENDATION_TIER_OBSERVE:
        open_status = "原则不买，只观察"

    avg_status = "优先于开盘直接买" if formal_candidate else "等待确认"
    if tier == adaptive.RECOMMENDATION_TIER_RESEARCH:
        avg_status = "只跟踪"
    elif tier == adaptive.RECOMMENDATION_TIER_AVOID:
        avg_status = "不建议"

    acceptance_status = "首选执行口径" if formal_candidate else "确认后才考虑"
    if tier == adaptive.RECOMMENDATION_TIER_RESEARCH:
        acceptance_status = "仅作为跟踪条件"
    elif tier == adaptive.RECOMMENDATION_TIER_AVOID:
        acceptance_status = "不触发买入"

    modes = {
        "next_open": {
            "name": "开盘直接买",
            "status": open_status,
            "trigger": (
                f"只接受平开到小高开，参考不高于{_price_text(high_open_price)}；"
                f"开盘后不破{_price_text(support_price)}附近支撑，且成交额/换手不是明显缩量。"
            ),
            "action": (
                "正式推荐才可用计划仓位的三分之一先试；观察候选和研究价值不使用开盘直接买。"
            ),
            "risk": "开盘买最大风险是被高开情绪带进去，若随后回落，当天受T+1限制不能立刻卖出。",
            "invalid": f"高开超过3%但不回踩、开盘后快速跌破{_price_text(support_price)}、或市场继续偏弱时放弃。",
        },
        "next_avg": {
            "name": "盘中均价/分批买",
            "status": avg_status,
            "trigger": (
                f"等待第一次回踩不破{_price_text(support_price)}或防守价{_price_text(stop_price)}，"
                "随后重新站上分时均线/开盘价并放量。"
            ),
            "action": "分两笔处理：第一笔在回踩企稳后，第二笔在重新放量突破当日分时压力后；不一次性满仓。",
            "risk": f"若日内状态继续表现为{tape_grade}，说明承接质量不足，盘中均价买也要降级为观察。",
            "invalid": "回踩跌破防守区、反弹无量、冲高留下明显上影线，取消当日买入计划。",
        },
        "acceptance": {
            "name": "承接确认买",
            "status": acceptance_status,
            "trigger": (
                "需要同时满足：高开不过度或高开后主动回踩；回踩不破关键支撑；"
                "开盘后能收回开盘价/分时均线；量能温和放大而不是缩量拉升。"
            ),
            "action": (
                f"这是最贴近当前模型的执行方式。满足后按{trade_plan.get('position_hint')}以内执行；"
                "若只是观察候选，最多小仓试错，未进正式推荐不加仓。"
            ),
            "risk": f"承接确认会少买很多票，但过滤高开不回踩、收盘偏弱、冲高回落的假强势。",
            "invalid": f"低开接近或超过2%（约低于{_price_text(weak_open_price)}）且不能快速收回，或跌破防守价{_price_text(stop_price)}。",
        },
    }
    abandon_rules = [
        f"高开超过3%（约高于{_price_text(high_open_price)}）且不回踩承接，不追。",
        f"低开接近或超过2%（约低于{_price_text(weak_open_price)}）且不能快速收回，不买。",
        f"跌破防守价{_price_text(stop_price)}或关键支撑{_price_text(support_price)}后不能收回，不买；已有仓位先降风险。",
        "拉升时量能不足、上影线明显变长、分时二次冲高失败，取消当日买点。",
        f"市场环境为{market_env}且行业/流动性不能继续配合时，不提高仓位；当前流动性判断为{liquidity_grade}。",
    ]
    full_text = (
        f"{overall} 开盘直接买:{modes['next_open']['status']}，{modes['next_open']['trigger']} "
        f"盘中均价:{modes['next_avg']['status']}，{modes['next_avg']['trigger']} "
        f"承接确认:{modes['acceptance']['status']}，{modes['acceptance']['trigger']} "
        f"放弃:{'；'.join(abandon_rules)}"
    )
    return {
        "overall": overall,
        "preferred_mode": "acceptance",
        "support_price": _round(support_price, 2),
        "high_open_price": _round(high_open_price, 2),
        "weak_open_price": _round(weak_open_price, 2),
        "suggested_buy_zone": suggested_buy_zone,
        "modes": modes,
        "abandon_rules": abandon_rules,
        "full_text": full_text,
    }


def _build_position_action_plan(
    candidate,
    trade_plan,
    operation,
    position_view,
    holding_stop,
    holding_t,
    recommendation_tier,
    formal_candidate,
    market,
    intraday_view,
    liquidity_view,
):
    latest_price = _number(candidate.get("latest_price"))
    ma5 = _number(candidate.get("ma5"))
    ma10 = _number(candidate.get("ma10"))
    ma20 = _number(candidate.get("ma20"))
    today_high = _number(candidate.get("today_high"))
    high_20d = _number(candidate.get("high_20d"))
    low_20d = _number(candidate.get("low_20d"))
    effective_stop = _number((holding_stop or {}).get("effective_stop_price"))
    dynamic_stop = _number((holding_stop or {}).get("dynamic_stop_price"))
    if effective_stop is None:
        effective_stop = _number(trade_plan.get("stop_price"))
    pnl_pct = _number((position_view or {}).get("pnl_pct"))
    has_cost = bool((position_view or {}).get("has_cost"))
    market_env = (market or {}).get("market_env") or "未知"
    tape_grade = (intraday_view or {}).get("tape_grade") or "未知"
    liquidity_grade = (liquidity_view or {}).get("grade") or "未知"
    trend_state = candidate.get("trend_state")
    risk_adjusted_score = _number(candidate.get("risk_adjusted_score"))
    risk_score = _number(candidate.get("risk_score"))
    stop_broken = bool(latest_price is not None and effective_stop is not None and latest_price <= effective_stop)
    below_ma20 = bool(latest_price is not None and ma20 is not None and latest_price < ma20)
    below_ma10 = bool(latest_price is not None and ma10 is not None and latest_price < ma10)
    weak_tape = tape_grade in ("冲高回落", "收盘偏弱")
    formal_or_observe = recommendation_tier in (
        adaptive.RECOMMENDATION_TIER_FORMAL,
        adaptive.RECOMMENDATION_TIER_OBSERVE,
    )

    near_resistance = _nearest_level_above(latest_price, [ma5, ma10, ma20, today_high, high_20d])
    rebound_reduce_low = None
    rebound_reduce_high = None
    if stop_broken:
        resistance_candidates = [level for level in [effective_stop, ma5, ma10, ma20] if level is not None and latest_price is not None and level > latest_price]
        if resistance_candidates:
            rebound_reduce_low = min(resistance_candidates)
            rebound_reduce_high = max(resistance_candidates[:2]) if len(resistance_candidates) >= 2 else rebound_reduce_low * 1.01

    sell_zone = (holding_t or {}).get("sell_zone") or {}
    buyback_zone = (holding_t or {}).get("buyback_zone") or {}
    reduce_strength_text = sell_zone.get("text")
    if not reduce_strength_text or reduce_strength_text == "--":
        resistance = near_resistance or (latest_price * 1.025 if latest_price else None)
        reduce_strength_text = _price_range_text(resistance, resistance * 1.015 if resistance else None)
    t_sell_zone_text = reduce_strength_text

    if stop_broken:
        stance = "风控处理"
        weakness_text = (
            f"现价已低于持仓有效防守{_price_text(effective_stop)}，反弹不能收回该线先减仓；"
            f"若继续跌破{_price_text(low_20d)}附近且不能收回，剩余仓位退出。"
        )
        add_status = "不加仓"
        add_trigger = "只有重新站回持仓有效防守和20日均线后，才重新评估，不做摊低。"
    elif below_ma20:
        stance = "弱势观察"
        weakness_text = (
            f"跌破20日均线{_price_text(ma20)}先降级为观察，不加仓；"
            f"若连续不能收回20日线，或再跌破持仓有效防守{_price_text(effective_stop)}，再减仓/退出。"
        )
        add_status = "不加仓"
        add_trigger = f"重新站回20日均线{_price_text(ma20)}并放量，才考虑恢复观察仓。"
    elif formal_candidate and market_env in ("强势", "偏强") and risk_score is not None and risk_score <= 2:
        stance = "可持有，承接确认后才考虑加仓"
        weakness_text = (
            f"跌破10日均线{_price_text(ma10)}先降级观察；"
            f"跌破持仓有效防守{_price_text(effective_stop)}且不能收回则减仓或退出。"
        )
        add_status = "条件加仓"
        add_trigger = f"仅限仍为正式推荐，且回踩不破{_price_text(ma10 or ma20 or effective_stop)}后重新放量走强；不在冲高时加。"
    elif formal_candidate:
        stance = "正式推荐，但等待承接修复"
        weakness_text = (
            f"仍是正式推荐，但市场或日内承接不配合；跌破10日均线{_price_text(ma10)}先降级观察，"
            f"跌破持仓有效防守{_price_text(effective_stop)}且不能收回则减仓或退出。"
        )
        add_status = "暂不加仓"
        add_trigger = (
            f"保持正式推荐且重新站稳{_price_text(ma10 or ma20 or effective_stop)}，同时市场/日内承接改善后才评估加仓；"
            "当前只按回踩承接执行，不扩大仓位。"
        )
    elif formal_or_observe and not weak_tape and risk_adjusted_score is not None and risk_adjusted_score >= 75:
        stance = "持有观察"
        weakness_text = (
            f"跌破20日均线{_price_text(ma20)}或持仓有效防守{_price_text(effective_stop)}不能收回，降低仓位。"
        )
        add_status = "不主动加仓"
        if formal_candidate:
            add_trigger = "虽然是正式推荐，但未满足强市场/强承接组合，不用观察信号扩大仓位。"
        else:
            add_trigger = "未进入正式推荐前，只能观察承接，不用观察信号扩大仓位。"
    else:
        stance = "只守不攻"
        weakness_text = (
            f"跌破或不能收回持仓有效防守{_price_text(effective_stop)}，先处理风险；"
            "未恢复正式推荐前不提高仓位。"
        )
        add_status = "不加仓"
        add_trigger = "模型层级、趋势结构和市场环境未同时改善前，不做新增买入。"

    if rebound_reduce_low is not None:
        rebound_reduce_text = _price_range_text(rebound_reduce_low, rebound_reduce_high)
        reduce_strength_text = rebound_reduce_text
        strength_action = (
            f"若反弹到{rebound_reduce_text}但量能不足或再次冲高回落，优先减仓。"
        )
    elif pnl_pct is not None and pnl_pct >= 8:
        strength_action = f"已有浮盈，涨到{reduce_strength_text}遇到放量滞涨或长上影，分批兑现。"
    else:
        strength_action = f"涨到{reduce_strength_text}附近若放量滞涨、上影变长或未突破压力，考虑减一部分。"

    t_direction = (holding_t or {}).get("direction") or "不做T"
    t_feasibility = (holding_t or {}).get("feasibility") or "不建议"
    if t_direction in ("正T", "反T", "只减不回补"):
        t_action = (holding_t or {}).get("action")
    else:
        t_action = "不做T，只按减仓线和加仓条件处理。"

    if not has_cost:
        stance = f"{stance}，但缺少持仓成本"

    return {
        "stance": stance,
        "has_cost": has_cost,
        "current_price": _round(latest_price, 2),
        "effective_stop_price": _round(effective_stop, 2),
        "dynamic_stop_price": _round(dynamic_stop, 2),
        "reduce_on_weakness": {
            "trigger_price": _round(effective_stop, 2),
            "action": weakness_text,
        },
        "reduce_on_strength": {
            "zone": reduce_strength_text,
            "action": strength_action,
        },
        "add_position": {
            "status": add_status,
            "trigger": add_trigger,
        },
        "t_plan": {
            "direction": t_direction,
            "feasibility": t_feasibility,
            "sell_zone": t_sell_zone_text,
            "buyback_zone": buyback_zone.get("text") or "--",
            "planned_t_ratio_pct": (holding_t or {}).get("planned_t_ratio_pct"),
            "action": t_action,
        },
        "notes": [
            f"趋势状态:{trend_state or '--'}",
            f"市场:{market_env}",
            f"日内:{tape_grade}",
            f"流动性:{liquidity_grade}",
        ],
    }


def _resolve_a_share_profile(stock_code, stock_name=None):
    code = str(stock_code or "")
    name = str(stock_name or "")
    is_st = name.startswith(("ST", "*ST", "S*ST"))

    if is_st:
        board = "风险警示"
        limit_pct = 5.0
    elif code.startswith(("688", "689")):
        board = "科创板"
        limit_pct = 20.0
    elif code.startswith(("300", "301", "302")):
        board = "创业板"
        limit_pct = 20.0
    elif code.startswith(("8", "4", "920")):
        board = "北交所"
        limit_pct = 30.0
    elif code.startswith(("600", "601", "603", "605", "000", "001", "002", "003")):
        board = "主板"
        limit_pct = 10.0
    else:
        board = "未知板块"
        limit_pct = 10.0

    return {
        "board": board,
        "limit_pct": limit_pct,
        "is_st": is_st,
        "t_plus_1": True,
        "note": f"{board}，常规涨跌幅约{limit_pct:g}%，A股T+1，买入当日通常不能卖出。",
    }


def _build_intraday_view(candidate, a_share_profile):
    close_price = _number(candidate.get("latest_price"))
    open_price = _number(candidate.get("today_open"))
    high_price = _number(candidate.get("today_high"))
    low_price = _number(candidate.get("today_low"))
    today_change = _number(candidate.get("today_change"))
    limit_pct = _number((a_share_profile or {}).get("limit_pct")) or 10.0

    prev_close = None
    if close_price is not None and today_change is not None and (1 + today_change / 100) != 0:
        prev_close = close_price / (1 + today_change / 100)

    open_gap_pct = (open_price / prev_close - 1) * 100 if open_price and prev_close else None
    close_from_open_pct = (close_price / open_price - 1) * 100 if close_price and open_price else None
    high_pullback_pct = (close_price / high_price - 1) * 100 if close_price and high_price else None
    low_recovery_pct = (close_price / low_price - 1) * 100 if close_price and low_price else None
    intraday_range_pct = (high_price / low_price - 1) * 100 if high_price and low_price else None
    close_position_pct = None
    if close_price is not None and high_price is not None and low_price is not None and high_price > low_price:
        close_position_pct = (close_price - low_price) / (high_price - low_price) * 100

    limit_up_price = prev_close * (1 + limit_pct / 100) if prev_close else None
    limit_down_price = prev_close * (1 - limit_pct / 100) if prev_close else None
    limit_up_distance_pct = (limit_up_price / close_price - 1) * 100 if limit_up_price and close_price else None
    limit_down_distance_pct = (close_price / limit_down_price - 1) * 100 if limit_down_price and close_price else None
    touched_limit_up = bool(high_price and limit_up_price and high_price >= limit_up_price * 0.995)
    near_limit_up = bool(today_change is not None and today_change >= limit_pct - 1.0)
    near_limit_down = bool(today_change is not None and today_change <= -limit_pct + 1.0)
    upper_shadow_pct = _round((_number(candidate.get("upper_shadow_ratio")) or 0) * 100, 2)

    if close_position_pct is not None and close_position_pct >= 70 and (close_from_open_pct or 0) >= 0:
        tape_grade = "承接较强"
    elif close_position_pct is not None and close_position_pct >= 55:
        tape_grade = "承接尚可"
    elif upper_shadow_pct is not None and upper_shadow_pct >= 35:
        tape_grade = "冲高回落"
    elif close_position_pct is not None and close_position_pct <= 35:
        tape_grade = "收盘偏弱"
    else:
        tape_grade = "中性"

    warnings = []
    if touched_limit_up:
        warnings.append("盘中接近涨停，次日容易分歧，不能追高")
    if near_limit_down:
        warnings.append("接近跌停约束，流动性和隔日风险偏高")
    if upper_shadow_pct is not None and upper_shadow_pct >= 35:
        warnings.append("上影线偏长，说明日内抛压较重")
    if close_position_pct is not None and close_position_pct <= 35:
        warnings.append("收盘靠近日内低位，承接不足")

    return {
        "prev_close": _round(prev_close, 2),
        "open_gap_pct": _round(open_gap_pct, 2),
        "close_from_open_pct": _round(close_from_open_pct, 2),
        "high_pullback_pct": _round(high_pullback_pct, 2),
        "low_recovery_pct": _round(low_recovery_pct, 2),
        "intraday_range_pct": _round(intraday_range_pct, 2),
        "close_position_pct": _round(close_position_pct, 2),
        "upper_shadow_pct": upper_shadow_pct,
        "limit_up_price": _round(limit_up_price, 2),
        "limit_down_price": _round(limit_down_price, 2),
        "limit_up_distance_pct": _round(limit_up_distance_pct, 2),
        "limit_down_distance_pct": _round(limit_down_distance_pct, 2),
        "touched_limit_up": touched_limit_up,
        "near_limit_up": near_limit_up,
        "near_limit_down": near_limit_down,
        "tape_grade": tape_grade,
        "warnings": warnings,
    }


def _build_liquidity_view(candidate, a_share_profile):
    amount = _number(candidate.get("today_amount"))
    amount_vs_avg_5d = _number(candidate.get("amount_vs_avg_5d"))
    amount_vs_avg_10d = _number(candidate.get("amount_vs_avg_10d"))
    turnover = _number(candidate.get("turnover_rate"))
    turnover_vs_avg_5d = _number(candidate.get("turnover_vs_avg_5d"))
    board = (a_share_profile or {}).get("board")
    weak_amount_floor = 30000000 if board == "北交所" else 80000000
    good_amount_floor = 120000000 if board == "北交所" else 200000000

    warnings = []
    if amount is not None and amount < weak_amount_floor:
        warnings.append("成交额偏低，实盘滑点和出入场难度较高")
    if turnover is not None and turnover < 1:
        warnings.append("换手偏低，短线资金参与度不足")
    if turnover is not None and turnover >= 12 and (turnover_vs_avg_5d or 0) >= 1.6:
        warnings.append("换手过热，容易出现分歧回撤")

    if amount is not None and amount >= good_amount_floor and turnover is not None and 1.5 <= turnover <= 10:
        grade = "流动性良好"
    elif warnings:
        grade = "流动性有瑕疵"
    else:
        grade = "流动性一般"

    return {
        "amount": _round(amount, 2),
        "amount_vs_avg_5d": _round(amount_vs_avg_5d, 2),
        "amount_vs_avg_10d": _round(amount_vs_avg_10d, 2),
        "turnover_rate": _round(turnover, 2),
        "turnover_vs_avg_5d": _round(turnover_vs_avg_5d, 2),
        "grade": grade,
        "warnings": warnings,
    }


def _signed_points(value):
    value = _number(value) or 0.0
    return f"{value:+.2f}"


def _component_label(name, detail=None):
    return f"{name}({detail})" if detail not in (None, "") else name


def _scorecard_component_text(scorecard):
    components = (scorecard or {}).get("components") or []
    if not components:
        return "；".join((scorecard or {}).get("details") or [])
    return "；".join(item.get("text") or "" for item in components if item.get("text"))


def _scorecard(formal_candidate, evidence, industry_view, market, candidate, intraday_view=None, liquidity_view=None):
    score = 0.0
    components = []

    def add_component(name, points, detail=None):
        nonlocal score
        points = _number(points) or 0.0
        score += points
        rounded_points = _round(points, 2)
        text = f"{_component_label(name, detail)}{_signed_points(rounded_points)}"
        components.append(
            {
                "name": name,
                "points": rounded_points,
                "detail": detail,
                "text": text,
            }
        )

    risk_adjusted_score = _number(candidate.get("risk_adjusted_score"))
    risk_score = _number(candidate.get("risk_score"))
    overlay_score = _number(candidate.get("risk_overlay_score")) or 0.0
    overlay_block = bool(candidate.get("risk_overlay_block_formal"))
    evidence_grade = ((evidence or {}).get("grade") or {}).get("grade")
    industry_grade = (industry_view or {}).get("strength_grade")
    market_env = (market or {}).get("market_env")
    tape_grade = (intraday_view or {}).get("tape_grade")
    liquidity_grade = (liquidity_view or {}).get("grade")

    add_component("正式推荐", 25 if formal_candidate else 0, "是" if formal_candidate else "否")
    if risk_adjusted_score is not None:
        risk_adjusted_points = max(0, min(25, (risk_adjusted_score - 55) / 25 * 25))
        add_component("风险调整分", risk_adjusted_points, _round(risk_adjusted_score, 2))
    else:
        add_component("风险调整分", 0, "缺失")
    if risk_score is not None:
        risk_points = max(0, 15 - risk_score * 4)
        add_component("风险分", risk_points, _round(risk_score, 2))
    else:
        add_component("风险分", 0, "缺失")
    if overlay_score:
        penalty = min(18, overlay_score * 1.2)
        add_component("事件风险", -penalty, _round(overlay_score, 2))
    if overlay_block:
        add_component("风险拦截", -15, "正式推荐拦截")

    evidence_points = 0
    if evidence_grade == "强":
        evidence_points = 20
    elif evidence_grade == "中":
        evidence_points = 14
    elif evidence_grade == "弱":
        evidence_points = 8
    add_component("证据", evidence_points, evidence_grade or "未知")

    industry_points = 0
    if industry_grade == "强共振":
        industry_points = 4
    elif industry_grade == "中性偏强":
        industry_points = 2
    elif industry_grade == "偏弱":
        industry_points = 0
    elif industry_grade == "弱共振":
        industry_points = -2
    add_component("行业", industry_points, industry_grade or "未知")

    market_points = 0
    if market_env in ("强势", "偏强"):
        market_points = 5
    elif market_env == "震荡":
        market_points = 2
    elif market_env == "偏弱":
        market_points = -5
    add_component("市场", market_points, market_env or "未知")

    tape_points = 0
    if tape_grade == "承接较强":
        tape_points = 2
    elif tape_grade in ("冲高回落", "收盘偏弱"):
        tape_points = -2
    add_component("日内", tape_points, tape_grade or "未知")

    liquidity_points = 0
    if liquidity_grade == "流动性良好":
        liquidity_points = 4
    elif liquidity_grade == "流动性有瑕疵":
        liquidity_points = -6
    add_component("流动性", liquidity_points, liquidity_grade or "未知")

    score = _round(max(0, min(100, score)), 1)
    if score >= 80:
        rating = "A"
    elif score >= 65:
        rating = "B"
    elif score >= 50:
        rating = "C"
    else:
        rating = "D"

    return {
        "score": score,
        "rating": rating,
        "components": components,
        "details": [item.get("text") for item in components if item.get("text")],
    }


def _load_saved_strategy(stock_code, trade_date):
    if not stock_code or not trade_date:
        return []

    rows = adaptive._query_frame(
        """
        SELECT trade_date, strategy_type, stock_code, stock_name, strategy_note
        FROM a_stock_strategy_result
        WHERE trade_date = %s AND stock_code = %s
        ORDER BY strategy_type
        """,
        params=[trade_date, stock_code],
    )
    if rows.empty:
        return []
    return rows.to_dict("records")


def _load_recent_saved_strategy(stock_code, trade_date, lookback_days=RECENT_SAVED_STRATEGY_LOOKBACK_DAYS):
    if not stock_code or not trade_date:
        return []

    rows = adaptive._query_frame(
        """
        SELECT trade_date, strategy_type, stock_code, stock_name, strategy_note
        FROM a_stock_strategy_result
        WHERE stock_code = %s
          AND trade_date < %s
          AND trade_date >= DATE_SUB(%s, INTERVAL %s DAY)
        ORDER BY trade_date DESC, strategy_type
        """,
        params=[stock_code, trade_date, trade_date, int(lookback_days)],
    )
    if rows.empty:
        return []
    rows["trade_date"] = rows["trade_date"].apply(_date_text)
    return rows.to_dict("records")


def _extract_saved_rank(saved_strategy):
    for row in saved_strategy or []:
        note = str(row.get("strategy_note") or "")
        if note.startswith("首选") or note.startswith("正式推荐首选"):
            return 1
        matched = re.search(r"候选(\d+)", note)
        if matched:
            return int(matched.group(1))
    return None


def _is_small_validation_strategy(row):
    strategy_type = str((row or {}).get("strategy_type") or "")
    note = str((row or {}).get("strategy_note") or "")
    return strategy_type in SMALL_VALIDATION_STRATEGY_TYPES or "小仓验证" in note or "层级:小仓验证" in note


def _is_long_runway_strategy(row):
    strategy_type = str((row or {}).get("strategy_type") or "")
    note = str((row or {}).get("strategy_note") or "")
    return (
        strategy_type in LONG_RUNWAY_STRATEGY_TYPES
        or "层级:中长期跟踪" in note
        or "长跑潜力模型" in note
    )


def _has_small_validation_strategy(saved_strategy):
    return any(_is_small_validation_strategy(row) for row in saved_strategy or [])


def _has_long_runway_strategy(saved_strategy):
    return any(_is_long_runway_strategy(row) for row in saved_strategy or [])


def _has_full_formal_strategy(saved_strategy):
    return any(
        not _is_small_validation_strategy(row) and not _is_long_runway_strategy(row)
        for row in saved_strategy or []
    )


def _latest_strategy_trade_date(saved_strategy):
    dates = [_date_text(row.get("trade_date")) for row in saved_strategy or [] if row.get("trade_date") is not None]
    dates = [date for date in dates if date]
    return max(dates) if dates else None


def _build_saved_strategy_hint(saved_strategy):
    has_small_validation = _has_small_validation_strategy(saved_strategy)
    has_long_runway = _has_long_runway_strategy(saved_strategy)
    if not has_small_validation and not has_long_runway:
        return ""
    hints = []
    if has_small_validation:
        if _has_full_formal_strategy(saved_strategy):
            hints.append("同时命中小仓验证策略和正式策略；以正式策略为主，但小仓验证备注仍需遵守承接确认。")
        else:
            hints.append("当前命中的是小仓验证策略，不等同普通正式推荐；只在次日承接确认后小仓试，弱于大盘或跌破防守位就放弃。")
    if has_long_runway:
        hints.append("当前命中长跑潜力模型的中长期跟踪；这不是短线正式买点，按备注里的阶段、仓位和防守规则跟踪。")
    return " ".join(hints)


def _build_recent_saved_strategy_hint(recent_saved_strategy):
    if not recent_saved_strategy:
        return ""

    latest_date = _latest_strategy_trade_date(recent_saved_strategy)
    latest_rows = [
        row for row in recent_saved_strategy
        if _date_text(row.get("trade_date")) == latest_date
    ]
    if _has_full_formal_strategy(latest_rows):
        return (
            f"最近一次正式推荐是{latest_date}；当前分析日未再次落库时，不等同新的买入信号，"
            "但已有仓位应按原推荐持有期、防守价和次日承接结果继续管理。"
        )
    if _has_small_validation_strategy(latest_rows):
        return (
            f"最近一次小仓验证信号是{latest_date}；当前分析日未再次落库时，只保留观察和防守管理。"
        )
    if _has_long_runway_strategy(latest_rows):
        return (
            f"最近一次长跑潜力跟踪是{latest_date}；这是中长期跟踪上下文，不等同当日短线正式推荐。"
        )
    return ""


def _load_strategy_memory(stock_code, trade_date):
    if not stock_code or not trade_date:
        return []

    rows = adaptive._query_frame(
        """
        SELECT trade_date, strategy_type, stock_code, stock_name, strategy_note
        FROM a_stock_strategy_result
        WHERE stock_code = %s
          AND trade_date <= %s
        ORDER BY trade_date, strategy_type
        """,
        params=[stock_code, trade_date],
    )
    if rows.empty:
        return []
    rows["trade_date"] = rows["trade_date"].apply(_date_text)
    return rows.to_dict("records")


def _load_stock_price_memory(stock_code, trade_date):
    if not stock_code or not trade_date:
        return pd.DataFrame()

    rows = adaptive._query_frame(
        """
        SELECT last_data_date, stock_code, stock_name, latest_price,
               today_open, today_high, today_low, today_change
        FROM a_stock_analysis_history
        WHERE stock_code = %s
          AND last_data_date <= %s
        ORDER BY last_data_date
        """,
        params=[stock_code, trade_date],
    )
    if rows.empty:
        return rows
    rows["last_data_date_text"] = rows["last_data_date"].apply(_date_text)
    for column in ["latest_price", "today_open", "today_high", "today_low", "today_change"]:
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    rows = rows.dropna(subset=["last_data_date_text", "latest_price"])
    rows = rows[rows["latest_price"] > 0].copy()
    return rows.reset_index(drop=True)


def _parse_saved_strategy_plan(note):
    text = str(note or "")

    defense = None
    matched = re.search(r"防守[:：]\s*([0-9]+(?:\.[0-9]+)?)", text)
    if matched:
        defense = _round(matched.group(1), 2)
    else:
        matched = re.search(r"跌破\s*([0-9]+(?:\.[0-9]+)?)\s*附近先撤", text)
        if matched:
            defense = _round(matched.group(1), 2)

    target_low = None
    target_high = None
    matched = re.search(
        r"(?:止盈|预期)[:：]?\s*([+-]?[0-9]+(?:\.[0-9]+)?)%?\s*(?:~|-|至)\s*([+-]?[0-9]+(?:\.[0-9]+)?)%?",
        text,
    )
    if matched:
        target_low = _round(matched.group(1), 2)
        target_high = _round(matched.group(2), 2)
    else:
        matched = re.search(r"(?:止盈|预期)[:：]?\s*([+-]?[0-9]+(?:\.[0-9]+)?)%", text)
        if matched:
            target_low = _round(matched.group(1), 2)

    hold_min = None
    hold_max = None
    hold_unit = None
    matched = re.search(r"(?:持有|跟踪)[:：]\s*(\d+)\s*(?:~|-|至)\s*(\d+)\s*(周|个交易日|交易日|天)?", text)
    if matched:
        hold_min = int(matched.group(1))
        hold_max = int(matched.group(2))
        hold_unit = matched.group(3) or "交易日"
    else:
        matched = re.search(r"(?:跟随|持有|跟踪)\s*(\d+)\s*(?:~|-|至)\s*(\d+)\s*(周|个交易日|交易日|天)?", text)
        if matched:
            hold_min = int(matched.group(1))
            hold_max = int(matched.group(2))
            hold_unit = matched.group(3) or "交易日"

    return {
        "defense_price": defense,
        "target_low_pct": target_low,
        "target_high_pct": target_high,
        "hold_min": hold_min,
        "hold_max": hold_max,
        "hold_unit": hold_unit,
        "needs_acceptance": "承接" in text,
    }


def _entry_acceptance_from_rows(signal_close, entry_row):
    entry_open = _number(entry_row.get("today_open"))
    entry_high = _number(entry_row.get("today_high"))
    entry_low = _number(entry_row.get("today_low"))
    entry_close = _number(entry_row.get("latest_price"))
    if not all(value is not None and value > 0 for value in [signal_close, entry_open, entry_high, entry_low, entry_close]):
        return {
            "entry_acceptance_ok": None,
            "entry_avg_price": None,
            "entry_gap_pct": None,
            "entry_close_position": None,
            "entry_close_from_open_pct": None,
        }

    entry_avg = (entry_open + entry_high + entry_low + entry_close) / 4
    entry_gap_pct = (entry_open - signal_close) / signal_close * 100
    entry_range = entry_high - entry_low
    entry_close_position = (entry_close - entry_low) / entry_range if entry_range > 0 else None
    entry_close_from_open_pct = (entry_close - entry_open) / entry_open * 100
    high_open_has_pullback = entry_gap_pct <= 3.0 or entry_low <= signal_close * 1.015
    acceptance_ok = (
        high_open_has_pullback
        and (entry_close_position is not None and entry_close_position >= 0.45)
        and entry_close_from_open_pct >= -2.5
    )

    return {
        "entry_acceptance_ok": bool(acceptance_ok),
        "entry_avg_price": _round(entry_avg, 2),
        "entry_gap_pct": _round(entry_gap_pct, 2),
        "entry_close_position": _round(entry_close_position * 100, 2) if entry_close_position is not None else None,
        "entry_close_from_open_pct": _round(entry_close_from_open_pct, 2),
    }


def _classify_saved_strategy_outcome(plan, metrics, strategy_type):
    if metrics.get("entry_date") is None:
        return "等待次日数据验证", "信号日之后还没有交易日，暂不能判断入场。"

    needs_acceptance = bool(plan.get("needs_acceptance") or strategy_type == getattr(adaptive, "ADAPTIVE_STRATEGY_TYPE", "adaptive_model"))
    if needs_acceptance and metrics.get("entry_acceptance_ok") is False:
        return "未满足入场承接", "按原计划应放弃当日买点；若仍有仓位，只按防守纪律管理。"

    if metrics.get("stop_broken_close"):
        return "防守失守", "后续收盘跌破防守价，按计划应降风险或退出。"
    if metrics.get("stop_touched_low"):
        return "盘中触及防守", "盘中触及过防守价，需核对当时是否快速收回；当前不宜加仓。"

    target_low = _number(plan.get("target_low_pct"))
    if target_low is not None and metrics.get("max_high_return_pct") is not None:
        if metrics["max_high_return_pct"] >= target_low:
            return "达到止盈区", "后续最高收益已进入止盈区，按计划应分批兑现或提高防守。"

    hold_max = plan.get("hold_max")
    hold_unit = plan.get("hold_unit")
    if hold_max and hold_unit != "周" and metrics.get("elapsed_trade_days") is not None:
        if metrics["elapsed_trade_days"] > int(hold_max):
            return "超过计划持有期", "已超过原计划持有窗口，若没有重新推荐，应降级为复盘观察。"

    current_return = metrics.get("current_return_pct")
    if current_return is not None and current_return > 0:
        return "持有期内表现正向", "未触发防守，当前按原计划继续用移动防守管理。"
    if current_return is not None:
        return "仍在验证", "当前收益未明显兑现，重点看防守是否保持和是否再次进入正式推荐。"
    return "数据不足", "缺少足够价格路径，无法完成表现复盘。"


def _evaluate_saved_strategy_memory(strategy_memory, price_memory, current_trade_date):
    if not strategy_memory:
        return {
            "items": [],
            "summary": {
                "total": 0,
                "formal_total": 0,
                "latest_formal_date": None,
                "target_hit_count": 0,
                "stop_broken_count": 0,
                "personal_guidance": "该股没有历史落库推荐，当前只能按今日模型和风险状态判断。",
            },
        }
    if price_memory.empty:
        return {
            "items": [],
            "summary": {
                "total": len(strategy_memory),
                "formal_total": 0,
                "latest_formal_date": None,
                "target_hit_count": 0,
                "stop_broken_count": 0,
                "personal_guidance": "该股有历史落库推荐，但缺少价格路径，无法复盘后续表现。",
            },
        }

    evaluations = []
    current_date_text = _date_text(current_trade_date)
    latest_row = price_memory.iloc[-1].to_dict()
    latest_price = _number(latest_row.get("latest_price"))

    for row in strategy_memory:
        signal_date = _date_text(row.get("trade_date"))
        strategy_type = str(row.get("strategy_type") or "")
        note = row.get("strategy_note") or ""
        plan = _parse_saved_strategy_plan(note)
        signal_rows = price_memory[price_memory["last_data_date_text"] == signal_date]
        future_rows = price_memory[price_memory["last_data_date_text"] > signal_date].copy()
        follow_rows = price_memory[
            (price_memory["last_data_date_text"] > signal_date)
            & (price_memory["last_data_date_text"] <= current_date_text)
        ].copy()

        signal_close = _number(signal_rows.iloc[-1].get("latest_price")) if not signal_rows.empty else None
        entry_row = future_rows.iloc[0].to_dict() if not future_rows.empty else {}
        entry_info = _entry_acceptance_from_rows(signal_close, entry_row)
        entry_date = _date_text(entry_row.get("last_data_date")) if entry_row else None
        entry_close = _number(entry_row.get("latest_price")) if entry_row else None
        entry_price = entry_info.get("entry_avg_price") or entry_close

        active_rows = follow_rows[follow_rows["last_data_date_text"] >= entry_date].copy() if entry_date else pd.DataFrame()
        if active_rows.empty:
            active_rows = follow_rows

        current_return = None
        max_high_return = None
        min_low_return = None
        stop_touched_low = False
        stop_broken_close = False
        target_hit = False
        if entry_price and latest_price:
            current_return = (latest_price - entry_price) / entry_price * 100
        if entry_price and not active_rows.empty:
            max_high = pd.to_numeric(active_rows["today_high"], errors="coerce").max()
            min_low = pd.to_numeric(active_rows["today_low"], errors="coerce").min()
            if pd.notna(max_high):
                max_high_return = (float(max_high) - entry_price) / entry_price * 100
            if pd.notna(min_low):
                min_low_return = (float(min_low) - entry_price) / entry_price * 100
            defense_price = _number(plan.get("defense_price"))
            if defense_price is not None:
                stop_touched_low = bool((pd.to_numeric(active_rows["today_low"], errors="coerce") <= defense_price).any())
                stop_broken_close = bool((pd.to_numeric(active_rows["latest_price"], errors="coerce") <= defense_price).any())
            target_low = _number(plan.get("target_low_pct"))
            if target_low is not None and max_high_return is not None:
                target_hit = bool(max_high_return >= target_low)

        metrics = {
            "entry_date": entry_date,
            "entry_price": _round(entry_price, 2),
            "entry_close": _round(entry_close, 2),
            "entry_acceptance_ok": entry_info.get("entry_acceptance_ok"),
            "entry_gap_pct": entry_info.get("entry_gap_pct"),
            "entry_close_position_pct": entry_info.get("entry_close_position"),
            "entry_close_from_open_pct": entry_info.get("entry_close_from_open_pct"),
            "elapsed_trade_days": int(len(follow_rows)),
            "current_price": _round(latest_price, 2),
            "current_return_pct": _round(current_return, 2),
            "max_high_return_pct": _round(max_high_return, 2),
            "min_low_return_pct": _round(min_low_return, 2),
            "target_hit": target_hit,
            "stop_touched_low": stop_touched_low,
            "stop_broken_close": stop_broken_close,
        }
        outcome, action = _classify_saved_strategy_outcome(plan, metrics, strategy_type)
        evaluations.append(
            {
                "trade_date": signal_date,
                "strategy_type": strategy_type,
                "stock_name": row.get("stock_name"),
                "kind": (
                    "小仓验证" if _is_small_validation_strategy(row)
                    else "中长期跟踪" if _is_long_runway_strategy(row)
                    else "正式推荐"
                ),
                "plan": plan,
                "metrics": metrics,
                "outcome": outcome,
                "action": action,
                "strategy_note": note,
            }
        )

    formal_items = [item for item in evaluations if item["kind"] == "正式推荐"]
    latest_formal = formal_items[-1] if formal_items else None
    target_hit_count = sum(1 for item in evaluations if item["metrics"].get("target_hit"))
    stop_broken_count = sum(1 for item in evaluations if item["metrics"].get("stop_broken_close"))
    accepted_count = sum(1 for item in evaluations if item["metrics"].get("entry_acceptance_ok") is True)

    if latest_formal:
        metrics = latest_formal["metrics"]
        plan = latest_formal["plan"]
        if latest_formal["outcome"] == "未满足入场承接":
            guidance = (
                f"最近正式推荐在{latest_formal['trade_date']}，但次日承接未通过；"
                "无仓不追，已有仓位只按防守处理。"
            )
        elif metrics.get("stop_broken_close"):
            guidance = (
                f"最近正式推荐在{latest_formal['trade_date']}，后续已收盘跌破防守"
                f"{plan.get('defense_price') or '--'}；当前以风险处理优先。"
            )
        elif metrics.get("target_hit"):
            guidance = (
                f"最近正式推荐在{latest_formal['trade_date']}，后续已进入止盈区；"
                "已有仓位应把防守上移，当前未新增正式推荐时不按新仓追买。"
            )
        else:
            guidance = (
                f"最近正式推荐在{latest_formal['trade_date']}，当前收益"
                f"{metrics.get('current_return_pct') if metrics.get('current_return_pct') is not None else '--'}%；"
                "未再次落库时按持仓管理，不当作新的买点。"
            )
    elif evaluations:
        guidance = "该股只有观察/中长期跟踪类落库记录，当前应按跟踪和防守管理，不等同短线正式买点。"
    else:
        guidance = "该股没有可复盘的落库推荐。"

    return {
        "items": evaluations,
        "summary": {
            "total": len(evaluations),
            "formal_total": len(formal_items),
            "latest_formal_date": latest_formal["trade_date"] if latest_formal else None,
            "accepted_count": accepted_count,
            "target_hit_count": target_hit_count,
            "stop_broken_count": stop_broken_count,
            "personal_guidance": guidance,
        },
    }


def _format_gap(value):
    rounded = _round(value, 2)
    if rounded is None:
        return None
    return f"{rounded}%"


def _safe_pct(numerator, denominator):
    numerator = _number(numerator)
    denominator = _number(denominator)
    if numerator is None or denominator is None or denominator == 0:
        return None
    return (numerator / denominator - 1) * 100


def _parse_date(value):
    text = _date_text(value)
    if not text:
        return None
    try:
        return dt.datetime.strptime(text[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _date_after(left, right):
    left_date = _parse_date(left)
    right_date = _parse_date(right)
    return bool(left_date and right_date and left_date > right_date)


def _realtime_symbol(stock_code):
    code = adaptive._normalize_stock_code(stock_code)
    if not code:
        return None
    if code.startswith(("6", "9")):
        prefix = "sh"
    elif code.startswith(("4", "8")):
        prefix = "bj"
    else:
        prefix = "sz"
    return f"{prefix}{code}"


def _parse_realtime_number(value):
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_sina_stock_quote(symbol, raw_text):
    fields = str(raw_text or "").split(",")
    if len(fields) < 32:
        return None

    quote_date = fields[30].strip()
    quote_time = fields[31].strip()
    quote_datetime = f"{quote_date} {quote_time}".strip()
    current = _parse_realtime_number(fields[3])
    open_price = _parse_realtime_number(fields[1])
    prev_close = _parse_realtime_number(fields[2])
    high_price = _parse_realtime_number(fields[4])
    low_price = _parse_realtime_number(fields[5])
    volume = _parse_realtime_number(fields[8])
    volume_hands = volume / 100.0 if volume is not None else None
    amount = _parse_realtime_number(fields[9])

    return {
        "symbol": symbol,
        "name": fields[0].strip(),
        "quote_date": quote_date or None,
        "quote_time": quote_time or None,
        "quote_datetime": quote_datetime if quote_date or quote_time else None,
        "open": _round(open_price, 2),
        "prev_close": _round(prev_close, 2),
        "current": _round(current, 2),
        "high": _round(high_price, 2),
        "low": _round(low_price, 2),
        "bid": _round(_parse_realtime_number(fields[6]), 2),
        "ask": _round(_parse_realtime_number(fields[7]), 2),
        "volume": _round(volume_hands, 2),
        "volume_unit": "手",
        "amount": _round(amount, 2),
        "change_pct": _round(_safe_pct(current, prev_close), 2),
        "open_gap_pct": _round(_safe_pct(open_price, prev_close), 2),
        "current_from_open_pct": _round(_safe_pct(current, open_price), 2),
        "low_from_prev_pct": _round(_safe_pct(low_price, prev_close), 2),
        "high_from_prev_pct": _round(_safe_pct(high_price, prev_close), 2),
    }


def _parse_sina_index_quote(symbol, raw_text):
    fields = str(raw_text or "").split(",")
    if len(fields) < 4:
        return None
    return {
        "symbol": symbol,
        "name": fields[0].strip(),
        "current": _round(_parse_realtime_number(fields[1]), 4),
        "change": _round(_parse_realtime_number(fields[2]), 4),
        "change_pct": _round(_parse_realtime_number(fields[3]), 2),
    }


def _fetch_sina_realtime_snapshot(stock_code, timeout=6):
    symbol = _realtime_symbol(stock_code)
    if not symbol:
        return {"success": False, "source": REALTIME_SOURCE_NAME, "reason": "invalid_stock_code"}

    index_symbols = ["s_sh000001", "s_sz399001", "s_sz399006"]
    symbols = ",".join([symbol] + index_symbols)
    url = REALTIME_SOURCE_URL.format(symbols=symbols)
    try:
        import requests

        response = requests.get(
            url,
            timeout=timeout,
            headers={
                "Referer": "https://finance.sina.com.cn",
                "User-Agent": "Mozilla/5.0",
            },
        )
        response.raise_for_status()
        response.encoding = "gb18030"
    except Exception as error:
        return {
            "success": False,
            "source": REALTIME_SOURCE_NAME,
            "url": url,
            "reason": f"{type(error).__name__}: {error}",
        }

    parsed = {}
    pattern = re.compile(r'var hq_str_([^=]+)="(.*?)";')
    for match in pattern.finditer(response.text or ""):
        parsed[match.group(1)] = match.group(2)

    quote = _parse_sina_stock_quote(symbol, parsed.get(symbol))
    if not quote or not quote.get("current") or quote.get("current") <= 0:
        return {
            "success": False,
            "source": REALTIME_SOURCE_NAME,
            "url": url,
            "reason": "realtime_quote_empty",
            "raw": parsed.get(symbol),
        }

    indices = []
    for index_symbol in index_symbols:
        index_quote = _parse_sina_index_quote(index_symbol, parsed.get(index_symbol))
        if index_quote:
            indices.append(index_quote)

    return {
        "success": True,
        "source": REALTIME_SOURCE_NAME,
        "url": url,
        "quote": quote,
        "indices": indices,
    }


def _parse_tencent_datetime(value):
    text = str(value or "").strip()
    if len(text) < 14:
        return None, None, None
    try:
        parsed = dt.datetime.strptime(text[:14], "%Y%m%d%H%M%S")
    except ValueError:
        return None, None, None
    return parsed.strftime("%Y-%m-%d"), parsed.strftime("%H:%M:%S"), parsed.strftime("%Y-%m-%d %H:%M:%S")


def _parse_tencent_stock_quote(symbol, raw_text):
    fields = str(raw_text or "").split("~")
    if len(fields) < 35:
        return None

    quote_date, quote_time, quote_datetime = _parse_tencent_datetime(fields[30] if len(fields) > 30 else None)
    current = _parse_realtime_number(fields[3])
    prev_close = _parse_realtime_number(fields[4])
    open_price = _parse_realtime_number(fields[5])
    high_price = _parse_realtime_number(fields[33])
    low_price = _parse_realtime_number(fields[34])
    volume = _parse_realtime_number(fields[36]) if len(fields) > 36 else None
    amount_10k = _parse_realtime_number(fields[37]) if len(fields) > 37 else None
    amount = amount_10k * 10000 if amount_10k is not None else None

    return {
        "symbol": symbol,
        "name": fields[1].strip(),
        "quote_date": quote_date,
        "quote_time": quote_time,
        "quote_datetime": quote_datetime,
        "open": _round(open_price, 2),
        "prev_close": _round(prev_close, 2),
        "current": _round(current, 2),
        "high": _round(high_price, 2),
        "low": _round(low_price, 2),
        "bid": _round(_parse_realtime_number(fields[9] if len(fields) > 9 else None), 2),
        "ask": _round(_parse_realtime_number(fields[19] if len(fields) > 19 else None), 2),
        "volume": _round(volume, 2),
        "volume_unit": "手",
        "amount": _round(amount, 2),
        "change_pct": _round(_parse_realtime_number(fields[32] if len(fields) > 32 else None), 2),
        "open_gap_pct": _round(_safe_pct(open_price, prev_close), 2),
        "current_from_open_pct": _round(_safe_pct(current, open_price), 2),
        "low_from_prev_pct": _round(_safe_pct(low_price, prev_close), 2),
        "high_from_prev_pct": _round(_safe_pct(high_price, prev_close), 2),
    }


def _fetch_tencent_realtime_snapshot(stock_code, timeout=6):
    symbol = _realtime_symbol(stock_code)
    if not symbol:
        return {"success": False, "source": REALTIME_FALLBACK_SOURCE_NAME, "reason": "invalid_stock_code"}

    url = REALTIME_FALLBACK_SOURCE_URL.format(symbol=symbol)
    try:
        import requests

        response = requests.get(
            url,
            timeout=timeout,
            headers={
                "Referer": "https://gu.qq.com",
                "User-Agent": "Mozilla/5.0",
            },
        )
        response.raise_for_status()
        response.encoding = "gbk"
    except Exception as error:
        return {
            "success": False,
            "source": REALTIME_FALLBACK_SOURCE_NAME,
            "url": url,
            "reason": f"{type(error).__name__}: {error}",
        }

    matched = re.search(r'v_[^=]+="(.*?)";', response.text or "")
    raw_text = matched.group(1) if matched else ""
    quote = _parse_tencent_stock_quote(symbol, raw_text)
    if not quote or not quote.get("current") or quote.get("current") <= 0:
        return {
            "success": False,
            "source": REALTIME_FALLBACK_SOURCE_NAME,
            "url": url,
            "reason": "realtime_quote_empty",
            "raw": raw_text,
        }

    return {
        "success": True,
        "source": REALTIME_FALLBACK_SOURCE_NAME,
        "url": url,
        "quote": quote,
        "indices": [],
    }


def _fetch_realtime_snapshot(stock_code):
    primary = _fetch_sina_realtime_snapshot(stock_code)
    if primary.get("success"):
        return primary
    fallback = _fetch_tencent_realtime_snapshot(stock_code)
    if fallback.get("success"):
        fallback["fallback_from"] = {
            "source": primary.get("source"),
            "reason": primary.get("reason"),
        }
        return fallback
    return primary


def _latest_formal_memory_item(strategy_memory_review):
    items = (strategy_memory_review or {}).get("items") or []
    formal_items = [item for item in items if item.get("kind") == "正式推荐"]
    return formal_items[-1] if formal_items else None


def _price_memory_value_on_date(price_memory, trade_date, column):
    if price_memory is None or price_memory.empty or not trade_date:
        return None
    date_text = _date_text(trade_date)
    rows = price_memory[price_memory["last_data_date_text"] == date_text]
    if rows.empty or column not in rows.columns:
        return None
    return _number(rows.iloc[-1].get(column))


def _build_realtime_acceptance(signal_close, quote):
    row = {
        "today_open": quote.get("open"),
        "today_high": quote.get("high"),
        "today_low": quote.get("low"),
        "latest_price": quote.get("current"),
    }
    result = _entry_acceptance_from_rows(signal_close, row)
    result["signal_close"] = _round(signal_close, 2)
    result["using_realtime_price"] = True
    result["note"] = "盘中用现价近似收盘位置，收盘前只能视为暂态判断。"
    return result


def _build_realtime_day_tape(quote, base_close):
    open_price = _number(quote.get("open"))
    high_price = _number(quote.get("high"))
    low_price = _number(quote.get("low"))
    current = _number(quote.get("current"))
    base_close = _number(base_close)

    close_position = None
    if current is not None and high_price is not None and low_price is not None and high_price > low_price:
        close_position = (current - low_price) / (high_price - low_price) * 100

    open_gap = _safe_pct(open_price, base_close)
    current_from_open = _safe_pct(current, open_price)
    high_open_has_pullback = None
    if base_close and open_gap is not None and low_price is not None:
        high_open_has_pullback = bool(open_gap <= 3.0 or low_price <= base_close * 1.015)

    acceptance_like = (
        bool(high_open_has_pullback)
        and close_position is not None
        and close_position >= 45
        and (current_from_open is not None and current_from_open >= -2.5)
    )

    if close_position is not None and close_position >= 70 and (current_from_open or 0) >= 0:
        grade = "盘中承接较强"
    elif acceptance_like:
        grade = "盘中承接尚可"
    elif close_position is not None and close_position <= 35:
        grade = "盘中承接偏弱"
    else:
        grade = "盘中中性"

    return {
        "base_close": _round(base_close, 2),
        "open_gap_pct": _round(open_gap, 2),
        "current_from_open_pct": _round(current_from_open, 2),
        "current_position_pct": _round(close_position, 2),
        "high_open_has_pullback": high_open_has_pullback,
        "acceptance_like": bool(acceptance_like),
        "grade": grade,
    }


def _market_realtime_summary(indices):
    if not indices:
        return "--"
    labels = []
    for item in indices:
        name = item.get("name") or item.get("symbol")
        change_pct = item.get("change_pct")
        labels.append(f"{name}{change_pct if change_pct is not None else '--'}%")
    return "，".join(labels)


def _format_level_item(name, value):
    rounded = _round(value, 2)
    if rounded is None:
        return None
    return {"name": name, "price": rounded, "text": f"{name}{rounded}"}


def _nearest_levels(current_price, level_items, direction, limit=4):
    current_price = _number(current_price)
    if current_price is None:
        return []
    prepared = []
    seen = set()
    for item in level_items:
        if not item:
            continue
        price = _number(item.get("price"))
        if price is None:
            continue
        key = (item.get("name"), price)
        if key in seen:
            continue
        seen.add(key)
        if direction == "support" and price <= current_price:
            distance = current_price - price
        elif direction == "pressure" and price >= current_price:
            distance = price - current_price
        else:
            continue
        prepared.append({**item, "distance": _round(distance, 2), "distance_pct": _round(_safe_pct(current_price, price), 2)})
    return sorted(prepared, key=lambda item: item["distance"])[:limit]


def _join_level_text(items):
    if not items:
        return "--"
    return "、".join(item.get("text") or f"{item.get('name')}{item.get('price')}" for item in items)


def _build_realtime_structure_view(mode, quote, day_tape, levels, target, has_trade_context=False):
    current = _number(quote.get("current"))
    open_price = _number(quote.get("open"))
    low_price = _number(quote.get("low"))
    high_price = _number(quote.get("high"))
    prior_low = _number(levels.get("prior_low"))
    prior_high = _number(levels.get("prior_high"))
    ma5 = _number(levels.get("ma5"))
    ma10 = _number(levels.get("ma10"))
    ma20 = _number(levels.get("ma20"))
    ma60 = _number(levels.get("ma60"))
    high_20d = _number(levels.get("high_20d"))
    low_20d = _number(levels.get("low_20d"))
    model_reference_stop = _number(levels.get("model_reference_stop"))
    effective_stop = _number(levels.get("effective_stop"))

    level_items = [
        _format_level_item("当日低点", low_price),
        _format_level_item("前日低点", prior_low),
        _format_level_item("MA5", ma5),
        _format_level_item("MA10", ma10),
        _format_level_item("MA20", ma20),
        _format_level_item("MA60", ma60),
        _format_level_item("20日低点", low_20d),
        _format_level_item("前日高点", prior_high),
        _format_level_item("当日高点", high_price),
        _format_level_item("20日高点", high_20d),
    ]
    if has_trade_context:
        level_items.append(_format_level_item("有效防守", effective_stop))

    supports = _nearest_levels(current, level_items, "support")
    pressures = _nearest_levels(current, level_items, "pressure")

    above_ma5 = bool(current is not None and ma5 is not None and current >= ma5)
    above_ma10 = bool(current is not None and ma10 is not None and current >= ma10)
    above_ma20 = bool(current is not None and ma20 is not None and current >= ma20)
    above_ma60 = bool(current is not None and ma60 is not None and current >= ma60)
    recovered_open = bool(current is not None and open_price is not None and current >= open_price)
    held_prior_low = bool(low_price is not None and prior_low is not None and low_price >= prior_low)

    observations = []
    if recovered_open and (day_tape or {}).get("acceptance_like"):
        observations.append("开盘后能收回并处在日内偏上位置，盘中承接不差。")
    elif not recovered_open:
        observations.append("现价仍低于开盘，盘中修复不足。")

    if held_prior_low:
        observations.append("盘中低点暂未跌破前一交易日低点，短线防守还没有被打穿。")
    elif prior_low is not None and low_price is not None:
        observations.append("盘中已经跌破前一交易日低点，说明承接质量要降级。")

    if above_ma5 and above_ma10 and above_ma20:
        structure = "均线修复较强"
        observations.append("现价在5/10/20日均线上方，短线结构较完整。")
    elif above_ma60 and not above_ma5:
        structure = "低位修复观察"
        observations.append("现价站在60日线附近或上方，但仍未收复5日线，属于低位修复观察。")
    elif not above_ma20:
        structure = "反弹未修复"
        observations.append("现价仍在20日线下方，趋势修复还不成立。")
    else:
        structure = "修复中"
        observations.append("结构在修复中，但还需要继续确认上方均线和压力位。")

    if mode == "no_formal_signal":
        summary = f"{structure}，但没有系统正式推荐，实时表现只作为观察，不自动升级为买点。"
        action_focus = "先看能否站回最近压力位并重新进入系统推荐；未满足前不按新仓买点处理。"
    elif mode == "entry_acceptance_check":
        summary = f"{structure}，当前重点是验证推荐次日承接是否成立。"
        action_focus = "只按原推荐计划分批，承接转弱或跌破防守就放弃后续买入。"
    else:
        summary = f"{structure}，当前重点是持仓纪律，而不是重新追买。"
        action_focus = "已有仓位按止盈/防守移动处理；无仓不把持仓信号当新买点。"

    return {
        "summary": summary,
        "structure": structure,
        "support_text": _join_level_text(supports),
        "pressure_text": _join_level_text(pressures),
        "supports": supports,
        "pressures": pressures,
        "observations": observations,
        "action_focus": action_focus,
    }


def _build_realtime_execution_view(
    stock_code,
    target,
    trade_plan,
    holding_stop_view,
    strategy_memory_review,
    price_memory,
    latest_trade_date,
    has_user_position=False,
    adaptive_drop_profile=None,
):
    snapshot = _fetch_realtime_snapshot(stock_code)
    if not snapshot.get("success"):
        return {
            "enabled": True,
            "success": False,
            "source": snapshot.get("source") or REALTIME_SOURCE_NAME,
            "reason": snapshot.get("reason") or "realtime_unavailable",
            "url": snapshot.get("url"),
        }

    quote = snapshot["quote"]
    quote_date = quote.get("quote_date")
    quote_current = _number(quote.get("current"))
    local_close = _number(target.get("latest_price"))
    quote_prev_close = _number(quote.get("prev_close"))
    prev_close_delta_pct = _safe_pct(quote_prev_close, local_close)
    data_consistency = {
        "local_close": _round(local_close, 2),
        "realtime_prev_close": _round(quote_prev_close, 2),
        "prev_close_delta_pct": _round(prev_close_delta_pct, 4),
        "tolerance_pct": REALTIME_PREV_CLOSE_TOLERANCE_PCT,
        "consistent": bool(
            local_close is not None
            and quote_prev_close is not None
            and prev_close_delta_pct is not None
            and abs(prev_close_delta_pct) <= REALTIME_PREV_CLOSE_TOLERANCE_PCT
        ),
    }
    if not data_consistency["consistent"]:
        return {
            "enabled": True,
            "success": False,
            "source": snapshot.get("source"),
            "url": snapshot.get("url"),
            "reason": "realtime_prev_close_mismatch",
            "data_consistency": data_consistency,
        }

    prev_close = quote_prev_close or local_close
    day_tape = _build_realtime_day_tape(quote, prev_close)
    latest_formal = _latest_formal_memory_item(strategy_memory_review)
    latest_formal_date = _date_text(latest_formal.get("trade_date")) if latest_formal else None
    plan = (latest_formal or {}).get("plan") or {}
    metrics = (latest_formal or {}).get("metrics") or {}
    formal_context = {
        "latest_formal_date": latest_formal_date,
        "strategy_type": latest_formal.get("strategy_type") if latest_formal else None,
        "outcome": latest_formal.get("outcome") if latest_formal else None,
        "entry_acceptance_ok": metrics.get("entry_acceptance_ok") if latest_formal else None,
        "entry_date": metrics.get("entry_date") if latest_formal else None,
        "entry_price": metrics.get("entry_price") if latest_formal else None,
        "target_low_pct": plan.get("target_low_pct"),
        "target_high_pct": plan.get("target_high_pct"),
        "defense_price": plan.get("defense_price"),
    }

    quote_is_after_local = _date_after(quote_date, latest_trade_date)
    is_entry_check = bool(
        latest_formal
        and latest_formal_date
        and latest_formal_date == _date_text(latest_trade_date)
        and quote_is_after_local
    )

    realtime_acceptance = None
    entry_price = _number(metrics.get("entry_price"))
    entry_accepted = metrics.get("entry_acceptance_ok")
    mode = "no_formal_signal"
    if is_entry_check:
        signal_close = _price_memory_value_on_date(price_memory, latest_formal_date, "latest_price")
        signal_close = signal_close or _number(target.get("latest_price"))
        realtime_acceptance = _build_realtime_acceptance(signal_close, quote)
        entry_accepted = realtime_acceptance.get("entry_acceptance_ok")
        entry_price = realtime_acceptance.get("entry_avg_price") if entry_accepted else None
        mode = "entry_acceptance_check"
    elif latest_formal:
        mode = "holding_management"

    target_low = _number(plan.get("target_low_pct"))
    target_high = _number(plan.get("target_high_pct"))
    effective_stop = _number((holding_stop_view or {}).get("effective_stop_price"))
    dynamic_stop = _number((holding_stop_view or {}).get("dynamic_stop_price"))
    model_reference_stop = effective_stop or _number(plan.get("defense_price")) or _number(trade_plan.get("stop_price"))
    has_formal_context = mode in {"entry_acceptance_check", "holding_management"}
    has_trade_context = bool(has_formal_context or has_user_position)
    if mode == "entry_acceptance_check":
        context_label = "正式推荐承接验证"
    elif mode == "holding_management":
        context_label = "历史正式推荐持仓管理"
    elif has_user_position:
        context_label = "用户持仓风控"
    else:
        context_label = "无推荐观察"
    defense_price = model_reference_stop if has_trade_context else None
    high_price = _number(quote.get("high"))
    low_price = _number(quote.get("low"))
    open_price = _number(quote.get("open"))

    current_return = _safe_pct(quote_current, entry_price) if entry_price else None
    high_return = _safe_pct(high_price, entry_price) if entry_price else None
    low_return = _safe_pct(low_price, entry_price) if entry_price else None
    max_high_return = high_return
    min_low_return = low_return
    if metrics.get("max_high_return_pct") is not None and max_high_return is not None:
        max_high_return = max(max_high_return, _number(metrics.get("max_high_return_pct")))
    if metrics.get("min_low_return_pct") is not None and min_low_return is not None:
        min_low_return = min(min_low_return, _number(metrics.get("min_low_return_pct")))

    stop_touched = bool(defense_price is not None and low_price is not None and low_price <= defense_price)
    stop_broken_now = bool(defense_price is not None and quote_current is not None and quote_current <= defense_price)
    target_low_hit = bool(target_low is not None and max_high_return is not None and max_high_return >= target_low)
    target_high_hit = bool(target_high is not None and current_return is not None and current_return >= target_high)

    ma5 = _number(target.get("ma5"))
    ma10 = _number(target.get("ma10"))
    ma20 = _number(target.get("ma20"))
    ma60 = _number(target.get("ma60"))
    prior_high = _number(target.get("today_high"))
    prior_low = _number(target.get("today_low"))
    high_20d = _number(target.get("high_20d"))
    low_20d = _number(target.get("low_20d"))

    key_levels = {
        "prev_close": _round(prev_close, 2),
        "prior_high": _round(prior_high, 2),
        "prior_low": _round(prior_low, 2),
        "ma5": _round(ma5, 2),
        "ma10": _round(ma10, 2),
        "ma20": _round(ma20, 2),
        "ma60": _round(ma60, 2),
        "high_20d": _round(high_20d, 2),
        "low_20d": _round(low_20d, 2),
        "dynamic_stop": _round(dynamic_stop, 2),
        "model_reference_stop": _round(model_reference_stop, 2),
        "effective_stop": _round(defense_price, 2),
        "has_trade_context": has_trade_context,
        "has_formal_context": has_formal_context,
        "has_user_position": bool(has_user_position),
        "execution_context_label": context_label,
        "open_recovered": bool(quote_current is not None and open_price is not None and quote_current >= open_price),
        "above_ma5": bool(quote_current is not None and ma5 is not None and quote_current >= ma5),
        "above_ma10": bool(quote_current is not None and ma10 is not None and quote_current >= ma10),
        "above_ma20": bool(quote_current is not None and ma20 is not None and quote_current >= ma20),
        "above_ma60": bool(quote_current is not None and ma60 is not None and quote_current >= ma60),
    }
    structure_view = _build_realtime_structure_view(
        mode,
        quote,
        day_tape,
        key_levels,
        target,
        has_trade_context=has_trade_context,
    )
    intraday_drop_view = _build_intraday_drop_view(
        {"success": True, "quote": quote},
        key_levels,
        adaptive_drop_profile=adaptive_drop_profile,
    )
    repair_watch_text = None
    if intraday_drop_view and intraday_drop_view.get("watch_text") != "关键线":
        repair_watch_text = intraday_drop_view.get("watch_text")
    elif structure_view.get("pressures"):
        repair_watch_text = _join_level_text((structure_view.get("pressures") or [])[:2])

    warnings = []
    if quote.get("open_gap_pct") is not None and quote["open_gap_pct"] <= -2 and quote_current and quote_current < prev_close:
        warnings.append("低开接近或超过2%且暂未收回前收，不满足主动新买纪律。")
    if quote.get("open_gap_pct") is not None and quote["open_gap_pct"] > 3:
        pullback_limit = prev_close * 1.015 if prev_close else None
        if pullback_limit and low_price and low_price > pullback_limit:
            warnings.append("高开超过3%且尚未回踩到前收+1.5%以内，不追。")
    if stop_touched and has_formal_context:
        warnings.append("盘中触及有效防守，必须观察能否快速收回。")
    if target_high_hit:
        warnings.append("实时收益已超过原计划止盈上沿，仓位纪律优先于继续进攻。")

    if mode == "entry_acceptance_check":
        if entry_accepted:
            decision = "次日承接暂通过"
            no_position_action = "可按原推荐计划分批执行；盘中判断未收盘，仍不能追高一次打满。"
            holding_action = "若已执行首笔，后续只在承接继续保持且未冲到压力滞涨时再考虑补第二笔。"
        else:
            decision = "次日承接暂未通过"
            no_position_action = "按承接纪律先不买，等重新收回开盘/分时均线并改善收盘位置。"
            holding_action = "若已有底仓，只按防守线管理，不加仓。"
    elif mode == "holding_management":
        if entry_accepted is False:
            decision = "原推荐入场承接未通过"
            no_position_action = "无仓不补追；该信号按历史纪律已经放弃买点。"
            holding_action = "已有仓位只按防守处理。"
        elif stop_broken_now:
            decision = "实时跌破有效防守"
            no_position_action = "无仓不参与。"
            holding_action = "按防守纪律先降风险，不等待模型重新解释。"
        elif stop_touched:
            decision = "盘中触及防守"
            no_position_action = "无仓不参与。"
            holding_action = "若不能快速收回防守线，先降风险；收回后也只观察，不加仓。"
        elif target_high_hit:
            decision = "超过止盈上沿"
            no_position_action = "无仓不追。"
            holding_action = "已有仓位优先兑现一部分，并把剩余仓位转为移动止盈管理。"
        elif target_low_hit:
            decision = "进入止盈区"
            no_position_action = "无仓不追。"
            holding_action = "已有仓位按计划分批兑现或提高防守，不再按新仓买点处理。"
        elif current_return is not None and current_return >= 0:
            decision = "持有期内正向"
            no_position_action = "未再次落库前不当作新买点。"
            holding_action = "防守未破则继续按原计划持有，回撤到防守附近先处理风险。"
        else:
            decision = "持有期内回撤"
            no_position_action = "无仓不接回撤。"
            holding_action = "先看有效防守和关键均线能否守住，未修复前不加仓。"
    else:
        decision = "无系统正式推荐实时观察"
        quote_change_pct = _number(quote.get("change_pct"))
        if quote_change_pct is not None and quote_change_pct < 0:
            no_position_action = "当前实时下跌且没有落库推荐，无仓不接下跌。"
            if repair_watch_text:
                holding_action = (
                    f"若本来有仓，先看能否收复{repair_watch_text}；"
                    "收不回不加仓，只按自己的成本/原计划风控处理。"
                )
            else:
                holding_action = "若本来有仓，只按自己的成本/原计划风控处理；未收复关键线不加仓。"
        elif quote_change_pct is not None and quote_change_pct > 0:
            no_position_action = "没有落库推荐上下文，实时走强也不自动转化为系统买点。"
            holding_action = "若本来有仓，只按自己的成本/原计划风控处理，不追着加仓。"
        else:
            no_position_action = "没有落库推荐上下文，无仓只观察，不主动开仓。"
            holding_action = "若本来有仓，只按自己的成本/原计划风控处理。"

    return {
        "enabled": True,
        "success": True,
        "source": snapshot.get("source"),
        "url": snapshot.get("url"),
        "local_base_date": _date_text(latest_trade_date),
        "quote": quote,
        "market_indices": snapshot.get("indices") or [],
        "market_summary": _market_realtime_summary(snapshot.get("indices") or []),
        "data_consistency": data_consistency,
        "mode": mode,
        "decision": decision,
        "no_position_action": no_position_action,
        "holding_action": holding_action,
        "warnings": warnings,
        "execution_context": {
            "mode": mode,
            "label": context_label,
            "has_trade_context": has_trade_context,
            "has_formal_context": has_formal_context,
            "has_user_position": bool(has_user_position),
            "allow_stop_action": has_trade_context,
        },
        "formal_context": formal_context,
        "day_tape": day_tape,
        "structure_view": structure_view,
        "intraday_drop_view": intraday_drop_view,
        "adaptive_drop_profile": {
            key: value
            for key, value in (adaptive_drop_profile or {}).items()
            if key not in {"event_stats", "low_samples"}
        },
        "realtime_acceptance": realtime_acceptance,
        "holding_metrics": {
            "entry_price": _round(entry_price, 2),
            "current_return_pct": _round(current_return, 2),
            "high_return_pct": _round(max_high_return, 2),
            "low_return_pct": _round(min_low_return, 2),
            "target_low_hit": target_low_hit,
            "target_high_hit": target_high_hit,
            "stop_touched": stop_touched,
            "stop_broken_now": stop_broken_now,
            "distance_to_stop_pct": _round(_safe_pct(quote_current, defense_price), 2) if defense_price else None,
        },
        "key_levels": key_levels,
        "discipline_note": "实时层只复用承接确认、低开放弃、高开不追、防守和止盈纪律；不改历史评分、不写库、不参与回测。",
    }


def _build_fail_reasons(candidate):
    reasons = []
    style = candidate.get("style")
    trend_state = candidate.get("trend_state")
    risk_adjusted_score = _round(candidate.get("risk_adjusted_score"), 2)
    risk_score = _round(candidate.get("risk_score"), 2)

    if style not in adaptive.ADAPTIVE_PRECISION_STYLES:
        style_label = adaptive.STYLE_LABELS.get(style) if style else "未形成稳定风格"
        reasons.append(f"风格不是当前精确模式允许的慢涨跟随，而是{style_label}")
    if trend_state not in adaptive.ADAPTIVE_PRECISION_TREND_STATES:
        reasons.append(f"趋势结构未达到周月同步或月线抬升要求，当前为{trend_state or '未知'}")
    if risk_adjusted_score is None or risk_adjusted_score < adaptive.ADAPTIVE_MIN_RISK_ADJUSTED_SCORE:
        reasons.append(f"风险调整分不足{adaptive.ADAPTIVE_MIN_RISK_ADJUSTED_SCORE:g}，当前为{risk_adjusted_score}")
    if risk_score is None or risk_score > adaptive.ADAPTIVE_MAX_RISK_SCORE:
        reasons.append(f"风险分高于{adaptive.ADAPTIVE_MAX_RISK_SCORE:g}，当前为{risk_score}")
    if candidate.get("risk_overlay_block_formal"):
        reasons.append(f"风险覆盖层拦截正式推荐：{candidate.get('risk_overlay_labels') or '特殊股票池/事件风险'}")
    elif candidate.get("risk_overlay_downgrade"):
        reasons.append(f"事件覆盖提示需降级处理：{candidate.get('risk_overlay_labels') or '特殊股票池/事件风险'}")
    return reasons


def _passes_precision_gate(candidate):
    return not _build_fail_reasons(candidate)


def _resolve_recommendation_tier(candidate, precision_candidate=False, saved_strategy=None, evidence=None):
    if candidate.get("risk_overlay_block_formal"):
        return {
            "tier": adaptive.RECOMMENDATION_TIER_RESEARCH,
            "reason": (
                "风险覆盖层识别到特殊股票池、公告/财务/解禁/龙虎榜等硬风险，"
                "即使价格形态较强，也不作为普通正式买点。"
            ),
        }

    if saved_strategy and _has_full_formal_strategy(saved_strategy):
        return {
            "tier": adaptive.RECOMMENDATION_TIER_FORMAL,
            "reason": "已进入当日策略结果表，说明系统级市场、健康度和落库校验通过。",
        }

    if saved_strategy and _has_small_validation_strategy(saved_strategy):
        return {
            "tier": adaptive.RECOMMENDATION_TIER_OBSERVE,
            "reason": "已进入当日小仓验证策略；系统级校验通过，但不等同普通正式推荐，必须等待次日承接确认。",
        }

    if saved_strategy and _has_long_runway_strategy(saved_strategy):
        return {
            "tier": adaptive.RECOMMENDATION_TIER_RESEARCH,
            "reason": "已进入长跑潜力模型的中长期跟踪；这是持续跟踪信号，不等同每日短线正式推荐。",
        }

    if candidate.get("risk_overlay_downgrade") and precision_candidate:
        return {
            "tier": adaptive.RECOMMENDATION_TIER_OBSERVE,
            "reason": "形态通过精确门槛，但事件覆盖提示要求先观察承接，不按无风险买点处理。",
        }

    if precision_candidate:
        return {
            "tier": adaptive.RECOMMENDATION_TIER_OBSERVE,
            "reason": "个股质量通过精确门槛，但未进入当日正式推荐，等待系统级确认或次日承接。",
        }

    risk_adjusted_score = _number(candidate.get("risk_adjusted_score"))
    evidence_grade = ((evidence or {}).get("grade") or {}).get("grade")
    if evidence_grade in ("强", "中") or (risk_adjusted_score is not None and risk_adjusted_score >= 65):
        return {
            "tier": adaptive.RECOMMENDATION_TIER_RESEARCH,
            "reason": "具备研究跟踪价值，但买点、风格、趋势或风险门槛尚未完整满足。",
        }

    return {
        "tier": adaptive.RECOMMENDATION_TIER_AVOID,
        "reason": "当前没有形成足够的可执行优势。",
    }


def _build_operation(
    candidate,
    trade_plan,
    recommendation_tier=None,
    formal_candidate=False,
    formal_rank=None,
    holding_stop_view=None,
):
    latest_price = _round(candidate.get("latest_price"), 2)
    dynamic_stop_price = _round(trade_plan.get("stop_price"), 2)
    stop_price = _round((holding_stop_view or {}).get("effective_stop_price"), 2)
    if stop_price is None:
        stop_price = dynamic_stop_price
    risk_adjusted_score = _round(candidate.get("risk_adjusted_score"), 2)
    risk_score = _round(candidate.get("risk_score"), 2)
    holding_trend_intact = _is_holding_trend_intact(candidate, trade_plan, stop_price=stop_price)

    if formal_candidate:
        rank_text = f"，模型排名第{formal_rank}" if formal_rank is not None else ""
        if stop_price is not None:
            holding_position = (
                f"已有仓位趋势和防守未破，可继续按计划持有；不要因为未确认的新机会主动卖出，"
                f"若跌破{stop_price}且不能快速收回，执行减仓或退出。"
                if holding_trend_intact
                else f"已有仓位按{stop_price}做防守；跌破后不能快速收回，执行减仓或退出。"
            )
        else:
            holding_position = "已有仓位可继续观察，但缺少有效防守价，不适合加仓。"
        return {
            "decision": f"可执行候选{rank_text}",
            "new_position": trade_plan.get("conclusion"),
            "holding_position": holding_position,
            "risk_control": trade_plan.get("stop_rule"),
        }

    fail_reasons = _build_fail_reasons(candidate)
    if recommendation_tier == adaptive.RECOMMENDATION_TIER_OBSERVE:
        holding_position = (
            f"已有仓位趋势和防守未破，可按{stop_price}继续持有观察；未转为正式推荐前不加仓，"
            "也不因其他非正式机会主动换出。"
            if stop_price is not None and holding_trend_intact
            else f"已有仓位可按{stop_price}做防守；未转为正式推荐前不加仓。"
            if stop_price is not None
            else "已有仓位只保留观察仓；未转为正式推荐前不加仓。"
        )
        return {
            "decision": "观察候选，暂不作为正式买入",
            "new_position": (
                "无仓只等触发：重新进入每日/自适应正式推荐，或次日缩量回踩后放量站稳关键均线。"
            ),
            "holding_position": holding_position,
            "risk_control": "系统级确认未完成；" + ("；".join(fail_reasons[:3]) if fail_reasons else trade_plan.get("stop_rule")),
        }

    if recommendation_tier == adaptive.RECOMMENDATION_TIER_RESEARCH:
        holding_position = (
            f"已有仓位趋势和防守未破，以{stop_price}为防守继续观察；这不是新买点，"
            "但也不等于必须卖出。"
            if stop_price is not None and holding_trend_intact
            else f"已有仓位以{stop_price}为防守；反弹不能放量站稳时降低仓位。"
            if stop_price is not None
            else "已有仓位只做跟踪，不满足买点前不加仓。"
        )
        return {
            "decision": "有研究价值，但不是买入信号",
            "new_position": "无仓不主动买入，只加入跟踪池，等趋势结构、风险分和系统推荐同时改善。",
            "holding_position": holding_position,
            "risk_control": "；".join(fail_reasons[:4]) if fail_reasons else trade_plan.get("stop_rule"),
        }

    near_quality = (
        risk_adjusted_score is not None
        and risk_adjusted_score >= 75
        and risk_score is not None
        and risk_score <= 4
    )

    if near_quality:
        decision = "观察等待，不进入正式买入候选"
        new_position = (
            "无仓不追；只有重新进入正式推荐池，或次日缩量回踩后放量站稳关键均线，才考虑小仓试错。"
        )
    else:
        decision = "暂不建议新开仓"
        new_position = "无仓先放弃，等待模型评分、趋势结构和风险分同时改善。"

    if stop_price is not None:
        if latest_price is not None and latest_price <= stop_price:
            holding = f"已有仓位已触及或跌破防守价{stop_price}，不加仓，优先减仓或退出。"
        elif holding_trend_intact:
            holding = (
                f"已有仓位趋势和防守未破，以{stop_price}为防守继续持有观察；"
                "没有正式卖出信号前，不因为别的观察票主动换出。"
            )
        else:
            holding = f"已有仓位以{stop_price}为防守；跌破后不能快速收回，减仓或退出。"
    else:
        holding = "已有仓位只保留观察仓；缺少清晰防守价时不做加仓。"

    return {
        "decision": decision,
        "new_position": new_position,
        "holding_position": holding,
        "risk_control": "；".join(fail_reasons[:4]) if fail_reasons else trade_plan.get("stop_rule"),
    }


def _is_holding_trend_intact(record, trade_plan, stop_price=None):
    latest_price = _number(record.get("latest_price"))
    stop_price = _number(stop_price)
    if stop_price is None:
        stop_price = _number(trade_plan.get("stop_price"))
    if latest_price is None:
        return False
    if stop_price is not None and latest_price <= stop_price:
        return False

    trend_state = record.get("trend_state")
    if trend_state in HOLDING_INTACT_TREND_STATES:
        return True

    price_vs_ma20 = _number(record.get("price_vs_ma20"))
    price_vs_ma60 = _number(record.get("price_vs_ma60"))
    ma20_slope_5d = _number(record.get("ma20_slope_5d"))
    ma60_slope_10d = _number(record.get("ma60_slope_10d"))
    above_near_ma20 = price_vs_ma20 is not None and price_vs_ma20 >= -0.015
    above_near_ma60 = price_vs_ma60 is not None and price_vs_ma60 >= -0.03
    ma20_not_broken = ma20_slope_5d is None or ma20_slope_5d >= -0.2
    ma60_not_broken = ma60_slope_10d is None or ma60_slope_10d >= -0.2
    return bool(above_near_ma20 and above_near_ma60 and ma20_not_broken and ma60_not_broken)


def _build_professional_view(
    formal_candidate,
    evidence,
    market,
    industry_view=None,
    scorecard=None,
    intraday_view=None,
    liquidity_view=None,
    recommendation_tier=None,
    risk_overlay_view=None,
    recent_saved_strategy=None,
):
    evidence_grade = ((evidence or {}).get("grade") or {}).get("grade")
    warnings = list(((evidence or {}).get("grade") or {}).get("warnings") or [])
    market_env = (market or {}).get("market_env")
    industry_grade = (industry_view or {}).get("strength_grade")
    rating = (scorecard or {}).get("rating")
    tape_grade = (intraday_view or {}).get("tape_grade")
    liquidity_grade = (liquidity_view or {}).get("grade")

    recent_full_formal = _has_full_formal_strategy(recent_saved_strategy)
    recent_formal_date = _latest_strategy_trade_date(recent_saved_strategy) if recent_full_formal else None

    if recommendation_tier == adaptive.RECOMMENDATION_TIER_FORMAL and evidence_grade in ("强", "中") and rating in ("A", "B"):
        stance = "可执行，但只按计划交易"
        sizing = "中等仓位；若市场转弱或开盘高开过多，不追价。"
    elif recommendation_tier == adaptive.RECOMMENDATION_TIER_FORMAL:
        stance = "可执行性偏谨慎"
        sizing = "轻到中等仓位；必须等回踩承接，不能直接追高。"
    elif recommendation_tier == adaptive.RECOMMENDATION_TIER_OBSERVE:
        stance = "观察候选，等待系统确认"
        sizing = "无仓等待正式推荐或次日承接确认；已有仓位按防守价管理，不加仓。"
    elif recommendation_tier == adaptive.RECOMMENDATION_TIER_RESEARCH and recent_full_formal:
        stance = f"{recent_formal_date}曾进系统正式推荐，当前未新增正式推荐"
        sizing = "无仓不按当前日追买；已有仓位按原推荐持有期、防守价和次日承接结果管理。"
    elif recommendation_tier == adaptive.RECOMMENDATION_TIER_RESEARCH:
        stance = "研究价值较高，但未进系统正式推荐"
        sizing = "无仓只跟踪；已有仓位按防守价管理。"
    elif evidence_grade in ("强", "中"):
        stance = "研究价值较高，但未进系统正式推荐"
        sizing = "无仓等待触发条件；已有仓位按防守价管理。"
    else:
        stance = "暂不具备专业买入条件"
        sizing = "无仓不参与；已有仓位以防守和减仓纪律为主。"

    if market_env in ("偏弱", "未知"):
        sizing = f"{sizing} 当前市场环境不支持提高仓位。"
    if industry_grade in ("弱共振", "偏弱"):
        sizing = f"{sizing} 行业共振不足，不提高仓位级别。"
    if tape_grade in ("冲高回落", "收盘偏弱"):
        sizing = f"{sizing} 日内承接不佳，只能等次日修复确认。"
    if liquidity_grade == "流动性有瑕疵":
        sizing = f"{sizing} 流动性有瑕疵，实际下单要降低仓位并使用限价。"
    if (risk_overlay_view or {}).get("block_formal"):
        stance = "风险覆盖层拦截，暂不作为普通买点"
        sizing = f"{sizing} 已触发特殊股票池/事件风控，宁可错过也不追高。"
    elif (risk_overlay_view or {}).get("downgrade"):
        sizing = f"{sizing} 事件覆盖提示只作降级和仓位约束，不等同于趋势看空；执行时必须等承接确认。"

    warnings.extend((intraday_view or {}).get("warnings") or [])
    warnings.extend((liquidity_view or {}).get("warnings") or [])
    if (risk_overlay_view or {}).get("labels"):
        warning_prefix = "硬风险覆盖" if (risk_overlay_view or {}).get("block_formal") else "事件覆盖提示"
        warnings.append(f"{warning_prefix}:{(risk_overlay_view or {}).get('labels')}")

    return {
        "stance": stance,
        "evidence_grade": evidence_grade,
        "industry_grade": industry_grade,
        "position_sizing": sizing,
        "warnings": list(dict.fromkeys(warnings)),
    }


def _summarize_key_data(record):
    latest_price = _round(record.get("latest_price"), 2)
    today_open = _round(record.get("today_open"), 2)
    today_high = _round(record.get("today_high"), 2)
    today_low = _round(record.get("today_low"), 2)
    ma20 = _round(record.get("ma20"), 2)
    ma60 = _round(record.get("ma60"), 2)
    price_vs_ma20 = _round((record.get("price_vs_ma20") or 0) * 100, 2) if record.get("price_vs_ma20") is not None else None
    price_vs_ma60 = _round((record.get("price_vs_ma60") or 0) * 100, 2) if record.get("price_vs_ma60") is not None else None

    return {
        "latest_price": latest_price,
        "today_open": today_open,
        "today_high": today_high,
        "today_low": today_low,
        "today_change": _format_gap(record.get("today_change")),
        "change_5d": _format_gap(record.get("change_5d")),
        "change_20d": _format_gap(record.get("change_20d")),
        "change_30d": _format_gap(record.get("change_30d")),
        "ma20": ma20,
        "ma60": ma60,
        "price_vs_ma20": _format_gap(price_vs_ma20),
        "price_vs_ma60": _format_gap(price_vs_ma60),
        "today_amp": _format_gap(record.get("today_amp")),
        "vr_today": _round(record.get("vr_today"), 2),
        "turnover_rate": _format_gap(record.get("turnover_rate")),
        "today_amount": _round(record.get("today_amount"), 2),
        "amount_vs_avg_5d": _round(record.get("amount_vs_avg_5d"), 2),
        "turnover_vs_avg_5d": _round(record.get("turnover_vs_avg_5d"), 2),
    }


def _resolve_trade_plan_for_record(record, market_env, horizon_profiles, style_horizon_profiles):
    scored = _score_single_record(record, horizon_profiles, style_horizon_profiles)
    style_name = scored.get("style") or scored.get("dominant_style")
    style_label = scored.get("style_label") or scored.get("dominant_style_label") or adaptive.STYLE_LABELS.get(style_name)
    trade_plan = adaptive._build_trade_plan(
        scored,
        market_env,
        style_label=style_label,
        style_name=style_name,
        trend_state=scored.get("trend_state"),
        adaptive_score=scored.get("adaptive_score"),
    )
    return scored, trade_plan


def _build_holding_stop_view(
    target,
    trade_plan,
    prepared,
    trade_date,
    market_env,
    horizon_profiles,
    style_horizon_profiles,
    cost_price=None,
    entry_date=None,
    prior_stop_price=None,
):
    dynamic_stop = _number(trade_plan.get("stop_price"))
    candidates = []
    warnings = []
    trailing_records = []
    entry_stop = None
    entry_date_text = _date_text(entry_date)
    trade_date_text = _date_text(trade_date)

    if dynamic_stop is not None:
        candidates.append(
            {
                "source": "当日策略防守",
                "date": trade_date_text,
                "stop_price": _round(dynamic_stop, 2),
            }
        )

    prior_stop = _number(prior_stop_price)
    if prior_stop is not None:
        candidates.append(
            {
                "source": "上次记录防守",
                "date": None,
                "stop_price": _round(prior_stop, 2),
            }
        )

    cost = _number(cost_price)
    cost_stop = None
    if cost is not None and cost > 0:
        cost_stop = cost * (1 - HOLDING_COST_MAX_LOSS_PCT / 100)
        candidates.append(
            {
                "source": f"成本最大亏损防守({HOLDING_COST_MAX_LOSS_PCT:g}%)",
                "date": None,
                "stop_price": _round(cost_stop, 2),
            }
        )

    if entry_date_text:
        stock_code = adaptive._normalize_stock_code(target.get("stock_code"))
        stock_history = prepared[prepared["stock_code"] == stock_code].copy()
        stock_history["last_data_date_text"] = stock_history["last_data_date"].apply(_date_text)
        stock_history = stock_history[
            (stock_history["last_data_date_text"] >= entry_date_text)
            & (stock_history["last_data_date_text"] <= trade_date_text)
        ].sort_values("last_data_date")

        if stock_history.empty:
            warnings.append(f"入场日{entry_date_text}不在当前样本窗口内，无法计算入场以来移动防守。")
        else:
            for _, history_row in stock_history.iterrows():
                _, day_plan = _resolve_trade_plan_for_record(
                    history_row.to_dict(),
                    market_env,
                    horizon_profiles,
                    style_horizon_profiles,
                )
                day_stop = _number(day_plan.get("stop_price"))
                if day_stop is None:
                    continue
                record = {
                    "date": _date_text(history_row.get("last_data_date")),
                    "stop_price": _round(day_stop, 2),
                }
                trailing_records.append(record)

            if trailing_records:
                entry_stop = trailing_records[0]["stop_price"]
                highest_record = max(trailing_records, key=lambda item: item["stop_price"])
                candidates.append(
                    {
                        "source": "入场以来移动防守",
                        "date": highest_record["date"],
                        "stop_price": highest_record["stop_price"],
                    }
                )
            else:
                warnings.append(f"入场日{entry_date_text}以来没有可用防守价。")
    elif prior_stop is None and cost_stop is None:
        warnings.append("未提供入场日或上次防守价，持仓防守只能按当日策略价，可能随状态下移。")

    if not candidates:
        return {
            "dynamic_stop_price": None,
            "effective_stop_price": None,
            "source": None,
            "source_date": None,
            "entry_date": entry_date_text,
            "entry_stop_price": entry_stop,
            "cost_price": _round(cost, 2),
            "cost_stop_price": _round(cost_stop, 2),
            "prior_stop_price": _round(prior_stop, 2),
            "locked": False,
            "warnings": warnings,
        }

    effective = max(candidates, key=lambda item: item["stop_price"])
    effective_stop = effective["stop_price"]
    dynamic_stop_rounded = _round(dynamic_stop, 2)
    locked = (
        dynamic_stop_rounded is not None
        and effective_stop is not None
        and effective_stop > dynamic_stop_rounded
    )
    if locked:
        warnings.append("持仓有效防守高于当日策略防守，本次不会跟随模型下移。")

    return {
        "dynamic_stop_price": dynamic_stop_rounded,
        "effective_stop_price": effective_stop,
        "source": effective["source"],
        "source_date": effective.get("date"),
        "entry_date": entry_date_text,
        "entry_stop_price": entry_stop,
        "cost_price": _round(cost, 2),
        "cost_stop_price": _round(cost_stop, 2),
        "prior_stop_price": _round(prior_stop, 2),
        "locked": locked,
        "candidates": candidates,
        "trailing_records": trailing_records,
        "warnings": warnings,
    }


def _load_analysis_context(normalized_code, stock_code, analysis_date, lookback_trade_days, status):
    history = _load_history_window(analysis_date=analysis_date, lookback_trade_days=lookback_trade_days)
    status(f"历史窗口加载完成: rows={len(history)}")
    if history.empty:
        return None, {
            "success": False,
            "reason": "history_empty",
            "stock_code": normalized_code,
            "message": "历史数据为空，无法分析。",
        }

    prepared = adaptive._prepare_short_term_history(history)
    status(f"特征准备完成: rows={len(prepared) if prepared is not None else 0}")
    if prepared is None or prepared.empty:
        return None, {
            "success": False,
            "reason": "prepared_history_empty",
            "stock_code": normalized_code,
            "message": "历史数据无法完成特征准备。",
        }

    latest_trade_date = pd.to_datetime(prepared["last_data_date"], errors="coerce").max()
    trade_date = _date_text(latest_trade_date)
    latest_snapshot = prepared[prepared["last_data_date"] == latest_trade_date].copy()
    status(f"最新快照定位完成: trade_date={trade_date}, rows={len(latest_snapshot)}")
    if latest_snapshot.empty:
        return None, {
            "success": False,
            "reason": "snapshot_empty",
            "stock_code": normalized_code,
            "message": "最新交易日快照为空。",
        }

    stock_rows = latest_snapshot[latest_snapshot["stock_code"] == normalized_code].copy()
    if stock_rows.empty:
        last_seen = prepared[prepared["stock_code"] == normalized_code]["last_data_date"].max()
        status(f"股票不在最新快照: last_seen={_date_text(last_seen)}")
        return None, {
            "success": False,
            "reason": "stock_not_in_latest_snapshot",
            "stock_code": normalized_code,
            "latest_trade_date": trade_date,
            "last_seen_date": _date_text(last_seen),
            "message": f"{normalized_code} 不在最新交易日 {trade_date} 的有效股票池中。",
        }

    horizon_profiles, style_horizon_profiles = _build_profiles(prepared)
    status("自适应画像构建完成")
    return {
        "stock_code": stock_code,
        "normalized_code": normalized_code,
        "history": history,
        "prepared": prepared,
        "latest_trade_date": latest_trade_date,
        "trade_date": trade_date,
        "latest_snapshot": latest_snapshot,
        "stock_rows": stock_rows,
        "horizon_profiles": horizon_profiles,
        "style_horizon_profiles": style_horizon_profiles,
        "raw_record": stock_rows.iloc[0].to_dict(),
    }, None


def _score_stock_context(context, status):
    normalized_code = context["normalized_code"]
    prepared = context["prepared"]
    latest_snapshot = context["latest_snapshot"]
    trade_date = context["trade_date"]
    horizon_profiles = context["horizon_profiles"]
    style_horizon_profiles = context["style_horizon_profiles"]

    target = _score_single_record(context["raw_record"], horizon_profiles, style_horizon_profiles)
    status(
        f"单票评分完成: adaptive_score={_round(target.get('adaptive_score'), 2)}, "
        f"risk_adjusted={_round(target.get('risk_adjusted_score'), 2)}"
    )
    status("风险覆盖开始: include_external=True")
    target_frame = risk_overlay.apply_risk_overlay_to_candidates(
        pd.DataFrame([target]),
        history=prepared,
        trade_date=trade_date,
        include_external=True,
        score_column="risk_adjusted_score",
        score_penalty_multiplier=adaptive.EXTERNAL_RISK_SCORE_PENALTY_MULTIPLIER,
    )
    if not target_frame.empty:
        target = target_frame.iloc[0].to_dict()
        target["risk_labels"] = adaptive._merge_risk_label_text(
            target.get("risk_labels"),
            target.get("risk_overlay_labels"),
        )
    status(
        f"风险覆盖完成: level={target.get('risk_overlay_level')}, "
        f"block={bool(target.get('risk_overlay_block_formal'))}, "
        f"downgrade={bool(target.get('risk_overlay_downgrade'))}"
    )

    market = _resolve_market_env(latest_snapshot)
    saved_strategy = _load_saved_strategy(normalized_code, trade_date)
    recent_saved_strategy = _load_recent_saved_strategy(normalized_code, trade_date)
    strategy_memory = _load_strategy_memory(normalized_code, trade_date)
    price_memory = _load_stock_price_memory(normalized_code, trade_date)
    strategy_memory_review = _evaluate_saved_strategy_memory(strategy_memory, price_memory, trade_date)
    saved_rank = _extract_saved_rank(saved_strategy)
    saved_strategy_hint = _build_saved_strategy_hint(saved_strategy)
    recent_saved_strategy_hint = _build_recent_saved_strategy_hint(recent_saved_strategy)
    precision_candidate = _passes_precision_gate(target)
    evidence = _build_evidence(prepared, target)
    status(
        f"历史证据完成: grade={((evidence or {}).get('grade') or {}).get('grade')}, "
        f"similar_sample={((evidence or {}).get('similar_sample') or {}).get('sample_size')}"
    )
    status(
        f"落库推荐复盘完成: total={strategy_memory_review.get('summary', {}).get('total')}, "
        f"formal={strategy_memory_review.get('summary', {}).get('formal_total')}"
    )
    recommendation_tier_info = _resolve_recommendation_tier(
        target,
        precision_candidate=precision_candidate,
        saved_strategy=saved_strategy,
        evidence=evidence,
    )
    recommendation_tier = recommendation_tier_info["tier"]
    formal_candidate = recommendation_tier == adaptive.RECOMMENDATION_TIER_FORMAL
    industry_view = _build_industry_view(latest_snapshot, target)
    a_share_profile = _resolve_a_share_profile(normalized_code, target.get("stock_name"))
    intraday_view = _build_intraday_view(target, a_share_profile)
    liquidity_view = _build_liquidity_view(target, a_share_profile)

    style_name = target.get("style") or target.get("dominant_style")
    style_label = target.get("style_label") or target.get("dominant_style_label") or adaptive.STYLE_LABELS.get(style_name)
    trade_plan = adaptive._build_trade_plan(
        target,
        market["market_env"],
        style_label=style_label,
        style_name=style_name,
        trend_state=target.get("trend_state"),
        adaptive_score=target.get("adaptive_score"),
    )
    return {
        "target": target,
        "market": market,
        "saved_strategy": saved_strategy,
        "recent_saved_strategy": recent_saved_strategy,
        "strategy_memory": strategy_memory,
        "price_memory": price_memory,
        "strategy_memory_review": strategy_memory_review,
        "saved_rank": saved_rank,
        "saved_strategy_hint": saved_strategy_hint,
        "recent_saved_strategy_hint": recent_saved_strategy_hint,
        "precision_candidate": precision_candidate,
        "evidence": evidence,
        "recommendation_tier_info": recommendation_tier_info,
        "recommendation_tier": recommendation_tier,
        "formal_candidate": formal_candidate,
        "industry_view": industry_view,
        "a_share_profile": a_share_profile,
        "intraday_view": intraday_view,
        "liquidity_view": liquidity_view,
        "style_name": style_name,
        "style_label": style_label,
        "trade_plan": trade_plan,
    }


def _build_execution_views(context, scored, cost_price, shares, available_shares, analysis_date, realtime, status):
    target = scored["target"]
    trade_plan = scored["trade_plan"]
    market = scored["market"]
    prepared = context["prepared"]
    trade_date = context["trade_date"]
    horizon_profiles = context["horizon_profiles"]
    style_horizon_profiles = context["style_horizon_profiles"]
    recommendation_tier = scored["recommendation_tier"]
    formal_candidate = scored["formal_candidate"]

    holding_stop_view = _build_holding_stop_view(
        target,
        trade_plan,
        prepared,
        trade_date,
        market["market_env"],
        horizon_profiles,
        style_horizon_profiles,
        cost_price=cost_price,
    )
    holding_trade_plan = dict(trade_plan)
    if holding_stop_view.get("effective_stop_price") is not None:
        holding_trade_plan["stop_price"] = holding_stop_view["effective_stop_price"]
    operation = _build_operation(
        target,
        trade_plan,
        recommendation_tier=recommendation_tier,
        formal_candidate=formal_candidate,
        formal_rank=scored["saved_rank"],
        holding_stop_view=holding_stop_view,
    )
    position_view = _build_position_view(
        target,
        trade_plan,
        cost_price=cost_price,
        holding_stop_view=holding_stop_view,
    )
    execution_scenarios = _build_execution_scenarios(
        target,
        trade_plan,
        formal_candidate,
        a_share_profile=scored["a_share_profile"],
        intraday_view=scored["intraday_view"],
    )
    buy_entry_strategy = _build_buy_entry_strategy(
        target,
        trade_plan,
        recommendation_tier,
        formal_candidate=formal_candidate,
        a_share_profile=scored["a_share_profile"],
        intraday_view=scored["intraday_view"],
        liquidity_view=scored["liquidity_view"],
        market=market,
    )
    holding_t_strategy = _build_holding_t_strategy(
        target,
        holding_trade_plan,
        recommendation_tier,
        formal_candidate,
        position_view,
        scored["intraday_view"],
        scored["liquidity_view"],
        market,
        shares=shares,
        available_shares=available_shares,
    )
    position_action_plan = _build_position_action_plan(
        target,
        trade_plan,
        operation,
        position_view,
        holding_stop_view,
        holding_t_strategy,
        recommendation_tier,
        formal_candidate,
        market,
        scored["intraday_view"],
        scored["liquidity_view"],
    )
    scorecard = _scorecard(
        formal_candidate,
        scored["evidence"],
        scored["industry_view"],
        market,
        target,
        intraday_view=scored["intraday_view"],
        liquidity_view=scored["liquidity_view"],
    )
    professional_view = _build_professional_view(
        formal_candidate,
        scored["evidence"],
        market,
        industry_view=scored["industry_view"],
        scorecard=scorecard,
        intraday_view=scored["intraday_view"],
        liquidity_view=scored["liquidity_view"],
        recommendation_tier=recommendation_tier,
        risk_overlay_view={
            "score": target.get("risk_overlay_score"),
            "level": target.get("risk_overlay_level"),
            "labels": target.get("risk_overlay_labels"),
            "block_formal": target.get("risk_overlay_block_formal"),
            "downgrade": target.get("risk_overlay_downgrade"),
        },
        recent_saved_strategy=scored["recent_saved_strategy"],
    )
    attack_defense_lines = _build_attack_defense_lines(
        target,
        trade_plan,
        holding_stop_view,
        recommendation_tier,
        formal_candidate,
        market,
        scorecard,
        professional_view,
        position_view,
    )
    adaptive_drop_profile = _build_adaptive_drop_profile(target, prepared, trade_date)
    realtime_execution = {}
    should_fetch_realtime = (analysis_date is None) if realtime is None else bool(realtime)
    if should_fetch_realtime:
        status("实时盘中覆盖开始")
        realtime_execution = _build_realtime_execution_view(
            context["normalized_code"],
            target,
            trade_plan,
            holding_stop_view,
            scored["strategy_memory_review"],
            scored["price_memory"],
            trade_date,
            has_user_position=bool(cost_price is not None or shares is not None or available_shares is not None),
            adaptive_drop_profile=adaptive_drop_profile,
        )
        if realtime_execution.get("success"):
            quote = realtime_execution.get("quote") or {}
            status(
                f"实时盘中覆盖完成: quote_time={quote.get('quote_datetime') or '--'}, "
                f"decision={realtime_execution.get('decision') or '--'}"
            )
        else:
            status(f"实时盘中覆盖不可用: {realtime_execution.get('reason') or '--'}")

    return {
        "holding_stop_view": holding_stop_view,
        "holding_trade_plan": holding_trade_plan,
        "operation": operation,
        "position_view": position_view,
        "execution_scenarios": execution_scenarios,
        "buy_entry_strategy": buy_entry_strategy,
        "holding_t_strategy": holding_t_strategy,
        "position_action_plan": position_action_plan,
        "scorecard": scorecard,
        "professional_view": professional_view,
        "attack_defense_lines": attack_defense_lines,
        "realtime_execution": realtime_execution,
    }


def _build_analysis_result(context, scored, views):
    target = scored["target"]
    trade_plan = scored["trade_plan"]
    holding_stop_view = views["holding_stop_view"]
    recommendation_tier = scored["recommendation_tier"]
    formal_candidate = scored["formal_candidate"]

    return {
        "success": True,
        "stock_code": context["normalized_code"],
        "stock_name": target.get("stock_name"),
        "industry": target.get("industry_1"),
        "latest_trade_date": context["trade_date"],
        "sample_start": _date_text(context["prepared"]["last_data_date"].min()),
        "sample_end": _date_text(context["prepared"]["last_data_date"].max()),
        "sample_trade_days": int(context["prepared"]["last_data_date"].nunique()),
        "market": scored["market"],
        "key_data": _summarize_key_data(target),
        "model": {
            "adaptive_score": _round(target.get("adaptive_score"), 2),
            "risk_adjusted_score": _round(target.get("risk_adjusted_score"), 2),
            "risk_score": _round(target.get("risk_score"), 2),
            "style": scored["style_name"],
            "style_label": scored["style_label"],
            "trend_state": target.get("trend_state"),
            "trend_detail": target.get("trend_detail"),
            "dominant_horizon": target.get("dominant_horizon"),
            "dominant_family": target.get("dominant_family"),
            "dominant_family_score": _round(target.get("dominant_family_score"), 2),
            "family_scores": target.get("family_scores") or {},
            "coverage": _round(target.get("coverage"), 4),
            "formal_candidate": formal_candidate,
            "precision_gate_candidate": scored["precision_candidate"],
            "recommendation_tier": recommendation_tier,
            "recommendation_tier_reason": scored["recommendation_tier_info"]["reason"],
            "formal_candidate_rank": scored["saved_rank"],
            "saved_strategy_hint": scored["saved_strategy_hint"],
            "recent_saved_strategy_hint": scored["recent_saved_strategy_hint"],
            "quality_fail_reasons": _build_fail_reasons(target),
            "reason": target.get("reason"),
            "risk_labels": target.get("risk_labels"),
            "no_buy_condition": target.get("no_buy_condition"),
        },
        "operation": views["operation"],
        "professional_view": views["professional_view"],
        "scorecard": views["scorecard"],
        "evidence": scored["evidence"],
        "industry_view": scored["industry_view"],
        "a_share_profile": scored["a_share_profile"],
        "intraday_view": scored["intraday_view"],
        "liquidity_view": scored["liquidity_view"],
        "position_view": views["position_view"],
        "holding_stop": holding_stop_view,
        "position_action_plan": views["position_action_plan"],
        "attack_defense_lines": views["attack_defense_lines"],
        "holding_t_strategy": views["holding_t_strategy"],
        "execution_scenarios": views["execution_scenarios"],
        "buy_entry_strategy": views["buy_entry_strategy"],
        "realtime_execution": views["realtime_execution"],
        "risk_overlay": {
            "score": _round(target.get("risk_overlay_score"), 2),
            "level": target.get("risk_overlay_level"),
            "labels": target.get("risk_overlay_labels"),
            "block_formal": bool(target.get("risk_overlay_block_formal")),
            "downgrade": bool(target.get("risk_overlay_downgrade")),
            "action": target.get("risk_overlay_action"),
            "note": target.get("risk_overlay_note"),
            "special_pool": target.get("special_pool"),
            "special_pool_label": target.get("special_pool_label"),
            "listing_trade_days": target.get("listing_trade_days"),
            "limit_up_days_5d": target.get("limit_up_days_5d"),
            "consecutive_limit_up_days": target.get("consecutive_limit_up_days"),
            "fundamental_event_context": target.get("fundamental_event_context"),
            "fundamental_note": target.get("fundamental_note"),
            "event_note": target.get("event_note"),
            "lhb_note": target.get("lhb_note"),
            "unlock_note": target.get("unlock_note"),
        },
        "trade_plan": {
            "action": trade_plan.get("action"),
            "entry": (
                "次日回踩承接确认再跟"
                if formal_candidate
                else "观察候选需等正式推荐或承接确认" if recommendation_tier == adaptive.RECOMMENDATION_TIER_OBSERVE
                else "未进入正式候选前不主动买入"
            ),
            "stop_price": trade_plan.get("stop_price"),
            "holding_effective_stop_price": holding_stop_view.get("effective_stop_price"),
            "expected_return": trade_plan.get("expected_return"),
            "hold_period": trade_plan.get("hold_period"),
            "position_hint": trade_plan.get("position_hint"),
            "full_note": trade_plan.get("conclusion"),
            "entry_strategy_note": views["buy_entry_strategy"].get("full_text"),
        },
        "saved_strategy": scored["saved_strategy"],
        "saved_strategy_hint": scored["saved_strategy_hint"],
        "recent_saved_strategy": scored["recent_saved_strategy"],
        "recent_saved_strategy_hint": scored["recent_saved_strategy_hint"],
        "strategy_memory_review": scored["strategy_memory_review"],
        "analyst_note": (
            "这是基于本地历史行情、全市场横截面评分和自适应模型的研究辅助结论；"
            "最终交易仍需结合你的仓位、成本和风险承受能力。"
        ),
    }


def analysis_per_stock_code(
    stock_code,
    analysis_date=None,
    lookback_trade_days=90,
    cost_price=None,
    shares=None,
    available_shares=None,
    emit_status=False,
    realtime=None,
):
    status = _build_status_emitter(emit_status)
    normalized_code = adaptive._normalize_stock_code(stock_code)
    if not normalized_code:
        status(f"股票代码无效: {stock_code}")
        return {
            "success": False,
            "reason": "invalid_stock_code",
            "message": f"股票代码无效: {stock_code}",
        }

    status(
        f"开始: stock_code={normalized_code}, "
        f"analysis_date={analysis_date or 'latest'}, lookback={lookback_trade_days}, "
        f"shares={shares if shares is not None else '--'}, "
        f"available_shares={available_shares if available_shares is not None else '--'}"
    )

    context, failure = _load_analysis_context(
        normalized_code,
        stock_code,
        analysis_date,
        lookback_trade_days,
        status,
    )
    if failure:
        return failure

    scored = _score_stock_context(context, status)
    views = _build_execution_views(
        context,
        scored,
        cost_price,
        shares,
        available_shares,
        analysis_date,
        realtime,
        status,
    )
    result = _build_analysis_result(context, scored, views)
    status(
        f"完成: tier={scored['recommendation_tier']}, formal={scored['formal_candidate']}, "
        f"decision={views['operation'].get('decision')}, "
        f"t_direction={views['holding_t_strategy'].get('direction')}"
    )
    return result


def _as_text(value, suffix=""):
    if value is None:
        return "--"
    return f"{value}{suffix}"


def _quote_current_label(quote):
    quote_datetime = (quote or {}).get("quote_datetime")
    if not quote_datetime:
        return "当前"
    try:
        time_text = str(quote_datetime).split()[-1]
        quote_time = dt.datetime.strptime(time_text[:8], "%H:%M:%S").time()
        if quote_time >= dt.time(15, 0):
            return "收盘"
    except (ValueError, TypeError):
        pass
    return "当前"


def _format_named_level(name, price):
    return f"{name} {_as_text(price)}" if name else _as_text(price)


def _append_unique_level(levels, name, price):
    if price is None:
        return
    rounded_price = _round(price, 2)
    key = (name, rounded_price)
    if any(item.get("key") == key for item in levels):
        return
    levels.append({"name": name, "price": rounded_price, "key": key})


def _build_intraday_drop_view(realtime, levels, adaptive_drop_profile=None):
    if not (realtime or {}).get("success"):
        return None

    quote = realtime.get("quote") or {}
    low_from_prev = _number(quote.get("low_from_prev_pct"))
    current_change = _number(quote.get("change_pct"))
    profile = adaptive_drop_profile or {}
    if low_from_prev is None or not profile.get("enabled"):
        return None

    trigger_rank = _number(profile.get("trigger_percentile")) or ADAPTIVE_DROP_TRIGGER_PERCENTILE
    sharp_rank = _number(profile.get("sharp_percentile")) or ADAPTIVE_DROP_SHARP_PERCENTILE
    extreme_rank = _number(profile.get("extreme_percentile")) or ADAPTIVE_DROP_EXTREME_PERCENTILE
    trigger_low_pct = _number(profile.get("trigger_low_pct"))
    sharp_low_pct = _number(profile.get("sharp_low_pct"))
    extreme_low_pct = _number(profile.get("extreme_low_pct"))
    low_samples = profile.get("low_samples") or []
    drop_percentile = _percentile_rank_lower(pd.Series(low_samples), low_from_prev) if low_samples else None
    is_triggered = (
        drop_percentile <= trigger_rank
        if drop_percentile is not None
        else trigger_low_pct is not None and low_from_prev <= trigger_low_pct
    )
    if not is_triggered:
        return None

    if drop_percentile is not None:
        if drop_percentile <= extreme_rank:
            event_bucket = "extreme"
            severity = "历史极端回撤"
        elif drop_percentile <= sharp_rank:
            event_bucket = "sharp"
            severity = "历史偏深回撤"
        else:
            event_bucket = "trigger"
            severity = "历史明显回撤"
    elif extreme_low_pct is not None and low_from_prev <= extreme_low_pct:
        event_bucket = "extreme"
        severity = "历史极端回撤"
    elif sharp_low_pct is not None and low_from_prev <= sharp_low_pct:
        event_bucket = "sharp"
        severity = "历史偏深回撤"
    else:
        event_bucket = "trigger"
        severity = "历史明显回撤"

    current_price = _number(quote.get("current"))
    low_price = _number(quote.get("low"))
    has_trade_context = bool(levels.get("has_trade_context"))
    stop_price = _number(levels.get("effective_stop") or levels.get("model_reference_stop")) if has_trade_context else None
    prior_low = _number(levels.get("prior_low"))
    ma5 = _number(levels.get("ma5"))
    ma10 = _number(levels.get("ma10"))
    ma20 = _number(levels.get("ma20"))

    stop_touched = bool(stop_price is not None and low_price is not None and low_price <= stop_price)
    stop_broken_now = bool(stop_price is not None and current_price is not None and current_price <= stop_price)

    watch_levels = []
    if stop_touched:
        _append_unique_level(watch_levels, "有效防守", stop_price)
    else:
        if prior_low is not None and low_price is not None and low_price <= prior_low:
            _append_unique_level(watch_levels, "前一日低点", prior_low)
        if ma5 is not None and low_price is not None and low_price <= ma5:
            _append_unique_level(watch_levels, "MA5", ma5)
        if not watch_levels and ma10 is not None and low_price is not None and low_price <= ma10:
            _append_unique_level(watch_levels, "MA10", ma10)
        if not watch_levels and ma20 is not None and low_price is not None and low_price <= ma20:
            _append_unique_level(watch_levels, "MA20", ma20)
        if not watch_levels:
            _append_unique_level(watch_levels, "有效防守", stop_price)

    watch_levels = watch_levels[:2]
    watch_prices = [item["price"] for item in watch_levels if item.get("price") is not None]
    watch_text = " / ".join(_format_named_level(item.get("name"), item.get("price")) for item in watch_levels) or "关键线"
    recovered_watch = bool(current_price is not None and watch_prices and current_price >= max(watch_prices))

    recovery_points = None
    recovery_ratio = None
    if current_change is not None:
        recovery_points = current_change - low_from_prev
        if low_from_prev < 0:
            recovery_ratio = recovery_points / abs(low_from_prev)

    if stop_broken_now:
        state = "跌破硬防守"
    elif stop_touched:
        state = "触及硬防守后收回" if current_price is not None and stop_price is not None and current_price > stop_price else "触及硬防守"
    elif recovered_watch:
        state = f"{severity}后站回{watch_text}"
    else:
        state = f"{severity}后未站回{watch_text}"

    event_stats = (profile.get("event_stats") or {}).get(event_bucket) or {}
    state_key = "recovered" if recovered_watch else "unrecovered"
    state_label = "站回关键线" if recovered_watch else "未站回关键线"
    state_stats = event_stats.get(state_key) or {}
    recovered_stats = event_stats.get("recovered") or {}
    unrecovered_stats = event_stats.get("unrecovered") or {}
    evaluated_count = int(_number(event_stats.get("evaluated_count")) or 0)
    state_count = int(_number(state_stats.get("count")) or 0)
    has_history_support = bool(event_stats.get("has_support") and state_count >= ADAPTIVE_DROP_MIN_GROUP_ROWS)
    avg_return = _number(state_stats.get("avg_return"))
    win_rate = _number(state_stats.get("win_rate"))
    recovered_avg = _number(recovered_stats.get("avg_return"))
    unrecovered_avg = _number(unrecovered_stats.get("avg_return"))
    recovery_has_edge = bool(
        recovered_avg is not None
        and unrecovered_avg is not None
        and recovered_avg > unrecovered_avg
    )
    risk_backed = bool(
        has_history_support
        and (
            (avg_return is not None and avg_return <= 0)
            or (win_rate is not None and win_rate < 50)
        )
    )
    repair_backed = bool(
        has_history_support
        and recovered_watch
        and (avg_return is not None and avg_return > 0)
        and (win_rate is None or win_rate >= 50)
    )

    if has_history_support:
        history_support_text = (
            f"回测支撑: 同类样本{evaluated_count}次，{state_label}样本{state_count}次，"
            f"5日均值{_as_text(_round(avg_return, 2), '%')}，胜率{_as_text(_round(win_rate, 2), '%')}。"
        )
    else:
        history_support_text = (
            f"回测支撑不足: 同类样本{evaluated_count}次，{state_label}样本{state_count}次，"
            "动作降级为观察和价格线纪律。"
        )

    break_word = "跌破这一区域后" if len(watch_levels) > 1 else "跌破后"
    stand_word = "重新站稳这一区域" if len(watch_levels) > 1 else f"重新站稳{watch_text}"
    if not has_trade_context:
        if not has_history_support:
            if recovered_watch:
                action = f"已站回{watch_text}，但同类样本不足；没有系统推荐前不开新仓。"
            else:
                action = f"{watch_text}只作观察线；没有系统推荐前不开新仓。"
        elif not recovered_watch:
            if risk_backed:
                action = f"同类未修复样本偏弱，无仓不买；有仓按自己的成本线降风险。"
            else:
                action = f"收盘仍没站回{watch_text}，无仓不买；有仓只按自己的成本线处理。"
        else:
            action = f"已站回{watch_text}，但没有系统推荐前不当作买点。"
    elif stop_broken_now:
        action = f"已经跌破{_format_named_level('有效防守', stop_price)}，先降风险；反弹不能收回不等。"
    elif not has_history_support:
        if recovered_watch:
            action = f"同类样本不足，动作不升级；后续只看{watch_text}和原防守，跌回关键线不加仓。"
        else:
            action = f"同类样本不足，动作不升级；只看{watch_text}和原防守，未站回不加仓。"
    elif not recovered_watch:
        if risk_backed:
            action = f"同类未修复样本偏弱，反弹不能站回{watch_text}，先降风险。"
        else:
            action = f"先看能否{stand_word}；回测没有给出强风险级别前，不加仓。"
    elif repair_backed and recovery_has_edge:
        action = f"下一交易日重点看{watch_text}；{break_word}收不回先减仓。"
    else:
        action = f"已站回{watch_text}，但同类修复优势不强；不加仓，跌回关键线先减仓。"

    return {
        "severity": severity,
        "state": state,
        "event_bucket": event_bucket,
        "drop_percentile": _round(drop_percentile, 2),
        "drop_rank_text": (
            f"低点回撤处在该股历史{_round(drop_percentile, 2)}分位"
            if drop_percentile is not None
            else None
        ),
        "low_change_pct": _round(low_from_prev, 2),
        "current_change_pct": _round(current_change, 2),
        "current_label": _quote_current_label(quote),
        "recovery_points": _round(recovery_points, 2),
        "recovery_ratio_pct": _round(recovery_ratio * 100, 2) if recovery_ratio is not None else None,
        "stop_price": _round(stop_price, 2),
        "stop_touched": stop_touched,
        "stop_broken_now": stop_broken_now,
        "has_trade_context": has_trade_context,
        "watch_levels": [{k: v for k, v in item.items() if k != "key"} for item in watch_levels],
        "watch_text": watch_text,
        "recovered_watch": recovered_watch,
        "history_support": has_history_support,
        "history_support_text": history_support_text,
        "history_event_stats": {
            "bucket": event_bucket,
            "evaluated_count": evaluated_count,
            "state": state_key,
            "state_count": state_count,
            "state_avg_return_5d": _round(avg_return, 2),
            "state_win_rate_5d": _round(win_rate, 2),
            "recovered_avg_return_5d": _round(recovered_avg, 2),
            "unrecovered_avg_return_5d": _round(unrecovered_avg, 2),
        },
        "action": action,
        "minute_data_note": "没有分钟线，不判断收回耗时，只按价格线判断修复是否有效。",
    }


def _format_realtime_unavailable(realtime):
    if not realtime:
        return None

    source = realtime.get("source") or "unknown"
    reason = realtime.get("reason") or "realtime_unavailable"
    consistency = realtime.get("data_consistency") or {}
    if consistency:
        return (
            f"{source}/{reason}，实时昨收{_as_text(consistency.get('realtime_prev_close'))} "
            f"vs 本地收盘{_as_text(consistency.get('local_close'))}，"
            f"偏差{_as_text(consistency.get('prev_close_delta_pct'), '%')}"
        )
    return f"{source}/{reason}"


def _first_non_empty(*values):
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _brief_join(items, limit=3):
    values = [str(item) for item in items or [] if item]
    if not values:
        return "--"
    if len(values) <= limit:
        return "；".join(values)
    return "；".join(values[:limit]) + f"；另{len(values) - limit}项"


def _compact_level_text(items, limit=2):
    values = []
    for item in items or []:
        if isinstance(item, dict):
            name = item.get("name")
            price = item.get("price")
            text = f"{name} {_as_text(price)}" if name else _as_text(price)
        else:
            text = str(item)
        if text:
            values.append(text)
    return "、".join(values[:limit]) if values else "--"


def _first_level_text(items):
    return _compact_level_text(items, limit=1)


def _execution_headline(result, realtime, operation, model):
    if not realtime.get("success"):
        return operation.get("decision") or "--"

    mode = realtime.get("mode")
    decision = realtime.get("decision") or operation.get("decision") or "--"
    tier = model.get("recommendation_tier")
    if mode == "holding_management":
        return "不是新买点；有仓按防守线管，无仓不接。"
    if mode == "entry_acceptance_check":
        if "通过" in decision:
            return "承接暂通过；只按原计划分批，不追高。"
        return "承接没通过；无仓先不买，有仓守防守。"
    if tier == adaptive.RECOMMENDATION_TIER_FORMAL:
        return "可执行候选；只等回踩承接，不追高。"
    if tier == adaptive.RECOMMENDATION_TIER_OBSERVE:
        return "观察候选；不是正式买点，只能小仓验证。"
    return "不是买点；只观察，不主动开仓。"


def _build_intraday_drop_rule(realtime, levels):
    drop_view = (realtime or {}).get("intraday_drop_view") or _build_intraday_drop_view(realtime, levels)
    if not drop_view:
        return None

    low_text = _as_text(drop_view.get("low_change_pct"), "%")
    current_text = _as_text(drop_view.get("current_change_pct"), "%")
    stop_text = _as_text(drop_view.get("stop_price"))

    base = (
        f"盘中最低跌到{low_text}，{drop_view.get('current_label') or '当前'}"
        f"收回到{current_text}，属于{drop_view.get('state')}。"
    )
    if drop_view.get("drop_rank_text"):
        base += f"{drop_view.get('drop_rank_text')}。"
    if drop_view.get("stop_price") is not None:
        if drop_view.get("stop_broken_now"):
            base += f"当前已跌破防守{stop_text}。"
        elif drop_view.get("stop_touched"):
            base += f"盘中碰过防守{stop_text}。"
        else:
            base += f"没有打到硬防守{stop_text}。"

    final_risk = ""
    if drop_view.get("stop_price") is not None and not drop_view.get("stop_broken_now"):
        final_risk = f"若跌破防守{stop_text}或收盘仍在关键线下，退出/大幅降仓。"

    return (
        f"{base}{drop_view.get('action') or ''}"
        f"{drop_view.get('history_support_text') or ''}"
        f"{drop_view.get('minute_data_note') or ''}{final_risk}"
    )


def _print_execution_report(result):
    if not result.get("success"):
        print(result.get("message") or result.get("reason") or "分析失败")
        return

    model = result.get("model") or {}
    operation = result.get("operation") or {}
    professional_view = result.get("professional_view") or {}
    scorecard = result.get("scorecard") or {}
    market = result.get("market") or {}
    trade_plan = result.get("trade_plan") or {}
    realtime = result.get("realtime_execution") or {}
    quote = realtime.get("quote") or {}
    structure = realtime.get("structure_view") or {}
    day_tape = realtime.get("day_tape") or {}
    levels = realtime.get("key_levels") or {}
    execution_context = realtime.get("execution_context") or {}
    has_trade_context = bool(execution_context.get("has_trade_context") or levels.get("has_trade_context"))
    holding_metrics = realtime.get("holding_metrics") or {}
    memory_summary = ((result.get("strategy_memory_review") or {}).get("summary") or {})
    evidence_grade = professional_view.get("evidence_grade")

    title_bits = [result.get("stock_code"), result.get("stock_name")]
    print(f"个股执行版: {' '.join(str(bit) for bit in title_bits if bit)}")
    if realtime.get("success"):
        print(
            f"实时: {quote.get('quote_datetime') or '--'} | "
            f"{_as_text(quote.get('current'))} ({_as_text(quote.get('change_pct'), '%')}) | "
            f"市场: {realtime.get('market_summary') or '--'}"
        )
    else:
        print(
            f"交易日: {result.get('latest_trade_date') or '--'} | "
            f"市场: {market.get('market_env') or '--'}({_as_text(market.get('market_change_5d'), '%')}, 广度{_as_text(market.get('market_breadth_5d'), '%')})"
        )
        realtime_unavailable = _format_realtime_unavailable(realtime)
        if realtime_unavailable:
            print(f"实时: 不可用 | {realtime_unavailable}")

    print()
    no_position_action = _first_non_empty(
        realtime.get("no_position_action"),
        operation.get("new_position"),
    )
    holding_action = _first_non_empty(
        realtime.get("holding_action"),
        operation.get("holding_position"),
    )
    stop_price = None
    support_text = "--"
    pressure_text = "--"
    strongest_pressure = "--"
    if realtime.get("success"):
        stop_price = levels.get("effective_stop") if has_trade_context else None
        support_text = _compact_level_text(structure.get("supports") or [], limit=2)
        pressure_text = _compact_level_text(structure.get("pressures") or [], limit=2)
        strongest_pressure = _first_level_text(structure.get("pressures") or [])
    else:
        buy_entry = result.get("buy_entry_strategy") or {}
        buy_zone = buy_entry.get("suggested_buy_zone") or {}
        stop_price = trade_plan.get("stop_price")
        support_text = ((buy_zone.get("watch_zone") or {}).get("text")) or "--"
        strongest_pressure = _as_text(buy_zone.get("pressure_price"))
        pressure_text = strongest_pressure

    print("结论:")
    print(_execution_headline(result, realtime, operation, model))

    print()
    print("现在怎么做:")
    print(f"- 无仓: {no_position_action or '--'}")
    print(f"- 持仓: {holding_action or '--'}")
    drop_rule = _build_intraday_drop_rule(realtime, levels)
    if drop_rule:
        print(f"- 急跌规则: {drop_rule}")
    if has_trade_context and stop_price is not None:
        print(f"- 错了就走: 跌破或不能收回 {_as_text(stop_price)}，先降风险。")
    if strongest_pressure != "--":
        print(f"- 重新评估: 先收复 {strongest_pressure}，并重新满足系统推荐/承接。")

    print()
    print("关键线:")
    if realtime.get("success"):
        if has_trade_context:
            print(
                f"- 现价 {_as_text(quote.get('current'))}；防守 {_as_text(stop_price)}；"
                f"支撑 {support_text}；压力 {pressure_text}"
            )
        else:
            print(
                f"- 现价 {_as_text(quote.get('current'))}；观察 {support_text}；压力 {pressure_text}"
            )
        if holding_metrics.get("entry_price") is not None:
            print(
                f"- 入场 {_as_text(holding_metrics.get('entry_price'))}；当前收益 {_as_text(holding_metrics.get('current_return_pct'), '%')}；"
                f"距防守 {_as_text(holding_metrics.get('distance_to_stop_pct'), '%')}"
            )
    else:
        print(f"- 防守 {_as_text(stop_price)}；观察区 {support_text}；压力 {pressure_text}")

    print()
    print("为什么:")
    latest_formal = memory_summary.get("latest_formal_date")
    system_bits = [f"系统{model.get('recommendation_tier') or '--'}"]
    if scorecard.get("rating"):
        system_bits.append(f"评级{scorecard.get('rating')}")
    if evidence_grade:
        system_bits.append(f"证据{evidence_grade}")
    if latest_formal:
        system_bits.append(f"最近正式推荐{latest_formal}")
    print("- " + "，".join(system_bits) + "。")
    score_text = _scorecard_component_text(scorecard)
    if score_text:
        print(
            f"- 评分构成: {score_text}；"
            f"合计{_as_text(scorecard.get('score'), '分')} -> {scorecard.get('rating') or '--'}。"
        )
    if realtime.get("success"):
        tape_grade = day_tape.get("grade") or "--"
        if isinstance(tape_grade, str) and tape_grade.startswith("盘中"):
            tape_grade = tape_grade[2:]
        open_gap_text = _as_text(day_tape.get("open_gap_pct"), "%")
        drop_view = realtime.get("intraday_drop_view") or {}
        drop_not_repaired = bool(drop_view and not drop_view.get("recovered_watch"))
        if not has_trade_context and ("偏弱" in str(tape_grade) or drop_not_repaired):
            print(
                f"- 盘中{tape_grade}，开盘缺口{open_gap_text}；没有系统推荐，且没有形成有效修复。"
            )
        elif not has_trade_context:
            print(
                f"- 盘中{tape_grade}，开盘缺口{open_gap_text}；只能作为观察，不能升级成新买点。"
            )
        else:
            print(
                f"- 盘中{tape_grade}，但开盘缺口{open_gap_text}，"
                f"只能说明承接修复，不能升级成新买点。"
            )
    if realtime.get("warnings"):
        print(f"- 提醒: {_brief_join(realtime.get('warnings'), limit=2)}")
    fail_reasons = model.get("quality_fail_reasons") or []
    if model.get("risk_labels"):
        print(f"- 风险: {model.get('risk_labels')}")
    elif fail_reasons:
        print(f"- 未通过: {_brief_join(fail_reasons, limit=2)}")

    print()
    print("更多细节: 加 --detail；机器读取: 加 --json")


def _print_report(result, detail=False):
    text = _render_report_text(result, detail=detail)
    if text:
        print(text)


def _render_report_text(result, detail=False):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        if detail:
            _print_detail_report(result)
        else:
            _print_execution_report(result)
    return output.getvalue().rstrip()


def _print_detail_report(result):
    if not result.get("success"):
        print(result.get("message") or result.get("reason") or "分析失败")
        return

    model = result["model"]
    operation = result["operation"]
    trade_plan = result["trade_plan"]
    key_data = result["key_data"]
    market = result["market"]
    professional_view = result.get("professional_view") or {}
    evidence = result.get("evidence") or {}
    scorecard = result.get("scorecard") or {}
    industry_view = result.get("industry_view") or {}
    a_share_profile = result.get("a_share_profile") or {}
    intraday_view = result.get("intraday_view") or {}
    liquidity_view = result.get("liquidity_view") or {}
    position_view = result.get("position_view") or {}
    holding_stop = result.get("holding_stop") or {}
    position_action = result.get("position_action_plan") or {}
    attack_defense = result.get("attack_defense_lines") or {}
    holding_t = result.get("holding_t_strategy") or {}
    scenarios = result.get("execution_scenarios") or {}
    buy_entry = result.get("buy_entry_strategy") or {}
    realtime_execution = result.get("realtime_execution") or {}
    overlay = result.get("risk_overlay") or {}

    print(f"个股分析: {result['stock_code']} {result.get('stock_name') or ''}")
    print(
        f"交易日: {result['latest_trade_date']} | 行业: {result.get('industry') or '--'} | "
        f"板块: {a_share_profile.get('board') or '--'} | "
        f"市场: {market['market_env']}({market['market_change_5d']}%, 广度{market['market_breadth_5d']}%)"
    )
    print(
        f"样本窗口: {result.get('sample_start') or '--'}~{result.get('sample_end') or '--'} | "
        f"{result.get('sample_trade_days') if result.get('sample_trade_days') is not None else '--'}个交易日"
    )
    print()
    print(f"结论: {operation['decision']}")
    print(
        f"推荐层级: {model.get('recommendation_tier') or '--'} | "
        f"{model.get('recommendation_tier_reason') or '--'}"
    )
    print(
        f"专业判断: {professional_view.get('stance') or '--'} | "
        f"交易质量评级: {scorecard.get('rating') or '--'}({scorecard.get('score') if scorecard.get('score') is not None else '--'}分) | "
        f"证据等级: {professional_view.get('evidence_grade') or '--'} | "
        f"行业: {professional_view.get('industry_grade') or '--'}"
    )
    print(f"仓位原则: {professional_view.get('position_sizing') or '--'}")
    print(f"无仓: {operation['new_position']}")
    print(f"持仓: {operation['holding_position']}")
    if holding_stop:
        print(
            f"持仓有效防守: {holding_stop.get('effective_stop_price') if holding_stop.get('effective_stop_price') is not None else '--'} "
            f"(当日策略防守{holding_stop.get('dynamic_stop_price') if holding_stop.get('dynamic_stop_price') is not None else '--'}，"
            f"来源:{holding_stop.get('source') or '--'}"
            f"{'/' + str(holding_stop.get('source_date')) if holding_stop.get('source_date') else ''})"
        )
        if holding_stop.get("warnings"):
            print(f"持仓防守说明: {'；'.join(holding_stop.get('warnings') or [])}")
    if position_view.get("has_cost"):
        print(
            f"成本视角: 成本{position_view.get('cost_price')}，现价{position_view.get('latest_price')}，"
            f"浮盈亏{position_view.get('pnl_pct')}%，距防守{position_view.get('distance_to_stop_pct')}%；"
            f"{position_view.get('advice')}"
        )
    if realtime_execution.get("success"):
        quote = realtime_execution.get("quote") or {}
        day_tape = realtime_execution.get("day_tape") or {}
        acceptance = realtime_execution.get("realtime_acceptance") or {}
        holding_metrics = realtime_execution.get("holding_metrics") or {}
        levels = realtime_execution.get("key_levels") or {}
        consistency = realtime_execution.get("data_consistency") or {}
        structure = realtime_execution.get("structure_view") or {}
        print()
        print("实时盘中:")
        print(
            f"- 数据源: {realtime_execution.get('source') or '--'} | "
            f"行情时间: {quote.get('quote_datetime') or '--'} | "
            f"本地基准日: {realtime_execution.get('local_base_date') or '--'}"
        )
        print(
            f"- 口径校验: 实时昨收{consistency.get('realtime_prev_close') if consistency.get('realtime_prev_close') is not None else '--'} "
            f"vs 本地收盘{consistency.get('local_close') if consistency.get('local_close') is not None else '--'}，"
            f"偏差{consistency.get('prev_close_delta_pct') if consistency.get('prev_close_delta_pct') is not None else '--'}%"
        )
        print(
            f"- 盘口: 开{quote.get('open') if quote.get('open') is not None else '--'}，"
            f"现{quote.get('current') if quote.get('current') is not None else '--'}，"
            f"高{quote.get('high') if quote.get('high') is not None else '--'}，"
            f"低{quote.get('low') if quote.get('low') is not None else '--'}，"
            f"涨跌{quote.get('change_pct') if quote.get('change_pct') is not None else '--'}%；"
            f"市场: {realtime_execution.get('market_summary') or '--'}"
        )
        print(
            f"- 盘中承接: {day_tape.get('grade') or '--'} | "
            f"开盘缺口{day_tape.get('open_gap_pct') if day_tape.get('open_gap_pct') is not None else '--'}%，"
            f"现价/开盘{day_tape.get('current_from_open_pct') if day_tape.get('current_from_open_pct') is not None else '--'}%，"
            f"当前位置{day_tape.get('current_position_pct') if day_tape.get('current_position_pct') is not None else '--'}%，"
            f"高开回踩{'是' if day_tape.get('high_open_has_pullback') is True else '否' if day_tape.get('high_open_has_pullback') is False else '--'}"
        )
        if structure:
            print(f"- 结构判断: {structure.get('summary') or '--'}")
            print(
                f"- 关键价位: 支撑/观察 {_join_level_text(structure.get('supports') or [])} | "
                f"压力/修复 {_join_level_text(structure.get('pressures') or [])}"
            )
            if structure.get("observations"):
                print(f"- 盘面依据: {'；'.join(structure.get('observations') or [])}")
        if acceptance:
            print(
                f"- 入场承接纪律: {'暂通过' if acceptance.get('entry_acceptance_ok') is True else '暂未通过' if acceptance.get('entry_acceptance_ok') is False else '--'} | "
                f"信号收盘{acceptance.get('signal_close') if acceptance.get('signal_close') is not None else '--'}，"
                f"入场均价估算{acceptance.get('entry_avg_price') if acceptance.get('entry_avg_price') is not None else '--'}，"
                f"缺口{acceptance.get('entry_gap_pct') if acceptance.get('entry_gap_pct') is not None else '--'}%，"
                f"位置{acceptance.get('entry_close_position') if acceptance.get('entry_close_position') is not None else '--'}%"
            )
        print(
            f"- 实时结论: {realtime_execution.get('decision') or '--'} | "
            f"无仓: {realtime_execution.get('no_position_action') or '--'} | "
            f"持仓: {realtime_execution.get('holding_action') or '--'}"
        )
        if holding_metrics.get("entry_price") is not None:
            print(
                f"- 持仓复盘: 入场{holding_metrics.get('entry_price')}，"
                f"当前收益{holding_metrics.get('current_return_pct') if holding_metrics.get('current_return_pct') is not None else '--'}%，"
                f"最高收益{holding_metrics.get('high_return_pct') if holding_metrics.get('high_return_pct') is not None else '--'}%，"
                f"最低收益{holding_metrics.get('low_return_pct') if holding_metrics.get('low_return_pct') is not None else '--'}%，"
                f"有效防守{levels.get('effective_stop') if levels.get('effective_stop') is not None else '--'}，"
                f"距防守{holding_metrics.get('distance_to_stop_pct') if holding_metrics.get('distance_to_stop_pct') is not None else '--'}%"
            )
        else:
            model_stop_text = (
                f" | 模型参考防守{levels.get('model_reference_stop')}"
                if realtime_execution.get("mode") != "no_formal_signal"
                and levels.get("model_reference_stop") is not None
                else ""
            )
            print(
                f"- 操作焦点: {structure.get('action_focus') or '--'}{model_stop_text}"
            )
        if realtime_execution.get("warnings"):
            print(f"- 实时提醒: {'；'.join(realtime_execution.get('warnings') or [])}")
        print(f"- 纪律说明: {realtime_execution.get('discipline_note') or '--'}")
    elif realtime_execution.get("enabled"):
        consistency = realtime_execution.get("data_consistency") or {}
        print()
        print("实时盘中:")
        print(
            f"- 不可用: {_format_realtime_unavailable(realtime_execution) or '--'}"
        )
        print(
            f"- 数据源: {realtime_execution.get('source') or '--'} | "
            f"URL: {realtime_execution.get('url') or '--'}"
        )
        if consistency:
            print(
                f"- 口径校验: 实时昨收{_as_text(consistency.get('realtime_prev_close'))} "
                f"vs 本地收盘{_as_text(consistency.get('local_close'))}，"
                f"偏差{_as_text(consistency.get('prev_close_delta_pct'), '%')}，"
                f"容忍{_as_text(consistency.get('tolerance_pct'), '%')}"
            )
    if attack_defense:
        technical_lines = attack_defense.get("technical") or {}
        professional_lines = attack_defense.get("professional") or {}
        print()
        print("攻防线:")
        print(
            f"- 技术侧: 防守 {technical_lines.get('defense_text') or '--'} | "
            f"进攻 {technical_lines.get('attack_zone') or '--'} | "
            f"{technical_lines.get('state') or '--'}"
        )
        if technical_lines.get("basis"):
            print(f"  依据: {'；'.join(technical_lines.get('basis') or [])}")
        print(f"  动作: {technical_lines.get('action') or '--'}")
        print(
            f"- 专业侧: 防守 {professional_lines.get('defense_text') or '--'} | "
            f"进攻 {professional_lines.get('attack_zone') or '--'} | "
            f"{professional_lines.get('state') or '--'}"
        )
        if professional_lines.get("basis"):
            print(f"  依据: {'；'.join(professional_lines.get('basis') or [])}")
        print(f"  动作: {professional_lines.get('action') or '--'}")
    if position_action:
        print()
        print("持仓操作区间:")
        print(f"- 结论: {position_action.get('stance') or '--'}")
        weakness = position_action.get("reduce_on_weakness") or {}
        strength = position_action.get("reduce_on_strength") or {}
        add_position = position_action.get("add_position") or {}
        t_plan = position_action.get("t_plan") or {}
        print(f"- 跌到/跌破: {weakness.get('trigger_price') if weakness.get('trigger_price') is not None else '--'} | {weakness.get('action') or '--'}")
        print(f"- 涨到: {strength.get('zone') or '--'} | {strength.get('action') or '--'}")
        print(f"- 加仓: {add_position.get('status') or '--'} | {add_position.get('trigger') or '--'}")
        print(
            f"- 做T: {t_plan.get('direction') or '--'}({t_plan.get('feasibility') or '--'}) | "
            f"卖出区{t_plan.get('sell_zone') or '--'}，买回区{t_plan.get('buyback_zone') or '--'}，"
            f"T仓{t_plan.get('planned_t_ratio_pct') if t_plan.get('planned_t_ratio_pct') is not None else '--'}%；"
            f"{t_plan.get('action') or '--'}"
        )
    print(f"风控: {operation['risk_control']}")
    if professional_view.get("warnings"):
        print(f"提醒: {'；'.join(professional_view['warnings'])}")
    if holding_t:
        print()
        print("持仓T策略:")
        print(
            f"- 可行性: {holding_t.get('feasibility') or '--'} | "
            f"方向: {holding_t.get('direction') or '--'} | "
            f"信心: {holding_t.get('confidence') or '--'}"
        )
        print(
            f"- T仓: {holding_t.get('planned_t_ratio_pct') if holding_t.get('planned_t_ratio_pct') is not None else '--'}% / "
            f"{holding_t.get('planned_t_shares') if holding_t.get('planned_t_shares') is not None else '--'}股；"
            f"可卖{holding_t.get('available_shares') if holding_t.get('available_shares') is not None else '--'}股"
        )
        print(
            f"- 卖出区: {(holding_t.get('sell_zone') or {}).get('text') or '--'} | "
            f"买回区: {(holding_t.get('buyback_zone') or {}).get('text') or '--'} | "
            f"防守: {holding_t.get('stop_price') if holding_t.get('stop_price') is not None else '--'}"
        )
        print(f"- 动作: {holding_t.get('action') or '--'}")
        print(f"- 收盘处理: {holding_t.get('close_rule') or '--'}")
        if holding_t.get("warnings"):
            print(f"- 提醒: {'；'.join(holding_t.get('warnings') or [])}")
    print()
    print("交易计划:")
    print(f"- A股规则: {scenarios.get('a_share_rule') or a_share_profile.get('note') or '--'}")
    print(f"- 入场: {trade_plan['entry']}")
    print(f"- 防守: {trade_plan['stop_price']}")
    if trade_plan.get("holding_effective_stop_price") != trade_plan.get("stop_price"):
        print(f"- 持仓有效防守: {trade_plan.get('holding_effective_stop_price')}")
    print(f"- 止盈: {trade_plan['expected_return']}")
    print(f"- 持有: {trade_plan['hold_period']}")
    print(f"- 仓位: {trade_plan['position_hint']}")
    if buy_entry:
        buy_zone = buy_entry.get("suggested_buy_zone") or {}
        if buy_zone:
            watch_zone = buy_zone.get("watch_zone") or {}
            support_zone = buy_zone.get("support_confirm_zone") or {}
            print("建议买入区间:")
            print(f"- 状态: {buy_zone.get('status') or '--'}")
            print(f"- 买入观察区: {watch_zone.get('text') or '--'}")
            print(f"- 强支撑确认区: {support_zone.get('text') or '--'}")
            print(
                f"- 失效/放弃: {buy_zone.get('invalid_price') if buy_zone.get('invalid_price') is not None else '--'} | "
                f"不追高: {buy_zone.get('no_chase_price') if buy_zone.get('no_chase_price') is not None else '--'} | "
                f"压力观察: {buy_zone.get('pressure_price') if buy_zone.get('pressure_price') is not None else '--'}"
            )
            print(f"- 执行: {buy_zone.get('action') or '--'}")
            if buy_zone.get("basis"):
                print(f"- 依据: {'；'.join(buy_zone.get('basis') or [])}")
        print("买入口径策略:")
        print(f"- 总原则: {buy_entry.get('overall') or '--'}")
        mode_labels = [("next_open", "开盘直接买"), ("next_avg", "盘中均价/分批买"), ("acceptance", "承接确认买")]
        modes = buy_entry.get("modes") or {}
        for mode_key, fallback_label in mode_labels:
            mode = modes.get(mode_key) or {}
            print(f"- {mode.get('name') or fallback_label}: {mode.get('status') or '--'}")
            print(f"  触发: {mode.get('trigger') or '--'}")
            print(f"  执行: {mode.get('action') or '--'}")
            print(f"  放弃: {mode.get('invalid') or '--'}")
        if buy_entry.get("abandon_rules"):
            print(f"- 放弃清单: {'；'.join(buy_entry.get('abandon_rules') or [])}")
    print("执行情景:")
    print(f"- 日内状态: {scenarios.get('intraday_condition') or '--'}")
    print(f"- 正常开盘: {scenarios.get('normal_open') or '--'}")
    print(f"- 高开: {scenarios.get('high_open') or '--'}")
    print(f"- 弱开: {scenarios.get('weak_open') or '--'}")
    print(f"- 破位: {scenarios.get('stop_break') or '--'}")
    print(f"- 止盈: {scenarios.get('take_profit') or '--'}")
    print()
    print("A股实战检查:")
    print(
        f"- 涨停价约{intraday_view.get('limit_up_price') or '--'}，跌停价约{intraday_view.get('limit_down_price') or '--'}，"
        f"距涨停{intraday_view.get('limit_up_distance_pct') if intraday_view.get('limit_up_distance_pct') is not None else '--'}%，"
        f"距跌停{intraday_view.get('limit_down_distance_pct') if intraday_view.get('limit_down_distance_pct') is not None else '--'}%"
    )
    print(
        f"- 开盘缺口{intraday_view.get('open_gap_pct') if intraday_view.get('open_gap_pct') is not None else '--'}%，"
        f"开收变化{intraday_view.get('close_from_open_pct') if intraday_view.get('close_from_open_pct') is not None else '--'}%，"
        f"收盘位置{intraday_view.get('close_position_pct') if intraday_view.get('close_position_pct') is not None else '--'}%，"
        f"上影占比{intraday_view.get('upper_shadow_pct') if intraday_view.get('upper_shadow_pct') is not None else '--'}%，"
        f"状态{intraday_view.get('tape_grade') or '--'}"
    )
    print(
        f"- 流动性: {liquidity_view.get('grade') or '--'}，"
        f"成交额{liquidity_view.get('amount') if liquidity_view.get('amount') is not None else '--'}，"
        f"成交额/5日均{liquidity_view.get('amount_vs_avg_5d') if liquidity_view.get('amount_vs_avg_5d') is not None else '--'}，"
        f"换手{liquidity_view.get('turnover_rate') if liquidity_view.get('turnover_rate') is not None else '--'}%，"
        f"换手/5日均{liquidity_view.get('turnover_vs_avg_5d') if liquidity_view.get('turnover_vs_avg_5d') is not None else '--'}"
    )
    print()
    print("基本面/事件/特殊股票池覆盖:")
    print(
        f"- 覆盖结论: {overlay.get('level') or '--'}风险，{overlay.get('labels') or '--'}；"
        f"{overlay.get('action') or '--'}"
    )
    print(
        f"- 特殊股票池: {overlay.get('special_pool_label') or '--'}，"
        f"样本内上市交易日{overlay.get('listing_trade_days') if overlay.get('listing_trade_days') is not None else '--'}，"
        f"5日涨停次数{overlay.get('limit_up_days_5d') if overlay.get('limit_up_days_5d') is not None else '--'}，"
        f"连续涨停{overlay.get('consecutive_limit_up_days') if overlay.get('consecutive_limit_up_days') is not None else '--'}"
    )
    print(f"- 财务: {overlay.get('fundamental_note') or '--'}")
    print(f"- 公告: {overlay.get('event_note') or '--'}")
    print(f"- 龙虎榜: {overlay.get('lhb_note') or '--'}")
    print(f"- 解禁: {overlay.get('unlock_note') or '--'}")
    print()
    print("模型判断:")
    print(
        f"- 分数: adaptive={model['adaptive_score']}, risk_adjusted={model['risk_adjusted_score']}, risk={model['risk_score']} | "
        f"主视角={model.get('dominant_horizon') or '--'}日，覆盖率={model.get('coverage') if model.get('coverage') is not None else '--'}"
    )
    print(f"- 风格: {model.get('style_label') or model.get('style') or '--'} | 趋势: {model.get('trend_state') or '--'}")
    family_scores = model.get("family_scores") or {}
    if family_scores:
        family_score_text = "；".join(
            f"{adaptive.FAMILY_LABELS.get(name, name)}={score}"
            for name, score in sorted(family_scores.items(), key=lambda item: item[1], reverse=True)
        )
        dominant_family = model.get("dominant_family")
        dominant_family_label = adaptive.FAMILY_LABELS.get(dominant_family, dominant_family) if dominant_family else "--"
        print(
            f"- 主导因子: {dominant_family_label}({model.get('dominant_family_score') if model.get('dominant_family_score') is not None else '--'}分) | "
            f"家族分: {family_score_text}"
        )
    formal_text = "是" if model.get("formal_candidate") else "否"
    precision_text = "是" if model.get("precision_gate_candidate") else "否"
    print(f"- 正式候选: {formal_text} | 精确门槛: {precision_text} | 落库排名: {model.get('formal_candidate_rank') or '--'}")
    print(f"- 原因: {model.get('reason') or '--'}")
    if model.get("quality_fail_reasons"):
        print(f"- 未通过项: {'；'.join(model.get('quality_fail_reasons') or [])}")
    print(f"- 风险: {model.get('risk_labels') or '--'} | 放弃条件: {model.get('no_buy_condition') or '--'}")
    print(f"- 评分构成: {_scorecard_component_text(scorecard) or '--'}")
    print()
    print("行业共振:")
    print(f"- {industry_view.get('summary') or '--'}")
    print(
        f"- 行业样本{industry_view.get('sample_count') or 0}，"
        f"5日广度{industry_view.get('industry_breadth_5d') if industry_view.get('industry_breadth_5d') is not None else '--'}%，"
        f"20日广度{industry_view.get('industry_breadth_20d') if industry_view.get('industry_breadth_20d') is not None else '--'}%，"
        f"个股5日行业分位{industry_view.get('stock_industry_rank_5d') if industry_view.get('stock_industry_rank_5d') is not None else '--'}"
    )
    print()
    print("历史证据:")
    similar = evidence.get("similar_sample") or {}
    style_trend = evidence.get("style_trend_sample") or {}
    print(
        f"- 相似形态样本: {similar.get('sample_size') or 0} 条，"
        f"平均相似度 {similar.get('avg_similarity') if similar.get('avg_similarity') is not None else '--'}"
    )
    for label, stats in (similar.get("stats") or {}).items():
        print(
            f"  {label}: 样本{stats.get('sample_count') or 0}，"
            f"均值{stats.get('avg_return')}%，中位{stats.get('median_return')}%，"
            f"胜率{stats.get('win_rate')}%，最差{stats.get('worst_return')}%，"
            f"收益因子{stats.get('profit_factor')}，95%区间{stats.get('confidence_interval') or '--'}"
        )
    print(f"- {style_trend.get('filter_label') or '同风格趋势样本'}: {style_trend.get('sample_size') or 0} 条")
    for label, stats in (style_trend.get("stats") or {}).items():
        print(
            f"  {label}: 样本{stats.get('sample_count') or 0}，"
            f"均值{stats.get('avg_return')}%，胜率{stats.get('win_rate')}%，"
            f"最差{stats.get('worst_return')}%，95%区间{stats.get('confidence_interval') or '--'}"
        )
    print()
    print("关键数据:")
    print(
        f"- 开{key_data['today_open']} 高{key_data['today_high']} 低{key_data['today_low']} 收{key_data['latest_price']} "
        f"今日{key_data['today_change']} 5日{key_data['change_5d']} "
        f"20日{key_data['change_20d']} 30日{key_data['change_30d']}"
    )
    print(
        f"- MA20={key_data['ma20']}({key_data['price_vs_ma20']}) "
        f"MA60={key_data['ma60']}({key_data['price_vs_ma60']}) "
        f"振幅={key_data['today_amp']} 量比={key_data['vr_today']} 换手={key_data['turnover_rate']} "
        f"成交额/5日均={key_data['amount_vs_avg_5d']} 换手/5日均={key_data['turnover_vs_avg_5d']}"
    )

    memory_review = result.get("strategy_memory_review") or {}
    memory_summary = memory_review.get("summary") or {}
    memory_items = memory_review.get("items") or []
    if memory_summary.get("total"):
        print()
        print("落库推荐复盘:")
        print(
            f"- 历史落库{memory_summary.get('total')}次，正式推荐{memory_summary.get('formal_total')}次，"
            f"最近正式推荐{memory_summary.get('latest_formal_date') or '--'}；"
            f"次日承接通过{memory_summary.get('accepted_count') or 0}次，"
            f"触达止盈{memory_summary.get('target_hit_count') or 0}次，"
            f"收盘跌破防守{memory_summary.get('stop_broken_count') or 0}次"
        )
        print(f"- 个性化结论: {memory_summary.get('personal_guidance') or '--'}")
        latest_items = memory_items[-8:]
        for item in latest_items:
            metrics = item.get("metrics") or {}
            plan = item.get("plan") or {}
            print(
                f"  {item.get('trade_date')} {item.get('kind')}/{item.get('strategy_type')}: "
                f"{item.get('outcome')}；入场日{metrics.get('entry_date') or '--'}，"
                f"承接{'是' if metrics.get('entry_acceptance_ok') is True else '否' if metrics.get('entry_acceptance_ok') is False else '--'}，"
                f"当前{metrics.get('current_return_pct') if metrics.get('current_return_pct') is not None else '--'}%，"
                f"最高{metrics.get('max_high_return_pct') if metrics.get('max_high_return_pct') is not None else '--'}%，"
                f"防守{plan.get('defense_price') if plan.get('defense_price') is not None else '--'}；"
                f"{item.get('action') or '--'}"
            )

    if result.get("saved_strategy"):
        print()
        print("已落库策略:")
        for item in result["saved_strategy"]:
            print(f"- {item.get('strategy_type')}: {item.get('strategy_note')}")
        if result.get("saved_strategy_hint"):
            print(f"提示: {result.get('saved_strategy_hint')}")
    elif result.get("recent_saved_strategy"):
        print()
        print("最近已落库策略:")
        latest_date = _latest_strategy_trade_date(result.get("recent_saved_strategy"))
        for item in result["recent_saved_strategy"]:
            if _date_text(item.get("trade_date")) != latest_date:
                continue
            print(f"- {item.get('trade_date')} {item.get('strategy_type')}: {item.get('strategy_note')}")
        if result.get("recent_saved_strategy_hint"):
            print(f"提示: {result.get('recent_saved_strategy_hint')}")


def main():
    parser = argparse.ArgumentParser(description="按单只股票代码输出当前操作建议")
    parser.add_argument("stock_code", help="股票代码，例如 301566")
    parser.add_argument("--date", dest="analysis_date", default=None, help="指定分析日期，例如 2026-05-15")
    parser.add_argument("--lookback", dest="lookback_trade_days", type=int, default=90, help="历史交易日窗口，默认90")
    parser.add_argument("--cost", dest="cost_price", type=float, default=None, help="持仓成本价，用于生成持仓处理建议")
    parser.add_argument("--shares", dest="shares", type=int, default=None, help="当前总持仓股数，用于生成持仓T策略")
    parser.add_argument("--available-shares", dest="available_shares", type=int, default=None, help="当前可卖股数；不填时按总持仓估算")
    parser.add_argument("--json", dest="as_json", action="store_true", help="输出JSON")
    parser.add_argument("--detail", dest="detail", action="store_true", help="输出完整详细报告；默认只输出执行版")
    parser.add_argument("--progress", dest="progress", action="store_true", help="显示分析进度；默认只输出结论")
    args = parser.parse_args()

    analysis_kwargs = {
        "stock_code": args.stock_code,
        "analysis_date": args.analysis_date,
        "lookback_trade_days": args.lookback_trade_days,
        "cost_price": args.cost_price,
        "shares": args.shares,
        "available_shares": args.available_shares,
        "emit_status": args.progress,
    }
    if args.progress:
        result = analysis_per_stock_code(**analysis_kwargs)
    else:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = analysis_per_stock_code(**analysis_kwargs)
    if args.as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        _print_report(result, detail=args.detail)

    return 0 if result.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
