"""自适应选股通用评分层。

作用:
    提供短线和长跑模型都能复用的画像训练、风格识别、单票评分、
    候选排序和交易计划辅助能力。这里关心“如何评价一只股票匹配历史
    赢家画像”，不负责查库、落库或整体运行调度。

流程:
    先按持有周期构建 horizon profile，比较赢家样本和弱势样本的特征
    分布差异；再为最新快照逐票计算画像匹配分、风格来源、主导周期和
    主导因子；最后结合短线风险分、精度过滤和排序规则生成候选列表。
"""

from .common import *


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


__all__ = [name for name in globals() if not name.startswith("__")]
