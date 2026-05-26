"""短线自适应选股模型。

作用:
    面向每日短线候选，学习 5 日和 10 日上涨股画像，并结合风格识别、
    趋势状态、市场环境和风险覆盖，对最新交易日股票池打分。默认产出
    观察候选；真正写入正式策略结果前，还需要健康验证。

流程:
    读取或接收已准备好的历史数据，构建通用画像和分风格画像；
    对最新快照评分并套用风险覆盖层；
    生成候选理由、持有计划和市场摘要；
    backtest_analysis_gu_piao_history_adaptive_model 提供 walk-forward 验证。
"""

from .common import *
from .scoring import *


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
            filter_downgraded=True,
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


def analysis_gu_piao_history_adaptive_model(
    start_date=None,
    end_date=None,
    top_candidate_count=TOP_CANDIDATE_COUNT,
    stock_code=None,
    history=None,
    history_prepared=False,
):
    func.logInfo(f"开始分析历史上涨票特征（{SHORT_TERM_MODEL_DISPLAY}）")
    print(f"开始分析历史上涨票特征（{SHORT_TERM_MODEL_DISPLAY}）")

    target_stock_code = _normalize_stock_code(stock_code)

    if history is None:
        history = _load_history(
            start_date=start_date,
            end_date=end_date,
            columns=LONG_RUNWAY_FRAME_COLUMNS,
        )
    if history.empty:
        func.logInfo("a_stock_analysis_history 没有可用历史数据")
        return {
            "model_version": MODEL_VERSION,
            "model_nature": MODEL_NATURE,
            "success": False,
            "reason": "history_empty",
            "top_candidates": [],
        }

    if not history_prepared:
        history = _prepare_short_term_history(history)

    latest_trade_date = history["last_data_date"].max()
    latest_trade_date_text = _to_date_text(latest_trade_date)
    latest_snapshot = history[history["last_data_date"] == latest_trade_date].copy()
    if latest_snapshot.empty:
        func.logInfo(f"{SHORT_TERM_MODEL_DISPLAY}没有可用于评分的最新快照")
        return {
            "model_version": MODEL_VERSION,
            "model_nature": MODEL_NATURE,
            "success": False,
            "reason": "snapshot_empty",
            "top_candidates": [],
        }

    snapshot_coverage = _assess_snapshot_coverage(latest_snapshot)
    if not snapshot_coverage["meets_min_coverage"]:
        func.logInfo(
            f"{SHORT_TERM_MODEL_DISPLAY}最新快照覆盖不足，跳过推荐: trade_date={snapshot_coverage['trade_date']}, "
            f"coverage={snapshot_coverage['coverage_ratio']}%, "
            f"trade_date_count={snapshot_coverage['trade_date_count']}, "
            f"universe_count={snapshot_coverage['universe_count']}"
        )
        return {
            "model_version": MODEL_VERSION,
            "model_nature": MODEL_NATURE,
            "success": False,
            "reason": "snapshot_coverage_below_threshold",
            "latest_trade_date": snapshot_coverage["trade_date"],
            "snapshot_coverage_ratio": snapshot_coverage["coverage_ratio"],
            "snapshot_trade_date_count": snapshot_coverage["trade_date_count"],
            "snapshot_universe_count": snapshot_coverage["universe_count"],
            "top_candidates": [],
        }

    horizon_profiles = {}
    style_horizon_profiles = {}
    for horizon_days in HORIZON_DAYS:
        profile = _build_horizon_profile(history, horizon_days)
        horizon_profiles[horizon_days] = profile
        style_horizon_profiles[horizon_days] = _build_style_horizon_profiles(history, horizon_days)
        func.logInfo(
            f"{SHORT_TERM_MODEL_DISPLAY}训练完成 horizon={horizon_days}d, sample_rows={profile['sample_rows']}, "
            f"sample_days={profile['sample_days']}, winner_rows={profile['winner_rows']}, "
            f"loser_rows={profile['loser_rows']}, positive_rate={profile['positive_rate']}%"
        )

    candidate_df = _score_candidates(latest_snapshot, horizon_profiles, style_horizon_profiles)
    if candidate_df.empty:
        func.logInfo("最新快照没有形成有效评分")
        return {
            "model_version": MODEL_VERSION,
            "model_nature": MODEL_NATURE,
            "success": False,
            "reason": "candidate_empty",
            "latest_trade_date": latest_trade_date_text,
            "top_candidates": [],
        }

    overlay_frame = risk_overlay.build_special_pool_overlay(history, trade_date=latest_trade_date_text)
    candidate_with_overlay = _apply_candidate_risk_overlay(
        candidate_df,
        history=history,
        trade_date=latest_trade_date_text,
        overlay_frame=overlay_frame,
        include_external=True,
        filter_blocked=False,
        score_penalty_multiplier=EXTERNAL_RISK_SCORE_PENALTY_MULTIPLIER,
    )
    if not candidate_with_overlay.empty:
        block_mask = (
            candidate_with_overlay["risk_overlay_block_formal"].fillna(False).astype(bool)
            if "risk_overlay_block_formal" in candidate_with_overlay.columns
            else pd.Series(False, index=candidate_with_overlay.index)
        )
        special_mask = (
            candidate_with_overlay["special_pool"].fillna("normal") != "normal"
            if "special_pool" in candidate_with_overlay.columns
            else pd.Series(False, index=candidate_with_overlay.index)
        )
        downgrade_mask = (
            candidate_with_overlay["risk_overlay_downgrade"].fillna(False).astype(bool)
            if "risk_overlay_downgrade" in candidate_with_overlay.columns
            else pd.Series(False, index=candidate_with_overlay.index)
        )
        special_pool_candidates_df = candidate_with_overlay[block_mask | downgrade_mask | special_mask].copy()
    else:
        special_pool_candidates_df = pd.DataFrame()
    candidate_df = _apply_candidate_risk_overlay(
        candidate_df,
        history=history,
        trade_date=latest_trade_date_text,
        overlay_frame=overlay_frame,
        include_external=True,
        filter_blocked=True,
        filter_downgraded=True,
        score_penalty_multiplier=EXTERNAL_RISK_SCORE_PENALTY_MULTIPLIER,
    )
    if candidate_df.empty:
        func.logInfo(f"{SHORT_TERM_MODEL_DISPLAY}最新快照经特殊股票池/事件风险覆盖后没有形成正式候选")
        return {
            "model_version": MODEL_VERSION,
            "model_nature": MODEL_NATURE,
            "success": False,
            "reason": "candidate_empty_after_risk_overlay",
            "latest_trade_date": latest_trade_date_text,
            "risk_overlay": risk_overlay.summarize_overlay(overlay_frame),
            "special_pool_candidates": special_pool_candidates_df.head(10).to_dict("records")
            if not special_pool_candidates_df.empty
            else [],
            "top_candidates": [],
        }

    top_families, top_features = _summarize_families(horizon_profiles)
    top_styles = _summarize_style_profiles(style_horizon_profiles)
    model_note = _compose_model_note({"top_families": top_families, "top_features": top_features, "top_styles": top_styles})

    market_change_5d = _round_or_none(pd.to_numeric(latest_snapshot["market_change_5d"], errors="coerce").mean(), 4)
    market_breadth_5d = _round_or_none(pd.to_numeric(latest_snapshot["market_breadth_5d"], errors="coerce").mean(), 2)
    market_regime = risk_overlay.classify_market_regime(market_change_5d, market_breadth_5d)
    market_env = risk_overlay.market_env_label(market_regime)

    candidate_df = _enrich_candidate_trade_plans(candidate_df, market_env)
    top_candidates_df = candidate_df.head(int(top_candidate_count)).copy()
    focus_candidate = None
    if target_stock_code:
        focus_df = candidate_df[candidate_df["stock_code"] == target_stock_code].head(1)
        if not focus_df.empty:
            focus_candidate = focus_df.iloc[0].to_dict()
    if focus_candidate is None and not top_candidates_df.empty:
        focus_candidate = top_candidates_df.iloc[0].to_dict()

    summary = {
        "model_version": MODEL_VERSION,
        "model_nature": MODEL_NATURE,
        "success": True,
        "sample_start": str(history["last_data_date"].min().date()),
        "sample_end": str(history["last_data_date"].max().date()),
        "trade_days": int(history["last_data_date"].nunique()),
        "history_rows": int(len(history)),
        "latest_trade_date": latest_trade_date_text,
        "snapshot_coverage_ratio": snapshot_coverage["coverage_ratio"],
        "snapshot_trade_date_count": snapshot_coverage["trade_date_count"],
        "snapshot_universe_count": snapshot_coverage["universe_count"],
        "requested_stock_code": target_stock_code,
        "market_env": market_env,
        "market_regime": market_regime,
        "market_change_5d": market_change_5d,
        "market_breadth_5d": market_breadth_5d,
        "recency_half_life_days": RECENCY_HALF_LIFE_DAYS,
        "model_definition": MODEL_DEFINITION,
        "winner_ratio": TOP_WINNER_RATIO,
        "loser_ratio": LOSER_RATIO,
        "top_families": top_families,
        "top_features": top_features,
        "top_styles": top_styles,
        "model_note": model_note,
        "top_industries": _summarize_industries(candidate_df),
        "risk_overlay": risk_overlay.summarize_overlay(overlay_frame),
        "recommendation_tiers": {
            "formal_recommendation": 0,
            "observation_candidate": int(len(top_candidates_df)),
            "research_value": int(min(len(special_pool_candidates_df), 10)) if not special_pool_candidates_df.empty else 0,
            "note": f"{SHORT_TERM_MODEL_DISPLAY}先产出观察候选；只有{ADAPTIVE_HEALTH_MODEL_DISPLAY}通过，才升级为正式推荐。",
        },
        "top_candidates": top_candidates_df.to_dict("records"),
        "special_pool_candidates": special_pool_candidates_df.head(10).to_dict("records")
        if not special_pool_candidates_df.empty
        else [],
        "focus_candidate": focus_candidate,
        "horizon_profiles": {
            f"{horizon_days}d": {
                "sample_rows": profile["sample_rows"],
                "sample_days": profile["sample_days"],
                "winner_rows": profile["winner_rows"],
                "loser_rows": profile["loser_rows"],
                "positive_rate": profile["positive_rate"],
                "top_features": profile["top_features"],
                "families": {
                    family_name: {
                        "label": FAMILY_LABELS.get(family_name, family_name),
                        "importance": family_data["importance"],
                        "normalized_importance": family_data["normalized_importance"],
                        "top_features": family_data["items"][:3],
                    }
                    for family_name, family_data in profile["families"].items()
                },
            }
            for horizon_days, profile in horizon_profiles.items()
        },
        "style_profiles": {
            f"{horizon_days}d": {
                style_name: {
                    "label": style_profile["style_label"],
                    "sample_rows": style_profile["sample_rows"],
                    "sample_days": style_profile["sample_days"],
                    "winner_rows": style_profile["winner_rows"],
                    "loser_rows": style_profile["loser_rows"],
                    "positive_rate": style_profile["positive_rate"],
                    "top_features": style_profile["top_features"],
                    "top_families": style_profile["families"],
                }
                for style_name, style_profile in profiles.items()
            }
            for horizon_days, profiles in style_horizon_profiles.items()
        },
    }

    func.logInfo(model_note)
    func.logInfo(
        f"{SHORT_TERM_MODEL_DISPLAY}分析完成: trade_days={summary['trade_days']}, history_rows={summary['history_rows']}, "
        f"market_env={market_env}, market_regime={market_regime}, top_candidates={len(summary['top_candidates'])}"
    )
    func.logInfo({
        "top_families": top_families[:3],
        "top_features": top_features[:5],
    })
    print(f"{SHORT_TERM_MODEL_DISPLAY}分析完毕")
    return summary


def backtest_analysis_gu_piao_history_adaptive_model(
    start_date=None,
    end_date=None,
    top_candidate_count=TOP_CANDIDATE_COUNT,
    history=None,
    history_prepared=False,
    eval_step=1,
):
    func.logInfo(f"开始回测{SHORT_TERM_MODEL_DISPLAY}（walk-forward）")
    print(f"开始回测{SHORT_TERM_MODEL_DISPLAY}（walk-forward）")

    if history is None:
        history = _load_history(start_date=start_date, end_date=end_date)
    if history.empty:
        func.logInfo("a_stock_analysis_history 没有可用历史数据")
        return {
            "model_version": MODEL_VERSION,
            "success": False,
            "reason": "history_empty",
            "backtest": {},
        }

    if not history_prepared:
        history = _prepare_short_term_history(history)

    trade_dates = history["last_data_date"].dropna().nunique()
    backtest_result = _backtest_horizon_profiles(
        history,
        top_candidate_count=top_candidate_count,
        eval_step=eval_step,
    )

    summary = {
        "model_version": MODEL_VERSION,
        "success": True,
        "trade_days": int(trade_dates),
        "eval_step": max(1, int(eval_step or 1)),
        "backtest": {
            f"{horizon_days}d": metrics
            for horizon_days, metrics in backtest_result.items()
        },
    }

    func.logInfo({
        "backtest": summary["backtest"],
    })
    print(f"{SHORT_TERM_MODEL_DISPLAY}回测完毕")
    return summary


__all__ = [name for name in globals() if not name.startswith("__")]
