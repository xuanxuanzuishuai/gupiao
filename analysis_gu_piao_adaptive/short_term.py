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

import time

from .common import *
from .scoring import *


def _build_point_in_time_profiles(history, as_of_date, emit_training_log=False):
    profile_history = _point_in_time_history_slice(
        history,
        as_of_date,
        tail_trade_days=SHORT_TERM_LOOKBACK_TRADE_DAYS,
    )
    horizon_profiles = {}
    style_horizon_profiles = {}
    as_of_ts = pd.to_datetime(as_of_date, errors="coerce")

    for horizon_days in HORIZON_DAYS:
        forward_date_col = f"forward_trade_date_{horizon_days}d"
        if profile_history.empty or forward_date_col not in profile_history.columns:
            horizon_frame = pd.DataFrame()
        else:
            forward_dates = pd.to_datetime(profile_history[forward_date_col], errors="coerce")
            horizon_frame = profile_history[forward_dates.notna() & forward_dates.le(as_of_ts)].copy()

        profile = _build_horizon_profile(horizon_frame, horizon_days)
        horizon_profiles[horizon_days] = profile
        style_horizon_profiles[horizon_days] = _build_style_horizon_profiles(horizon_frame, horizon_days)
        if emit_training_log:
            func.logInfo(
                f"{SHORT_TERM_MODEL_DISPLAY}as-of训练完成 horizon={horizon_days}d, "
                f"as_of={_to_date_text(as_of_date)}, sample_rows={profile['sample_rows']}, "
                f"sample_days={profile['sample_days']}, winner_rows={profile['winner_rows']}, "
                f"loser_rows={profile['loser_rows']}, positive_rate={profile['positive_rate']}%"
            )

    return profile_history, horizon_profiles, style_horizon_profiles


def _resolve_backtest_eval_trade_dates(history, eval_step=1, eval_window_trade_days=None):
    trade_dates = sorted(history["last_data_date"].dropna().unique())
    if eval_window_trade_days:
        trade_dates = trade_dates[-int(eval_window_trade_days):]
    return trade_dates[:: max(1, int(eval_step or 1))], trade_dates


def _empty_backtest_metrics():
    return {
        horizon_days: {
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
        for horizon_days in HORIZON_DAYS
    }


def _backtest_horizon_profiles(
    history,
    top_candidate_count=TOP_CANDIDATE_COUNT,
    eval_step=1,
    eval_window_trade_days=None,
):
    eval_trade_dates, window_trade_dates = _resolve_backtest_eval_trade_dates(
        history,
        eval_step=eval_step,
        eval_window_trade_days=eval_window_trade_days,
    )
    total_eval_trade_dates = len(eval_trade_dates)
    progress_started_at = time.monotonic()
    metrics = _empty_backtest_metrics()
    daily_cache_hits = 0
    daily_cache_misses = 0
    overlay_frame = risk_overlay.build_special_pool_overlay(history, all_dates=True)

    for eval_index, eval_date in enumerate(eval_trade_dates, start=1):
        evaluated_days_so_far = max(
            int(metrics[horizon_days].get("evaluated_days") or 0)
            for horizon_days in HORIZON_DAYS
        )
        _emit_runtime_status(
            f"{SHORT_TERM_MODEL_DISPLAY}walk-forward进度: "
            f"{eval_index}/{total_eval_trade_dates}, "
            f"eval_date={_to_date_text(eval_date)}, "
            f"effective_eval_days={evaluated_days_so_far}, "
            f"cache_hit={daily_cache_hits}, cache_miss={daily_cache_misses}, "
            f"elapsed={int(time.monotonic() - progress_started_at)}s"
        )
        snapshot = history[history["last_data_date"] == eval_date].copy()
        if snapshot.empty or len(snapshot) < MIN_DAILY_ROWS:
            continue

        decision_context = _point_in_time_history_slice(
            history,
            eval_date,
            tail_trade_days=ADAPTIVE_DAILY_DECISION_CONTEXT_TRADE_DAYS,
        )
        decision = _load_adaptive_daily_decision_cache(
            decision_context,
            eval_date,
            top_candidate_count,
            include_external=False,
        )
        if decision:
            daily_cache_hits += 1
        else:
            daily_cache_misses += 1
            eval_date_text = _to_date_text(eval_date)
            eval_overlay_frame = overlay_frame
            if eval_date_text and overlay_frame is not None and not overlay_frame.empty and "trade_date" in overlay_frame.columns:
                eval_overlay_frame = overlay_frame[overlay_frame["trade_date"] == eval_date_text].copy()
            decision = analysis_gu_piao_history_adaptive_model(
                top_candidate_count=top_candidate_count,
                history=decision_context,
                history_prepared=True,
                as_of_date=eval_date,
                include_external=False,
                emit_status=False,
                use_daily_decision_cache=False,
                enforce_snapshot_coverage=False,
                precomputed_overlay_frame=eval_overlay_frame,
            )
            if decision:
                _save_adaptive_daily_decision_cache(
                    decision_context,
                    eval_date,
                    top_candidate_count,
                    decision,
                    include_external=False,
                )

        if not decision.get("success"):
            continue

        scored = pd.DataFrame(decision.get("top_candidates") or []).head(int(top_candidate_count)).copy()
        if scored.empty or "stock_code" not in scored.columns:
            continue

        top_stock_codes = scored["stock_code"].dropna().tolist()
        if not top_stock_codes:
            continue

        market_regime = decision.get("market_regime") or risk_overlay.classify_market_regime(
            pd.to_numeric(snapshot.get("market_change_5d"), errors="coerce").mean(),
            pd.to_numeric(snapshot.get("market_breadth_5d"), errors="coerce").mean(),
        )

        for horizon_days in HORIZON_DAYS:
            future_col = f"forward_return_{horizon_days}d"
            if future_col not in snapshot.columns:
                continue

            universe_returns = pd.to_numeric(snapshot[future_col], errors="coerce").dropna()
            top_returns = pd.to_numeric(
                snapshot[snapshot["stock_code"].isin(top_stock_codes)][future_col],
                errors="coerce",
            ).dropna()
            if universe_returns.empty or top_returns.empty:
                continue

            top_subset = scored.merge(
                snapshot[["stock_code", future_col]],
                on="stock_code",
                how="left",
            )

            metrics[horizon_days]["evaluated_days"] += 1
            metrics[horizon_days]["universe_return_sum"] += float(universe_returns.mean())
            metrics[horizon_days]["universe_win_sum"] += float((universe_returns > 0).mean())
            metrics[horizon_days]["universe_count"] += 1
            metrics[horizon_days]["top_return_sum"] += float(top_returns.mean())
            metrics[horizon_days]["top_win_sum"] += float((top_returns > 0).mean())
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
            regime_stat["universe_return_sum"] += float(universe_returns.mean())
            regime_stat["universe_win_sum"] += float((universe_returns > 0).mean())
            regime_stat["top_return_sum"] += float(top_returns.mean())
            regime_stat["top_win_sum"] += float((top_returns > 0).mean())
            regime_stat["top_count"] += 1

            style_series = scored["style"].fillna("generic") if "style" in scored.columns else pd.Series([], dtype=object)
            for style_name, count in style_series.value_counts().items():
                metrics[horizon_days]["top_style_counts"][style_name] = metrics[horizon_days]["top_style_counts"].get(style_name, 0) + int(count)

            trend_series = scored["trend_state"].fillna("未知") if "trend_state" in scored.columns else pd.Series([], dtype=object)
            for trend_state, count in trend_series.value_counts().items():
                metrics[horizon_days]["top_trend_state_counts"][trend_state] = metrics[horizon_days]["top_trend_state_counts"].get(trend_state, 0) + int(count)

            group_key = top_subset["style"].fillna("generic") if "style" in top_subset.columns else pd.Series("generic", index=top_subset.index)
            for style_name, group in top_subset.groupby(group_key):
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

    result["_meta"] = {
        "daily_decision_cache_hits": daily_cache_hits,
        "daily_decision_cache_misses": daily_cache_misses,
        "eval_trade_dates": int(total_eval_trade_dates),
        "window_trade_days": int(len(window_trade_dates)),
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
    as_of_date=None,
    include_external=True,
    emit_status=True,
    use_daily_decision_cache=False,
    enforce_snapshot_coverage=True,
    precomputed_overlay_frame=None,
):
    if emit_status:
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

    history["last_data_date"] = pd.to_datetime(history["last_data_date"], errors="coerce")
    history = history[history["last_data_date"].notna()].copy()
    latest_trade_date = pd.to_datetime(as_of_date, errors="coerce") if as_of_date is not None else history["last_data_date"].max()
    if pd.isna(latest_trade_date):
        latest_trade_date = history["last_data_date"].max()
    latest_trade_date_text = _to_date_text(latest_trade_date)

    decision_context = _point_in_time_history_slice(
        history,
        latest_trade_date,
        tail_trade_days=ADAPTIVE_DAILY_DECISION_CONTEXT_TRADE_DAYS,
    )
    if use_daily_decision_cache and not include_external:
        cached_decision = _load_adaptive_daily_decision_cache(
            decision_context,
            latest_trade_date,
            top_candidate_count,
            include_external=False,
        )
        if cached_decision:
            cached_decision = dict(cached_decision)
            cached_decision["daily_decision_cache"] = "hit"
            return cached_decision

    latest_snapshot = history[history["last_data_date"] == latest_trade_date].copy()
    if latest_snapshot.empty:
        if emit_status:
            func.logInfo(f"{SHORT_TERM_MODEL_DISPLAY}没有可用于评分的最新快照")
        return {
            "model_version": MODEL_VERSION,
            "model_nature": MODEL_NATURE,
            "success": False,
            "reason": "snapshot_empty",
            "latest_trade_date": latest_trade_date_text,
            "top_candidates": [],
        }

    if enforce_snapshot_coverage:
        snapshot_coverage = _assess_snapshot_coverage(latest_snapshot)
    else:
        snapshot_coverage = {
            "trade_date": latest_trade_date_text,
            "trade_date_count": int(len(latest_snapshot)),
            "universe_count": int(len(latest_snapshot)),
            "coverage_ratio": 100.0,
            "meets_min_coverage": True,
        }
    if not snapshot_coverage["meets_min_coverage"]:
        if emit_status:
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

    profile_history, horizon_profiles, style_horizon_profiles = _build_point_in_time_profiles(
        history,
        latest_trade_date,
        emit_training_log=emit_status,
    )

    candidate_df = _score_candidates(latest_snapshot, horizon_profiles, style_horizon_profiles)
    if candidate_df.empty:
        if emit_status:
            func.logInfo("最新快照没有形成有效评分")
        return {
            "model_version": MODEL_VERSION,
            "model_nature": MODEL_NATURE,
            "success": False,
            "reason": "candidate_empty",
            "latest_trade_date": latest_trade_date_text,
            "top_candidates": [],
        }

    risk_context = _point_in_time_history_slice(
        history,
        latest_trade_date,
        tail_trade_days=ADAPTIVE_RISK_CONTEXT_TRADE_DAYS,
    )
    overlay_frame = (
        precomputed_overlay_frame
        if precomputed_overlay_frame is not None and not precomputed_overlay_frame.empty
        else risk_overlay.build_special_pool_overlay(risk_context, trade_date=latest_trade_date_text)
    )
    candidate_with_overlay = _apply_candidate_risk_overlay(
        candidate_df,
        history=risk_context,
        trade_date=latest_trade_date_text,
        overlay_frame=overlay_frame,
        include_external=include_external,
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
        history=risk_context,
        trade_date=latest_trade_date_text,
        overlay_frame=overlay_frame,
        include_external=include_external,
        filter_blocked=True,
        filter_downgraded=True,
        score_penalty_multiplier=EXTERNAL_RISK_SCORE_PENALTY_MULTIPLIER,
    )
    if candidate_df.empty:
        if emit_status:
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
        "decision_context": "point_in_time",
        "as_of_date": latest_trade_date_text,
        "sample_start": str(profile_history["last_data_date"].min().date()) if not profile_history.empty else None,
        "sample_end": str(profile_history["last_data_date"].max().date()) if not profile_history.empty else latest_trade_date_text,
        "trade_days": int(profile_history["last_data_date"].nunique()) if not profile_history.empty else 0,
        "history_rows": int(len(profile_history)),
        "risk_context_start": str(risk_context["last_data_date"].min().date()) if not risk_context.empty else None,
        "risk_context_trade_days": int(risk_context["last_data_date"].nunique()) if not risk_context.empty else 0,
        "risk_context_rows": int(len(risk_context)),
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

    if use_daily_decision_cache and not include_external:
        _save_adaptive_daily_decision_cache(
            decision_context,
            latest_trade_date,
            top_candidate_count,
            summary,
            include_external=False,
        )

    if emit_status:
        func.logInfo(model_note)
        func.logInfo(
            f"{SHORT_TERM_MODEL_DISPLAY}分析完成: trade_days={summary['trade_days']}, history_rows={summary['history_rows']}, "
            f"risk_context_trade_days={summary['risk_context_trade_days']}, "
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
    eval_window_trade_days=None,
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
        raw_trade_days = int(history["last_data_date"].dropna().nunique()) if "last_data_date" in history.columns else 0
        _emit_runtime_status(
            f"{SHORT_TERM_MODEL_DISPLAY}回测特征准备开始: rows={len(history)}, trade_days={raw_trade_days}"
        )
        history = _prepare_short_term_history(history)
        prepared_trade_days = int(history["last_data_date"].dropna().nunique()) if "last_data_date" in history.columns else 0
        _emit_runtime_status(
            f"{SHORT_TERM_MODEL_DISPLAY}回测特征准备完成: rows={len(history)}, trade_days={prepared_trade_days}"
        )

    cached_summary = _load_adaptive_backtest_cache(
        history,
        top_candidate_count,
        eval_step,
        eval_window_trade_days=eval_window_trade_days,
    )
    if cached_summary:
        print(f"{SHORT_TERM_MODEL_DISPLAY}回测使用缓存")
        return cached_summary

    _, window_trade_dates = _resolve_backtest_eval_trade_dates(
        history,
        eval_step=eval_step,
        eval_window_trade_days=eval_window_trade_days,
    )
    context_trade_dates = history["last_data_date"].dropna().nunique()
    backtest_result = _backtest_horizon_profiles(
        history,
        top_candidate_count=top_candidate_count,
        eval_step=eval_step,
        eval_window_trade_days=eval_window_trade_days,
    )
    backtest_meta = backtest_result.pop("_meta", {})

    summary = {
        "model_version": MODEL_VERSION,
        "success": True,
        "decision_context": "point_in_time_daily",
        "trade_days": int(len(window_trade_dates)),
        "context_trade_days": int(context_trade_dates),
        "eval_step": max(1, int(eval_step or 1)),
        "eval_window_trade_days": int(eval_window_trade_days or 0),
        "daily_decision_cache_hits": int(backtest_meta.get("daily_decision_cache_hits") or 0),
        "daily_decision_cache_misses": int(backtest_meta.get("daily_decision_cache_misses") or 0),
        "backtest": {
            f"{horizon_days}d": metrics
            for horizon_days, metrics in backtest_result.items()
        },
    }

    func.logInfo({
        "backtest": summary["backtest"],
    })
    _save_adaptive_backtest_cache(
        history,
        top_candidate_count,
        eval_step,
        summary,
        eval_window_trade_days=eval_window_trade_days,
    )
    print(f"{SHORT_TERM_MODEL_DISPLAY}回测完毕")
    return summary


__all__ = [name for name in globals() if not name.startswith("__")]
