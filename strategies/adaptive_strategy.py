"""Short-term adaptive strategy public workflow."""

from . import common as core

for _name in dir(core):
    if not _name.startswith("__") and _name not in globals():
        globals()[_name] = getattr(core, _name)


def _long_runway_strategy_module():
    from . import long_runway_strategy

    return long_runway_strategy


def clear_adaptive_strategy_results(trade_date):
    _clear_adaptive_strategy_rows(trade_date)


def persist_adaptive_candidates_to_strategy_result(
    workflow,
    max_picks=DAILY_ADAPTIVE_TOP_PICK_COUNT,
    health_check=None,
):
    short_term = (workflow or {}).get("short_term_recommendation") or {}
    trade_date = short_term.get("latest_trade_date")
    _clear_adaptive_strategy_rows(trade_date)

    if not short_term.get("success"):
        _emit_runtime_status(
            f"{SHORT_TERM_MODEL_DISPLAY}落库跳过: trade_date={trade_date}, "
            f"reason={short_term.get('reason') or 'short_term_recommendation_failed'}"
        )
        return {
            "success": False,
            "reason": short_term.get("reason") or "short_term_recommendation_failed",
            "saved_count": 0,
            "trade_date": trade_date,
        }

    if not trade_date:
        return {"success": False, "reason": "latest_trade_date_missing", "saved_count": 0}

    health_snapshot = (
        health_check
        or (workflow or {}).get("adaptive_signal_health")
        or (workflow or {}).get("adaptive_backtest_health")
        or {}
    )
    if not health_snapshot:
        _emit_runtime_status(
            f"{ADAPTIVE_HEALTH_MODEL_DISPLAY}缺少历史验证健康度，拒绝{SHORT_TERM_MODEL_DISPLAY}落库"
        )
        return {
            "success": False,
            "reason": "adaptive_health_missing",
            "saved_count": 0,
            "trade_date": trade_date,
            "health": {},
        }

    if not health_snapshot.get("enabled"):
        reject_message = (
            f"{ADAPTIVE_HEALTH_MODEL_DISPLAY}未达标，拒绝{SHORT_TERM_MODEL_DISPLAY}落库: trade_date={trade_date}, "
            f"mode={health_snapshot.get('mode')}, "
            f"avg_return={health_snapshot.get('avg_return')}, "
            f"trade_win_rate={health_snapshot.get('trade_win_rate')}, "
            f"evaluated_trades={health_snapshot.get('evaluated_trades')}, "
            f"avg_top_return={health_snapshot.get('avg_top_return')}, "
            f"avg_top_win_rate={health_snapshot.get('avg_top_win_rate')}, "
            f"excess_return={health_snapshot.get('excess_return')}, "
            f"reasons={health_snapshot.get('failure_reasons')}"
        )
        _emit_runtime_status(reject_message)
        return {
            "success": False,
            "reason": "adaptive_health_rejected",
            "saved_count": 0,
            "trade_date": trade_date,
            "health": health_snapshot,
        }

    top_candidates = short_term.get("top_candidates") or []
    if not top_candidates:
        _emit_runtime_status(f"{SHORT_TERM_MODEL_DISPLAY}落库完成: trade_date={trade_date}, saved_count=0, reason=candidate_empty")
        return {"success": True, "reason": "candidate_empty", "saved_count": 0}

    effective_max_picks = _effective_adaptive_max_picks(max_picks, health_snapshot)
    if effective_max_picks <= 0:
        _emit_runtime_status(
            f"{SHORT_TERM_MODEL_DISPLAY}落库完成: trade_date={trade_date}, saved_count=0, "
            f"reason=adaptive_health_policy_no_capacity"
        )
        return {
            "success": True,
            "reason": "adaptive_health_policy_no_capacity",
            "saved_count": 0,
            "trade_date": trade_date,
            "health": health_snapshot,
        }

    candidate_frame = _apply_adaptive_health_policy_to_candidates(
        pd.DataFrame(top_candidates),
        health_snapshot,
    )

    selected_candidates = []
    seen_codes = set()
    for candidate in candidate_frame.to_dict("records"):
        stock_code = _normalize_scalar(candidate.get("stock_code"))
        if not stock_code or stock_code in seen_codes:
            continue
        seen_codes.add(stock_code)
        selected_candidates.append(candidate)
        if len(selected_candidates) >= effective_max_picks:
            break

    final_filter_reason = None
    if selected_candidates:
        final_frame = _apply_candidate_risk_overlay(
            pd.DataFrame(selected_candidates),
            trade_date=trade_date,
            include_external=True,
            filter_blocked=False,
            filter_downgraded=False,
            # top_candidates already went through external risk penalty when the
            # short-term recommendation was built; the final pass confirms
            # current hard blocks and notes without double-penalizing score.
            score_penalty_multiplier=0.0,
        )
        if final_frame.empty:
            final_filter_reason = "candidate_empty_after_final_score_gate"
            selected_candidates = []
        else:
            block_mask = final_frame["risk_overlay_block_formal"].fillna(False).astype(bool)
            downgrade_mask = final_frame["risk_overlay_downgrade"].fillna(False).astype(bool)
            blocked_frame = final_frame[block_mask].copy()
            if not blocked_frame.empty:
                blocked_text = "；".join(
                    f"{row.get('stock_code')} {row.get('stock_name')} labels={row.get('risk_overlay_labels')}"
                    for _, row in blocked_frame.head(5).iterrows()
                )
                _emit_runtime_status(f"{SHORT_TERM_MODEL_DISPLAY}落库最终硬风控拦截: {blocked_text}")
            if downgrade_mask.any():
                downgrade_text = "；".join(
                    f"{row.get('stock_code')} {row.get('stock_name')} labels={row.get('risk_overlay_labels')}"
                    for _, row in final_frame[downgrade_mask].head(5).iterrows()
                )
                _emit_runtime_status(f"{SHORT_TERM_MODEL_DISPLAY}落库事件降级提示: {downgrade_text}")

            final_frame = final_frame[~block_mask].copy()
            if final_frame.empty and block_mask.any():
                final_filter_reason = "candidate_empty_after_final_hard_block"
            if "risk_adjusted_score" in final_frame.columns:
                before_score_gate_count = int(len(final_frame))
                final_frame = final_frame[
                    pd.to_numeric(final_frame["risk_adjusted_score"], errors="coerce").fillna(0)
                    >= ADAPTIVE_MIN_RISK_ADJUSTED_SCORE
                ].copy()
                if final_frame.empty and before_score_gate_count > 0:
                    final_filter_reason = "candidate_empty_after_final_score_gate"
            final_frame = _apply_adaptive_health_policy_to_candidates(final_frame, health_snapshot)
            selected_candidates = final_frame.to_dict("records")

    if not selected_candidates:
        final_filter_reason = final_filter_reason or "candidate_empty_after_final_risk_overlay"
        _emit_runtime_status(
            f"{SHORT_TERM_MODEL_DISPLAY}落库完成: trade_date={trade_date}, saved_count=0, "
            f"reason={final_filter_reason}"
        )
        return {
            "success": True,
            "reason": final_filter_reason,
            "saved_count": 0,
            "trade_date": trade_date,
            "health": health_snapshot,
        }

    saved_count = 0
    total_count = len(selected_candidates)
    for index, candidate in enumerate(selected_candidates, start=1):
        func.executeInsert(
            "a_stock_strategy_result",
            _build_adaptive_strategy_record(
                candidate,
                trade_date,
                rank=index,
                total=total_count,
                health_snapshot=health_snapshot,
            ),
        )
        saved_count += 1

    _emit_runtime_status(
        f"{ADAPTIVE_STRATEGY_LABEL}写入完成: model={SHORT_TERM_MODEL_DISPLAY}, "
        f"trade_date={trade_date}, saved_count={saved_count}, strategy_type={ADAPTIVE_STRATEGY_TYPE}"
    )
    return {
        "success": True,
        "reason": None,
        "saved_count": saved_count,
        "trade_date": trade_date,
        "strategy_type": ADAPTIVE_STRATEGY_TYPE,
        "health": health_snapshot,
        "effective_max_picks": effective_max_picks,
    }


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
        filter_downgraded=False,
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


def run_adaptive_model_workflow(
    start_date=None,
    end_date=None,
    target_trade_date=None,
    top_candidate_count=TOP_CANDIDATE_COUNT,
    stock_code=None,
    include_long_runway=False,
    include_backtest=False,
    persist_strategy_result=False,
    long_runway_use_cache=True,
    rebuild_long_runway_cache=False,
):
    target_trade_date_text = _to_date_text(target_trade_date)
    end_date_text = _to_date_text(end_date)
    if target_trade_date_text and end_date_text and target_trade_date_text != end_date_text:
        raise ValueError(
            f"target_trade_date 与 end_date 不一致: target_trade_date={target_trade_date_text}, end_date={end_date_text}"
        )
    effective_end_date = target_trade_date_text or end_date

    history_tail_trade_days = None
    short_history_can_tail = not include_backtest and (not include_long_runway or long_runway_use_cache)
    if start_date is None and end_date is None and short_history_can_tail:
        history_tail_trade_days = SHORT_TERM_LOOKBACK_TRADE_DAYS
        if persist_strategy_result and not stock_code:
            history_tail_trade_days = max(
                int(SHORT_TERM_LOOKBACK_TRADE_DAYS),
                int(ADAPTIVE_BACKTEST_HEALTH_LOOKBACK_TRADE_DAYS),
            )

    shared_history = _load_history(
        start_date=start_date,
        end_date=effective_end_date,
        tail_trade_days=history_tail_trade_days,
        columns=LONG_RUNWAY_FRAME_COLUMNS,
    )

    prepared_short_term_history = None
    if shared_history is not None and not shared_history.empty:
        prepared_short_term_history = _prepare_short_term_history(shared_history)

    short_term_history = prepared_short_term_history if prepared_short_term_history is not None else shared_history
    if (
        history_tail_trade_days
        and history_tail_trade_days > SHORT_TERM_LOOKBACK_TRADE_DAYS
        and (not include_long_runway or long_runway_use_cache)
        and not include_backtest
    ):
        short_term_history = _tail_trade_days_frame(short_term_history, SHORT_TERM_LOOKBACK_TRADE_DAYS)

    short_term_result = analysis_gu_piao_history_adaptive_model(
        start_date=start_date,
        end_date=effective_end_date,
        top_candidate_count=top_candidate_count,
        stock_code=stock_code,
        history=short_term_history,
        history_prepared=prepared_short_term_history is not None,
    )
    short_term_latest_trade_date = _to_date_text(short_term_result.get("latest_trade_date"))
    target_trade_date_mismatch = (
        bool(target_trade_date_text)
        and bool(short_term_latest_trade_date)
        and short_term_latest_trade_date != target_trade_date_text
    )
    if target_trade_date_mismatch:
        _emit_runtime_status(
            f"{SHORT_TERM_MODEL_DISPLAY}交易日不一致，跳过后续落库: "
            f"expected={target_trade_date_text}, actual={short_term_latest_trade_date}"
        )

    long_runway_result = None
    if include_long_runway and not target_trade_date_mismatch:
        long_runway_strategy = _long_runway_strategy_module()
        long_runway_history = None if long_runway_use_cache and start_date is None else shared_history
        long_runway_result = long_runway_strategy.analysis_gu_piao_history_long_runway_model(
            end_date=effective_end_date,
            top_candidate_count=top_candidate_count,
            stock_code=stock_code,
            history=long_runway_history,
            use_cache=long_runway_use_cache,
            rebuild_cache=rebuild_long_runway_cache,
        )

    long_term_result = None
    if include_backtest and not target_trade_date_mismatch:
        long_term_result = backtest_analysis_gu_piao_history_adaptive_model(
            start_date=start_date,
            end_date=effective_end_date,
            top_candidate_count=top_candidate_count,
            history=prepared_short_term_history if prepared_short_term_history is not None else shared_history,
            history_prepared=prepared_short_term_history is not None,
        )

    long_runway_backtest_result = None
    if include_backtest and include_long_runway and not target_trade_date_mismatch:
        long_runway_strategy = _long_runway_strategy_module()
        long_runway_backtest_result = long_runway_strategy.backtest_analysis_gu_piao_history_long_runway_model(
            start_date=start_date,
            end_date=effective_end_date,
            top_candidate_count=top_candidate_count,
            history=shared_history,
        )

    workflow = {
        "model_version": MODEL_VERSION,
        "model_definition": MODEL_DEFINITION,
        "short_term_recommendation": short_term_result,
        "long_runway_recommendation": long_runway_result,
        "long_term_backtest": long_term_result,
        "long_runway_backtest": long_runway_backtest_result,
    }

    should_persist_strategy = (
        persist_strategy_result
        and start_date is None
        and end_date is None
        and not stock_code
    )
    if should_persist_strategy:
        if target_trade_date_text:
            clear_adaptive_strategy_results(target_trade_date_text)

        if target_trade_date_mismatch:
            workflow["strategy_save_result"] = {
                "success": False,
                "reason": "latest_trade_date_mismatch",
                "trade_date": short_term_latest_trade_date,
                "expected_trade_date": target_trade_date_text,
                "saved_count": 0,
            }
            return workflow

        if not short_term_result.get("success"):
            workflow["strategy_save_result"] = persist_adaptive_candidates_to_strategy_result(
                workflow,
                health_check={},
            )
            return workflow

        _emit_runtime_status(
            f"{ADAPTIVE_HEALTH_MODEL_DISPLAY}: 开始读取{SHORT_TERM_MODEL_DISPLAY}实盘信号健康度, "
            f"trade_date={short_term_result.get('latest_trade_date')}"
        )
        adaptive_signal_health = _load_adaptive_signal_health(short_term_result.get("latest_trade_date"))
        _emit_runtime_status(
            f"{ADAPTIVE_HEALTH_MODEL_DISPLAY}实盘信号: model={SHORT_TERM_MODEL_DISPLAY}, enabled={adaptive_signal_health.get('enabled')}, "
            f"evaluated_trades={adaptive_signal_health.get('evaluated_trades')}, "
            f"avg_return={adaptive_signal_health.get('avg_return')}, "
            f"win_rate={adaptive_signal_health.get('trade_win_rate')}, "
            f"reasons={adaptive_signal_health.get('failure_reasons')}"
        )
        backtest_history = prepared_short_term_history
        backtest_history_prepared = prepared_short_term_history is not None
        if backtest_history is None or backtest_history.empty:
            backtest_history = _load_history(
                end_date=effective_end_date,
                tail_trade_days=ADAPTIVE_BACKTEST_HEALTH_LOOKBACK_TRADE_DAYS,
                columns=LONG_RUNWAY_FRAME_COLUMNS,
            )
            backtest_history_prepared = False
        backtest_trade_days = (
            int(backtest_history["last_data_date"].dropna().nunique())
            if backtest_history is not None and not backtest_history.empty
            else 0
        )
        _emit_runtime_status(
            f"{ADAPTIVE_HEALTH_MODEL_DISPLAY}walk-forward回测: model={SHORT_TERM_MODEL_DISPLAY}, 开始, trade_days={backtest_trade_days}, "
            f"eval_step=1, timeout={ADAPTIVE_PERSIST_BACKTEST_TIMEOUT_SECONDS}s"
        )
        try:
            adaptive_backtest = _run_with_wall_timeout(
                ADAPTIVE_PERSIST_BACKTEST_TIMEOUT_SECONDS,
                backtest_analysis_gu_piao_history_adaptive_model,
                top_candidate_count=max(int(top_candidate_count), int(DAILY_ADAPTIVE_TOP_PICK_COUNT)),
                history=backtest_history,
                history_prepared=backtest_history_prepared,
                eval_step=1,
            )
            _emit_runtime_status(
                f"{ADAPTIVE_HEALTH_MODEL_DISPLAY}walk-forward回测: model={SHORT_TERM_MODEL_DISPLAY}, 完成, success={adaptive_backtest.get('success')}, "
                f"trade_days={adaptive_backtest.get('trade_days')}, eval_step={adaptive_backtest.get('eval_step')}"
            )
        except TimeoutError as error:
            adaptive_backtest = {
                "model_version": MODEL_VERSION,
                "success": False,
                "reason": f"persist_backtest_{error}",
                "trade_days": int(backtest_history["last_data_date"].dropna().nunique())
                if backtest_history is not None and not backtest_history.empty
                else 0,
                "eval_step": 1,
                "backtest": {},
            }
            _emit_runtime_status(
                f"{ADAPTIVE_HEALTH_MODEL_DISPLAY}walk-forward回测超时: model={SHORT_TERM_MODEL_DISPLAY}, "
                f"reason={adaptive_backtest['reason']}"
            )
        adaptive_backtest_health = _evaluate_adaptive_backtest_health(adaptive_backtest)
        adaptive_health = _combine_adaptive_health(adaptive_signal_health, adaptive_backtest_health)
        _emit_runtime_status(
            f"{ADAPTIVE_HEALTH_MODEL_DISPLAY}判定: model={SHORT_TERM_MODEL_DISPLAY}, enabled={adaptive_health.get('enabled')}, "
            f"mode={adaptive_health.get('mode')}, mode_label={adaptive_health.get('mode_label')}, "
            f"confidence_weight={adaptive_health.get('confidence_weight')}, "
            f"max_pick_ratio={adaptive_health.get('max_pick_ratio')}, "
            f"reasons={adaptive_health.get('failure_reasons')}"
        )

        workflow["adaptive_signal_health"] = adaptive_signal_health
        workflow["adaptive_backtest"] = adaptive_backtest
        workflow["adaptive_backtest_health"] = adaptive_backtest_health
        workflow["adaptive_health"] = adaptive_health
        workflow["strategy_save_result"] = persist_adaptive_candidates_to_strategy_result(
            workflow,
            health_check=adaptive_health,
        )
        save_result = workflow["strategy_save_result"]
        _emit_runtime_status(
            f"{SHORT_TERM_MODEL_DISPLAY}策略结果表: success={save_result.get('success')}, "
            f"trade_date={save_result.get('trade_date')}, saved_count={save_result.get('saved_count')}, "
            f"effective_max_picks={save_result.get('effective_max_picks')}, "
            f"reason={save_result.get('reason')}"
        )
        if include_long_runway:
            long_runway_strategy = _long_runway_strategy_module()
            workflow["long_runway_save_result"] = long_runway_strategy.persist_long_runway_candidates_to_strategy_result(
                long_runway_result,
                trade_date=target_trade_date_text,
            )
            long_runway_save = workflow["long_runway_save_result"]
            _emit_runtime_status(
                f"{LONG_RUNWAY_MODEL_DISPLAY}策略结果表: success={long_runway_save.get('success')}, "
                f"trade_date={long_runway_save.get('trade_date')}, "
                f"saved_count={long_runway_save.get('saved_count')}, "
                f"strategy_type={long_runway_save.get('strategy_type')}, tier=中长期跟踪"
            )

    return workflow


def _print_short_term_section(result):
    print(f"{SHORT_TERM_MODEL_DISPLAY}推荐:")
    print(f"  模型版本: {result.get('model_version')}")
    print(f"  模型属性: {result.get('model_nature')}")
    print(f"  模型定义: {result.get('model_definition')}")
    print(
        f"  市场环境: {result.get('market_env')}, "
        f"5日均涨幅: {result.get('market_change_5d')}, "
        f"5日广度: {result.get('market_breadth_5d')}"
    )
    print(
        f"  历史样本: {result.get('trade_days')} 日, "
        f"{result.get('history_rows')} 条, "
        f"最新交易日: {result.get('latest_trade_date')}"
    )
    print(f"  模型结论: {result.get('model_note')}")
    overlay = result.get("risk_overlay") or {}
    if overlay:
        print(
            f"  风险覆盖: blocked={overlay.get('blocked_formal')}, "
            f"downgraded={overlay.get('downgraded')}, 市场={overlay.get('market_regime')}"
        )

    focus_candidate = result.get("focus_candidate")
    if focus_candidate:
        print("  单票判断:")
        print(
            f"    {focus_candidate.get('stock_code')} {focus_candidate.get('stock_name')} "
            f"style={focus_candidate.get('style_label')} trend={focus_candidate.get('trend_state')} "
            f"score={focus_candidate.get('adaptive_score')}"
        )
        print(f"    conclusion={focus_candidate.get('conclusion')}")
        print(
            f"    hold={focus_candidate.get('hold_period')}, expected={focus_candidate.get('expected_return')}, "
            f"stop={focus_candidate.get('stop_price')}, action={focus_candidate.get('action')}, "
            f"position={focus_candidate.get('position_hint')}"
        )
        print(f"    detail={focus_candidate.get('trend_detail')}")
        print(f"    reason={focus_candidate.get('reason')}")

    print("  当前候选:")
    for index, item in enumerate(result.get("top_candidates", [])[:5], start=1):
        print(
            f"    {index}. {item.get('stock_code')} {item.get('stock_name')} "
            f"score={item.get('adaptive_score')} "
            f"style={item.get('style_label')} "
            f"trend={item.get('trend_state')} "
            f"hold={item.get('hold_period')} "
            f"action={item.get('action')} "
            f"reason={item.get('reason')}"
        )
    special_candidates = result.get("special_pool_candidates") or []
    if special_candidates:
        print("  特殊股票池观察:")
        for index, item in enumerate(special_candidates[:5], start=1):
            print(
                f"    {index}. {item.get('stock_code')} {item.get('stock_name')} "
                f"{item.get('special_pool_label')} risk={item.get('risk_overlay_score')} "
                f"labels={item.get('risk_overlay_labels')}"
            )


def _print_long_term_section(result):
    if not result or not result.get("success"):
        print(f"{SHORT_TERM_MODEL_DISPLAY}长周期验证: 暂无有效回测结果")
        return

    print(f"{SHORT_TERM_MODEL_DISPLAY}长周期验证:")
    print(f"  回测交易日: {result.get('trade_days')}")
    for horizon_label, metrics in result.get("backtest", {}).items():
        print(
            f"  {horizon_label}: "
            f"universe_return={metrics.get('avg_universe_return')}%, "
            f"top_return={metrics.get('avg_top_return')}%, "
            f"universe_win_rate={metrics.get('avg_universe_win_rate')}%, "
            f"top_win_rate={metrics.get('avg_top_win_rate')}%"
        )

        style_stats = metrics.get("style_return_stats") or {}
        if style_stats:
            top_styles = list(style_stats.items())[:3]
            style_text = "；".join(
                f"{style_name}:avg={stat.get('avg_return')}%,win={stat.get('win_rate')}%,count={stat.get('count')}"
                for style_name, stat in top_styles
            )
            print(f"    风格回测: {style_text}")
        trend_counts = metrics.get("top_trend_state_counts") or {}
        if trend_counts:
            top_trends = list(trend_counts.items())[:3]
            trend_text = "；".join(f"{trend_name}:{count}" for trend_name, count in top_trends)
            print(f"    主要趋势: {trend_text}")
        regime_stats = metrics.get("market_regime_stats") or {}
        if regime_stats:
            regime_text = "；".join(
                f"{stat.get('label')}:top={stat.get('avg_top_return')}%,win={stat.get('avg_top_win_rate')}%,days={stat.get('evaluated_days')}"
                for _, stat in regime_stats.items()
            )
            print(f"    市场分层: {regime_text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"{SHORT_TERM_MODEL_DISPLAY}工作流")
    parser.add_argument("--start-date", dest="start_date", default=None, help="分析起始日期，格式 YYYY-MM-DD")
    parser.add_argument("--end-date", dest="end_date", default=None, help="分析结束日期，格式 YYYY-MM-DD")
    parser.add_argument("--top-candidate-count", dest="top_candidate_count", type=int, default=TOP_CANDIDATE_COUNT)
    parser.add_argument("--stock-code", dest="stock_code", default=None, help="指定单只股票代码做分析")
    parser.add_argument("--include-long-runway", dest="include_long_runway", action="store_true", help=f"显式执行{LONG_RUNWAY_MODEL_DISPLAY}")
    parser.add_argument("--no-long-runway-cache", dest="no_long_runway_cache", action="store_true", help=f"禁用{LONG_RUNWAY_MODEL_DISPLAY}缓存")
    parser.add_argument("--rebuild-long-runway-cache", dest="rebuild_long_runway_cache", action="store_true", help=f"重建{LONG_RUNWAY_MODEL_DISPLAY}缓存")
    parser.add_argument("--skip-backtest", dest="skip_backtest", action="store_true", help="兼容旧参数：跳过长周期回测")
    parser.add_argument("--include-backtest", dest="include_backtest", action="store_true", help="显式执行长周期回测")
    parser.add_argument("--skip-save", dest="skip_save", action="store_true", help="仅分析，不写入最终推荐表")
    args = parser.parse_args()

    should_include_backtest = bool(args.include_backtest) and not bool(args.skip_backtest)
    should_persist_outputs = not bool(args.skip_save) and not bool(args.stock_code)
    workflow = run_adaptive_model_workflow(
        start_date=args.start_date,
        end_date=args.end_date,
        top_candidate_count=args.top_candidate_count,
        stock_code=args.stock_code,
        include_long_runway=bool(args.include_long_runway),
        include_backtest=should_include_backtest,
        persist_strategy_result=should_persist_outputs,
        long_runway_use_cache=not bool(args.no_long_runway_cache),
        rebuild_long_runway_cache=bool(args.rebuild_long_runway_cache),
    )

    short_term = workflow.get("short_term_recommendation") or {}
    long_runway_result = workflow.get("long_runway_recommendation") or {}
    long_term = workflow.get("long_term_backtest")
    long_runway_backtest = workflow.get("long_runway_backtest")

    if short_term.get("success"):
        _print_short_term_section(short_term)
    else:
        print(f"{SHORT_TERM_MODEL_DISPLAY}失败: {short_term.get('reason')}")

    if not args.include_long_runway:
        print(f"{LONG_RUNWAY_MODEL_DISPLAY}: 已跳过")
    elif long_runway_result.get("success"):
        _long_runway_strategy_module()._print_long_runway_section(long_runway_result)
    else:
        print(f"{LONG_RUNWAY_MODEL_DISPLAY}失败: {long_runway_result.get('reason')}")

    if should_include_backtest:
        _print_long_term_section(long_term)
        if args.include_long_runway:
            _long_runway_strategy_module()._print_long_runway_backtest_section(long_runway_backtest)
    else:
        print(f"{SHORT_TERM_MODEL_DISPLAY}长周期验证: 已跳过")

    strategy_save_result = workflow.get("strategy_save_result")
    if strategy_save_result and strategy_save_result.get("success"):
        print(
            f"{SHORT_TERM_MODEL_DISPLAY}策略结果表: 已写入 {strategy_save_result.get('saved_count')} 条, "
            f"类型={strategy_save_result.get('strategy_type')}"
        )
    elif should_persist_outputs:
        print(f"{SHORT_TERM_MODEL_DISPLAY}策略结果表: 未写入")

    long_runway_save_result = workflow.get("long_runway_save_result")
    if long_runway_save_result and long_runway_save_result.get("success"):
        print(
            f"{LONG_RUNWAY_MODEL_DISPLAY}策略结果表: 已写入 {long_runway_save_result.get('saved_count')} 条, "
            f"类型={long_runway_save_result.get('strategy_type')}, 层级=中长期跟踪"
        )
