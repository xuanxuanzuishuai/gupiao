"""自适应策略结果落库层。

作用:
    负责把短线自适应候选和长跑跟踪候选写入 a_stock_strategy_result，
    并在写入前应用健康度策略、最终风险覆盖确认、去重和数量限制。
    它只处理持久化规则，不重新训练模型。

流程:
    短线落库先读取 workflow 里的推荐结果和健康验证结果；
    若健康度未通过则拒绝写入，通过后按政策调整候选分数和仓位提示；
    最终再过一遍风险覆盖硬拦截和降级过滤；
    长跑落库则按中长期跟踪层级生成完整 strategy_note 后写表。
"""

from .common import *


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


def clear_adaptive_strategy_results(trade_date):
    _clear_adaptive_strategy_rows(trade_date)


def persist_adaptive_candidates_to_strategy_result(
    workflow,
    max_picks=DAILY_ADAPTIVE_TOP_PICK_COUNT,
    health_check=None,
):
    short_term = (workflow or {}).get("short_term_recommendation") or {}
    trade_date = short_term.get("latest_trade_date")

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
                _emit_runtime_status(f"{SHORT_TERM_MODEL_DISPLAY}落库事件降级剔除: {downgrade_text}")

            final_frame = final_frame[~block_mask & ~downgrade_mask].copy()
            if final_frame.empty and block_mask.any():
                final_filter_reason = "candidate_empty_after_final_hard_block"
            elif final_frame.empty and downgrade_mask.any():
                final_filter_reason = "candidate_empty_after_final_downgrade"
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
        _clear_adaptive_strategy_rows(trade_date)
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
    _clear_adaptive_strategy_rows(trade_date)
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


def _clear_long_runway_strategy_rows(trade_date):
    trade_date = _to_date_text(trade_date)
    if not trade_date:
        return
    func.executeDelete("a_stock_strategy_result", {"trade_date": trade_date, "strategy_type": LONG_RUNWAY_STRATEGY_TYPE})


def clear_long_runway_strategy_results(trade_date):
    _clear_long_runway_strategy_rows(trade_date)


def _json_compact(value):
    if value is None:
        return None
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def _build_long_runway_strategy_note(candidate, summary, rank, total):
    stage_scores = _json_compact(candidate.get("runway_stage_scores"))
    parts = [
        "层级:中长期跟踪",
        f"模型:{LONG_RUNWAY_MODEL_DISPLAY}({LONG_RUNWAY_MODEL_VERSION})",
        "性质:历史长期赢家画像匹配，不等同每日短线正式推荐",
        f"排名:{rank}/{total}",
        f"市场:{summary.get('market_env') or '--'}",
        f"样本:{summary.get('sample_start') or '--'}~{summary.get('sample_end') or '--'}",
        (
            f"分数:总分{candidate.get('runway_total_score')}, "
            f"画像{candidate.get('adaptive_score')}, "
            f"质量{candidate.get('runway_quality_score')}, "
            f"风险{candidate.get('runway_risk_score')}"
        ),
        (
            f"周期:60日{candidate.get('score_60d')}, "
            f"120日{candidate.get('score_120d')}, "
            f"252日{candidate.get('score_252d')}, "
            f"主导{candidate.get('dominant_horizon')}日/{candidate.get('dominant_family')}"
        ),
        (
            f"阶段:{candidate.get('runway_stage_label') or '--'}, "
            f"信念:{candidate.get('runway_conviction') or '--'}, "
            f"动作:{candidate.get('runway_action') or '--'}, "
            f"仓位:{candidate.get('runway_position_hint') or '--'}"
        ),
        (
            f"风险覆盖:{candidate.get('risk_overlay_labels') or '无明显特殊池风险'}, "
            f"覆盖动作:{candidate.get('risk_overlay_action') or '--'}"
        ),
        (
            f"跟踪:{candidate.get('runway_hold_period') or '--'}, "
            f"预期:{candidate.get('runway_expected_return') or '--'}, "
            f"防守:{candidate.get('runway_stop_price') or '--'}"
        ),
        f"结论:{candidate.get('runway_conclusion') or '--'}",
        f"放弃/退出:{candidate.get('runway_stop_rule') or '--'}",
        f"重新跟踪:{candidate.get('runway_reenter_rule') or '--'}",
        f"原因:{candidate.get('reason') or '--'}",
    ]
    if stage_scores:
        parts.append(f"阶段分:{stage_scores}")

    strategy_note = "；".join(str(part) for part in parts if part is not None)
    if len(strategy_note) > ADAPTIVE_NOTE_MAX_LENGTH:
        strategy_note = strategy_note[: ADAPTIVE_NOTE_MAX_LENGTH - 3] + "..."
    return strategy_note


def _build_long_runway_strategy_record(candidate, summary, trade_date, rank, total):
    change_30d_value = _normalize_scalar(candidate.get("ret_20d"))
    if change_30d_value is None:
        change_30d_value = _normalize_scalar(candidate.get("change_30d"))

    return {
        "trade_date": trade_date,
        "strategy_type": LONG_RUNWAY_STRATEGY_TYPE,
        "stock_code": _normalize_scalar(candidate.get("stock_code")),
        "stock_name": _normalize_scalar(candidate.get("stock_name")),
        "today_change": _normalize_scalar(candidate.get("today_change")),
        "industry": _normalize_scalar(candidate.get("industry_1")) or "",
        "change_30d": change_30d_value,
        "vr_today": _normalize_scalar(candidate.get("vr_today")),
        "vr_30d": _normalize_scalar(candidate.get("vr_30d")),
        "today_amp": _normalize_scalar(candidate.get("today_amp")),
        "amp_30d": _normalize_scalar(candidate.get("amp_30d")),
        "stock_rank": _normalize_scalar(candidate.get("stock_rank")),
        "today_amount": _normalize_scalar(candidate.get("today_amount")),
        "turnover_rate": _normalize_scalar(candidate.get("turnover_rate")),
        "strategy_note": _build_long_runway_strategy_note(candidate, summary, rank, total),
    }


def persist_long_runway_candidates_to_strategy_result(summary, trade_date=None):
    summary = summary or {}
    effective_trade_date = _to_date_text(trade_date or summary.get("latest_trade_date"))
    _clear_long_runway_strategy_rows(effective_trade_date)

    if not summary.get("success"):
        _emit_runtime_status(
            f"{LONG_RUNWAY_MODEL_DISPLAY}中长期跟踪落库跳过: "
            f"trade_date={effective_trade_date}, reason={summary.get('reason') or 'long_runway_failed'}"
        )
        return {
            "success": False,
            "reason": summary.get("reason") or "long_runway_failed",
            "saved_count": 0,
            "trade_date": effective_trade_date,
            "strategy_type": LONG_RUNWAY_STRATEGY_TYPE,
        }

    if not effective_trade_date:
        return {
            "success": False,
            "reason": "latest_trade_date_missing",
            "saved_count": 0,
            "strategy_type": LONG_RUNWAY_STRATEGY_TYPE,
        }

    top_candidates = summary.get("top_candidates") or []
    if not top_candidates:
        _emit_runtime_status(
            f"{LONG_RUNWAY_MODEL_DISPLAY}中长期跟踪落库完成: "
            f"trade_date={effective_trade_date}, saved_count=0, reason=candidate_empty"
        )
        return {
            "success": True,
            "reason": "candidate_empty",
            "saved_count": 0,
            "trade_date": effective_trade_date,
            "strategy_type": LONG_RUNWAY_STRATEGY_TYPE,
        }

    selected_candidates = []
    seen_codes = set()
    for candidate in top_candidates:
        stock_code = _normalize_scalar(candidate.get("stock_code"))
        if not stock_code or stock_code in seen_codes:
            continue
        seen_codes.add(stock_code)
        selected_candidates.append(candidate)

    saved_count = 0
    total_count = len(selected_candidates)
    for index, candidate in enumerate(selected_candidates, start=1):
        insert_result = func.executeInsert(
            "a_stock_strategy_result",
            _build_long_runway_strategy_record(
                candidate,
                summary,
                effective_trade_date,
                rank=index,
                total=total_count,
            ),
        )
        if insert_result.get("insertResult"):
            saved_count += 1

    _emit_runtime_status(
        f"{LONG_RUNWAY_MODEL_DISPLAY}中长期跟踪写入完成: "
        f"trade_date={effective_trade_date}, saved_count={saved_count}, "
        f"strategy_type={LONG_RUNWAY_STRATEGY_TYPE}, 层级=中长期跟踪"
    )
    return {
        "success": True,
        "reason": None if saved_count else "insert_failed_or_empty",
        "saved_count": saved_count,
        "trade_date": effective_trade_date,
        "strategy_type": LONG_RUNWAY_STRATEGY_TYPE,
        "tier": "中长期跟踪",
    }


__all__ = [name for name in globals() if not name.startswith("__")]
