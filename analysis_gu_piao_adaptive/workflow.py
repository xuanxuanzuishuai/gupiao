"""自适应选股总工作流。

作用:
    串联短线模型、长跑模型、回测健康验证和策略结果落库，是每日任务
    和手动分析调用的主调度层。它负责决定“本次运行哪些模块”，不直接
    实现具体打分或落库细节。

流程:
    先按日期参数加载共享历史数据，并尽量复用准备好的短线历史；
    然后运行短线推荐，必要时运行长跑推荐和回测；
    若要求落库，则执行实盘信号健康度、walk-forward 健康度和最终写表；
    返回统一 workflow 字典供 CLI、主流程或其他脚本消费。
"""

from .common import *
from .short_term import *
from .long_runway import *
from .persistence import *


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
            adaptive_backtest_context_trade_days = _adaptive_backtest_context_trade_days(ADAPTIVE_STRATEGY_TYPE)
            history_tail_trade_days = max(
                int(SHORT_TERM_LOOKBACK_TRADE_DAYS),
                int(adaptive_backtest_context_trade_days),
            )

    shared_history_columns = LONG_RUNWAY_FRAME_COLUMNS if include_long_runway else SHORT_TERM_FRAME_COLUMNS
    shared_history = _load_history(
        start_date=start_date,
        end_date=effective_end_date,
        tail_trade_days=history_tail_trade_days,
        columns=shared_history_columns,
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
        and not persist_strategy_result
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
        long_runway_history = None if long_runway_use_cache and start_date is None else shared_history
        long_runway_result = analysis_gu_piao_history_long_runway_model(
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
        long_runway_backtest_result = backtest_analysis_gu_piao_history_long_runway_model(
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
        adaptive_backtest_window_trade_days = _adaptive_backtest_health_lookback_trade_days(ADAPTIVE_STRATEGY_TYPE)
        adaptive_backtest_context_trade_days = _adaptive_backtest_context_trade_days(ADAPTIVE_STRATEGY_TYPE)
        if backtest_history is None or backtest_history.empty:
            backtest_history = _load_history(
                end_date=effective_end_date,
                tail_trade_days=adaptive_backtest_context_trade_days,
                columns=SHORT_TERM_FRAME_COLUMNS,
            )
            backtest_history_prepared = False
        backtest_trade_days = (
            int(backtest_history["last_data_date"].dropna().nunique())
            if backtest_history is not None and not backtest_history.empty
            else 0
        )
        _emit_runtime_status(
            f"{ADAPTIVE_HEALTH_MODEL_DISPLAY}walk-forward回测: model={SHORT_TERM_MODEL_DISPLAY}, 开始, trade_days={backtest_trade_days}, "
            f"window={adaptive_backtest_window_trade_days}, eval_step=1, timeout={ADAPTIVE_PERSIST_BACKTEST_TIMEOUT_SECONDS}s"
        )
        try:
            adaptive_backtest = _run_with_wall_timeout(
                ADAPTIVE_PERSIST_BACKTEST_TIMEOUT_SECONDS,
                backtest_analysis_gu_piao_history_adaptive_model,
                top_candidate_count=max(int(top_candidate_count), int(DAILY_ADAPTIVE_TOP_PICK_COUNT)),
                history=backtest_history,
                history_prepared=backtest_history_prepared,
                eval_step=1,
                eval_window_trade_days=adaptive_backtest_window_trade_days,
            )
            _emit_runtime_status(
                f"{ADAPTIVE_HEALTH_MODEL_DISPLAY}walk-forward回测: model={SHORT_TERM_MODEL_DISPLAY}, 完成, success={adaptive_backtest.get('success')}, "
                f"trade_days={adaptive_backtest.get('trade_days')}, window={adaptive_backtest_window_trade_days}, "
                f"eval_step={adaptive_backtest.get('eval_step')}"
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
                "eval_window_trade_days": adaptive_backtest_window_trade_days,
                "backtest": {},
            }
            _emit_runtime_status(
                f"{ADAPTIVE_HEALTH_MODEL_DISPLAY}walk-forward回测超时: model={SHORT_TERM_MODEL_DISPLAY}, "
                f"reason={adaptive_backtest['reason']}"
            )
        adaptive_backtest_health = _evaluate_adaptive_backtest_health(
            adaptive_backtest,
            strategy_type=ADAPTIVE_STRATEGY_TYPE,
        )
        adaptive_health = _combine_adaptive_health(adaptive_signal_health, adaptive_backtest_health)
        _emit_runtime_status(
            f"{ADAPTIVE_HEALTH_MODEL_DISPLAY}判定: model={SHORT_TERM_MODEL_DISPLAY}, enabled={adaptive_health.get('enabled')}, "
            f"mode={adaptive_health.get('mode')}, mode_label={adaptive_health.get('mode_label')}, "
            f"confidence_weight={adaptive_health.get('confidence_weight')}, "
            f"max_pick_ratio={adaptive_health.get('max_pick_ratio')}, "
            f"signal_health_available={adaptive_health.get('signal_health_available')}, "
            f"ignored_signal_reasons={adaptive_health.get('ignored_signal_failure_reasons')}, "
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
            workflow["long_runway_save_result"] = persist_long_runway_candidates_to_strategy_result(
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


__all__ = [name for name in globals() if not name.startswith("__")]
