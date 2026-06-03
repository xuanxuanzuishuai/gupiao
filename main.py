"""每日股票分析主流程。

作用:
    串联数据抓取、风险覆盖刷新、固定策略推荐和自适应策略推荐，是
    每日自动运行时最外层的编排入口。它只负责阶段顺序、前置校验和
    失败时清理结果，不直接实现具体策略算法。

流程:
    先运行 get_gu_piao_info 抓取行情并确认目标交易日；
    再刷新自适应风险覆盖表，确保策略使用最新风控结果；
    然后运行每日固定策略推荐；
    接着运行短线自适应模型工作流并按健康验证结果决定是否落库；
    最后输出行业热点、关注上升板块和龙头股报告。
"""

import get_gu_piao_info
import analysis_industry_hotspot
import analysis_intraday_focus
import analysis_gu_piao_adaptive_risk_overlay_model as risk_overlay
from analysis_gu_piao_data_print_result import analysis_gu_piao_data, clear_daily_strategy_results
from analysis_gu_piao_history_adaptive_model import (
    LONG_RUNWAY_MODEL_DISPLAY,
    SHORT_TERM_MODEL_DISPLAY,
    analysis_gu_piao_history_long_runway_model,
    clear_adaptive_strategy_results,
    clear_long_runway_strategy_results,
    persist_long_runway_candidates_to_strategy_result,
    run_adaptive_model_workflow,
)


def _to_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_trade_date_text(value):
    if value is None:
        return None

    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")

    text = str(value).strip()
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        return text[:10]
    return text or None


def run_stage(stage_name, callback, *args, **kwargs):
    print(f"[阶段开始] {stage_name}", flush=True)
    try:
        result = callback(*args, **kwargs)
    except Exception as error:
        print(f"[阶段失败] {stage_name}: {type(error).__name__}: {error}", flush=True)
        raise
    print(f"[阶段完成] {stage_name}", flush=True)
    return result


def print_long_runway_summary(result):
    if not (result or {}).get("success"):
        print(f"{LONG_RUNWAY_MODEL_DISPLAY}观察: success=False, reason={(result or {}).get('reason')}", flush=True)
        return

    print(
        f"{LONG_RUNWAY_MODEL_DISPLAY}观察: "
        f"trade_date={result.get('latest_trade_date')}, "
        f"cache_mode={result.get('cache_mode') or 'disabled'}, "
        f"top_candidates={len(result.get('top_candidates') or [])}, "
        f"stage_summary={result.get('stage_summary') or {}}, "
        "mode=中长期跟踪, 写入a_stock_strategy_result但不等同短线正式推荐",
        flush=True,
    )


def clear_managed_strategy_results(trade_date):
    trade_date = _normalize_trade_date_text(trade_date)
    if not trade_date:
        return

    clear_daily_strategy_results(trade_date)
    clear_adaptive_strategy_results(trade_date)
    clear_long_runway_strategy_results(trade_date)


def require_fetch_ready(fetch_summary):
    target_trade_date = _normalize_trade_date_text((fetch_summary or {}).get("target_trade_date"))
    if not target_trade_date:
        raise RuntimeError("抓取阶段没有拿到有效交易日，停止后续推荐，避免使用旧数据")

    history_saved_count = _to_int((fetch_summary or {}).get("history_saved_count"))
    if history_saved_count <= 0:
        clear_managed_strategy_results(target_trade_date)
        raise RuntimeError(
            f"交易日 {target_trade_date} 历史入库为0，停止后续推荐，避免基于缺失历史落库"
        )

    return target_trade_date


def require_risk_overlay_ready(risk_overlay_summary, target_trade_date):
    target_trade_date = _normalize_trade_date_text(target_trade_date)
    adaptive_model = (risk_overlay_summary or {}).get("adaptive_model") or {}
    if (risk_overlay_summary or {}).get("adaptive_enabled") and not (risk_overlay_summary or {}).get("adaptive_success"):
        clear_managed_strategy_results(target_trade_date)
        reason = adaptive_model.get("reason") or (risk_overlay_summary or {}).get("save_result", {}).get("reason")
        raise RuntimeError(
            f"交易日 {target_trade_date} {risk_overlay.DEFAULT_ADAPTIVE_RISK_MODEL_DISPLAY}失败: {reason or 'unknown'}"
        )

    if not (risk_overlay_summary or {}).get("success"):
        clear_managed_strategy_results(target_trade_date)
        reason = (risk_overlay_summary or {}).get("save_result", {}).get("reason")
        raise RuntimeError(f"交易日 {target_trade_date} 风险覆盖刷新失败: {reason or 'unknown'}")

    saved_count = _to_int((risk_overlay_summary or {}).get("saved_count"))
    if saved_count <= 0:
        clear_managed_strategy_results(target_trade_date)
        raise RuntimeError(f"交易日 {target_trade_date} 风险覆盖落库为0，停止推荐落库")

    overlay_trade_date = _normalize_trade_date_text((risk_overlay_summary or {}).get("trade_date"))
    if overlay_trade_date and overlay_trade_date != target_trade_date:
        clear_managed_strategy_results(target_trade_date)
        raise RuntimeError(
            f"风险覆盖交易日不一致: expected={target_trade_date}, actual={overlay_trade_date}"
        )


def build_intraday_focus_report(target_trade_date):
    target_trade_date = _normalize_trade_date_text(target_trade_date)
    result = analysis_intraday_focus.build_intraday_focus(
        trade_date=target_trade_date,
        industry_date=target_trade_date,
    )
    markdown = analysis_intraday_focus.render_markdown(result)
    output_path = analysis_intraday_focus.save_report(markdown, trade_date=result.get("target_trade_date"))
    action_counts = result.get("action_group_counts") or {}
    print(
        "盘中关注池报告: "
        f"output_path={output_path}, "
        f"candidate_count={result.get('candidate_count')}, "
        f"可操作={action_counts.get('可操作', 0)}, "
        f"观察={action_counts.get('观察', 0)}, "
        f"回避={action_counts.get('回避', 0)}, "
        f"放弃={action_counts.get('放弃', 0)}",
        flush=True,
    )
    return {
        "success": True,
        "output_path": str(output_path),
        "candidate_count": result.get("candidate_count"),
        "action_group_counts": action_counts,
        "target_trade_date": result.get("target_trade_date"),
        "industry_date": result.get("industry_date"),
    }


if __name__ == "__main__":
    fetch_summary = run_stage("抓取个股行情与历史入库", get_gu_piao_info.get_gu_piao_info)

    target_trade_date = require_fetch_ready(fetch_summary)
    # target_trade_date = '2026-05-15'
    risk_overlay_summary = run_stage(
        "刷新风险覆盖表",
        risk_overlay.refresh_latest_risk_overlay,
        trade_date=target_trade_date,
    )
    require_risk_overlay_ready(risk_overlay_summary, target_trade_date)
    print(
        "风险覆盖刷新: "
        f"trade_date={risk_overlay_summary.get('trade_date')}, "
        f"saved_count={risk_overlay_summary.get('saved_count')}",
        flush=True,
    )
    adaptive_risk_model = (risk_overlay_summary or {}).get("adaptive_model") or {}
    print(
        f"{risk_overlay.DEFAULT_ADAPTIVE_RISK_MODEL_DISPLAY}: "
        f"enabled={risk_overlay_summary.get('adaptive_enabled')}, "
        f"success={risk_overlay_summary.get('adaptive_success')}, "
        f"report_path={adaptive_risk_model.get('report_path') or '--'}",
        flush=True,
    )

    run_stage("每日固定策略推荐", analysis_gu_piao_data, expected_trade_date=target_trade_date)

    run_stage(
        f"每日{SHORT_TERM_MODEL_DISPLAY}推荐",
        run_adaptive_model_workflow,
        target_trade_date=target_trade_date,
        top_candidate_count=3,
        include_long_runway=False,
        include_backtest=False,
        persist_strategy_result=True,
    )
    industry_hotspot_summary = run_stage(
        "行业热点资金分析",
        analysis_industry_hotspot.run_industry_hotspot_analysis,
        end_date=target_trade_date,
    )
    industry_output_paths = (industry_hotspot_summary or {}).get("output_paths") or {}
    print(
        "行业热点资金分析: "
        f"report_path={industry_output_paths.get('report_path') or '--'}, "
        f"board_csv={industry_output_paths.get('board_csv_path') or '--'}, "
        f"leader_csv={industry_output_paths.get('leader_csv_path') or '--'}",
        flush=True,
    )
    intraday_focus_summary = run_stage(
        "盘中关注池报告",
        build_intraday_focus_report,
        target_trade_date,
    )
    print(
        f"盘中关注池最新报告: {intraday_focus_summary.get('output_path') or '--'}",
        flush=True,
    )
    # long_runway_summary = run_stage(
    #     f"每日{LONG_RUNWAY_MODEL_DISPLAY}中长期跟踪",
    #     analysis_gu_piao_history_long_runway_model,
    #     end_date=target_trade_date,
    #     top_candidate_count=10,
    # )
    # print_long_runway_summary(long_runway_summary)
    # long_runway_save_result = persist_long_runway_candidates_to_strategy_result(
    #     long_runway_summary,
    #     trade_date=target_trade_date,
    # )
    # print(
    #     f"{LONG_RUNWAY_MODEL_DISPLAY}策略结果表: "
    #     f"success={long_runway_save_result.get('success')}, "
    #     f"trade_date={long_runway_save_result.get('trade_date')}, "
    #     f"saved_count={long_runway_save_result.get('saved_count')}, "
    #     f"strategy_type={long_runway_save_result.get('strategy_type')}, "
    #     "tier=中长期跟踪",
    #     flush=True,
    # )
