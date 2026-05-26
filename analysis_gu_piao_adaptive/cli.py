"""自适应选股命令行入口。

作用:
    解析命令行参数，调用 workflow.run_adaptive_model_workflow，并把短线
    推荐、长跑跟踪、回测和落库结果打印成便于人工阅读的控制台输出。

流程:
    main() 读取参数，决定是否执行长跑模型、回测和落库；
    workflow 返回结构化结果后，本模块只负责格式化展示，不参与打分、
    风控、健康验证或数据库写入逻辑。
"""

import argparse

from .common import *
from .workflow import *


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


def _print_long_runway_backtest_section(result):
    if not result or not result.get("success"):
        print(f"{LONG_RUNWAY_MODEL_DISPLAY}回测: 暂无有效回测结果")
        return

    print(f"{LONG_RUNWAY_MODEL_DISPLAY}回测:")
    print(
        f"  回测交易日: {result.get('trade_days')}, "
        f"调仓步长: 每{result.get('rebalance_trade_days')}个交易日"
    )
    for horizon_label, metrics in result.get("backtest", {}).items():
        print(
            f"  {horizon_label}: "
            f"universe_return={metrics.get('avg_universe_return')}%, "
            f"top_return={metrics.get('avg_top_return')}%, "
            f"universe_win_rate={metrics.get('avg_universe_win_rate')}%, "
            f"top_win_rate={metrics.get('avg_top_win_rate')}%"
        )

        stage_stats = metrics.get("stage_return_stats") or {}
        if stage_stats:
            top_stages = list(stage_stats.items())[:3]
            stage_text = "；".join(
                f"{stage}:avg={stat.get('avg_return')}%,win={stat.get('win_rate')}%,count={stat.get('count')}"
                for stage, stat in top_stages
            )
            print(f"    阶段回测: {stage_text}")

        conviction_stats = metrics.get("conviction_return_stats") or {}
        if conviction_stats:
            top_convictions = list(conviction_stats.items())[:3]
            conviction_text = "；".join(
                f"{conviction}:avg={stat.get('avg_return')}%,win={stat.get('win_rate')}%,count={stat.get('count')}"
                for conviction, stat in top_convictions
            )
            print(f"    信念回测: {conviction_text}")


def _print_long_runway_section(result):
    if not result or not result.get("success"):
        print(f"{LONG_RUNWAY_MODEL_DISPLAY}: 暂无有效结果")
        return

    print(f"{LONG_RUNWAY_MODEL_DISPLAY}:")
    print(f"  模型版本: {result.get('long_runway_model_version')}")
    print(
        f"  市场环境: {result.get('market_env')}, "
        f"60日均涨幅: {result.get('market_ret_60d')}, "
        f"60日广度: {result.get('market_breadth_60d')}, "
        f"120日均涨幅: {result.get('market_ret_120d')}, "
        f"120日广度: {result.get('market_breadth_120d')}"
    )
    print(
        f"  历史样本: {result.get('trade_days')} 日, "
        f"{result.get('history_rows')} 条, "
        f"最新交易日: {result.get('latest_trade_date')}"
    )
    print(f"  缓存: mode={result.get('cache_mode')}, path={result.get('cache_path')}")
    print(f"  模型结论: {result.get('long_runway_note')}")
    if result.get("stage_summary"):
        stage_text = "；".join(f"{stage}:{count}" for stage, count in list(result["stage_summary"].items())[:5])
        print(f"  阶段分布: {stage_text}")

    focus_candidate = result.get("focus_candidate")
    if focus_candidate:
        print("  单票长跑判断:")
        print(
            f"    {focus_candidate.get('stock_code')} {focus_candidate.get('stock_name')} "
            f"stage={focus_candidate.get('runway_stage_label')} score={focus_candidate.get('runway_total_score')} "
            f"quality={focus_candidate.get('runway_quality_score')} risk={focus_candidate.get('runway_risk_score')} "
            f"action={focus_candidate.get('runway_action')}"
        )
        print(
            f"    hold={focus_candidate.get('runway_hold_period')}, expected={focus_candidate.get('runway_expected_return')}, "
            f"stop={focus_candidate.get('runway_stop_price')}, position={focus_candidate.get('runway_position_hint')}, "
            f"conviction={focus_candidate.get('runway_conviction')}"
        )
        print(f"    detail={focus_candidate.get('runway_stage_detail')}")
        print(f"    conclusion={focus_candidate.get('runway_conclusion')}")
        print(f"    reason={focus_candidate.get('reason')}")

    print("  当前长跑候选:")
    for index, item in enumerate(result.get("top_candidates", [])[:5], start=1):
        print(
            f"    {index}. {item.get('stock_code')} {item.get('stock_name')} "
            f"score={item.get('runway_total_score')} quality={item.get('runway_quality_score')} "
            f"stage={item.get('runway_stage_label')} conviction={item.get('runway_conviction')} "
            f"hold={item.get('runway_hold_period')} action={item.get('runway_action')} "
            f"reason={item.get('reason')}"
        )


def main():
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
    long_runway = workflow.get("long_runway_recommendation") or {}
    long_term = workflow.get("long_term_backtest")
    long_runway_backtest = workflow.get("long_runway_backtest")

    if short_term.get("success"):
        _print_short_term_section(short_term)
    else:
        print(f"{SHORT_TERM_MODEL_DISPLAY}失败: {short_term.get('reason')}")

    if not args.include_long_runway:
        print(f"{LONG_RUNWAY_MODEL_DISPLAY}: 已跳过")
    elif long_runway.get("success"):
        _print_long_runway_section(long_runway)
    else:
        print(f"{LONG_RUNWAY_MODEL_DISPLAY}失败: {long_runway.get('reason')}")

    if should_include_backtest:
        _print_long_term_section(long_term)
        if args.include_long_runway:
            _print_long_runway_backtest_section(long_runway_backtest)
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


if __name__ == "__main__":
    main()

__all__ = [name for name in globals() if not name.startswith("__")]
