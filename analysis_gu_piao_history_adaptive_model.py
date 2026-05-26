"""自适应选股模型外部入口。

作用:
    作为根目录的兼容门面，对外统一暴露短线自适应选股模型、长跑潜力
    模型、回测、健康验证和策略结果落库入口。包内实现已经拆到
    analysis_gu_piao_adaptive/，但旧调用方仍可 import 本文件。

流程:
    从 common、scoring、short_term、long_runway、persistence、
    workflow 和 cli 重新导出原有函数与常量；
    其他脚本继续调用 run_adaptive_model_workflow 或具体模型函数；
    直接命令行运行本文件时，转到 analysis_gu_piao_adaptive.cli.main。
"""

from analysis_gu_piao_adaptive.common import *
from analysis_gu_piao_adaptive.scoring import *
from analysis_gu_piao_adaptive.short_term import *
from analysis_gu_piao_adaptive.long_runway import *
from analysis_gu_piao_adaptive.persistence import *
from analysis_gu_piao_adaptive.workflow import *
from analysis_gu_piao_adaptive.cli import (
    _print_long_runway_backtest_section,
    _print_long_runway_section,
    _print_long_term_section,
    _print_short_term_section,
    main,
)


if __name__ == "__main__":
    main()
