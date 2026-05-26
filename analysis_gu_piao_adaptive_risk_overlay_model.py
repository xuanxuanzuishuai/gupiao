"""自适应风险覆盖外部入口。

作用:
    作为根目录唯一风险入口，对外统一暴露自适应风险覆盖刷新能力，以及
    风险分支滚动回测和动态规则校准能力。其他脚本统一把它 import 成
    risk_overlay 使用，避免再直接依赖包内实现。

流程:
    从 analysis_gu_piao_adaptive_risk.overlay 导出风险覆盖刷新、候选
    风控、市场分层和落库相关 API；
    从 analysis_gu_piao_adaptive_risk.adaptive_model 导出滚动回测、
    Markdown 报告和动态配置生成 API；
    直接命令行运行本文件时，默认执行风险覆盖刷新入口 main()。
"""

from analysis_gu_piao_adaptive_risk import adaptive_model as _adaptive_model
from analysis_gu_piao_adaptive_risk import overlay as _overlay


def _export_module(module, overwrite=False):
    for name in dir(module):
        if name.startswith("__"):
            continue
        if overwrite or name not in globals():
            globals()[name] = getattr(module, name)


_export_module(_overlay, overwrite=True)
_export_module(_adaptive_model, overwrite=False)

run_adaptive_risk_overlay_model = _adaptive_model.run_adaptive_risk_overlay_model
backtest_risk_overlay_branches = _adaptive_model.backtest_risk_overlay_branches
format_markdown_report = _adaptive_model.format_markdown_report
adaptive_model_main = _adaptive_model.main
main = _overlay.main


if __name__ == "__main__":
    main()
