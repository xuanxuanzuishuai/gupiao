"""自适应风险覆盖包。

作用:
    收纳所有风险覆盖实现，根目录只保留
    analysis_gu_piao_adaptive_risk_overlay_model.py 作为外部入口，让系统
    认知上统一为“自适应风险覆盖”。

流程:
    overlay 负责把特殊池、事件和外部上下文转换为风险字段；
    adaptive_model 负责滚动回测这些风险分支并生成动态配置；
    外部脚本统一 import analysis_gu_piao_adaptive_risk_overlay_model。
"""
