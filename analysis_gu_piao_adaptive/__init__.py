"""自适应选股模型包。

作用:
    汇总短线自适应模型、长跑潜力模型、通用评分、落库和工作流入口。
    外部通常不直接依赖包内文件，而是继续通过
    analysis_gu_piao_history_adaptive_model.py 这个兼容入口调用。

流程:
    common 提供数据和特征基础能力，scoring 提供画像和打分能力；
    short_term 与 long_runway 分别完成两类策略分析；
    persistence 负责策略结果落库，workflow 负责整体调度。
"""

from .common import *
from .scoring import *
from .short_term import *
from .long_runway import *
from .persistence import *
from .workflow import *
