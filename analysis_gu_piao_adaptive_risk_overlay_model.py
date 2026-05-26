import argparse
import json
from pathlib import Path

import pandas as pd

import analysis_gu_piao_risk_overlay as risk_overlay
import func


DEFAULT_TAIL_TRADE_DAYS = 260
DEFAULT_HOLDING_DAYS = (1, 3, 5, 10, 20)
DEFAULT_ENTRY_OFFSET_DAYS = 1
DEFAULT_FEE_BPS = 3
DEFAULT_SLIPPAGE_BPS = 5
ADAPTIVE_MODEL_VERSION = "adaptive_risk_overlay_v1"
ADAPTIVE_MODEL_LABEL = "自适应风险覆盖模型"
ADAPTIVE_MODEL_DISPLAY = ADAPTIVE_MODEL_LABEL
DEFAULT_FOCUS_HOLD_DAYS = 10

ADAPTIVE_RULE_DEFAULTS = {
    "recent_ipo": {
        "source_branch": "recent_ipo_0_20d",
        "label": "上市20日内新股",
        "base_action": "block",
        "base_score": 8.0,
        "min_samples": 300,
    },
    "sub_new": {
        "source_branch": "sub_new_21_120d",
        "label": "次新股",
        "base_action": "downgrade",
        "base_score": 3.0,
        "min_samples": 500,
    },
    "limit_chain": {
        "source_branch": "limit_chain",
        "label": "连续涨停/连板",
        "base_action": "block",
        "base_score": 7.0,
        "min_samples": 300,
    },
    "overheated": {
        "source_branch": "overheated",
        "label": "短线涨幅过热",
        "base_action": "block",
        "base_score": 6.0,
        "min_samples": 300,
    },
    "high_volatility": {
        "source_branch": "high_volatility",
        "label": "高波动高换手",
        "base_action": "downgrade",
        "base_score": 4.0,
        "min_samples": 1000,
    },
    "turnover_extreme_35": {
        "source_branch": "turnover_extreme_35",
        "label": "换手极高",
        "base_action": "downgrade",
        "base_score": 2.0,
        "min_samples": 200,
    },
}

BRANCH_METADATA = {
    "market_strong": {
        "label": "强市",
        "group": "市场状态",
        "source": "classify_market_regime: change>2 且 breadth>=55",
        "expected_action": "描述性分层",
        "theory": "市场动量与市场宽度同步改善时，趋势延续概率通常更高，但短线拥挤后也可能出现回撤。",
    },
    "market_neutral": {
        "label": "震荡市",
        "group": "市场状态",
        "source": "classify_market_regime: change>-1.5 且 breadth>=43",
        "expected_action": "描述性分层",
        "theory": "宽度尚未失守时适合降低仓位弹性、保留精选机会。",
    },
    "market_weak": {
        "label": "弱市",
        "group": "市场状态",
        "source": "classify_market_regime: 其他",
        "expected_action": "描述性分层",
        "theory": "弱市中个股左尾风险通常上升，但极端弱势后的均值回归可能提高短期反弹收益。",
    },
    "board_10pct_main": {
        "label": "主板/默认10%涨跌幅",
        "group": "交易制度",
        "source": "_board_limit_pct: default 10",
        "expected_action": "描述性分层",
        "theory": "10%涨跌幅制度压缩单日波动，连续涨停时更容易形成排队和流动性约束。",
    },
    "board_20pct_chinext_star": {
        "label": "创业板/科创板20%涨跌幅",
        "group": "交易制度",
        "source": "_board_limit_pct: 300/301/688/689",
        "expected_action": "描述性分层",
        "theory": "20%涨跌幅扩大单日价格发现空间，收益弹性和左尾波动通常同时更高。",
    },
    "board_30pct_bse": {
        "label": "北交所30%涨跌幅",
        "group": "交易制度",
        "source": "_board_limit_pct: 8/4/9",
        "expected_action": "描述性分层",
        "theory": "30%涨跌幅叠加流动性差异，短线收益分布更容易呈厚尾。",
    },
    "recent_ipo_0_20d": {
        "label": "上市20日内新股",
        "group": "新股/次新",
        "source": "_resolve_row_overlay: probable_new_listing and listing_trade_days<=20",
        "expected_action": "拦截",
        "theory": "新股早期缺少稳定筹码结构，估值锚弱，IPO/新股文献长期显示早期交易阶段回撤和换手风险更高。",
    },
    "sub_new_21_120d": {
        "label": "21-120日次新股",
        "group": "新股/次新",
        "source": "_resolve_row_overlay: 20<listing_trade_days<=120",
        "expected_action": "降级观察",
        "theory": "次新股仍处于估值再定价和筹码换手阶段，适合等待承接确认。",
    },
    "consecutive_limit_up_ge_2": {
        "label": "连续涨停>=2",
        "group": "连板/过热",
        "source": "_resolve_row_overlay: consecutive_limit_up_days>=2",
        "expected_action": "拦截",
        "theory": "连续涨停后的短线收益高度依赖流动性和情绪接力，断板时左尾风险陡增。",
    },
    "limit_up_days_5d_ge_2": {
        "label": "5日内涨停>=2",
        "group": "连板/过热",
        "source": "_resolve_row_overlay: limit_up_days_5d>=2",
        "expected_action": "拦截",
        "theory": "短期多次涨停代表拥挤交易和情绪加速，后续更容易出现波动放大。",
    },
    "limit_chain": {
        "label": "连续涨停/连板综合",
        "group": "连板/过热",
        "source": "_resolve_row_overlay: consecutive_limit_up_days>=2 or limit_up_days_5d>=2",
        "expected_action": "拦截",
        "theory": "连板综合分支聚焦情绪交易最拥挤区域，目标是避开无法合理成交和断板回撤。",
    },
    "overheated_5d_ge_35": {
        "label": "5日涨幅>=35%",
        "group": "连板/过热",
        "source": "_resolve_row_overlay: change_5d>=35",
        "expected_action": "拦截",
        "theory": "极短期涨幅过大后，追高收益需要更高胜率补偿，实盘更容易遭遇均值回归。",
    },
    "overheated_10d_ge_60": {
        "label": "10日涨幅>=60%",
        "group": "连板/过热",
        "source": "_resolve_row_overlay: change_10d>=60",
        "expected_action": "拦截",
        "theory": "10日维度过热通常意味着风险收益比恶化，尤其在成交拥挤时。",
    },
    "overheated_20d_ge_100": {
        "label": "20日涨幅>=100%",
        "group": "连板/过热",
        "source": "_resolve_row_overlay: change_20d>=100",
        "expected_action": "拦截",
        "theory": "翻倍级短期涨幅往往伴随估值和情绪双重透支，除非有基本面突变支撑。",
    },
    "overheated_limit_up_high_vol": {
        "label": "涨停附近且高波动",
        "group": "连板/过热",
        "source": "_resolve_row_overlay: today_change>=limit_pct-0.4 and is_high_volatility",
        "expected_action": "拦截",
        "theory": "涨停附近叠加高波动/高换手，常见于分歧加大阶段，次日承接不确定性高。",
    },
    "overheated": {
        "label": "短线过热综合",
        "group": "连板/过热",
        "source": "_resolve_row_overlay: is_overheated",
        "expected_action": "拦截",
        "theory": "综合过热分支用于过滤极端涨幅或涨停高波动样本，控制追高左尾。",
    },
    "today_amp_ge_12": {
        "label": "当日振幅>=12%",
        "group": "波动/换手",
        "source": "_resolve_row_overlay: today_amp>=12",
        "expected_action": "降级观察",
        "theory": "单日振幅过大说明多空分歧和成交冲击上升，后续收益分布通常更厚尾。",
    },
    "amp_5d_ge_9": {
        "label": "5日均振幅>=9%",
        "group": "波动/换手",
        "source": "_resolve_row_overlay: amp_5d>=9",
        "expected_action": "降级观察",
        "theory": "持续高振幅代表状态不稳，适合用承接确认替代直接加分。",
    },
    "amp_10d_ge_8": {
        "label": "10日均振幅>=8%",
        "group": "波动/换手",
        "source": "_resolve_row_overlay: amp_10d>=8",
        "expected_action": "降级观察",
        "theory": "更长窗口高振幅说明波动风险不是单日偶发，仓位和入场价格容错要求更高。",
    },
    "turnover_ge_25": {
        "label": "换手率>=25%",
        "group": "波动/换手",
        "source": "_resolve_row_overlay: turnover_rate>=25",
        "expected_action": "降级观察",
        "theory": "高换手可代表资金关注，也可代表筹码松动；单独不应硬杀，但需要承接确认。",
    },
    "high_volatility": {
        "label": "高波动/高换手综合",
        "group": "波动/换手",
        "source": "_resolve_row_overlay: is_high_volatility",
        "expected_action": "降级观察",
        "theory": "高波动股票在大量实证研究中长期风险补偿不足，短线也容易放大左尾。",
    },
    "turnover_extreme_35": {
        "label": "极高换手>=35%",
        "group": "波动/换手",
        "source": "_resolve_row_overlay: turnover_rate>=35",
        "expected_action": "降级观察",
        "theory": "极端换手同时包含主升分歧和出货风险，样本通常需要单独检验，不能只凭方向判断。",
    },
    "normal_pool": {
        "label": "普通股票池",
        "group": "综合处置",
        "source": "_resolve_row_overlay: not special_pools",
        "expected_action": "继续评估",
        "theory": "未触发特殊风险时，交还主策略模型排序，不额外惩罚。",
    },
    "block_formal": {
        "label": "正式推荐拦截",
        "group": "综合处置",
        "source": "_resolve_row_overlay: block_formal",
        "expected_action": "拦截",
        "theory": "把新股、连板、过热等高左尾分支统一挡在正式推荐外，强调风险预算优先。",
    },
    "downgrade_observe": {
        "label": "降级观察",
        "group": "综合处置",
        "source": "_resolve_row_overlay: downgrade_to_observation",
        "expected_action": "降级观察",
        "theory": "对高波动和次新等非硬风险样本降权而非删除，降低误杀主升机会。",
    },
    "continue_evaluate": {
        "label": "继续普通评估",
        "group": "综合处置",
        "source": "_resolve_row_overlay: else action",
        "expected_action": "继续评估",
        "theory": "没有触发硬风险或观察风险时，风险覆盖层不应干扰主模型。",
    },
    "risk_level_high": {
        "label": "风险等级高",
        "group": "综合处置",
        "source": "_resolve_row_overlay: block_formal or score>=8",
        "expected_action": "拦截",
        "theory": "高风险等级应对应更差的后续收益或更深左尾，否则权重需要下调。",
    },
    "risk_level_mid": {
        "label": "风险等级中",
        "group": "综合处置",
        "source": "_resolve_row_overlay: score>=4",
        "expected_action": "降级观察",
        "theory": "中风险等级重点看左尾和胜率是否恶化，用于决定降权强度。",
    },
    "risk_level_low": {
        "label": "风险等级低",
        "group": "综合处置",
        "source": "_resolve_row_overlay: default low",
        "expected_action": "继续评估",
        "theory": "低风险池应承担主策略的主要候选来源。",
    },
}

EXTERNAL_BRANCH_ASSESSMENT = [
    {
        "branch": "fundamental_negative_profit",
        "label": "最新报告期净利润为负",
        "source": "_extract_latest_fundamental: netprofit<0",
        "theory": "盈利为负会削弱估值锚和融资能力，通常应降低正式推荐优先级。",
        "backtest_status": "缺少point-in-time财报快照，当前不能严肃回测。",
    },
    {
        "branch": "fundamental_deduct_negative",
        "label": "扣非净利润为负",
        "source": "_extract_latest_fundamental: deduct_netprofit<0",
        "theory": "扣非亏损比一次性损益更能反映主营质量不足。",
        "backtest_status": "缺少point-in-time财报快照，当前不能严肃回测。",
    },
    {
        "branch": "fundamental_netprofit_yoy_down",
        "label": "净利同比<-30%",
        "source": "_extract_latest_fundamental: netprofit_yoy<-30",
        "theory": "盈利大幅下滑通常对应基本面预期下修，适合风险加分。",
        "backtest_status": "缺少point-in-time财报快照，当前不能严肃回测。",
    },
    {
        "branch": "fundamental_high_debt_or_negative_roe",
        "label": "高负债或ROE为负",
        "source": "_extract_latest_fundamental: debt_ratio>=75 or roe<0",
        "theory": "资产负债率偏高和ROE为负会提高财务弹性风险。",
        "backtest_status": "缺少point-in-time财报快照，当前不能严肃回测。",
    },
    {
        "branch": "event_risk_keywords",
        "label": "公告风险关键词",
        "source": "_extract_recent_events: 异常波动/停牌核查/减持/立案/处罚/监管/问询/诉讼/冻结/风险提示/预亏/亏损",
        "theory": "监管、诉讼、处罚、减持和亏损公告属于事件冲击，短期通常提高波动和跳空风险。",
        "backtest_status": "缺少历史公告快照和发布日期对齐表，当前不能严肃回测。",
    },
    {
        "branch": "event_positive_keywords",
        "label": "公告正面关键词",
        "source": "_extract_recent_events: 预增/扭亏/中标/签订/回购/增持/分红",
        "theory": "正面事件可能改善预期，但也需要区分是否已被价格提前反映。",
        "backtest_status": "缺少历史公告快照和发布日期对齐表，当前不能严肃回测。",
    },
    {
        "branch": "lhb_active",
        "label": "龙虎榜活跃/高频",
        "source": "_extract_lhb_context: recent_count>=2, >=5, >=8",
        "theory": "龙虎榜高频代表交易拥挤和资金博弈增强，适合提示波动风险。",
        "backtest_status": "缺少历史龙虎榜明细快照表，当前不能严肃回测。",
    },
    {
        "branch": "unlock_pressure",
        "label": "限售股解禁压力",
        "source": "_extract_unlock_context: ratio>=3, >=8, >=20且30日内",
        "theory": "大比例近端解禁带来潜在供给冲击，文献通常观察到解禁窗口附近异常收益承压。",
        "backtest_status": "缺少历史解禁计划point-in-time快照，当前不能严肃回测。",
    },
    {
        "branch": "external_merge",
        "label": "外部覆盖合并",
        "source": "_merge_external_context: external_score>=3降级，block_formal硬拦截",
        "theory": "把财务、事件、龙虎榜、解禁统一折算为风险预算，避免单一行情指标漏掉非价格风险。",
        "backtest_status": "依赖以上外部快照，当前只能做实时校验，不能做历史因果回测。",
    },
]


def _round_or_none(value, digits=4):
    if value is None or pd.isna(value):
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _load_backtest_history(tail_trade_days, end_date, max_shift):
    load_trade_days = int(tail_trade_days) + int(max_shift) + 5
    return risk_overlay.load_recent_history(
        tail_trade_days=load_trade_days,
        end_date=end_date,
    )


def _prepare_history(history, holding_days, entry_offset_days, fee_bps, slippage_bps):
    if history is None or history.empty:
        return pd.DataFrame()

    frame = history.copy()
    frame["last_data_date"] = pd.to_datetime(frame["last_data_date"], errors="coerce")
    frame["stock_code"] = frame["stock_code"].apply(risk_overlay._normalize_stock_code)
    frame = frame[frame["last_data_date"].notna() & frame["stock_code"].notna()].copy()
    if frame.empty:
        return frame

    numeric_columns = [
        "latest_price",
        "today_change",
        "change_5d",
        "change_10d",
        "change_20d",
        "today_amp",
        "amp_5d",
        "amp_10d",
        "amp_20d",
        "turnover_rate",
        "today_amount",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")

    frame = frame.sort_values(["stock_code", "last_data_date"]).reset_index(drop=True)
    grouped = frame.groupby("stock_code", sort=False)
    entry_shift = int(entry_offset_days)
    frame["entry_close"] = grouped["latest_price"].shift(-entry_shift)
    round_trip_cost_pct = (float(fee_bps) + float(slippage_bps)) * 2 / 100
    for hold_days in holding_days:
        exit_shift = entry_shift + int(hold_days)
        gross_return = grouped["latest_price"].shift(-exit_shift) / frame["entry_close"] * 100 - 100
        frame[f"return_{hold_days}d"] = gross_return - round_trip_cost_pct

    return frame


def _add_market_regime(frame):
    if frame.empty:
        return frame

    regime = (
        frame.groupby("last_data_date", sort=True)
        .agg(
            market_change_5d=("change_5d", "mean"),
            market_breadth_5d=("change_5d", lambda series: float(series.gt(0).mean() * 100)),
            universe_count=("stock_code", "count"),
        )
        .reset_index()
    )
    regime["market_regime_segment"] = regime.apply(
        lambda row: risk_overlay.classify_market_regime(row["market_change_5d"], row["market_breadth_5d"]),
        axis=1,
    )
    return frame.merge(regime, on="last_data_date", how="left")


def _add_overlay_features(frame):
    if frame.empty:
        return frame

    frame["board_limit_pct"] = frame.apply(
        lambda row: risk_overlay._board_limit_pct(
            row.get("stock_code"),
            stock_name=row.get("stock_name"),
            trade_date=row.get("last_data_date"),
        ),
        axis=1,
    )

    frame["listing_trade_days"] = frame.groupby("stock_code", sort=False).cumcount() + 1
    global_first_trade_date = frame["last_data_date"].min()
    first_seen = frame.groupby("stock_code", sort=False)["last_data_date"].transform("min")
    frame["first_seen_date"] = first_seen
    frame["probable_new_listing"] = first_seen > (global_first_trade_date + pd.Timedelta(days=7))

    frame["is_limit_up"] = frame["today_change"] >= (frame["board_limit_pct"] - 0.4)
    frame["limit_up_days_5d"] = (
        frame.groupby("stock_code", sort=False)["is_limit_up"]
        .rolling(5, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
        .fillna(0)
    )
    frame["limit_up_days_10d"] = (
        frame.groupby("stock_code", sort=False)["is_limit_up"]
        .rolling(10, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
        .fillna(0)
    )
    limit_break_group = (~frame["is_limit_up"].fillna(False)).groupby(frame["stock_code"], sort=False).cumsum()
    frame["consecutive_limit_up_days"] = (
        frame["is_limit_up"]
        .fillna(False)
        .astype(int)
        .groupby([frame["stock_code"], limit_break_group], sort=False)
        .cumsum()
    )

    frame["recent_ipo_0_20d"] = frame["probable_new_listing"] & frame["listing_trade_days"].between(1, 20)
    frame["sub_new_21_120d"] = frame["probable_new_listing"] & frame["listing_trade_days"].between(21, 120)
    frame["consecutive_limit_up_ge_2"] = frame["consecutive_limit_up_days"].ge(2)
    frame["limit_up_days_5d_ge_2"] = frame["limit_up_days_5d"].ge(2)
    frame["limit_chain"] = frame["consecutive_limit_up_ge_2"] | frame["limit_up_days_5d_ge_2"]

    frame["today_amp_ge_12"] = frame["today_amp"].ge(12)
    frame["amp_5d_ge_9"] = frame["amp_5d"].ge(9)
    frame["amp_10d_ge_8"] = frame["amp_10d"].ge(8)
    frame["turnover_ge_25"] = frame["turnover_rate"].ge(25)
    frame["high_volatility"] = (
        frame["today_amp_ge_12"]
        | frame["amp_5d_ge_9"]
        | frame["amp_10d_ge_8"]
        | frame["turnover_ge_25"]
    )

    frame["overheated_5d_ge_35"] = frame["change_5d"].ge(35)
    frame["overheated_10d_ge_60"] = frame["change_10d"].ge(60)
    frame["overheated_20d_ge_100"] = frame["change_20d"].ge(100)
    frame["overheated_limit_up_high_vol"] = frame["today_change"].ge(frame["board_limit_pct"] - 0.4) & frame[
        "high_volatility"
    ]
    frame["overheated"] = (
        frame["overheated_5d_ge_35"]
        | frame["overheated_10d_ge_60"]
        | frame["overheated_20d_ge_100"]
        | frame["overheated_limit_up_high_vol"]
    )
    frame["turnover_extreme_35"] = frame["turnover_rate"].ge(35)

    frame["risk_overlay_score"] = 0.0
    score_rules = [
        ("recent_ipo_0_20d", 8.0),
        ("sub_new_21_120d", 3.0),
        ("limit_chain", 7.0),
        ("overheated", 6.0),
        ("high_volatility", 4.0),
        ("turnover_extreme_35", 2.0),
    ]
    for column, score in score_rules:
        frame.loc[frame[column].fillna(False), "risk_overlay_score"] += score

    frame["block_formal"] = frame["recent_ipo_0_20d"] | frame["limit_chain"] | frame["overheated"]
    frame["downgrade_observe"] = (
        frame["sub_new_21_120d"] | frame["high_volatility"] | frame["turnover_extreme_35"]
    )
    frame["normal_pool"] = ~frame[
        ["recent_ipo_0_20d", "sub_new_21_120d", "limit_chain", "overheated", "high_volatility"]
    ].fillna(False).any(axis=1)
    frame["continue_evaluate"] = ~frame["block_formal"] & ~frame["downgrade_observe"]

    frame["risk_level"] = "低"
    frame.loc[frame["risk_overlay_score"].ge(4), "risk_level"] = "中"
    frame.loc[frame["block_formal"] | frame["risk_overlay_score"].ge(8), "risk_level"] = "高"
    frame["risk_level_high"] = frame["risk_level"].eq("高")
    frame["risk_level_mid"] = frame["risk_level"].eq("中")
    frame["risk_level_low"] = frame["risk_level"].eq("低")

    frame["market_strong"] = frame["market_regime_segment"].eq("strong")
    frame["market_neutral"] = frame["market_regime_segment"].eq("neutral")
    frame["market_weak"] = frame["market_regime_segment"].eq("weak")
    frame["board_10pct_main"] = frame["board_limit_pct"].eq(10)
    frame["board_20pct_chinext_star"] = frame["board_limit_pct"].eq(20)
    frame["board_30pct_bse"] = frame["board_limit_pct"].eq(30)
    return frame


def _restrict_signal_window(frame, tail_trade_days, max_shift, complete_only):
    if frame.empty:
        return frame
    trade_dates = sorted(frame["last_data_date"].dropna().unique())
    if complete_only and len(trade_dates) > max_shift:
        eligible_dates = trade_dates[: -int(max_shift)]
    else:
        eligible_dates = trade_dates
    eligible_dates = eligible_dates[-int(tail_trade_days) :]
    return frame[frame["last_data_date"].isin(eligible_dates)].copy()


def _branch_verdict(metadata, metrics):
    action = metadata.get("expected_action")
    ten_day = metrics.get("10d") or {}
    five_day = metrics.get("5d") or {}
    n = ten_day.get("n") or five_day.get("n") or 0
    if n < 100:
        return "样本不足，先观察"

    excess = ten_day.get("excess_vs_nonhit")
    avg_return = ten_day.get("avg_return")
    win_rate = ten_day.get("win_rate")
    if action == "拦截":
        if excess is not None and excess <= -1.0:
            return "强支持拦截"
        if excess is not None and excess < 0 and (win_rate is None or win_rate < 48):
            return "支持拦截"
        if avg_return is not None and avg_return < 0:
            return "部分支持拦截"
        return "需要复核拦截阈值"
    if action == "降级观察":
        if excess is not None and excess <= -0.5:
            return "支持降级"
        if win_rate is not None and win_rate < 48:
            return "部分支持降级"
        return "需要复核降级强度"
    if action == "继续评估":
        if excess is not None and excess >= 0:
            return "支持继续评估"
        return "需要复核普通池质量"
    return "描述性结果"


def _evidence_level(metric, min_samples):
    n = int((metric or {}).get("n") or 0)
    if n < int(min_samples or 0):
        return "insufficient"

    excess = (metric or {}).get("excess_vs_nonhit")
    avg_return = (metric or {}).get("avg_return")
    win_rate = (metric or {}).get("win_rate")
    loss_rate = (metric or {}).get("loss_rate_5pct")

    excess = float(excess) if excess is not None and not pd.isna(excess) else None
    avg_return = float(avg_return) if avg_return is not None and not pd.isna(avg_return) else None
    win_rate = float(win_rate) if win_rate is not None and not pd.isna(win_rate) else None
    loss_rate = float(loss_rate) if loss_rate is not None and not pd.isna(loss_rate) else None

    if (
        (excess is not None and excess <= -1.0)
        or (avg_return is not None and avg_return < -0.5)
        or (win_rate is not None and win_rate < 42)
        or (loss_rate is not None and loss_rate >= 40)
    ):
        return "strong_negative"
    if (
        (excess is not None and excess <= -0.5)
        or (win_rate is not None and win_rate < 46)
        or (loss_rate is not None and loss_rate >= 32)
    ):
        return "negative"
    if (
        (excess is not None and excess < 0)
        or (win_rate is not None and win_rate < 48)
        or (loss_rate is not None and loss_rate >= 28)
    ):
        return "mixed_negative"
    return "weak_or_positive"


def _calibrate_action_and_score(default_rule, evidence_level):
    base_action = default_rule.get("base_action")
    base_score = float(default_rule.get("base_score") or 0.0)

    if evidence_level == "insufficient":
        if base_action == "block":
            return "downgrade", round(max(4.0, base_score * 0.65), 2), True
        return "observe", 0.0, True

    if evidence_level in {"strong_negative", "negative"}:
        return base_action, round(base_score, 2), True

    if evidence_level == "mixed_negative":
        if base_action == "block":
            return "downgrade", round(max(4.0, base_score * 0.65), 2), True
        return "downgrade", round(max(2.0, base_score * 0.75), 2), True

    if base_action == "block":
        return "observe", round(max(1.0, base_score * 0.25), 2), True
    return "observe", 0.0, True


def _action_label(action):
    return {
        "block": "拦截正式推荐",
        "downgrade": "降级观察",
        "observe": "仅标记观察",
    }.get(action, action or "未知")


def _build_adaptive_config(branch_summary, sample, focus_hold_days=DEFAULT_FOCUS_HOLD_DAYS):
    summary_by_branch = {item.get("branch"): item for item in branch_summary or []}
    focus_label = f"{int(focus_hold_days)}d"
    branch_rules = {}
    for rule_key, default_rule in ADAPTIVE_RULE_DEFAULTS.items():
        source_branch = default_rule.get("source_branch")
        source_summary = summary_by_branch.get(source_branch) or {}
        metric = (source_summary.get("metrics") or {}).get(focus_label) or {}
        evidence = _evidence_level(metric, default_rule.get("min_samples"))
        action, score, enabled = _calibrate_action_and_score(default_rule, evidence)
        branch_rules[rule_key] = {
            "label": default_rule.get("label"),
            "source_branch": source_branch,
            "base_action": default_rule.get("base_action"),
            "base_score": default_rule.get("base_score"),
            "action": action,
            "action_label": _action_label(action),
            "score": score,
            "enabled": bool(enabled),
            "evidence_level": evidence,
            "min_samples": int(default_rule.get("min_samples") or 0),
            "metrics": metric,
            "verdict": source_summary.get("verdict"),
        }

    return {
        "model_version": ADAPTIVE_MODEL_VERSION,
        "focus_hold_days": int(focus_hold_days),
        "sample_start": (sample or {}).get("sample_start"),
        "sample_end": (sample or {}).get("sample_end"),
        "signal_trade_days": (sample or {}).get("signal_trade_days"),
        "signal_rows": (sample or {}).get("signal_rows"),
        "branch_rules": branch_rules,
    }


def _summarize_branch(frame, branch, holding_days):
    mask = frame[branch].fillna(False).astype(bool) if branch in frame.columns else pd.Series(False, index=frame.index)
    nonhit_mask = ~mask
    metadata = BRANCH_METADATA.get(branch, {})
    metrics = {}
    for hold_days in holding_days:
        return_col = f"return_{hold_days}d"
        hit_returns = pd.to_numeric(frame.loc[mask, return_col], errors="coerce").dropna()
        nonhit_returns = pd.to_numeric(frame.loc[nonhit_mask, return_col], errors="coerce").dropna()
        universe_returns = pd.to_numeric(frame[return_col], errors="coerce").dropna()
        if hit_returns.empty:
            metrics[f"{hold_days}d"] = {
                "n": 0,
                "avg_return": None,
                "median_return": None,
                "win_rate": None,
                "q10_return": None,
                "q05_return": None,
                "loss_rate_5pct": None,
                "excess_vs_nonhit": None,
                "excess_vs_universe": None,
            }
            continue
        metrics[f"{hold_days}d"] = {
            "n": int(len(hit_returns)),
            "avg_return": _round_or_none(hit_returns.mean(), 4),
            "median_return": _round_or_none(hit_returns.median(), 4),
            "win_rate": _round_or_none((hit_returns > 0).mean() * 100, 2),
            "q10_return": _round_or_none(hit_returns.quantile(0.10), 4),
            "q05_return": _round_or_none(hit_returns.quantile(0.05), 4),
            "loss_rate_5pct": _round_or_none((hit_returns <= -5).mean() * 100, 2),
            "excess_vs_nonhit": _round_or_none(hit_returns.mean() - nonhit_returns.mean(), 4)
            if not nonhit_returns.empty
            else None,
            "excess_vs_universe": _round_or_none(hit_returns.mean() - universe_returns.mean(), 4)
            if not universe_returns.empty
            else None,
        }

    return {
        "branch": branch,
        "label": metadata.get("label", branch),
        "group": metadata.get("group", "未分组"),
        "source": metadata.get("source"),
        "expected_action": metadata.get("expected_action"),
        "theory": metadata.get("theory"),
        "hit_rows": int(mask.sum()),
        "hit_stock_count": int(frame.loc[mask, "stock_code"].nunique()) if "stock_code" in frame.columns else 0,
        "metrics": metrics,
        "verdict": _branch_verdict(metadata, metrics),
    }


def _latest_vector_summary(frame):
    if frame.empty:
        return {}
    latest_date = frame["last_data_date"].max()
    latest = frame[frame["last_data_date"].eq(latest_date)].copy()
    branch_counts = {
        branch: int(latest[branch].fillna(False).astype(bool).sum())
        for branch in BRANCH_METADATA
        if branch in latest.columns
    }
    market_label = None
    if latest["market_regime_segment"].notna().any():
        market_label = risk_overlay.market_regime_label(latest["market_regime_segment"].dropna().iloc[0])
    return {
        "trade_date": str(pd.to_datetime(latest_date).date()),
        "total": int(len(latest)),
        "blocked_formal": int(latest["block_formal"].fillna(False).astype(bool).sum()),
        "downgraded": int(latest["downgrade_observe"].fillna(False).astype(bool).sum()),
        "market_regime": market_label,
        "branch_counts": branch_counts,
    }


def backtest_risk_overlay_branches(
    tail_trade_days=DEFAULT_TAIL_TRADE_DAYS,
    end_date=None,
    holding_days=DEFAULT_HOLDING_DAYS,
    entry_offset_days=DEFAULT_ENTRY_OFFSET_DAYS,
    fee_bps=DEFAULT_FEE_BPS,
    slippage_bps=DEFAULT_SLIPPAGE_BPS,
    complete_only=True,
):
    normalized_holding_days = tuple(sorted({int(day) for day in holding_days if int(day) > 0}))
    if not normalized_holding_days:
        raise ValueError("holding_days 至少需要一个大于0的周期")
    entry_offset_days = int(entry_offset_days)
    if entry_offset_days < 1:
        raise ValueError("entry_offset_days 必须大于等于1，避免使用信号日收盘后不可执行口径")

    max_shift = entry_offset_days + max(normalized_holding_days)
    history = _load_backtest_history(tail_trade_days, end_date=end_date, max_shift=max_shift)
    prepared = _prepare_history(
        history,
        holding_days=normalized_holding_days,
        entry_offset_days=entry_offset_days,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )
    prepared = _add_market_regime(prepared)
    prepared = _add_overlay_features(prepared)
    signal_frame = _restrict_signal_window(prepared, tail_trade_days, max_shift=max_shift, complete_only=complete_only)
    if signal_frame.empty:
        return {
            "success": False,
            "reason": "history_empty",
            "sample": {},
            "branch_summary": [],
            "external_branch_assessment": EXTERNAL_BRANCH_ASSESSMENT,
        }

    branch_summary = [
        _summarize_branch(signal_frame, branch, normalized_holding_days)
        for branch in BRANCH_METADATA.keys()
    ]

    sample = {
        "sample_start": str(signal_frame["last_data_date"].min().date()),
        "sample_end": str(signal_frame["last_data_date"].max().date()),
        "loaded_rows": int(len(prepared)),
        "signal_rows": int(len(signal_frame)),
        "signal_trade_days": int(signal_frame["last_data_date"].nunique()),
        "stock_count": int(signal_frame["stock_code"].nunique()),
        "tail_trade_days": int(tail_trade_days),
        "holding_days": list(normalized_holding_days),
        "entry_offset_days": int(entry_offset_days),
        "entry_mode": f"信号日t之后第{entry_offset_days}个交易日收盘价买入",
        "fee_bps": float(fee_bps),
        "slippage_bps": float(slippage_bps),
        "round_trip_cost_pct": _round_or_none((float(fee_bps) + float(slippage_bps)) * 2 / 100, 4),
        "complete_only": bool(complete_only),
        "assumption": "风险分支只使用信号日及以前的行情字段，收益使用t+entry_offset入场，t+entry_offset+holding_days收盘退出。",
    }

    adaptive_config = _build_adaptive_config(
        branch_summary,
        sample,
        focus_hold_days=DEFAULT_FOCUS_HOLD_DAYS if DEFAULT_FOCUS_HOLD_DAYS in normalized_holding_days else max(normalized_holding_days),
    )

    return {
        "success": True,
        "sample": sample,
        "branch_summary": branch_summary,
        "adaptive_config": adaptive_config,
        "external_branch_assessment": EXTERNAL_BRANCH_ASSESSMENT,
        "latest_vector_summary": _latest_vector_summary(prepared),
    }


def run_adaptive_risk_overlay_model(
    tail_trade_days=DEFAULT_TAIL_TRADE_DAYS,
    end_date=None,
    holding_days=DEFAULT_HOLDING_DAYS,
    focus_hold_days=DEFAULT_FOCUS_HOLD_DAYS,
    entry_offset_days=DEFAULT_ENTRY_OFFSET_DAYS,
    fee_bps=DEFAULT_FEE_BPS,
    slippage_bps=DEFAULT_SLIPPAGE_BPS,
    complete_only=True,
):
    normalized_holding_days = tuple(sorted({int(day) for day in holding_days if int(day) > 0}))
    if int(focus_hold_days) not in normalized_holding_days:
        normalized_holding_days = tuple(sorted({*normalized_holding_days, int(focus_hold_days)}))

    result = backtest_risk_overlay_branches(
        tail_trade_days=tail_trade_days,
        end_date=end_date,
        holding_days=normalized_holding_days,
        entry_offset_days=entry_offset_days,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        complete_only=complete_only,
    )
    if result.get("success"):
        result["adaptive_config"] = _build_adaptive_config(
            result.get("branch_summary") or [],
            result.get("sample") or {},
            focus_hold_days=focus_hold_days,
        )
    return result


def _metric_text(branch_result, hold_label):
    metric = (branch_result.get("metrics") or {}).get(hold_label) or {}
    if not metric or not metric.get("n"):
        return "n=0"
    return (
        f"n={metric.get('n')}, 均值={metric.get('avg_return')}%, 胜率={metric.get('win_rate')}%, "
        f"q10={metric.get('q10_return')}%, <=-5%={metric.get('loss_rate_5pct')}%, "
        f"超额={metric.get('excess_vs_nonhit')}%"
    )


def format_markdown_report(result, focus_hold_days=10):
    if not result.get("success"):
        return f"# {ADAPTIVE_MODEL_DISPLAY}\n\n运行失败: {result.get('reason') or '--'}\n"

    sample = result.get("sample") or {}
    adaptive_config = result.get("adaptive_config") or {}
    hold_label = f"{int(focus_hold_days)}d"
    lines = [
        f"# {ADAPTIVE_MODEL_DISPLAY}",
        "",
        "## 样本与口径",
        "",
        f"- 样本区间: {sample.get('sample_start')} 至 {sample.get('sample_end')}",
        f"- 信号交易日: {sample.get('signal_trade_days')}，信号样本: {sample.get('signal_rows')}，股票数: {sample.get('stock_count')}",
        f"- 入场: {sample.get('entry_mode')}；持有周期: {sample.get('holding_days')}",
        f"- 成本: fee_bps={sample.get('fee_bps')}，slippage_bps={sample.get('slippage_bps')}，双边成本={sample.get('round_trip_cost_pct')}%",
        f"- 假设: {sample.get('assumption')}",
        "",
        "## 动态规则配置",
        "",
        f"| 规则 | 基础动作 | 动态动作 | 动态分数 | 证据 | 样本 | {hold_label}均值 | {hold_label}胜率 | {hold_label}超额 |",
        "|---|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for rule in (adaptive_config.get("branch_rules") or {}).values():
        metric = rule.get("metrics") or {}
        lines.append(
            f"| {rule.get('label')} | {_action_label(rule.get('base_action'))} | {rule.get('action_label')} | "
            f"{rule.get('score')} | {rule.get('evidence_level')} | {metric.get('n') or 0} | "
            f"{metric.get('avg_return')} | {metric.get('win_rate')} | {metric.get('excess_vs_nonhit')} |"
        )

    lines.extend(
        [
            "",
        "## 行情分支验证",
        "",
        "| 分支 | 分组 | 动作 | 理论依据 | "
        + f"{hold_label}回测 | 结论 |",
        "|---|---|---|---|---|---|",
        ]
    )
    for item in result.get("branch_summary") or []:
        theory = (item.get("theory") or "").replace("|", "/")
        lines.append(
            f"| {item.get('label')} | {item.get('group')} | {item.get('expected_action')} | "
            f"{theory} | {_metric_text(item, hold_label)} | {item.get('verdict')} |"
        )

    lines.extend(
        [
            "",
            "## 外部分支状态",
            "",
            "| 分支 | 来源 | 理论依据 | 回测状态 |",
            "|---|---|---|---|",
        ]
    )
    for item in result.get("external_branch_assessment") or []:
        lines.append(
            f"| {item.get('label')} | {item.get('source')} | "
            f"{(item.get('theory') or '').replace('|', '/')} | {item.get('backtest_status')} |"
        )

    latest = result.get("latest_vector_summary") or {}
    if latest:
        lines.extend(
            [
                "",
                "## 最新截面",
                "",
                f"- 交易日: {latest.get('trade_date')}",
                f"- 覆盖数量: {latest.get('total')}",
                f"- 拦截正式推荐: {latest.get('blocked_formal')}",
                f"- 降级观察: {latest.get('downgraded')}",
                f"- 市场分层: {latest.get('market_regime')}",
            ]
        )
    return "\n".join(lines) + "\n"


def _print_console_summary(result, focus_hold_days=10):
    if not result.get("success"):
        print(f"{ADAPTIVE_MODEL_DISPLAY}失败: {result.get('reason') or '--'}")
        return
    sample = result.get("sample") or {}
    adaptive_config = result.get("adaptive_config") or {}
    hold_label = f"{int(focus_hold_days)}d"
    print(f"{ADAPTIVE_MODEL_DISPLAY}完成")
    print(f"- 样本区间: {sample.get('sample_start')} 至 {sample.get('sample_end')}")
    print(f"- 信号交易日: {sample.get('signal_trade_days')}，信号样本: {sample.get('signal_rows')}")
    print(f"- 入场/成本: {sample.get('entry_mode')}，双边成本{sample.get('round_trip_cost_pct')}%")
    if adaptive_config:
        print("- 动态规则:")
        for rule in (adaptive_config.get("branch_rules") or {}).values():
            metric = rule.get("metrics") or {}
            print(
                f"  {rule.get('label')}: {rule.get('action_label')}, score={rule.get('score')}, "
                f"evidence={rule.get('evidence_level')}, n={metric.get('n') or 0}"
            )
    print("")
    print(f"{'分支':<24} {'动作':<8} {'n':>8} {'均值%':>10} {'胜率%':>8} {'q10%':>10} {'超额%':>10}  结论")
    print("-" * 100)
    for item in result.get("branch_summary") or []:
        metric = (item.get("metrics") or {}).get(hold_label) or {}
        print(
            f"{item.get('label'):<24} "
            f"{str(item.get('expected_action') or ''):<8} "
            f"{int(metric.get('n') or 0):>8} "
            f"{str(metric.get('avg_return')):>10} "
            f"{str(metric.get('win_rate')):>8} "
            f"{str(metric.get('q10_return')):>10} "
            f"{str(metric.get('excess_vs_nonhit')):>10}  "
            f"{item.get('verdict')}"
        )


def main():
    parser = argparse.ArgumentParser(description=f"{ADAPTIVE_MODEL_DISPLAY}：滚动回测并生成动态分支规则")
    parser.add_argument("--tail-trade-days", type=int, default=DEFAULT_TAIL_TRADE_DAYS, help="回测信号交易日数量，默认260")
    parser.add_argument("--end-date", default=None, help="历史数据截止交易日，例如2026-05-21")
    parser.add_argument("--holding-days", default="1,3,5,10,20", help="逗号分隔持有周期，默认1,3,5,10,20")
    parser.add_argument("--entry-offset-days", type=int, default=DEFAULT_ENTRY_OFFSET_DAYS, help="信号后第几个交易日入场，默认1")
    parser.add_argument("--fee-bps", type=float, default=DEFAULT_FEE_BPS, help="单边手续费bps，默认3")
    parser.add_argument("--slippage-bps", type=float, default=DEFAULT_SLIPPAGE_BPS, help="单边滑点bps，默认5")
    parser.add_argument("--include-incomplete-tail", action="store_true", help="包含尾部尚无完整远期收益的信号日")
    parser.add_argument("--focus-hold-days", type=int, default=10, help="控制台和Markdown重点展示周期，默认10")
    parser.add_argument("--output-md", default=None, help="可选，输出Markdown报告路径")
    parser.add_argument("--json", action="store_true", help="输出完整JSON")
    args = parser.parse_args()

    holding_days = tuple(int(day.strip()) for day in args.holding_days.split(",") if day.strip())
    result = run_adaptive_risk_overlay_model(
        tail_trade_days=args.tail_trade_days,
        end_date=args.end_date,
        holding_days=holding_days,
        focus_hold_days=args.focus_hold_days,
        entry_offset_days=args.entry_offset_days,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        complete_only=not args.include_incomplete_tail,
    )

    if args.output_md:
        output_path = Path(args.output_md).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(format_markdown_report(result, focus_hold_days=args.focus_hold_days), encoding="utf-8")
        func.logInfo(f"风险覆盖层分支回测报告已输出: {output_path}")

    if args.json:
        print(json.dumps(result, ensure_ascii=False, default=str))
    else:
        _print_console_summary(result, focus_hold_days=args.focus_hold_days)
        if args.output_md:
            print(f"\nMarkdown报告: {Path(args.output_md).expanduser().resolve()}")


if __name__ == "__main__":
    main()
