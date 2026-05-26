import argparse
import json
import math
import re
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pymysql

import func


DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "rootroot",
    "database": "gu_piao",
    "charset": "utf8mb4",
}

RISK_OVERLAY_TABLE = "a_stock_risk_overlay"
DEFAULT_HISTORY_TAIL_DAYS = 260
DEFAULT_ADAPTIVE_RISK_TAIL_DAYS = 260
DEFAULT_ADAPTIVE_RISK_FOCUS_HOLD_DAYS = 10
DEFAULT_ADAPTIVE_RISK_REPORT_DIR = "log"
DEFAULT_ADAPTIVE_RISK_REPORT_PREFIX = "adaptive_risk_overlay"
DEFAULT_ADAPTIVE_RISK_MODEL_VERSION = "adaptive_risk_overlay_v1"
DEFAULT_ADAPTIVE_RISK_MODEL_DISPLAY = "自适应风险覆盖模型"
LATEST_ADAPTIVE_RISK_CONFIG = None
DEFAULT_EVENT_LOOKBACK_DAYS = 45
DEFAULT_UNLOCK_LOOKAHEAD_DAYS = 180
OVERLAY_HISTORY_COLUMNS = [
    "last_data_date",
    "stock_code",
    "stock_name",
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

MARKET_REGIME_LABELS = {
    "strong": "强市",
    "neutral": "震荡市",
    "weak": "弱市",
    "unknown": "未知",
}
MARKET_ENV_LABELS = {
    "strong": "强势",
    "neutral": "震荡",
    "weak": "弱势",
    "unknown": "未知",
}

SPECIAL_POOL_LABELS = {
    "normal": "普通股票池",
    "recent_ipo": "上市20日内新股",
    "sub_new": "次新股",
    "limit_chain": "连续涨停/连板高波动",
    "high_volatility": "高波动票",
    "overheated": "短线过热票",
}
EXTERNAL_FORCE_DOWNGRADE_LABELS = {"减持", "大比例解禁", "近端超大比例解禁"}


def _normalize_stock_code(value):
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    matched = re.search(r"(\d{6})", text)
    return matched.group(1) if matched else None


def _to_date_text(value):
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.strftime("%Y-%m-%d")


def _to_compact_date(value):
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.strftime("%Y%m%d")


def _round_or_none(value, digits=4):
    if value is None or pd.isna(value):
        return None
    try:
        if math.isinf(float(value)):
            return None
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _to_float(value):
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_bool(value):
    return bool(value) if value is not None and not pd.isna(value) else False


def _query_frame(sql, params=None):
    connection = None
    try:
        connection = pymysql.connect(**DB_CONFIG)
        with connection.cursor() as cursor:
            cursor.execute(sql, params or [])
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
        return pd.DataFrame.from_records(rows, columns=columns)
    except Exception as error:
        func.logInfo(f"风险覆盖层查询失败: {error}")
        return pd.DataFrame()
    finally:
        if connection:
            connection.close()


def load_recent_history(tail_trade_days=DEFAULT_HISTORY_TAIL_DAYS, end_date=None):
    params = []
    date_filter = ""
    if end_date:
        date_filter = "WHERE last_data_date <= %s"
        params.append(end_date)

    selected_columns = ", ".join(f"`{column}`" for column in OVERLAY_HISTORY_COLUMNS)
    sql = f"""
        SELECT {selected_columns}
        FROM a_stock_analysis_history
        WHERE last_data_date IN (
            SELECT last_data_date
            FROM (
                SELECT DISTINCT last_data_date
                FROM a_stock_analysis_history
                {date_filter}
                ORDER BY last_data_date DESC
                LIMIT %s
            ) recent_trade_days
        )
    """
    params.append(max(1, int(tail_trade_days or DEFAULT_HISTORY_TAIL_DAYS)))
    return _query_frame(sql, params=params)


def classify_market_regime(market_change_5d=None, market_breadth_5d=None):
    change = _to_float(market_change_5d)
    breadth = _to_float(market_breadth_5d)
    if change is None or breadth is None:
        return "unknown"
    if change >= 1.0 and breadth >= 55:
        return "strong"
    if change <= -0.5 or breadth < 45:
        return "weak"
    return "neutral"


def market_regime_label(regime):
    return MARKET_REGIME_LABELS.get(regime or "unknown", "未知")


def market_env_label(regime):
    return MARKET_ENV_LABELS.get(regime or "unknown", "未知")


def build_market_regime_frame(history):
    if history is None or history.empty:
        return pd.DataFrame(
            columns=[
                "last_data_date",
                "market_change_5d",
                "market_breadth_5d",
                "market_regime_segment",
                "market_regime_label",
            ]
        )

    frame = history.copy()
    if "last_data_date" not in frame.columns:
        return pd.DataFrame()
    frame["last_data_date"] = pd.to_datetime(frame["last_data_date"], errors="coerce")
    frame["change_5d"] = pd.to_numeric(frame.get("change_5d"), errors="coerce")
    frame = frame[frame["last_data_date"].notna()].copy()
    if frame.empty:
        return pd.DataFrame()

    frame["_change_5d_up"] = frame["change_5d"].gt(0).astype(float) * 100
    regime = (
        frame.groupby("last_data_date", sort=True)
        .agg(
            market_change_5d=("change_5d", "mean"),
            market_breadth_5d=("_change_5d_up", "mean"),
            universe_count=("stock_code", "count"),
        )
        .reset_index()
    )
    regime["market_regime_segment"] = regime.apply(
        lambda row: classify_market_regime(row.get("market_change_5d"), row.get("market_breadth_5d")),
        axis=1,
    )
    regime["market_regime_label"] = regime["market_regime_segment"].map(market_regime_label)
    return regime


def _board_limit_pct(stock_code):
    code = _normalize_stock_code(stock_code) or ""
    if code.startswith(("300", "301", "688", "689")):
        return 20.0
    if code.startswith(("8", "4", "9")):
        return 30.0
    return 10.0


def _ensure_history_columns(frame):
    for column in OVERLAY_HISTORY_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame


def _drop_duplicate_columns(frame, context=""):
    if frame is None or frame.empty or not frame.columns.has_duplicates:
        return frame
    duplicated = frame.columns[frame.columns.duplicated()].tolist()
    func.logInfo(f"风险覆盖层发现重复列{context}: {duplicated}")
    return frame.loc[:, ~frame.columns.duplicated()].copy()


def _adaptive_rule(adaptive_config, rule_key, default_action, default_score):
    rules = ((adaptive_config or {}).get("branch_rules") or {}) if isinstance(adaptive_config, dict) else {}
    rule = rules.get(rule_key) or {}
    if not rule:
        return {
            "action": default_action,
            "score": float(default_score or 0.0),
            "enabled": True,
            "action_label": default_action,
            "evidence_level": "static",
        }
    return {
        "action": rule.get("action") or default_action,
        "score": _to_float(rule.get("score")) if rule.get("score") is not None else float(default_score or 0.0),
        "enabled": bool(rule.get("enabled", True)),
        "action_label": rule.get("action_label") or rule.get("action") or default_action,
        "evidence_level": rule.get("evidence_level") or "unknown",
    }


def _apply_adaptive_branch(
    *,
    condition,
    adaptive_config,
    rule_key,
    default_action,
    default_score,
    label,
    pool=None,
    state=None,
):
    if not condition:
        return None
    rule = _adaptive_rule(adaptive_config, rule_key, default_action, default_score)
    if not rule.get("enabled"):
        return {
            "score": 0.0,
            "block": False,
            "downgrade": False,
            "label": label,
            "pool": pool,
            "adaptive_note": f"{label}:动态停用/{rule.get('evidence_level')}",
        }
    action = rule.get("action")
    return {
        "score": _to_float(rule.get("score")) or 0.0,
        "block": action == "block",
        "downgrade": action == "downgrade",
        "label": label,
        "pool": pool,
        "adaptive_note": f"{label}:{rule.get('action_label')}/{rule.get('evidence_level')}",
    }


def _resolve_row_overlay(row, adaptive_config=None):
    risk_score = 0.0
    labels = []
    special_pools = []
    block_formal = False
    downgrade_to_observation = False

    listing_trade_days = _to_float(row.get("listing_trade_days")) or 0
    probable_new_listing = bool(row.get("probable_new_listing"))
    limit_up_days_5d = _to_float(row.get("limit_up_days_5d")) or 0
    consecutive_limit_up_days = _to_float(row.get("consecutive_limit_up_days")) or 0
    today_change = _to_float(row.get("today_change")) or 0
    change_5d = _to_float(row.get("change_5d")) or 0
    change_10d = _to_float(row.get("change_10d")) or 0
    change_20d = _to_float(row.get("change_20d")) or 0
    today_amp = _to_float(row.get("today_amp")) or 0
    amp_5d = _to_float(row.get("amp_5d")) or 0
    amp_10d = _to_float(row.get("amp_10d")) or 0
    turnover_rate = _to_float(row.get("turnover_rate")) or 0
    limit_pct = _to_float(row.get("board_limit_pct")) or 10.0

    is_recent_ipo = probable_new_listing and 0 < listing_trade_days <= 20
    is_sub_new = probable_new_listing and 20 < listing_trade_days <= 120
    is_limit_chain = consecutive_limit_up_days >= 2 or limit_up_days_5d >= 2
    is_high_volatility = (
        today_amp >= 12
        or amp_5d >= 9
        or amp_10d >= 8
        or turnover_rate >= 25
    )
    is_overheated = (
        change_5d >= 35
        or change_10d >= 60
        or change_20d >= 100
        or (today_change >= limit_pct - 0.4 and is_high_volatility)
    )

    adaptive_notes = []
    branch_results = []
    if is_recent_ipo:
        branch_results.append(
            _apply_adaptive_branch(
                condition=True,
                adaptive_config=adaptive_config,
                rule_key="recent_ipo",
                default_action="block",
                default_score=8.0,
                label="上市20日内新股",
                pool="recent_ipo",
            )
        )
    elif is_sub_new:
        branch_results.append(
            _apply_adaptive_branch(
                condition=True,
                adaptive_config=adaptive_config,
                rule_key="sub_new",
                default_action="downgrade",
                default_score=3.0,
                label="次新股",
                pool="sub_new",
            )
        )

    branch_results.extend(
        [
            _apply_adaptive_branch(
                condition=is_limit_chain,
                adaptive_config=adaptive_config,
                rule_key="limit_chain",
                default_action="block",
                default_score=7.0,
                label="连续涨停/连板",
                pool="limit_chain",
            ),
            _apply_adaptive_branch(
                condition=is_overheated,
                adaptive_config=adaptive_config,
                rule_key="overheated",
                default_action="block",
                default_score=6.0,
                label="短线涨幅过热",
                pool="overheated",
            ),
            _apply_adaptive_branch(
                condition=is_high_volatility,
                adaptive_config=adaptive_config,
                rule_key="high_volatility",
                default_action="downgrade",
                default_score=4.0,
                label="高波动高换手",
                pool="high_volatility",
            ),
            _apply_adaptive_branch(
                condition=turnover_rate >= 35,
                adaptive_config=adaptive_config,
                rule_key="turnover_extreme_35",
                default_action="downgrade",
                default_score=2.0,
                label="换手极高",
            ),
        ]
    )

    for branch_result in [item for item in branch_results if item]:
        risk_score += _to_float(branch_result.get("score")) or 0.0
        block_formal = block_formal or bool(branch_result.get("block"))
        downgrade_to_observation = downgrade_to_observation or bool(branch_result.get("downgrade"))
        if branch_result.get("pool"):
            special_pools.append(branch_result["pool"])
        if branch_result.get("label"):
            labels.append(branch_result["label"])
        if branch_result.get("adaptive_note") and adaptive_config:
            adaptive_notes.append(branch_result["adaptive_note"])

    if not special_pools:
        special_pools.append("normal")

    special_pools = list(dict.fromkeys(special_pools))
    labels = list(dict.fromkeys(labels))
    risk_level = "高" if block_formal or risk_score >= 8 else "中" if risk_score >= 4 else "低"
    if block_formal:
        action = "不进入普通正式推荐，单独放入特殊股票池观察承接"
    elif downgrade_to_observation:
        action = "不加分，最多作为观察候选，等待次日承接确认"
    else:
        action = "可按普通策略继续评估"

    return {
        "special_pool": special_pools[0],
        "special_pool_label": "、".join(SPECIAL_POOL_LABELS.get(item, item) for item in special_pools),
        "special_pool_tags": ",".join(special_pools),
        "risk_overlay_score": _round_or_none(risk_score, 2),
        "risk_overlay_level": risk_level,
        "risk_overlay_labels": "、".join(labels) if labels else "无明显特殊池风险",
        "risk_overlay_block_formal": bool(block_formal),
        "risk_overlay_downgrade": bool(downgrade_to_observation),
        "risk_overlay_action": action,
        "adaptive_risk_overlay_note": "；".join(dict.fromkeys(adaptive_notes)) if adaptive_notes else "",
    }


def build_special_pool_overlay(history, trade_date=None, all_dates=False, adaptive_config=None):
    if adaptive_config is None:
        adaptive_config = LATEST_ADAPTIVE_RISK_CONFIG
    if history is None or history.empty:
        history = load_recent_history(end_date=trade_date)
    if history is None or history.empty:
        return pd.DataFrame()

    available_columns = [column for column in OVERLAY_HISTORY_COLUMNS if column in history.columns]
    frame = _ensure_history_columns(history[available_columns].copy())
    frame["stock_code"] = frame["stock_code"].apply(_normalize_stock_code)
    frame = frame[frame["stock_code"].notna()].copy()
    if frame.empty:
        return pd.DataFrame()

    frame["last_data_date"] = pd.to_datetime(frame["last_data_date"], errors="coerce")
    frame = frame[frame["last_data_date"].notna()].copy()
    if frame.empty:
        return pd.DataFrame()

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
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.sort_values(["stock_code", "last_data_date"]).reset_index(drop=True)
    frame["listing_trade_days"] = frame.groupby("stock_code").cumcount() + 1
    global_first_trade_date = frame["last_data_date"].min()
    first_seen = frame.groupby("stock_code")["last_data_date"].transform("min")
    frame["first_seen_date"] = first_seen
    frame["probable_new_listing"] = first_seen > (global_first_trade_date + pd.Timedelta(days=7))
    frame["board_limit_pct"] = frame["stock_code"].apply(_board_limit_pct)
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
        frame["is_limit_up"].fillna(False).astype(int).groupby([frame["stock_code"], limit_break_group], sort=False).cumsum()
    )

    if all_dates:
        pass
    elif trade_date:
        trade_date_ts = pd.to_datetime(trade_date, errors="coerce")
        if not pd.isna(trade_date_ts):
            frame = frame[frame["last_data_date"] == trade_date_ts].copy()
    else:
        latest_date = frame["last_data_date"].max()
        frame = frame[frame["last_data_date"] == latest_date].copy()

    if frame.empty:
        return pd.DataFrame()

    overlay_records = []
    for row in frame.to_dict("records"):
        overlay = _resolve_row_overlay(row, adaptive_config=adaptive_config)
        risk_note = (
            f"特殊池:{overlay['special_pool_label']}；风险:{overlay['risk_overlay_labels']}；"
            f"处理:{overlay['risk_overlay_action']}"
        )
        if overlay.get("adaptive_risk_overlay_note"):
            risk_note += f"；动态风控:{overlay['adaptive_risk_overlay_note']}"
        overlay_records.append(
            {
                "trade_date": _to_date_text(row.get("last_data_date")),
                "stock_code": row.get("stock_code"),
                "stock_name": row.get("stock_name"),
                "first_seen_date": _to_date_text(row.get("first_seen_date")),
                "listing_trade_days": int(row.get("listing_trade_days") or 0),
                "probable_new_listing": bool(row.get("probable_new_listing")),
                "board_limit_pct": _round_or_none(row.get("board_limit_pct"), 2),
                "limit_up_days_5d": int(row.get("limit_up_days_5d") or 0),
                "limit_up_days_10d": int(row.get("limit_up_days_10d") or 0),
                "consecutive_limit_up_days": int(row.get("consecutive_limit_up_days") or 0),
                **overlay,
                "risk_overlay_note": risk_note,
                "fundamental_status": "not_fetched",
                "event_status": "not_fetched",
                "fundamental_note": "未拉取财务数据，当前只使用行情与特殊股票池风控。",
                "event_note": "未拉取公告/龙虎榜/解禁数据，当前只使用行情与特殊股票池风控。",
                "lhb_note": "",
                "unlock_note": "",
            }
        )

    result = pd.DataFrame(overlay_records)
    market_frame = build_market_regime_frame(history)
    if not market_frame.empty:
        market_frame["trade_date"] = market_frame["last_data_date"].apply(_to_date_text)
        result = result.merge(
            market_frame[
                [
                    "trade_date",
                    "market_change_5d",
                    "market_breadth_5d",
                    "market_regime_segment",
                    "market_regime_label",
                ]
            ],
            on="trade_date",
            how="left",
        )
    else:
        result["market_regime_segment"] = "unknown"
        result["market_regime_label"] = "未知"
    return result


def _eastmoney_symbol(stock_code):
    code = _normalize_stock_code(stock_code) or ""
    if code.startswith(("6", "9")):
        return f"{code}.SH"
    if code.startswith(("8", "4")):
        return f"{code}.BJ"
    return f"{code}.SZ"


def _column_value(row, *columns):
    for column in columns:
        if column in row and row[column] is not None and not pd.isna(row[column]):
            return row[column]
    return None


def _parse_individual_info(df):
    result = {}
    if df is None or df.empty:
        return result
    key_col = "item" if "item" in df.columns else "项目" if "项目" in df.columns else df.columns[0]
    value_col = "value" if "value" in df.columns else "值" if "值" in df.columns else df.columns[-1]
    for _, row in df.iterrows():
        key = str(row.get(key_col) or "").strip()
        value = row.get(value_col)
        if not key:
            continue
        result[key] = value
    return result


def _match_info_value(info, keywords):
    for key, value in info.items():
        if any(keyword in key for keyword in keywords):
            return value
    return None


def _extract_latest_fundamental(ak, stock_code):
    context = {
        "status": "missing",
        "risk_score": 0.0,
        "positive_score": 0.0,
        "block_formal": False,
        "labels": [],
        "note": "财务数据未获取。",
        "raw": {},
    }
    try:
        df = ak.stock_financial_analysis_indicator_em(symbol=_eastmoney_symbol(stock_code), indicator="按报告期")
    except Exception as error:
        context["note"] = f"财务数据获取失败:{error}"
        return context

    if df is None or df.empty:
        context["note"] = "财务数据为空。"
        return context

    data = df.copy()
    if "REPORT_DATE" in data.columns:
        data["REPORT_DATE"] = pd.to_datetime(data["REPORT_DATE"], errors="coerce")
        data = data.sort_values("REPORT_DATE", ascending=False)
    latest = data.iloc[0].to_dict()
    report_date = _to_date_text(_column_value(latest, "REPORT_DATE", "日期"))
    revenue_yoy = _to_float(_column_value(latest, "TOTALOPERATEREVETZ", "主营业务收入增长率(%)"))
    netprofit_yoy = _to_float(_column_value(latest, "PARENTNETPROFITTZ", "净利润增长率(%)"))
    deduct_yoy = _to_float(_column_value(latest, "KCFJCXSYJLRTZ"))
    netprofit = _to_float(_column_value(latest, "PARENTNETPROFIT"))
    deduct_netprofit = _to_float(_column_value(latest, "KCFJCXSYJLR", "扣除非经常性损益后的净利润(元)"))
    roe = _to_float(_column_value(latest, "ROEJQ", "净资产收益率(%)"))
    gross_margin = _to_float(_column_value(latest, "XSMLL", "销售毛利率(%)"))
    debt_ratio = _to_float(_column_value(latest, "ZCFZL", "资产负债率(%)"))

    labels = []
    risk_score = 0.0
    positive_score = 0.0
    block_formal = False

    if netprofit is not None and netprofit < 0:
        risk_score += 5.0
        block_formal = True
        labels.append("最新报告期净利润为负")
    if deduct_netprofit is not None and deduct_netprofit < 0:
        risk_score += 4.0
        labels.append("扣非净利润为负")
    if netprofit_yoy is not None and netprofit_yoy < -30:
        risk_score += 3.0
        labels.append("净利同比大幅下滑")
    if debt_ratio is not None and debt_ratio >= 75:
        risk_score += 2.0
        labels.append("资产负债率偏高")
    if roe is not None and roe < 0:
        risk_score += 2.0
        labels.append("ROE为负")

    if revenue_yoy is not None and revenue_yoy >= 20:
        positive_score += 1.5
    if netprofit_yoy is not None and netprofit_yoy >= 20:
        positive_score += 2.0
    if roe is not None and roe >= 10:
        positive_score += 1.5

    note_bits = [f"报告期{report_date or '--'}"]
    if revenue_yoy is not None:
        note_bits.append(f"营收同比{_round_or_none(revenue_yoy, 2)}%")
    if netprofit_yoy is not None:
        note_bits.append(f"净利同比{_round_or_none(netprofit_yoy, 2)}%")
    if netprofit is not None:
        note_bits.append(f"归母净利{_round_or_none(netprofit / 100000000, 2)}亿")
    if roe is not None:
        note_bits.append(f"ROE{_round_or_none(roe, 2)}%")
    if debt_ratio is not None:
        note_bits.append(f"负债率{_round_or_none(debt_ratio, 2)}%")
    if gross_margin is not None:
        note_bits.append(f"毛利率{_round_or_none(gross_margin, 2)}%")

    context.update(
        {
            "status": "ok",
            "risk_score": _round_or_none(risk_score, 2),
            "positive_score": _round_or_none(positive_score, 2),
            "block_formal": bool(block_formal),
            "labels": labels,
            "note": "；".join(note_bits) + (f"；风险:{'、'.join(labels)}" if labels else "；财务未见硬伤"),
            "raw": {
                "report_date": report_date,
                "revenue_yoy": _round_or_none(revenue_yoy, 2),
                "netprofit_yoy": _round_or_none(netprofit_yoy, 2),
                "deduct_netprofit_yoy": _round_or_none(deduct_yoy, 2),
                "netprofit": netprofit,
                "deduct_netprofit": deduct_netprofit,
                "roe": _round_or_none(roe, 2),
                "gross_margin": _round_or_none(gross_margin, 2),
                "debt_ratio": _round_or_none(debt_ratio, 2),
            },
        }
    )
    return context


def _extract_recent_events(ak, stock_code, trade_date=None, lookback_days=DEFAULT_EVENT_LOOKBACK_DAYS):
    context = {
        "status": "missing",
        "risk_score": 0.0,
        "positive_score": 0.0,
        "block_formal": False,
        "labels": [],
        "note": "公告数据未获取。",
        "recent_items": [],
    }
    end_ts = _resolve_notice_query_end_date(trade_date)
    begin_ts = end_ts - pd.Timedelta(days=int(lookback_days or DEFAULT_EVENT_LOOKBACK_DAYS))
    try:
        df = ak.stock_individual_notice_report(
            security=stock_code,
            symbol="全部",
            begin_date=begin_ts.strftime("%Y%m%d"),
            end_date=end_ts.strftime("%Y%m%d"),
        )
    except Exception as error:
        context["note"] = f"公告数据获取失败:{error}"
        return context

    if df is None or df.empty:
        context["note"] = "近期开奖公告为空。"
        return context

    title_col = "公告标题" if "公告标题" in df.columns else df.columns[2]
    type_col = "公告类型" if "公告类型" in df.columns else None
    date_col = "公告日期" if "公告日期" in df.columns else None
    url_col = "网址" if "网址" in df.columns else None
    rows = []
    labels = []
    risk_score = 0.0
    positive_score = 0.0
    block_formal = False

    risk_patterns = [
        ("严重异常波动", 6.0, True),
        ("异常波动", 3.0, False),
        ("停牌核查", 5.0, True),
        ("减持", 2.0, False),
        ("立案", 8.0, True),
        ("处罚", 6.0, True),
        ("监管", 4.0, False),
        ("问询", 4.0, False),
        ("诉讼", 4.0, False),
        ("冻结", 4.0, False),
        ("风险提示", 4.0, False),
        ("预亏", 5.0, True),
        ("亏损", 4.0, False),
    ]
    positive_patterns = [
        ("预增", 2.0),
        ("扭亏", 2.5),
        ("中标", 1.5),
        ("签订", 1.2),
        ("回购", 1.5),
        ("增持", 1.5),
        ("分红", 1.0),
    ]

    data = df.copy()
    if date_col:
        data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
        data = data.sort_values(date_col, ascending=False)

    for _, row in data.head(12).iterrows():
        title = str(row.get(title_col) or "")
        event_type = str(row.get(type_col) or "") if type_col else ""
        item_date = _to_date_text(row.get(date_col)) if date_col else None
        rows.append(
            {
                "date": item_date,
                "type": event_type,
                "title": title,
                "url": row.get(url_col) if url_col else None,
            }
        )
        text = f"{title} {event_type}"
        for keyword, score, hard_block in risk_patterns:
            if keyword in text:
                risk_score += score
                block_formal = block_formal or hard_block
                labels.append(keyword)
        for keyword, score in positive_patterns:
            if keyword in text:
                positive_score += score

    labels = list(dict.fromkeys(labels))
    note = f"近{lookback_days}日公告{len(data)}条"
    if rows:
        note += f"；最新:{rows[0].get('date') or '--'} {rows[0].get('title') or '--'}"
    if labels:
        note += f"；事件提示:{'、'.join(labels[:5])}"
    else:
        note += "；未识别出硬风险公告关键词"

    context.update(
        {
            "status": "ok",
            "risk_score": _round_or_none(min(risk_score, 20.0), 2),
            "positive_score": _round_or_none(min(positive_score, 8.0), 2),
            "block_formal": bool(block_formal),
            "labels": labels[:8],
            "note": note,
            "recent_items": rows,
        }
    )
    return context


def _resolve_notice_query_end_date(trade_date=None):
    end_ts = pd.to_datetime(trade_date, errors="coerce")
    current_day = pd.Timestamp.today().normalize()
    if pd.isna(end_ts):
        return current_day + pd.Timedelta(days=1)

    end_ts = end_ts.normalize()
    if end_ts >= current_day:
        return end_ts + pd.Timedelta(days=1)
    return end_ts


def _extract_lhb_context(ak, stock_code, trade_date=None, lookback_days=60):
    context = {
        "status": "missing",
        "risk_score": 0.0,
        "positive_score": 0.0,
        "block_formal": False,
        "labels": [],
        "note": "龙虎榜数据未获取。",
    }
    end_ts = pd.to_datetime(trade_date, errors="coerce")
    if pd.isna(end_ts):
        end_ts = pd.Timestamp.today()
    begin_ts = end_ts - pd.Timedelta(days=int(lookback_days or 60))
    try:
        df = ak.stock_lhb_stock_detail_date_em(symbol=stock_code)
    except Exception as error:
        context["note"] = f"龙虎榜数据获取失败:{error}"
        return context

    if df is None or df.empty:
        context["note"] = "近阶段未查询到龙虎榜记录。"
        context["status"] = "ok"
        return context

    date_col = "交易日" if "交易日" in df.columns else df.columns[-1]
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    recent = data[(data[date_col] >= begin_ts) & (data[date_col] <= end_ts)].copy()
    recent_count = int(len(recent))
    risk_score = 0.0
    labels = []
    block_formal = False
    if recent_count >= 5:
        risk_score += 4.0
        labels.append("龙虎榜高频")
    elif recent_count >= 2:
        risk_score += 2.0
        labels.append("龙虎榜活跃")
    if recent_count >= 8:
        block_formal = True

    latest_date = _to_date_text(recent[date_col].max()) if not recent.empty else None
    context.update(
        {
            "status": "ok",
            "risk_score": _round_or_none(risk_score, 2),
            "positive_score": 0.0,
            "block_formal": bool(block_formal),
            "labels": labels,
            "note": f"近{lookback_days}日龙虎榜{recent_count}次" + (f"，最近{latest_date}" if latest_date else ""),
        }
    )
    return context


def _extract_unlock_context(ak, stock_code, trade_date=None, lookahead_days=DEFAULT_UNLOCK_LOOKAHEAD_DAYS):
    context = {
        "status": "missing",
        "risk_score": 0.0,
        "positive_score": 0.0,
        "block_formal": False,
        "labels": [],
        "note": "解禁数据未获取。",
    }
    begin_ts = pd.to_datetime(trade_date, errors="coerce")
    if pd.isna(begin_ts):
        begin_ts = pd.Timestamp.today()
    end_ts = begin_ts + pd.Timedelta(days=int(lookahead_days or DEFAULT_UNLOCK_LOOKAHEAD_DAYS))
    try:
        df = ak.stock_restricted_release_detail_em(
            start_date=begin_ts.strftime("%Y%m%d"),
            end_date=end_ts.strftime("%Y%m%d"),
        )
    except Exception as error:
        context["note"] = f"解禁数据获取失败:{error}"
        return context

    if df is None or df.empty:
        context["status"] = "ok"
        context["note"] = f"未来{lookahead_days}日未查询到解禁记录。"
        return context

    code_col = "股票代码" if "股票代码" in df.columns else None
    date_col = "解禁时间" if "解禁时间" in df.columns else None
    ratio_col = "占解禁前流通市值比例" if "占解禁前流通市值比例" in df.columns else None
    value_col = "实际解禁市值" if "实际解禁市值" in df.columns else None
    if not code_col:
        context["note"] = "解禁数据字段不完整。"
        return context

    data = df.copy()
    data[code_col] = data[code_col].astype(str).str.extract(r"(\d{6})", expand=False)
    target = data[data[code_col] == stock_code].copy()
    if target.empty:
        context["status"] = "ok"
        context["note"] = f"未来{lookahead_days}日未查询到该股解禁。"
        return context

    if date_col:
        target[date_col] = pd.to_datetime(target[date_col], errors="coerce")
        target = target.sort_values(date_col)
    ratio = pd.to_numeric(target.get(ratio_col), errors="coerce").max() if ratio_col else None
    release_value = pd.to_numeric(target.get(value_col), errors="coerce").max() if value_col else None
    nearest_ts = target[date_col].min() if date_col else None
    nearest_date = _to_date_text(nearest_ts) if date_col else None
    days_to_unlock = None
    if nearest_ts is not None and not pd.isna(nearest_ts):
        days_to_unlock = int((pd.Timestamp(nearest_ts).normalize() - begin_ts.normalize()).days)
    risk_score = 0.0
    labels = []
    block_formal = False
    ratio_pct = None
    if ratio is not None and not pd.isna(ratio):
        ratio_pct = float(ratio) * 100 if float(ratio) <= 1 else float(ratio)
        if ratio_pct >= 20 and days_to_unlock is not None and days_to_unlock <= 30:
            risk_score += 6.0
            block_formal = True
            labels.append("近端超大比例解禁")
        elif ratio_pct >= 8:
            risk_score += 3.0 if days_to_unlock is not None and days_to_unlock <= 30 else 2.0
            labels.append("大比例解禁")
        elif ratio_pct >= 3:
            risk_score += 1.5
            labels.append("中等比例解禁")

    context.update(
        {
            "status": "ok",
            "risk_score": _round_or_none(risk_score, 2),
            "positive_score": 0.0,
            "block_formal": bool(block_formal),
            "labels": labels,
            "note": (
                f"未来{lookahead_days}日有解禁，最近{nearest_date or '--'}，"
                f"占流通市值约{_round_or_none(ratio_pct, 2) if ratio_pct is not None else '--'}%，"
                f"解禁市值约{_round_or_none((release_value or 0) / 100000000, 2) if release_value is not None else '--'}亿"
            ),
        }
    )
    return context


def fetch_external_stock_context(
    stock_code,
    stock_name=None,
    trade_date=None,
    event_lookback_days=DEFAULT_EVENT_LOOKBACK_DAYS,
    unlock_lookahead_days=DEFAULT_UNLOCK_LOOKAHEAD_DAYS,
):
    code = _normalize_stock_code(stock_code)
    if not code:
        return {"status": "invalid_stock_code", "risk_score": 0.0, "positive_score": 0.0}

    try:
        import akshare as ak
    except Exception as error:
        return {
            "status": "akshare_unavailable",
            "risk_score": 0.0,
            "positive_score": 0.0,
            "block_formal": False,
            "labels": [],
            "note": f"akshare不可用:{error}",
        }

    info_note = ""
    valuation = {}
    try:
        info_df = ak.stock_individual_info_em(symbol=code, timeout=6)
        info = _parse_individual_info(info_df)
        valuation = {
            "pe": _match_info_value(info, ["市盈率", "PE"]),
            "pb": _match_info_value(info, ["市净率", "PB"]),
            "total_mv": _match_info_value(info, ["总市值"]),
            "float_mv": _match_info_value(info, ["流通市值"]),
            "listing_date": _match_info_value(info, ["上市时间", "上市日期"]),
        }
        if any(value is not None for value in valuation.values()):
            info_note = (
                f"估值/基础:PE={valuation.get('pe') or '--'}, PB={valuation.get('pb') or '--'}, "
                f"总市值={valuation.get('total_mv') or '--'}, 上市时间={valuation.get('listing_date') or '--'}"
            )
    except Exception as error:
        info_note = f"估值/基础数据获取失败:{error}"

    fundamental = _extract_latest_fundamental(ak, code)
    events = _extract_recent_events(
        ak,
        code,
        trade_date=trade_date,
        lookback_days=event_lookback_days,
    )
    lhb = _extract_lhb_context(ak, code, trade_date=trade_date)
    unlock = _extract_unlock_context(
        ak,
        code,
        trade_date=trade_date,
        lookahead_days=unlock_lookahead_days,
    )

    risk_score = sum(
        _to_float(item.get("risk_score")) or 0.0
        for item in [fundamental, events, lhb, unlock]
    )
    positive_score = sum(
        _to_float(item.get("positive_score")) or 0.0
        for item in [fundamental, events, lhb, unlock]
    )
    labels = []
    for item in [fundamental, events, lhb, unlock]:
        labels.extend(item.get("labels") or [])
    labels = list(dict.fromkeys(labels))
    block_formal = any(_normalize_bool(item.get("block_formal")) for item in [fundamental, events, lhb, unlock])

    notes = [note for note in [info_note, fundamental.get("note"), events.get("note"), lhb.get("note"), unlock.get("note")] if note]
    return {
        "status": "ok",
        "stock_code": code,
        "stock_name": stock_name,
        "risk_score": _round_or_none(min(risk_score, 30.0), 2),
        "positive_score": _round_or_none(min(positive_score, 10.0), 2),
        "block_formal": bool(block_formal),
        "labels": labels[:10],
        "note": " | ".join(notes),
        "valuation": valuation,
        "fundamental": fundamental,
        "events": events,
        "lhb": lhb,
        "unlock": unlock,
    }


def _overlay_for_candidates(candidate_df, history=None, trade_date=None, overlay_frame=None):
    if candidate_df is None or candidate_df.empty:
        return pd.DataFrame()
    if overlay_frame is None or overlay_frame.empty:
        overlay_frame = build_special_pool_overlay(history, trade_date=trade_date)
    if overlay_frame is None or overlay_frame.empty:
        return pd.DataFrame()
    overlay = overlay_frame.copy()
    overlay["stock_code"] = overlay["stock_code"].apply(_normalize_stock_code)
    if "trade_date" not in overlay.columns:
        overlay["trade_date"] = trade_date
    if trade_date:
        overlay = overlay[overlay["trade_date"].astype(str) == str(trade_date)].copy()
    return overlay


def _merge_external_context(row, external_context):
    base_score = _to_float(row.get("risk_overlay_score")) or 0.0
    external_score = _to_float(external_context.get("risk_score")) or 0.0
    labels = []
    existing_labels = str(row.get("risk_overlay_labels") or "").strip()
    if existing_labels and existing_labels != "无明显特殊池风险":
        labels.extend([item for item in existing_labels.split("、") if item])
    labels.extend(external_context.get("labels") or [])
    labels = list(dict.fromkeys(labels))

    block_formal = _normalize_bool(row.get("risk_overlay_block_formal")) or _normalize_bool(
        external_context.get("block_formal")
    )
    force_downgrade = any(label in EXTERNAL_FORCE_DOWNGRADE_LABELS for label in labels)
    downgrade = _normalize_bool(row.get("risk_overlay_downgrade")) or external_score >= 3 or force_downgrade
    total_score = min(base_score + external_score, 40.0)
    row["risk_overlay_score"] = _round_or_none(total_score, 2)
    row["risk_overlay_level"] = "高" if block_formal or total_score >= 8 else "中" if downgrade or total_score >= 4 else "低"
    row["risk_overlay_labels"] = "、".join(labels) if labels else "无明显特殊池风险"
    row["risk_overlay_block_formal"] = block_formal
    row["risk_overlay_downgrade"] = downgrade
    if block_formal:
        row["risk_overlay_action"] = "不进入普通正式推荐，单独放入特殊股票池观察承接"
    elif downgrade:
        row["risk_overlay_action"] = "不进入普通正式推荐，降级观察并等待事件供给压力释放"
    else:
        row["risk_overlay_action"] = row.get("risk_overlay_action") or "可按普通策略继续评估"
    row["fundamental_event_context"] = external_context
    row["fundamental_note"] = (external_context.get("fundamental") or {}).get("note")
    row["event_note"] = (external_context.get("events") or {}).get("note")
    row["lhb_note"] = (external_context.get("lhb") or {}).get("note")
    row["unlock_note"] = (external_context.get("unlock") or {}).get("note")
    base_note_parts = [
        part
        for part in str(row.get("risk_overlay_note") or "").split("；")
        if part and not part.startswith("处理:")
    ]
    row["risk_overlay_note"] = "；".join(
        [
            *base_note_parts,
            f"外部覆盖:{external_context.get('note') or '--'}",
            f"处理:{row['risk_overlay_action']}",
        ]
    ).strip("；")
    return row


def apply_risk_overlay_to_candidates(
    candidate_df,
    history=None,
    trade_date=None,
    overlay_frame=None,
    include_external=False,
    score_column=None,
    filter_blocked=False,
    filter_downgraded=False,
    score_penalty_multiplier=1.0,
):
    if candidate_df is None or candidate_df.empty:
        return candidate_df

    candidates = _drop_duplicate_columns(candidate_df.copy(), context=":candidate_input")
    candidates["stock_code"] = candidates["stock_code"].apply(_normalize_stock_code)
    overlay = _overlay_for_candidates(candidates, history=history, trade_date=trade_date, overlay_frame=overlay_frame)
    if overlay.empty:
        candidates["risk_overlay_score"] = 0.0
        candidates["risk_overlay_level"] = "低"
        candidates["risk_overlay_labels"] = "风险覆盖数据缺失"
        candidates["risk_overlay_block_formal"] = False
        candidates["risk_overlay_downgrade"] = False
        candidates["risk_overlay_note"] = "风险覆盖数据缺失，本次不因覆盖层加分。"
    else:
        merge_columns = [
            "stock_code",
            "trade_date",
            "special_pool",
            "special_pool_label",
            "special_pool_tags",
            "listing_trade_days",
            "first_seen_date",
            "board_limit_pct",
            "limit_up_days_5d",
            "limit_up_days_10d",
            "consecutive_limit_up_days",
            "risk_overlay_score",
            "risk_overlay_level",
            "risk_overlay_labels",
            "risk_overlay_block_formal",
            "risk_overlay_downgrade",
            "risk_overlay_action",
            "risk_overlay_note",
            "market_regime_segment",
            "market_regime_label",
            "market_change_5d",
            "market_breadth_5d",
            "fundamental_status",
            "event_status",
            "fundamental_note",
            "event_note",
            "lhb_note",
            "unlock_note",
        ]
        merge_columns = [column for column in merge_columns if column in overlay.columns]
        overlay_managed_columns = [column for column in merge_columns if column != "stock_code"]
        stale_columns = [
            column
            for column in [*overlay_managed_columns, *(f"{column}_overlay" for column in overlay_managed_columns)]
            if column in candidates.columns
        ]
        if stale_columns:
            candidates = candidates.drop(columns=stale_columns)
        candidates = candidates.merge(
            overlay[merge_columns].drop_duplicates(subset=["stock_code"], keep="last"),
            on="stock_code",
            how="left",
            suffixes=("", "_overlay"),
        )
        candidates = _drop_duplicate_columns(candidates, context=":after_overlay_merge")

        fill_values = {
            "risk_overlay_score": 0.0,
            "risk_overlay_level": "低",
            "risk_overlay_labels": "无明显特殊池风险",
            "risk_overlay_block_formal": False,
            "risk_overlay_downgrade": False,
            "risk_overlay_action": "可按普通策略继续评估",
            "risk_overlay_note": "特殊池:普通股票池；风险:无明显特殊池风险；处理:可按普通策略继续评估",
            "special_pool": "normal",
            "special_pool_label": SPECIAL_POOL_LABELS["normal"],
            "special_pool_tags": "normal",
            "market_regime_segment": "unknown",
            "market_regime_label": "未知",
        }
        for column, value in fill_values.items():
            if column not in candidates.columns:
                candidates[column] = value
            else:
                candidates[column] = candidates[column].fillna(value)

    if include_external:
        candidates = _drop_duplicate_columns(candidates, context=":before_external_context")
        rows = []
        for row in candidates.to_dict("records"):
            external = fetch_external_stock_context(
                row.get("stock_code"),
                stock_name=row.get("stock_name"),
                trade_date=trade_date or row.get("trade_date"),
            )
            rows.append(_merge_external_context(row, external))
        candidates = _drop_duplicate_columns(pd.DataFrame(rows), context=":after_external_context")

    if score_column and score_column in candidates.columns:
        original_column = f"{score_column}_before_risk_overlay"
        candidates[original_column] = pd.to_numeric(candidates[score_column], errors="coerce")
        penalty_multiplier = 1.0 if score_penalty_multiplier is None else float(score_penalty_multiplier)
        penalty = pd.to_numeric(candidates["risk_overlay_score"], errors="coerce").fillna(0) * float(
            penalty_multiplier
        )
        candidates[score_column] = (candidates[original_column] - penalty).clip(lower=0)

    candidates["risk_overlay_block_formal"] = candidates["risk_overlay_block_formal"].fillna(False).astype(bool)
    candidates["risk_overlay_downgrade"] = candidates["risk_overlay_downgrade"].fillna(False).astype(bool)

    if filter_blocked:
        candidates = candidates[~candidates["risk_overlay_block_formal"]].copy()

    if filter_downgraded:
        candidates = candidates[~candidates["risk_overlay_downgrade"]].copy()

    return candidates


def enrich_candidate_record(record, history=None, trade_date=None, include_external=True):
    if not record:
        return record
    frame = pd.DataFrame([record])
    enriched = apply_risk_overlay_to_candidates(
        frame,
        history=history,
        trade_date=trade_date,
        include_external=include_external,
    )
    if enriched.empty:
        return record
    return enriched.iloc[0].to_dict()


def append_overlay_note(note, row, max_length=1600):
    overlay_note = row.get("risk_overlay_note")
    if not overlay_note:
        return note
    merged = f"{note}；风险覆盖:{overlay_note}"
    if max_length and len(merged) > max_length:
        return merged[: max_length - 3] + "..."
    return merged


def ensure_risk_overlay_table():
    connection = None
    try:
        connection = pymysql.connect(**DB_CONFIG)
        with connection.cursor() as cursor:
            cursor.execute("SHOW TABLES LIKE %s", [RISK_OVERLAY_TABLE])
            exists = cursor.fetchone() is not None
        if not exists:
            func.logInfo("风险覆盖表不存在，请先完成数据库初始化后再运行风险覆盖刷新任务")
        return exists
    except Exception as error:
        func.logInfo(f"风险覆盖表检查失败: {error}")
        return False
    finally:
        if connection:
            connection.close()


def upsert_risk_overlay(overlay_frame):
    if overlay_frame is None or overlay_frame.empty:
        return {"success": False, "saved_count": 0, "reason": "overlay_empty"}
    if not ensure_risk_overlay_table():
        return {"success": False, "saved_count": 0, "reason": "ensure_table_failed"}

    columns = [
        "trade_date",
        "stock_code",
        "stock_name",
        "first_seen_date",
        "listing_trade_days",
        "board_limit_pct",
        "limit_up_days_5d",
        "limit_up_days_10d",
        "consecutive_limit_up_days",
        "special_pool",
        "special_pool_label",
        "special_pool_tags",
        "risk_overlay_score",
        "risk_overlay_level",
        "risk_overlay_labels",
        "risk_overlay_block_formal",
        "risk_overlay_downgrade",
        "risk_overlay_action",
        "risk_overlay_note",
        "market_regime_segment",
        "market_regime_label",
        "market_change_5d",
        "market_breadth_5d",
        "fundamental_status",
        "event_status",
        "fundamental_note",
        "event_note",
        "lhb_note",
        "unlock_note",
    ]
    frame = overlay_frame.copy()
    for column in columns:
        if column not in frame.columns:
            frame[column] = None
    frame = frame[columns].copy()
    frame = frame.where(pd.notna(frame), None)

    placeholders = ", ".join(["%s"] * len(columns))
    update_clause = ", ".join(
        f"{column}=VALUES({column})" for column in columns if column not in {"trade_date", "stock_code"}
    )
    sql = f"""
        INSERT INTO {RISK_OVERLAY_TABLE} ({", ".join(columns)})
        VALUES ({placeholders})
        ON DUPLICATE KEY UPDATE {update_clause}
    """
    connection = None
    try:
        connection = pymysql.connect(**DB_CONFIG)
        with connection.cursor() as cursor:
            cursor.executemany(sql, [tuple(row[column] for column in columns) for row in frame.to_dict("records")])
        connection.commit()
        return {"success": True, "saved_count": int(len(frame)), "reason": None}
    except Exception as error:
        func.logInfo(f"风险覆盖表写入失败: {error}")
        return {"success": False, "saved_count": 0, "reason": str(error)}
    finally:
        if connection:
            connection.close()


def _default_adaptive_report_path(result=None, trade_date=None):
    report_date = _to_date_text(trade_date)
    if not report_date:
        report_date = ((result or {}).get("latest_vector_summary") or {}).get("trade_date")
    if not report_date:
        report_date = (result or {}).get("sample", {}).get("sample_end")
    if not report_date:
        report_date = datetime.now().strftime("%Y-%m-%d")
    compact_date = _to_compact_date(report_date) or datetime.now().strftime("%Y%m%d")
    return str(Path(DEFAULT_ADAPTIVE_RISK_REPORT_DIR) / f"{DEFAULT_ADAPTIVE_RISK_REPORT_PREFIX}_{compact_date}.md")


def _run_adaptive_risk_overlay_model(
    trade_date=None,
    adaptive_tail_trade_days=DEFAULT_ADAPTIVE_RISK_TAIL_DAYS,
    adaptive_focus_hold_days=DEFAULT_ADAPTIVE_RISK_FOCUS_HOLD_DAYS,
    adaptive_report_path=None,
):
    try:
        import analysis_gu_piao_adaptive_risk_overlay_model as adaptive_model
    except Exception as error:
        func.logInfo(f"{DEFAULT_ADAPTIVE_RISK_MODEL_DISPLAY}加载失败: {error}")
        return {"success": False, "reason": f"import_failed:{error}"}

    model_display = getattr(adaptive_model, "ADAPTIVE_MODEL_DISPLAY", DEFAULT_ADAPTIVE_RISK_MODEL_DISPLAY)
    try:
        result = adaptive_model.run_adaptive_risk_overlay_model(
            tail_trade_days=adaptive_tail_trade_days,
            end_date=trade_date,
            focus_hold_days=adaptive_focus_hold_days,
        )
        if result.get("success"):
            output_target = adaptive_report_path or _default_adaptive_report_path(
                result=result,
                trade_date=trade_date,
            )
            output_path = Path(output_target).expanduser()
            if not output_path.is_absolute():
                output_path = Path.cwd() / output_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                adaptive_model.format_markdown_report(result, focus_hold_days=adaptive_focus_hold_days),
                encoding="utf-8",
            )
            result["report_path"] = str(output_path)
            func.logInfo(
                f"{model_display}完成: trade_date={trade_date}, "
                f"report_path={result.get('report_path')}, has_config={bool(result.get('adaptive_config'))}"
            )
        else:
            func.logInfo(
                f"{model_display}未通过: trade_date={trade_date}, reason={result.get('reason')}"
            )
        return result
    except Exception as error:
        func.logInfo(f"{model_display}运行失败: {error}")
        return {"success": False, "reason": str(error)}


def refresh_latest_risk_overlay(
    trade_date=None,
    tail_trade_days=DEFAULT_HISTORY_TAIL_DAYS,
    use_adaptive=True,
    adaptive_tail_trade_days=DEFAULT_ADAPTIVE_RISK_TAIL_DAYS,
    adaptive_focus_hold_days=DEFAULT_ADAPTIVE_RISK_FOCUS_HOLD_DAYS,
    adaptive_report_path=None,
):
    global LATEST_ADAPTIVE_RISK_CONFIG
    adaptive_result = None
    adaptive_config = None
    if use_adaptive:
        adaptive_result = _run_adaptive_risk_overlay_model(
            trade_date=trade_date,
            adaptive_tail_trade_days=adaptive_tail_trade_days,
            adaptive_focus_hold_days=adaptive_focus_hold_days,
            adaptive_report_path=adaptive_report_path,
        )
        if adaptive_result.get("success"):
            adaptive_config = adaptive_result.get("adaptive_config")
            LATEST_ADAPTIVE_RISK_CONFIG = adaptive_config
        else:
            LATEST_ADAPTIVE_RISK_CONFIG = None
            return {
                "success": False,
                "trade_date": trade_date,
                "overlay_count": 0,
                "saved_count": 0,
                "save_result": {
                    "success": False,
                    "saved_count": 0,
                    "reason": "adaptive_risk_overlay_failed",
                },
                "risk_summary": {},
                "adaptive_enabled": True,
                "adaptive_success": False,
                "adaptive_model": {
                    "success": False,
                    "reason": adaptive_result.get("reason"),
                    "report_path": adaptive_result.get("report_path"),
                    "config": adaptive_result.get("adaptive_config"),
                },
            }
        if not adaptive_config:
            LATEST_ADAPTIVE_RISK_CONFIG = None
            return {
                "success": False,
                "trade_date": trade_date,
                "overlay_count": 0,
                "saved_count": 0,
                "save_result": {
                    "success": False,
                    "saved_count": 0,
                    "reason": "adaptive_risk_config_missing",
                },
                "risk_summary": {},
                "adaptive_enabled": True,
                "adaptive_success": False,
                "adaptive_model": {
                    "success": True,
                    "reason": "adaptive_risk_config_missing",
                    "report_path": adaptive_result.get("report_path"),
                    "config": adaptive_result.get("adaptive_config"),
                },
            }
    else:
        LATEST_ADAPTIVE_RISK_CONFIG = None

    history = load_recent_history(tail_trade_days=tail_trade_days, end_date=trade_date)
    overlay = build_special_pool_overlay(history, trade_date=trade_date, adaptive_config=adaptive_config)
    save_result = upsert_risk_overlay(overlay)
    return {
        "success": bool(save_result.get("success")),
        "trade_date": overlay["trade_date"].iloc[0] if not overlay.empty else trade_date,
        "overlay_count": int(len(overlay)),
        "saved_count": int(save_result.get("saved_count") or 0),
        "save_result": save_result,
        "risk_summary": summarize_overlay(overlay),
        "adaptive_enabled": bool(use_adaptive),
        "adaptive_success": bool((adaptive_result or {}).get("success")),
        "adaptive_model": {
            "success": bool((adaptive_result or {}).get("success")),
            "reason": (adaptive_result or {}).get("reason"),
            "report_path": (adaptive_result or {}).get("report_path"),
            "config": (adaptive_result or {}).get("adaptive_config"),
        }
        if use_adaptive
        else None,
    }


def summarize_overlay(overlay_frame):
    if overlay_frame is None or overlay_frame.empty:
        return {}
    frame = overlay_frame.copy()
    return {
        "total": int(len(frame)),
        "blocked_formal": int(frame["risk_overlay_block_formal"].fillna(False).astype(bool).sum())
        if "risk_overlay_block_formal" in frame.columns
        else 0,
        "downgraded": int(frame["risk_overlay_downgrade"].fillna(False).astype(bool).sum())
        if "risk_overlay_downgrade" in frame.columns
        else 0,
        "special_pool_counts": frame.get("special_pool_label", pd.Series(dtype=object)).fillna("未知").value_counts().to_dict(),
        "market_regime": frame.get("market_regime_label", pd.Series(dtype=object)).dropna().iloc[0]
        if "market_regime_label" in frame.columns and frame["market_regime_label"].notna().any()
        else "未知",
    }


def main():
    parser = argparse.ArgumentParser(description="刷新A股自适应风险覆盖表 a_stock_risk_overlay")
    parser.add_argument("--trade-date", dest="trade_date", default=None, help="指定交易日，例如 2026-05-18；默认取本地历史表最新交易日")
    parser.add_argument("--tail-trade-days", dest="tail_trade_days", type=int, default=DEFAULT_HISTORY_TAIL_DAYS, help="行情风险计算窗口，默认260个交易日")
    parser.add_argument("--static-risk-overlay", dest="static_risk_overlay", action="store_true", help="跳过滚动回测，使用静态风险覆盖规则")
    parser.add_argument("--adaptive-tail-trade-days", dest="adaptive_tail_trade_days", type=int, default=DEFAULT_ADAPTIVE_RISK_TAIL_DAYS, help="自适应回测窗口，默认260个交易日")
    parser.add_argument("--adaptive-focus-hold-days", dest="adaptive_focus_hold_days", type=int, default=DEFAULT_ADAPTIVE_RISK_FOCUS_HOLD_DAYS, help="动态规则校准关注持有周期，默认10日")
    parser.add_argument("--adaptive-report-path", dest="adaptive_report_path", default=None, help=f"{DEFAULT_ADAPTIVE_RISK_MODEL_DISPLAY}Markdown报告路径；默认按交易日输出到log/adaptive_risk_overlay_YYYYMMDD.md")
    parser.add_argument("--skip-adaptive-report", dest="skip_adaptive_report", action="store_true", help=f"不输出{DEFAULT_ADAPTIVE_RISK_MODEL_DISPLAY}Markdown报告")
    parser.add_argument("--json", dest="json_output", action="store_true", help="以JSON格式输出结果")
    args = parser.parse_args()

    result = refresh_latest_risk_overlay(
        trade_date=args.trade_date,
        tail_trade_days=args.tail_trade_days,
        use_adaptive=not args.static_risk_overlay,
        adaptive_tail_trade_days=args.adaptive_tail_trade_days,
        adaptive_focus_hold_days=args.adaptive_focus_hold_days,
        adaptive_report_path=None if args.skip_adaptive_report else args.adaptive_report_path,
    )
    if args.json_output:
        print(json.dumps(result, ensure_ascii=False, default=str))
        return

    summary = result.get("risk_summary") or {}
    print("风险覆盖层刷新完成" if result.get("success") else "风险覆盖层刷新失败")
    print(f"- 交易日: {result.get('trade_date') or '--'}")
    print(f"- 覆盖数量: {result.get('overlay_count')}")
    print(f"- 入库数量: {result.get('saved_count')}")
    print(f"- 市场分层: {summary.get('market_regime') or '--'}")
    print(f"- 拦截正式推荐: {summary.get('blocked_formal')}")
    print(f"- 降级观察: {summary.get('downgraded')}")
    adaptive_model = result.get("adaptive_model") or {}
    if result.get("adaptive_enabled"):
        config = adaptive_model.get("config") or {}
        model_version = config.get("model_version") or DEFAULT_ADAPTIVE_RISK_MODEL_VERSION
        print(f"- 自适应风险覆盖模型({model_version}): {'成功' if adaptive_model.get('success') else '失败'}")
        if adaptive_model.get("report_path"):
            print(f"- 自适应风险覆盖报告: {adaptive_model.get('report_path')}")
        for rule in (config.get("branch_rules") or {}).values():
            print(
                f"  · {rule.get('label')}: {rule.get('action_label')}, "
                f"score={rule.get('score')}, evidence={rule.get('evidence_level')}"
            )
    if not result.get("success"):
        print(f"- 失败原因: {(result.get('save_result') or {}).get('reason') or result.get('reason')}")


if __name__ == "__main__":
    main()
