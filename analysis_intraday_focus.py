"""盘中关注池分析。

作用:
    读取近20个交易日的策略落库结果和当前分析交易日行业热点报告，
    同时比较近几日行业热点延续性，盘中拉取实时行情，
    输出今天值得关注、只观察或应回避的个股。它只做临时判断，不写入
    a_stock_analysis_history，避免污染收盘模型和回测。

流程:
    先确定库里最新分析交易日；
    再读取 a_stock_strategy_result 和 log/industry_hotspot/YYYY-MM-DD；
    然后批量请求 Sina 实时行情；
    最后按策略来源、行业热度、龙头身份和盘中承接状态综合排序。
"""

import argparse
import contextlib
import datetime as dt
import html
import io
import json
import math
import re
from pathlib import Path

import pandas as pd
import pymysql
import requests

from analysis_gu_piao_adaptive_risk import overlay as event_overlay


PROJECT_ROOT = Path(__file__).resolve().parent
INTRADAY_REPORT_DIR = PROJECT_ROOT / "log" / "intraday_focus"
EMOTION_WATCH_DIR = PROJECT_ROOT / "log" / "emotion_leader_watch"
SINA_REALTIME_URL = "https://hq.sinajs.cn/list={symbols}"
SINA_HEADERS = {
    "Referer": "https://finance.sina.com.cn",
    "User-Agent": "Mozilla/5.0",
}
DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "rootroot",
    "database": "gu_piao",
    "charset": "utf8mb4",
}
HOT_VERSION_SCORE = {
    "主线热点": 18,
    "强势热点": 15,
    "观察": 7,
}
LEADER_TYPE_SCORE = {
    "涨停龙头": 16,
    "容量中军": 11,
    "弹性领涨": 10,
    "放量趋势": 9,
    "观察核心": 5,
}
DEFAULT_STRATEGY_LOOKBACK_TRADE_DAYS = 20
DEFAULT_INDUSTRY_REVIEW_DAYS = 5
DEFAULT_EVENT_LOOKBACK_DAYS = 14
DEFAULT_EVENT_SCAN_LIMIT = 25
DEFAULT_HTML_HIDE_CODE_PREFIXES = ("688", "920")
EVENT_RISK_PATTERNS = [
    ("终止重大资产重组", 10.0, True),
    ("终止筹划重大资产重组", 10.0, True),
    ("终止收购", 6.0, False),
    ("终止购买资产", 6.0, False),
    ("严重异常波动", 8.0, True),
    ("停牌核查", 8.0, True),
    ("立案", 10.0, True),
    ("处罚", 8.0, True),
    ("监管", 5.0, False),
    ("问询", 5.0, False),
    ("诉讼", 5.0, False),
    ("冻结", 5.0, False),
    ("风险提示", 5.0, False),
    ("异常波动", 3.0, False),
    ("减持", 4.0, False),
    ("预亏", 6.0, True),
    ("亏损", 4.0, False),
]
EVENT_POSITIVE_PATTERNS = [
    ("预增", 2.0),
    ("扭亏", 2.5),
    ("中标", 1.5),
    ("签订合同", 1.5),
    ("重大合同", 2.0),
    ("回购", 1.5),
    ("增持", 1.5),
]
EVENT_NEGATIVE_CATALYST_PHRASES = [
    "不生产",
    "不涉及",
    "无规模化",
    "暂停状态",
    "紧急澄清",
    "澄清",
]
EVENT_HIGH_TRUTH_PATTERNS = [
    ("获得批复", 4.0, "监管/审批落地"),
    ("获批", 3.5, "监管/审批落地"),
    ("核准", 3.5, "监管/审批落地"),
    ("完成交割", 4.5, "交易完成交割"),
    ("完成过户", 4.0, "资产/股权已过户"),
    ("完成工商变更", 4.0, "工商变更完成"),
    ("正式投产", 3.5, "产能投产"),
    ("投产", 2.5, "产能投产"),
    ("量产", 2.5, "产能量产"),
    ("签订合同", 3.0, "合同落地"),
    ("重大合同", 3.5, "重大合同落地"),
    ("中标", 2.5, "订单/项目中标"),
    ("订单", 2.0, "订单验证"),
    ("业绩预增", 3.0, "业绩验证"),
    ("预增", 2.0, "业绩验证"),
    ("扭亏", 2.5, "业绩改善"),
    ("回购完成", 2.5, "回购执行"),
    ("增持完成", 2.5, "增持执行"),
]
EVENT_LOW_TRUTH_PATTERNS = [
    ("拟", -2.0, "拟/筹划阶段"),
    ("计划", -1.5, "计划阶段"),
    ("筹划", -2.0, "筹划阶段"),
    ("意向", -3.0, "意向性表述"),
    ("框架协议", -2.5, "框架协议待兑现"),
    ("战略合作", -2.0, "战略合作待兑现"),
    ("有望", -1.5, "预期性表述"),
    ("预计", -1.0, "预期性表述"),
    ("互动平台", -2.0, "互动平台口径"),
    ("投资者关系", -1.5, "投资者关系口径"),
    ("传闻", -4.0, "传闻口径"),
    ("网传", -4.0, "传闻口径"),
    ("澄清", -4.0, "澄清/否认"),
    ("不涉及", -4.0, "不涉及题材"),
    ("风险提示", -3.0, "风险提示"),
    ("异常波动", -2.0, "异动提示"),
]


def _date_text(value):
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    text = str(value).strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}", text):
        return text[:10]
    if re.match(r"^\d{8}$", text):
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return text or None


def _parse_date(value):
    text = _date_text(value)
    if not text:
        return None
    try:
        return dt.datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        return None


def _to_float(value):
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().replace("%", "").replace(",", "")
    if not text or text.lower() in {"nan", "none", "null", "--"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y", "是", "高"}


def _round(value, digits=2):
    value = _to_float(value)
    if value is None:
        return None
    return round(value, digits)


def _normalize_code(value):
    text = str(value or "").strip()
    matched = re.search(r"(\d{6})", text)
    if matched:
        return matched.group(1)
    digits = re.sub(r"\D", "", text)
    if 1 <= len(digits) <= 6:
        return digits.zfill(6)
    return None


def _realtime_symbol(stock_code):
    code = _normalize_code(stock_code)
    if not code:
        return None
    if code.startswith(("920", "8", "4")):
        return f"bj{code}"
    if code.startswith(("6", "9")):
        return f"sh{code}"
    return f"sz{code}"


def _connect():
    return pymysql.connect(**DB_CONFIG)


def _query_frame(sql, params=None):
    connection = _connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, params or [])
            rows = cursor.fetchall()
            columns = [column[0] for column in cursor.description or []]
        return pd.DataFrame(list(rows), columns=columns)
    finally:
        connection.close()


def _latest_history_date():
    where = "WHERE CHAR_LENGTH(last_data_date) = 10 AND last_data_date BETWEEN '2000-01-01' AND '2100-12-31'"
    frame = _query_frame(
        f"""
        SELECT MAX(last_data_date) AS trade_date
        FROM a_stock_analysis_history
        {where}
        """,
    )
    if frame.empty:
        return None
    return _date_text(frame["trade_date"].iloc[0])


def _latest_strategy_date(max_date):
    frame = _query_frame(
        """
        SELECT MAX(trade_date) AS trade_date
        FROM a_stock_strategy_result
        WHERE trade_date <= %s
        """,
        [_date_text(max_date)],
    )
    if frame.empty:
        return None
    return _date_text(frame["trade_date"].iloc[0])


def _recent_history_trade_dates(max_date, lookback_trade_days=DEFAULT_STRATEGY_LOOKBACK_TRADE_DAYS):
    max_date = _date_text(max_date)
    if not max_date:
        return []
    frame = _query_frame(
        """
        SELECT last_data_date
        FROM (
            SELECT DISTINCT last_data_date
            FROM a_stock_analysis_history
            WHERE CHAR_LENGTH(last_data_date) = 10
              AND last_data_date BETWEEN '2000-01-01' AND '2100-12-31'
              AND last_data_date <= %s
            ORDER BY last_data_date DESC
            LIMIT %s
        ) recent_days
        ORDER BY last_data_date
        """,
        [max_date, int(lookback_trade_days or DEFAULT_STRATEGY_LOOKBACK_TRADE_DAYS)],
    )
    if frame.empty:
        return []
    return [_date_text(value) for value in frame["last_data_date"].tolist() if _date_text(value)]


def _extract_stop_price(strategy_note):
    text = str(strategy_note or "")
    patterns = [
        r"防守[:：]\s*([0-9]+(?:\.[0-9]+)?)",
        r"跌破([0-9]+(?:\.[0-9]+)?)附近",
        r"跌破或不能收回\s*([0-9]+(?:\.[0-9]+)?)",
    ]
    for pattern in patterns:
        matched = re.search(pattern, text)
        if matched:
            return _round(matched.group(1), 2)
    return None


def _strategy_age_map(strategy_dates):
    latest_first = list(reversed([_date_text(value) for value in strategy_dates if _date_text(value)]))
    return {date_value: idx for idx, date_value in enumerate(latest_first)}


def _load_strategy_candidates(
    strategy_date=None,
    target_trade_date=None,
    lookback_trade_days=DEFAULT_STRATEGY_LOOKBACK_TRADE_DAYS,
):
    if strategy_date:
        strategy_dates = [_date_text(strategy_date)]
    else:
        strategy_dates = _recent_history_trade_dates(target_trade_date, lookback_trade_days=lookback_trade_days)
    strategy_dates = [date_value for date_value in strategy_dates if date_value]
    if not strategy_dates:
        return pd.DataFrame()
    placeholders = ", ".join(["%s"] * len(strategy_dates))
    frame = _query_frame(
        f"""
        SELECT
            s.trade_date,
            s.strategy_type,
            s.stock_code,
            s.stock_name,
            s.today_change AS strategy_today_change,
            s.industry,
            s.today_amount AS strategy_today_amount,
            s.turnover_rate AS strategy_turnover_rate,
            s.strategy_note,
            h.latest_price AS ref_close,
            h.today_open AS ref_open,
            h.today_high AS ref_high,
            h.today_low AS ref_low,
            h.ma5,
            h.ma10,
            h.ma20,
            h.amount_avg_5d,
            h.turnover_avg_5d
        FROM a_stock_strategy_result s
        LEFT JOIN a_stock_analysis_history h
         ON h.stock_code COLLATE utf8mb4_general_ci = s.stock_code COLLATE utf8mb4_general_ci
         AND h.last_data_date REGEXP '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'
         AND h.last_data_date COLLATE utf8mb4_general_ci = CAST(s.trade_date AS CHAR) COLLATE utf8mb4_general_ci
        WHERE s.trade_date IN ({placeholders})
        ORDER BY s.trade_date DESC, s.strategy_type, s.id
        """,
        strategy_dates,
    )
    if frame.empty:
        return frame
    frame["stock_code"] = frame["stock_code"].apply(_normalize_code)
    frame["stop_price"] = frame["strategy_note"].apply(_extract_stop_price)
    age_map = _strategy_age_map(strategy_dates)
    frame["strategy_date_text"] = frame["trade_date"].apply(_date_text)
    frame["strategy_age_trade_days"] = frame["strategy_date_text"].map(age_map)
    return frame


def _latest_risk_overlay_date(max_date):
    max_date = _date_text(max_date)
    if not max_date:
        return None
    frame = _query_frame(
        """
        SELECT MAX(trade_date) AS trade_date
        FROM a_stock_risk_overlay
        WHERE trade_date <= %s
        """,
        [max_date],
    )
    if frame.empty:
        return None
    return _date_text(frame["trade_date"].iloc[0])


def _load_risk_overlay_for_codes(stock_codes, trade_date):
    codes = sorted({code for code in (_normalize_code(value) for value in stock_codes or []) if code})
    if not codes:
        return None, {}
    resolved_date = _latest_risk_overlay_date(trade_date)
    if not resolved_date:
        return None, {}

    placeholders = ", ".join(["%s"] * len(codes))
    frame = _query_frame(
        f"""
        SELECT
            trade_date,
            stock_code,
            stock_name,
            special_pool,
            special_pool_label,
            special_pool_tags,
            risk_overlay_score,
            risk_overlay_level,
            risk_overlay_labels,
            risk_overlay_block_formal,
            risk_overlay_downgrade,
            risk_overlay_action,
            risk_overlay_note,
            market_regime_label,
            fundamental_status,
            event_status,
            fundamental_note,
            event_note,
            lhb_note,
            unlock_note
        FROM a_stock_risk_overlay
        WHERE trade_date = %s
          AND stock_code IN ({placeholders})
        """,
        [resolved_date, *codes],
    )
    if frame.empty:
        return resolved_date, {}
    frame["stock_code"] = frame["stock_code"].apply(_normalize_code)
    return resolved_date, {
        row["stock_code"]: row
        for row in frame.to_dict("records")
        if row.get("stock_code")
    }


def _industry_dir_for_date(trade_date):
    wanted = _parse_date(trade_date)
    root = PROJECT_ROOT / "log" / "industry_hotspot"
    if not root.exists():
        return None
    dirs = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        parsed = _parse_date(child.name)
        if not parsed:
            continue
        if wanted and parsed > wanted:
            continue
        dirs.append((parsed, child))
    if not dirs:
        return None
    return sorted(dirs, key=lambda item: item[0])[-1][1]


def _recent_industry_dirs(end_date, review_days=DEFAULT_INDUSTRY_REVIEW_DAYS):
    wanted = _parse_date(end_date)
    root = PROJECT_ROOT / "log" / "industry_hotspot"
    if not root.exists():
        return []
    dirs = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        parsed = _parse_date(child.name)
        if not parsed:
            continue
        if wanted and parsed > wanted:
            continue
        dirs.append((parsed, child))
    dirs = sorted(dirs, key=lambda item: item[0], reverse=True)[: int(review_days or DEFAULT_INDUSTRY_REVIEW_DAYS)]
    return [child for _, child in sorted(dirs, key=lambda item: item[0])]


def _read_csv_if_exists(path):
    if not path or not Path(path).exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def _load_industry_reports(trade_date):
    daily_dir = _industry_dir_for_date(trade_date)
    if not daily_dir:
        return daily_dir, pd.DataFrame(), pd.DataFrame()
    boards = _read_csv_if_exists(daily_dir / "industry_hotspot_boards.csv")
    leaders = _read_csv_if_exists(daily_dir / "industry_hotspot_leaders.csv")
    return daily_dir, boards, leaders


def _emotion_watch_dir_for_date(trade_date):
    wanted = _parse_date(trade_date)
    if not EMOTION_WATCH_DIR.exists():
        return None
    dirs = []
    for child in EMOTION_WATCH_DIR.iterdir():
        if not child.is_dir():
            continue
        parsed = _parse_date(child.name)
        if not parsed:
            continue
        if wanted and parsed > wanted:
            continue
        dirs.append((parsed, child))
    if not dirs:
        return None
    return sorted(dirs, key=lambda item: item[0])[-1][1]


def _load_emotion_leader_watch(trade_date, limit=10):
    daily_dir = _emotion_watch_dir_for_date(trade_date)
    if not daily_dir:
        return None, [], 0

    frame = _read_csv_if_exists(daily_dir / "emotion_leader_watch_latest.csv")
    if frame.empty:
        return daily_dir.name, [], 0

    frame = frame.copy()
    frame["stock_code"] = frame["stock_code"].apply(_normalize_code)
    frame = frame[frame["stock_code"].notna()].copy()
    if frame.empty:
        return daily_dir.name, [], 0

    stage_order = {"A早期胚子": 1, "B右侧确认": 2, "C龙头已成": 3, "D过热回避": 4, "观察": 5}
    frame["_stage_order"] = frame["stage"].map(stage_order).fillna(9)
    frame["_score_sort"] = frame["score"].apply(_to_float).fillna(-999)
    frame = frame.sort_values(["_stage_order", "_score_sort"], ascending=[True, False])
    total_count = int(len(frame))
    if limit:
        limit = int(limit)
        selected_frames = []
        selected_indexes = set()
        a_rows = frame[frame["stage"] == "A早期胚子"].head(min(6, limit)).copy()
        if not a_rows.empty:
            selected_frames.append(a_rows)
            selected_indexes.update(a_rows.index.tolist())
        remaining_limit = max(0, limit - len(a_rows))
        b_cap = 8 if a_rows.empty else 6
        b_rows = frame[frame["stage"] == "B右侧确认"].head(min(b_cap, remaining_limit)).copy()
        if not b_rows.empty:
            selected_frames.append(b_rows)
            selected_indexes.update(b_rows.index.tolist())
        if selected_frames:
            selected = pd.concat(selected_frames, ignore_index=False)
            if len(selected) < limit:
                remaining = frame[
                    (~frame.index.isin(selected_indexes))
                    & (~frame["stage"].isin(["C龙头已成", "D过热回避", "观察"]))
                ].head(limit - len(selected))
                if not remaining.empty:
                    selected = pd.concat([selected, remaining], ignore_index=False)
            frame = selected.sort_values(["_stage_order", "_score_sort"], ascending=[True, False]).head(limit).copy()
        else:
            frame = frame[~frame["stage"].isin(["C龙头已成", "D过热回避", "观察"])].head(limit).copy()
    frame = frame.drop(columns=["_stage_order", "_score_sort"], errors="ignore")
    return daily_dir.name, frame.to_dict("records"), total_count


def _load_industry_review(end_date, review_days=DEFAULT_INDUSTRY_REVIEW_DAYS):
    review_dirs = _recent_industry_dirs(end_date, review_days=review_days)
    board_frames = []
    leader_frames = []
    for daily_dir in review_dirs:
        boards = _read_csv_if_exists(daily_dir / "industry_hotspot_boards.csv")
        leaders = _read_csv_if_exists(daily_dir / "industry_hotspot_leaders.csv")
        if not boards.empty:
            boards = boards.copy()
            boards["_review_date"] = daily_dir.name
            boards["_board_name"] = boards.apply(_board_name, axis=1)
            board_frames.append(boards)
        if not leaders.empty:
            leaders = leaders.copy()
            leaders["_review_date"] = daily_dir.name
            leaders["股票代码"] = leaders["股票代码"].apply(_normalize_code)
            leader_frames.append(leaders)
    boards = pd.concat(board_frames, ignore_index=True) if board_frames else pd.DataFrame()
    leaders = pd.concat(leader_frames, ignore_index=True) if leader_frames else pd.DataFrame()
    return review_dirs, boards, leaders


def _board_name(row):
    if row is None:
        return ""
    return str(row.get("板块名称") or row.get("industry_name") or "").strip()


def _is_focus_board(row):
    hot_rank = _to_float(row.get("热点排名") or row.get("hot_rank"))
    attention_rank = _to_float(row.get("关注排名") or row.get("attention_rank"))
    version = str(row.get("热点版本") or "")
    advice = str(row.get("参与建议") or "")
    if "回避" in advice:
        return False
    if version in {"主线热点", "强势热点"}:
        return True
    if hot_rank is not None and hot_rank <= 20:
        return True
    return bool(attention_rank is not None and attention_rank <= 20)


def _board_rank_tuple(row):
    hot_rank = _to_float(row.get("热点排名") or row.get("hot_rank"))
    attention_rank = _to_float(row.get("关注排名") or row.get("attention_rank"))
    hot_score = _to_float(row.get("热点分") or row.get("hot_score"))
    attention_score = _to_float(row.get("关注度分") or row.get("attention_score"))
    return (
        hot_rank if hot_rank is not None else 999,
        attention_rank if attention_rank is not None else 999,
        -(hot_score or 0),
        -(attention_score or 0),
    )


def _rank_score_by_rank(rank, max_score=10, floor=30):
    rank = _to_float(rank)
    if rank is None:
        return 0
    if rank <= 1:
        return max_score
    if rank >= floor:
        return 0
    return round(max_score * (floor - rank) / (floor - 1), 2)


def _board_lookup(boards):
    if boards is None or boards.empty:
        return {}
    lookup = {}
    for row in boards.to_dict("records"):
        name = _board_name(row)
        if not name:
            continue
        if name not in lookup or _board_rank_tuple(row) < _board_rank_tuple(lookup[name]):
            lookup[name] = row
    return lookup


def _score_board(board):
    if board is None:
        return {"score": 0.0, "reasons": [], "warnings": []}

    score = 0.0
    reasons = []
    warnings = []
    version = str(board.get("热点版本") or "")
    shape = str(board.get("参与形态") or "")
    advice = str(board.get("参与建议") or "")
    risk = str(board.get("风险提示") or "")

    if version in HOT_VERSION_SCORE:
        score += HOT_VERSION_SCORE[version]
        reasons.append(version)
    score += _rank_score_by_rank(board.get("热点排名"), max_score=12, floor=35)
    score += _rank_score_by_rank(board.get("关注排名"), max_score=10, floor=35)

    if "主线延续" in shape:
        score += 10
        reasons.append("主线延续")
    elif "加速高潮" in shape:
        score -= 8
        warnings.append("加速高潮")
    elif "放量下跌" in shape:
        score -= 12
        warnings.append("放量下跌")

    if "优先跟踪" in advice:
        score += 10
        reasons.append("报告建议优先跟踪")
    if "等分歧" in advice or "二次确认" in advice:
        score += 4
        reasons.append("适合等分歧确认")
    if "谨慎追高" in advice:
        score -= 8
        warnings.append("报告提示谨慎追高")
    if "回避" in advice:
        score -= 25
        warnings.append("报告提示回避")

    breadth_today = _to_float(board.get("上涨广度"))
    breadth_5d = _to_float(board.get("5日广度"))
    amount_ratio = _to_float(board.get("额比5日"))
    top3_share = _to_float(board.get("前三成交占比"))
    attention_delta = _to_float(board.get("关注较昨日"))
    attention_delta_5d = _to_float(board.get("关注较5日"))

    if breadth_today is not None:
        if breadth_today >= 60:
            score += 7
            reasons.append("板块广度强")
        elif breadth_today >= 40:
            score += 4
        elif breadth_today < 20:
            score -= 7
            warnings.append("板块广度弱")
    if breadth_5d is not None and breadth_5d >= 50:
        score += 4
    if amount_ratio is not None:
        if amount_ratio >= 1.5:
            score += 7
            reasons.append("资金放大")
        elif amount_ratio >= 1.1:
            score += 4
        elif amount_ratio < 0.8:
            score -= 4
            warnings.append("资金不足")
    if attention_delta is not None and attention_delta >= 15:
        score += 5
        reasons.append("关注度上升")
    elif attention_delta is not None and attention_delta <= -15:
        score -= 4
    if attention_delta_5d is not None and attention_delta_5d >= 20:
        score += 5
    if top3_share is not None:
        if top3_share >= 85:
            score -= 8
            warnings.append("前三成交过度集中")
        elif top3_share >= 70:
            score -= 4
            warnings.append("成交集中")
    if "成交过度集中" in risk:
        score -= 4
        warnings.append("成交过度集中")

    return {"score": round(score, 2), "reasons": reasons, "warnings": warnings}


def _leader_stability(leaders, board_name):
    if leaders is None or leaders.empty or not board_name:
        return {
            "leaders": [],
            "summary": "--",
            "tier_label": "--",
            "tier_score": 0.0,
            "tier_advice": "--",
            "strong_leader_count": 0,
            "limit_leader_count": 0,
        }
    frame = leaders[leaders["板块名称"].astype(str) == str(board_name)].copy()
    if frame.empty:
        return {
            "leaders": [],
            "summary": "--",
            "tier_label": "--",
            "tier_score": 0.0,
            "tier_advice": "--",
            "strong_leader_count": 0,
            "limit_leader_count": 0,
        }
    grouped = []
    for code, group in frame.groupby("股票代码", sort=False):
        code = _normalize_code(code)
        if not code:
            continue
        group = group.sort_values("_review_date")
        latest = group.iloc[-1].to_dict()
        grouped.append(
            {
                "stock_code": code,
                "stock_name": latest.get("股票名称") or code,
                "appear_days": int(group["_review_date"].nunique()),
                "latest_rank": _to_float(latest.get("板块内龙头排名")),
                "latest_type": latest.get("龙头类型"),
                "latest_change": latest.get("今日涨幅"),
            }
        )
    grouped = sorted(
        grouped,
        key=lambda item: (
            -item["appear_days"],
            item["latest_rank"] if item["latest_rank"] is not None else 999,
        ),
    )
    summary_bits = [
        f"{item['stock_name']}({item['stock_code']}, 龙头持续{item['appear_days']}日)"
        for item in grouped[:3]
    ]
    latest_date = str(frame["_review_date"].max()) if "_review_date" in frame.columns else None
    latest_frame = frame[frame["_review_date"].astype(str) == latest_date].copy() if latest_date else frame.copy()
    strong_count = 0
    limit_count = 0
    weak_count = 0
    for _, row in latest_frame.iterrows():
        change = _to_float(row.get("今日涨幅"))
        leader_type = str(row.get("龙头类型") or "")
        if change is not None and change <= 0:
            weak_count += 1
        if "涨停龙头" in leader_type or (change is not None and change >= 9.5):
            limit_count += 1
        if (
            "涨停龙头" in leader_type
            or "容量中军" in leader_type
            or "放量趋势" in leader_type
            or (change is not None and change >= 3)
        ):
            strong_count += 1

    if limit_count >= 1 and strong_count >= 3:
        tier_label = "梯队强"
        tier_score = 10.0
        tier_advice = "涨停龙头、容量/趋势股共振，可优先看分歧后的承接"
    elif limit_count >= 1 and strong_count >= 2:
        tier_label = "梯队尚可"
        tier_score = 6.0
        tier_advice = "有龙头也有跟随，继续看扩散能否维持"
    elif limit_count >= 1:
        tier_label = "单龙带动"
        tier_score = -3.0
        tier_advice = "龙头强但梯队薄，追高要等板块跟随"
    elif strong_count >= 2:
        tier_label = "趋势梯队"
        tier_score = 4.0
        tier_advice = "无涨停龙头但趋势跟随尚可，适合低吸确认"
    elif weak_count >= max(2, len(latest_frame) // 2):
        tier_label = "梯队弱"
        tier_score = -8.0
        tier_advice = "龙头名单内弱股偏多，板块扩散不足"
    else:
        tier_label = "梯队普通"
        tier_score = 0.0
        tier_advice = "梯队强度普通，按个股承接处理"

    return {
        "leaders": grouped[:5],
        "summary": " / ".join(summary_bits) or "--",
        "tier_label": tier_label,
        "tier_score": tier_score,
        "tier_advice": tier_advice,
        "strong_leader_count": int(strong_count),
        "limit_leader_count": int(limit_count),
    }


def _theme_lifecycle(item):
    if not item:
        return {
            "theme_lifecycle": "--",
            "theme_lifecycle_score": 0.0,
            "theme_lifecycle_advice": "--",
            "theme_lifecycle_warnings": [],
        }
    shape = str(item.get("latest_shape") or "")
    conclusion = str(item.get("conclusion") or "")
    appear_days = int(item.get("appear_days") or 0)
    total_days = int(item.get("total_days") or 0)
    mainline_days = int(item.get("mainline_days") or 0)
    latest_hot_rank = _to_float(item.get("latest_hot_rank"))
    avg_hot_rank = _to_float(item.get("avg_hot_rank"))
    leader_tier = str(item.get("leader_tier_label") or "")
    warnings = []

    if "放量下跌" in shape or conclusion == "分歧回避":
        stage = "分歧退潮"
        score = -18.0
        advice = "先等风险释放，不能用龙头反抽替代板块修复"
    elif "加速高潮" in shape or conclusion == "过热不追":
        stage = "高潮加速"
        score = -10.0
        advice = "不追加速段，只等开板换手后的二次承接"
    elif "高位分化" in shape:
        stage = "高位分化"
        score = -12.0
        advice = "强弱分化后容易回撤，只看核心辨识度"
    elif "低位放量启动" in shape or (appear_days <= 1 and latest_hot_rank is not None and latest_hot_rank <= 20):
        stage = "启动"
        score = 8.0
        advice = "看次日放量延续和板块扩散，首分歧承接更关键"
    elif "主线延续" in shape and appear_days >= 2 and mainline_days >= 2:
        stage = "主升延续"
        score = 14.0
        advice = "主线延续，优先等核心回踩承接或弱转强"
    elif "广度扩散" in shape or conclusion == "持续稳定关注":
        stage = "扩散发酵"
        score = 10.0
        advice = "板块扩散中，优先选有量有承接的前排"
    elif "少数股拉动" in shape:
        stage = "单点试探"
        score = -3.0
        advice = "先看是否从单龙扩散到跟涨股"
        warnings.append("少数股拉动")
    elif "缩量反弹" in shape:
        stage = "缩量反弹"
        score = -6.0
        advice = "资金确认不足，不作为优先方向"
    elif appear_days >= 3 and latest_hot_rank is not None and avg_hot_rank is not None and latest_hot_rank > avg_hot_rank + 8:
        stage = "降温"
        score = -8.0
        advice = "排名走弱，降低仓位和预期"
    elif total_days and appear_days >= min(3, total_days):
        stage = "稳定观察"
        score = 4.0
        advice = "有持续度但缺强确认，等龙头和广度共振"
    else:
        stage = "观察"
        score = 0.0
        advice = "生命周期不够清晰，按盘口和个股结构确认"

    if leader_tier == "梯队强" and score > 0:
        score += 3.0
    elif leader_tier == "梯队弱" and score > 0:
        score -= 5.0
        warnings.append("梯队弱削弱生命周期")

    return {
        "theme_lifecycle": stage,
        "theme_lifecycle_score": round(score, 2),
        "theme_lifecycle_advice": advice,
        "theme_lifecycle_warnings": warnings,
    }


def _top_leader_code_on_date(board_group, date_value):
    if not date_value:
        return None
    rows = board_group[board_group["_review_date"].astype(str) == str(date_value)].copy()
    if rows.empty:
        return None
    rows["_leader_rank_tmp"] = rows["板块内龙头排名"].apply(_to_float)
    rows = rows.sort_values("_leader_rank_tmp", na_position="last")
    return _normalize_code(rows.iloc[0].get("股票代码"))


def _leader_transition_lookup(leaders):
    if leaders is None or leaders.empty or "_review_date" not in leaders.columns:
        return {}
    frame = leaders.copy()
    frame["股票代码"] = frame["股票代码"].apply(_normalize_code)
    frame = frame[frame["股票代码"].notna()].copy()
    if frame.empty:
        return {}

    lookup = {}
    for board_name, board_group in frame.groupby("板块名称", sort=False):
        board_name = str(board_name or "").strip()
        if not board_name:
            continue
        board_group = board_group.copy()
        board_group["_review_date_text"] = board_group["_review_date"].astype(str)
        board_dates = sorted(date for date in board_group["_review_date_text"].dropna().unique() if date)
        if not board_dates:
            continue
        latest_board_date = board_dates[-1]
        previous_board_date = board_dates[-2] if len(board_dates) >= 2 else None
        previous_top_code = _top_leader_code_on_date(board_group, previous_board_date)

        for code, group in board_group.groupby("股票代码", sort=False):
            code = _normalize_code(code)
            if not code:
                continue
            group = group.sort_values("_review_date_text")
            latest = group.iloc[-1].to_dict()
            latest_date = str(latest.get("_review_date_text") or latest.get("_review_date") or "")
            appeared_dates = sorted(date for date in group["_review_date_text"].dropna().unique() if date)
            previous_rows = group[group["_review_date_text"] < latest_date]
            previous = previous_rows.iloc[-1].to_dict() if not previous_rows.empty else {}
            latest_rank = _to_float(latest.get("板块内龙头排名"))
            previous_rank = _to_float(previous.get("板块内龙头排名"))
            rank_change = None
            if latest_rank is not None and previous_rank is not None:
                rank_change = previous_rank - latest_rank
            is_latest_report = latest_date == latest_board_date
            is_new = is_latest_report and len(appeared_dates) == 1
            is_top_switch = bool(
                is_latest_report
                and latest_rank == 1
                and previous_top_code
                and previous_top_code != code
            )
            lookup[(board_name, code)] = {
                "board_name": board_name,
                "stock_code": code,
                "stock_name": latest.get("股票名称") or code,
                "appear_days": int(len(appeared_dates)),
                "latest_date": latest_date,
                "first_date": appeared_dates[0] if appeared_dates else None,
                "latest_rank": latest_rank,
                "previous_rank": previous_rank,
                "rank_change": rank_change,
                "latest_type": latest.get("龙头类型"),
                "previous_type": previous.get("龙头类型"),
                "is_latest_report": is_latest_report,
                "is_new": is_new,
                "is_top_switch": is_top_switch,
                "previous_top_code": previous_top_code,
            }
    return lookup


def _board_trend_review(boards, leaders=None, total_days=0):
    if boards is None or boards.empty:
        return {}
    total_days = int(total_days or boards["_review_date"].nunique() or 0)
    review = {}
    for name, group in boards.groupby("_board_name", sort=False):
        name = str(name or "").strip()
        if not name:
            continue
        group = group.sort_values("_review_date")
        latest = group.iloc[-1].to_dict()
        first = group.iloc[0].to_dict()
        appear_days = int(group["_review_date"].nunique())
        hot_ranks = [_to_float(value) for value in group.get("热点排名", pd.Series(dtype=object)).tolist()]
        attention_ranks = [_to_float(value) for value in group.get("关注排名", pd.Series(dtype=object)).tolist()]
        hot_ranks = [value for value in hot_ranks if value is not None]
        attention_ranks = [value for value in attention_ranks if value is not None]
        latest_hot_rank = _to_float(latest.get("热点排名"))
        latest_attention_rank = _to_float(latest.get("关注排名"))
        first_hot_rank = _to_float(first.get("热点排名"))
        first_attention_rank = _to_float(first.get("关注排名"))
        mainline_days = int(
            group["热点版本"].astype(str).isin(["主线热点", "强势热点"]).sum()
            if "热点版本" in group.columns
            else 0
        )
        board_score = _score_board(latest)
        stability_score = board_score["score"]
        stability_score += min(24, appear_days * 7)
        if total_days and appear_days >= min(3, total_days):
            stability_score += 8
        if mainline_days >= 2:
            stability_score += 8
        if latest_hot_rank is not None and latest_hot_rank <= 10:
            stability_score += 10
        elif latest_hot_rank is not None and latest_hot_rank <= 20:
            stability_score += 5
        if hot_ranks and sum(hot_ranks) / len(hot_ranks) <= 20:
            stability_score += 8
        if first_hot_rank is not None and latest_hot_rank is not None and latest_hot_rank < first_hot_rank:
            stability_score += 6
        if (
            first_attention_rank is not None
            and latest_attention_rank is not None
            and latest_attention_rank < first_attention_rank
        ):
            stability_score += 5

        warnings = list(board_score["warnings"])
        latest_shape = str(latest.get("参与形态") or "")
        latest_advice = str(latest.get("参与建议") or "")
        if "放量下跌" in latest_shape or "回避" in latest_advice:
            stability_score -= 18
            warnings.append("昨日分歧/回避")
        elif "加速高潮" in latest_shape or "谨慎追高" in latest_advice:
            stability_score -= 8
            warnings.append("昨日高潮/不宜追")

        if "昨日分歧/回避" in warnings:
            conclusion = "分歧回避"
        elif "昨日高潮/不宜追" in warnings:
            conclusion = "过热不追"
        elif stability_score >= 72 and appear_days >= 2:
            conclusion = "持续稳定关注"
        elif stability_score >= 58:
            conclusion = "可跟踪但等确认"
        else:
            conclusion = "弱观察"

        leader_info = _leader_stability(leaders, name)
        review_item = {
            "board_name": name,
            "score": round(stability_score, 2),
            "conclusion": conclusion,
            "appear_days": appear_days,
            "total_days": total_days,
            "mainline_days": mainline_days,
            "latest_hot_rank": latest_hot_rank,
            "latest_attention_rank": latest_attention_rank,
            "avg_hot_rank": round(sum(hot_ranks) / len(hot_ranks), 2) if hot_ranks else None,
            "avg_attention_rank": round(sum(attention_ranks) / len(attention_ranks), 2) if attention_ranks else None,
            "latest_version": latest.get("热点版本"),
            "latest_shape": latest.get("参与形态"),
            "latest_advice": latest.get("参与建议"),
            "reasons": list(dict.fromkeys(board_score["reasons"])),
            "warnings": list(dict.fromkeys(warnings)),
            "leader_summary": leader_info["summary"],
            "leaders": leader_info["leaders"],
            "leader_tier_label": leader_info.get("tier_label"),
            "leader_tier_score": leader_info.get("tier_score"),
            "leader_tier_advice": leader_info.get("tier_advice"),
            "strong_leader_count": leader_info.get("strong_leader_count"),
            "limit_leader_count": leader_info.get("limit_leader_count"),
        }
        lifecycle = _theme_lifecycle(review_item)
        review_item.update(lifecycle)
        review[name] = review_item
    return review


def _top_board_reviews(board_review, limit=8):
    values = list((board_review or {}).values())
    values = sorted(
        values,
        key=lambda item: (
            -(item.get("score") or 0),
            item.get("latest_hot_rank") or 999,
            -(item.get("appear_days") or 0),
        ),
    )
    return values[: int(limit or 8)]


def _focus_board_names(boards, board_limit=25):
    if boards is None or boards.empty:
        return []
    frame = boards.copy()
    frame["_board_name"] = frame.apply(_board_name, axis=1)
    frame = frame[frame["_board_name"].astype(bool)].copy()
    frame = frame[frame.apply(_is_focus_board, axis=1)].copy()
    if frame.empty:
        return []
    frame["_sort_hot_rank"] = frame.apply(lambda row: _board_rank_tuple(row)[0], axis=1)
    frame["_sort_attention_rank"] = frame.apply(lambda row: _board_rank_tuple(row)[1], axis=1)
    frame["_board_score"] = frame.apply(lambda row: _score_board(row)["score"], axis=1)
    frame = frame.sort_values(["_board_score", "_sort_hot_rank", "_sort_attention_rank"], ascending=[False, True, True])
    return frame.head(int(board_limit or 25))["_board_name"].tolist()


def _load_industry_leader_candidates(leaders, boards=None, board_limit=25, leaders_per_board=3):
    if leaders is None or leaders.empty:
        return pd.DataFrame()
    frame = leaders.copy()
    frame["股票代码"] = frame["股票代码"].apply(_normalize_code)
    frame = frame[frame["股票代码"].notna()].copy()
    frame["_hot_rank"] = frame["热点排名"].apply(_to_float) if "热点排名" in frame.columns else math.inf
    frame["_leader_rank"] = frame["板块内龙头排名"].apply(_to_float) if "板块内龙头排名" in frame.columns else 999
    focus_boards = _focus_board_names(boards, board_limit=board_limit)
    if focus_boards:
        frame = frame[frame["板块名称"].isin(focus_boards)].copy()
    else:
        frame = frame[frame.apply(_is_focus_board, axis=1)].copy()
    frame = frame.sort_values(["_hot_rank", "板块名称", "_leader_rank"]).groupby("板块名称", sort=False).head(
        int(leaders_per_board)
    )
    return frame


def _fetch_sina_quotes(stock_codes, timeout=8, chunk_size=80):
    symbols = []
    symbol_to_code = {}
    for code in stock_codes:
        symbol = _realtime_symbol(code)
        if not symbol:
            continue
        symbols.append(symbol)
        symbol_to_code[symbol] = code

    quotes = {}
    for start in range(0, len(symbols), chunk_size):
        chunk = symbols[start : start + chunk_size]
        url = SINA_REALTIME_URL.format(symbols=",".join(chunk))
        try:
            response = requests.get(url, headers=SINA_HEADERS, timeout=timeout)
            response.raise_for_status()
            response.encoding = "gb18030"
        except Exception:
            continue
        for matched in re.finditer(r'var hq_str_([^=]+)="(.*?)";', response.text or ""):
            symbol, raw = matched.group(1), matched.group(2)
            code = symbol_to_code.get(symbol)
            quote = _parse_sina_quote(symbol, raw)
            if code and quote:
                quotes[code] = quote
    return quotes


def _parse_sina_quote(symbol, raw):
    fields = str(raw or "").split(",")
    if len(fields) < 32:
        return None
    current = _to_float(fields[3])
    prev_close = _to_float(fields[2])
    open_price = _to_float(fields[1])
    high = _to_float(fields[4])
    low = _to_float(fields[5])
    amount = _to_float(fields[9])
    volume = _to_float(fields[8])
    if current is None or current <= 0:
        return None
    return {
        "symbol": symbol,
        "name": fields[0].strip(),
        "open": _round(open_price, 2),
        "prev_close": _round(prev_close, 2),
        "current": _round(current, 2),
        "high": _round(high, 2),
        "low": _round(low, 2),
        "bid": _round(fields[6], 2),
        "ask": _round(fields[7], 2),
        "volume_hands": _round((volume or 0) / 100.0, 2),
        "amount": _round(amount, 2),
        "change_pct": _round((current / prev_close - 1) * 100 if prev_close else None, 2),
        "open_gap_pct": _round((open_price / prev_close - 1) * 100 if prev_close else None, 2),
        "current_from_open_pct": _round((current / open_price - 1) * 100 if open_price else None, 2),
        "low_from_prev_pct": _round((low / prev_close - 1) * 100 if prev_close else None, 2),
        "high_from_prev_pct": _round((high / prev_close - 1) * 100 if prev_close else None, 2),
        "quote_datetime": f"{fields[30].strip()} {fields[31].strip()}".strip(),
    }


def _fetch_market_summary():
    symbols = "s_sh000001,s_sz399001,s_sz399006"
    url = SINA_REALTIME_URL.format(symbols=symbols)
    try:
        response = requests.get(url, headers=SINA_HEADERS, timeout=6)
        response.raise_for_status()
        response.encoding = "gb18030"
    except Exception:
        return []
    indices = []
    for matched in re.finditer(r'var hq_str_([^=]+)="(.*?)";', response.text or ""):
        fields = matched.group(2).split(",")
        if len(fields) >= 4:
            indices.append(
                {
                    "name": fields[0].strip(),
                    "current": _round(fields[1], 2),
                    "change": _round(fields[2], 2),
                    "change_pct": _round(fields[3], 2),
                }
            )
    return indices


def _quiet_call(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return func(*args, **kwargs)


def _event_date_range(lookback_days):
    end_ts = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    begin_ts = end_ts - pd.Timedelta(days=int(lookback_days or DEFAULT_EVENT_LOOKBACK_DAYS))
    return begin_ts, end_ts


def _event_dt(value):
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else parsed


def _event_item_date(value):
    parsed = _event_dt(value)
    if parsed is None:
        return None
    return parsed.strftime("%Y-%m-%d")


def _load_notice_items(ak, stock_code, begin_ts, end_ts, max_items=8):
    try:
        frame = _quiet_call(
            ak.stock_individual_notice_report,
            security=stock_code,
            symbol="全部",
            begin_date=begin_ts.strftime("%Y%m%d"),
            end_date=end_ts.strftime("%Y%m%d"),
        )
    except Exception as error:
        return [], f"公告获取失败:{error}"
    if frame is None or frame.empty:
        return [], "近期开奖公告为空"

    title_col = "公告标题" if "公告标题" in frame.columns else None
    if title_col is None:
        title_candidates = [column for column in frame.columns if "标题" in str(column) or "名称" in str(column)]
        title_col = title_candidates[0] if title_candidates else frame.columns[min(2, len(frame.columns) - 1)]
    type_col = "公告类型" if "公告类型" in frame.columns else None
    date_col = "公告日期" if "公告日期" in frame.columns else None
    url_col = "网址" if "网址" in frame.columns else None

    data = frame.copy()
    if date_col:
        data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
        data = data[(data[date_col] >= begin_ts) & (data[date_col] <= end_ts)].copy()
        data = data.sort_values(date_col, ascending=False)
    items = []
    for _, row in data.head(int(max_items or 8)).iterrows():
        title = str(row.get(title_col) or "").strip()
        if not title:
            continue
        items.append(
            {
                "kind": "公告",
                "date": _event_item_date(row.get(date_col)) if date_col else None,
                "title": title,
                "content": "",
                "type": str(row.get(type_col) or "").strip() if type_col else "",
                "source": "东方财富公告",
                "url": row.get(url_col) if url_col else None,
            }
        )
    return items, f"公告{len(items)}条"


def _load_news_items(ak, stock_code, begin_ts, end_ts, max_items=8):
    try:
        frame = _quiet_call(ak.stock_news_em, symbol=stock_code)
    except Exception as error:
        return [], f"新闻获取失败:{error}"
    if frame is None or frame.empty:
        return [], "近阶段新闻为空"

    title_col = "新闻标题" if "新闻标题" in frame.columns else None
    content_col = "新闻内容" if "新闻内容" in frame.columns else None
    date_col = "发布时间" if "发布时间" in frame.columns else None
    source_col = "文章来源" if "文章来源" in frame.columns else None
    url_col = "新闻链接" if "新闻链接" in frame.columns else None
    if title_col is None:
        title_candidates = [column for column in frame.columns if "标题" in str(column)]
        title_col = title_candidates[0] if title_candidates else frame.columns[0]

    data = frame.copy()
    if date_col:
        data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
        data = data[(data[date_col] >= begin_ts) & (data[date_col] <= end_ts)].copy()
        data = data.sort_values(date_col, ascending=False)
    items = []
    for _, row in data.head(int(max_items or 8)).iterrows():
        title = str(row.get(title_col) or "").strip()
        if not title:
            continue
        items.append(
            {
                "kind": "新闻",
                "date": _event_item_date(row.get(date_col)) if date_col else None,
                "title": title,
                "content": str(row.get(content_col) or "").strip() if content_col else "",
                "type": "",
                "source": str(row.get(source_col) or "东方财富新闻").strip() if source_col else "东方财富新闻",
                "url": row.get(url_col) if url_col else None,
            }
        )
    return items, f"新闻{len(items)}条"


def _dedupe_event_items(items):
    seen = set()
    unique = []
    for item in items:
        key = (item.get("date"), item.get("title"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _match_event_text(text):
    text = _clean_event_text(text)
    catalyst_hits = event_overlay._match_event_catalysts(text)
    risk_labels = []
    risk_score = 0.0
    block_formal = False
    positive_score = 0.0
    for keyword, score, hard_block in EVENT_RISK_PATTERNS:
        if keyword in text:
            risk_score += score
            block_formal = block_formal or hard_block
            risk_labels.append(keyword)
    for keyword, score in EVENT_POSITIVE_PATTERNS:
        if keyword in text:
            positive_score += score
    if any(phrase in text for phrase in EVENT_NEGATIVE_CATALYST_PHRASES):
        catalyst_hits = [
            hit
            for hit in catalyst_hits
            if hit.get("label") not in {"战略合作/题材", "产能释放"}
        ]
        risk_score += 3.0
        risk_labels.append("题材澄清")
    return catalyst_hits, risk_labels, risk_score, block_formal, positive_score


def _is_company_specific_event_item(item, stock_code, stock_name=None):
    if (item or {}).get("kind") == "公告":
        return True
    title = str((item or {}).get("title") or "")
    name = str(stock_name or "").strip()
    code = _normalize_code(stock_code)
    return bool((name and name in title) or (code and code in title))


def _clean_event_text(text):
    text = str(text or "")
    patterns = [
        r"免责声明[:：]?\s*本文基于AI生产[^。；\n]*[。；]?",
        r"本文基于AI生产[^。；\n]*[。；]?",
        r"不构成任何投资建议",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text


def _bounded(value, lower, upper):
    value = _to_float(value)
    if value is None:
        return lower
    return max(lower, min(upper, value))


def _event_item_truth_score(item, matched=None, risk_labels=None, positive_score=0.0):
    item = item or {}
    matched = matched or []
    risk_labels = risk_labels or []
    text = _clean_event_text(
        " ".join(
            str(part or "")
            for part in [item.get("title"), item.get("content"), item.get("type"), item.get("source")]
        )
    )
    score = 0.0
    reasons = []
    warnings = []

    if item.get("kind") == "公告":
        score += 5.0
        reasons.append("公告来源")
    elif item.get("kind") == "新闻":
        score += 1.0
        warnings.append("新闻来源")

    source = str(item.get("source") or "")
    if "公告" in source:
        score += 1.0
    if any(keyword in text for keyword in ["交易所", "上交所", "深交所", "北交所", "证监"]):
        score += 2.0
        reasons.append("监管/交易所口径")

    for keyword, weight, label in EVENT_HIGH_TRUTH_PATTERNS:
        if keyword in text:
            score += weight
            reasons.append(label)
    for keyword, weight, label in EVENT_LOW_TRUTH_PATTERNS:
        if keyword in text:
            score += weight
            warnings.append(label)

    if matched:
        score += min(3.0, len(matched) * 1.0)
        reasons.append("命中事件催化")
    if positive_score:
        score += min(2.0, _to_float(positive_score) or 0.0)
    if risk_labels:
        score -= min(6.0, len(risk_labels) * 2.0)
        warnings.append("伴随风险提示")

    event_date = _parse_date(item.get("date"))
    if event_date:
        days = (dt.date.today() - event_date).days
        if days <= 3:
            score += 1.0
            reasons.append("近3日事件")
        elif days <= 7:
            score += 0.5

    has_hard_landing = any(keyword in text for keyword, _, _ in EVENT_HIGH_TRUTH_PATTERNS)
    is_soft_disclosure = any(keyword in text for keyword in ["投资者关系活动记录", "业绩说明会", "现金分红说明会"])
    if is_soft_disclosure and not has_hard_landing:
        score = min(score, 5.5)
        warnings.append("交流纪要不等于硬催化")

    score = round(_bounded(score, -10.0, 16.0), 2)
    return {
        "score": score,
        "grade": _event_truth_grade(score, has_catalyst=bool(matched or positive_score), has_risk=bool(risk_labels)),
        "reasons": list(dict.fromkeys(reasons)),
        "warnings": list(dict.fromkeys(warnings)),
    }


def _event_truth_grade(score, has_catalyst=False, has_risk=False, block_formal=False):
    score = _to_float(score) or 0.0
    if block_formal:
        return "风险"
    if not has_catalyst and has_risk:
        return "风险"
    if not has_catalyst:
        return "无催化"
    if score >= 10:
        return "A强兑现"
    if score >= 6:
        return "B较可信"
    if score >= 2:
        return "C题材观察"
    return "D弱相关"


def _event_truth_display(context):
    if not context:
        return "--"
    grade = str(context.get("event_truth_grade") or "").strip()
    score = _to_float(context.get("event_truth_score"))
    if not grade:
        return "--"
    if score is None:
        return grade
    return f"{grade}({round(score, 1)})"


def _build_event_context(stock_code, stock_name=None, lookback_days=DEFAULT_EVENT_LOOKBACK_DAYS):
    begin_ts, end_ts = _event_date_range(lookback_days)
    try:
        import akshare as ak
    except Exception as error:
        return {
            "status": "akshare_unavailable",
            "stock_code": stock_code,
            "stock_name": stock_name,
            "note": f"akshare不可用:{error}",
            "items": [],
            "event_catalyst": event_overlay._build_event_catalyst_summary([], 0.0),
            "event_truth_score": 0.0,
            "event_truth_grade": "未扫描",
            "event_truth_reasons": [],
            "event_truth_warnings": [],
            "risk_score": 0.0,
            "positive_score": 0.0,
            "risk_labels": [],
            "block_formal": False,
        }

    notice_items, notice_note = _load_notice_items(ak, stock_code, begin_ts, end_ts)
    news_items, news_note = _load_news_items(ak, stock_code, begin_ts, end_ts)
    items = _dedupe_event_items([*notice_items, *news_items])

    catalyst_hits = []
    catalyst_score = 0.0
    risk_score = 0.0
    positive_score = 0.0
    risk_labels = []
    block_formal = False
    matched_items = []
    truth_items = []
    for item in items:
        if not _is_company_specific_event_item(item, stock_code, stock_name=stock_name):
            continue
        text = " ".join(
            str(part or "")
            for part in [item.get("title"), item.get("content"), item.get("type"), item.get("source")]
        )
        matched, item_risk_labels, item_risk_score, item_block, item_positive_score = _match_event_text(text)
        item_truth = _event_item_truth_score(
            item,
            matched=matched,
            risk_labels=item_risk_labels,
            positive_score=item_positive_score,
        )
        if matched:
            item["catalysts"] = [match.get("label") for match in matched if match.get("label")]
            for match in matched:
                catalyst_score += _to_float(match.get("score")) or 0.0
                catalyst_hits.append(
                    {
                        **match,
                        "date": item.get("date"),
                        "title": item.get("title"),
                        "source": item.get("source"),
                        "kind": item.get("kind"),
                    }
                )
        if item_risk_labels:
            item["risk_labels"] = item_risk_labels
        if matched or item_risk_labels or item_positive_score:
            item["truth_grade"] = item_truth["grade"]
            item["truth_score"] = item_truth["score"]
            truth_items.append(item_truth)
        if matched or item_risk_labels or item_positive_score:
            matched_items.append(item)
        risk_labels.extend(item_risk_labels)
        risk_score += item_risk_score
        positive_score += item_positive_score
        block_formal = block_formal or item_block

    risk_labels = list(dict.fromkeys(risk_labels))
    catalyst_summary = event_overlay._build_event_catalyst_summary(catalyst_hits, catalyst_score)
    truth_items = sorted(truth_items, key=lambda item: item.get("score") or 0.0, reverse=True)
    best_truth = truth_items[0] if truth_items else {}
    event_truth_score = best_truth.get("score", 0.0)
    event_truth_grade = _event_truth_grade(
        event_truth_score,
        has_catalyst=bool(catalyst_summary.get("primary") or positive_score),
        has_risk=bool(risk_labels),
        block_formal=block_formal,
    )
    event_truth_reasons = list(
        dict.fromkeys(reason for item in truth_items for reason in (item.get("reasons") or []))
    )
    event_truth_warnings = list(
        dict.fromkeys(warning for item in truth_items for warning in (item.get("warnings") or []))
    )
    latest = (matched_items or items or [{}])[0]
    note_bits = [
        f"近{lookback_days}日",
        notice_note,
        news_note,
    ]
    if latest:
        note_bits.append(f"最新:{latest.get('date') or '--'} {latest.get('title') or '--'}")
    if risk_labels:
        note_bits.append(f"风险:{'、'.join(risk_labels[:5])}")
    if catalyst_summary.get("primary"):
        note_bits.append(f"催化:{catalyst_summary.get('level')}/{catalyst_summary.get('primary')}")
    if event_truth_grade not in {"无催化", "未扫描"}:
        note_bits.append(f"真实性:{event_truth_grade}")

    return {
        "status": "ok",
        "stock_code": stock_code,
        "stock_name": stock_name,
        "note": "；".join(note_bits),
        "items": items[:12],
        "latest_item": latest,
        "event_catalyst": catalyst_summary,
        "event_truth_score": round(event_truth_score, 2),
        "event_truth_grade": event_truth_grade,
        "event_truth_reasons": event_truth_reasons[:8],
        "event_truth_warnings": event_truth_warnings[:8],
        "risk_score": round(min(risk_score, 30.0), 2),
        "positive_score": round(min(positive_score + (catalyst_summary.get("score") or 0.0), 12.0), 2),
        "risk_labels": risk_labels[:10],
        "block_formal": bool(block_formal),
    }


def _event_adjustment(context):
    if not context:
        return 0.0, [], []
    catalyst = context.get("event_catalyst") or {}
    level = str(catalyst.get("level") or "无")
    primary = catalyst.get("primary")
    risk_score = _to_float(context.get("risk_score")) or 0.0
    positive_score = _to_float(context.get("positive_score")) or 0.0
    block_formal = bool(context.get("block_formal"))
    truth_grade = str(context.get("event_truth_grade") or "")
    score = 0.0
    reasons = []
    warnings = []

    if primary and level == "强":
        score += 10
        reasons.append(f"强事件催化:{primary}")
    elif primary and level == "中":
        score += 6
        reasons.append(f"中事件催化:{primary}")
    elif primary and level == "弱":
        score += 3
        reasons.append(f"弱事件催化:{primary}")
    elif positive_score > 0:
        score += min(3, positive_score)
        reasons.append("公告/新闻偏正面")

    if primary or positive_score > 0:
        if truth_grade == "A强兑现":
            score += 3
            reasons.append("事件真实性高")
        elif truth_grade == "B较可信":
            score += 1
            reasons.append("事件较可信")
        elif truth_grade == "C题材观察":
            score -= 2
            warnings.append("事件仍偏题材观察")
        elif truth_grade == "D弱相关":
            score -= 5
            warnings.append("事件弱相关或待验证")

    if block_formal:
        score -= 25
        warnings.append("硬风险事件")
    elif risk_score >= 8:
        score -= 14
        warnings.append("事件风险较高")
    elif risk_score >= 4:
        score -= 8
        warnings.append("事件风险需降级")
    elif risk_score > 0:
        score -= 3
        warnings.append("事件风险提醒")

    labels = context.get("risk_labels") or []
    if labels:
        warnings.append("事件风险:" + "、".join(labels[:4]))
    return round(score, 2), reasons, warnings


def _event_display(context):
    if not context:
        return "--"
    catalyst = context.get("event_catalyst") or {}
    primary = catalyst.get("primary")
    if primary:
        return f"{catalyst.get('level') or '--'}:{primary}"
    labels = context.get("risk_labels") or []
    if labels:
        return "风险:" + "、".join(labels[:2])
    if context.get("status") == "ok":
        return "无明确催化"
    return context.get("status") or "--"


def _event_trade_note(context):
    if not context:
        return "--"
    catalyst = context.get("event_catalyst") or {}
    if catalyst.get("trade_note"):
        return catalyst.get("trade_note")
    labels = context.get("risk_labels") or []
    if labels:
        return "事件风险存在，先降级看承接"
    return "未识别出明确事件催化，按盘面和板块处理"


def _append_text(existing, additions):
    bits = [] if not existing or existing == "--" else str(existing).split("；")
    bits.extend(additions or [])
    bits = [bit for bit in bits if bit and bit != "--"]
    return "；".join(dict.fromkeys(bits)) or "--"


def _apply_event_context_to_rows(rows, event_contexts):
    updated = []
    for row in rows:
        row = dict(row)
        context = event_contexts.get(row.get("stock_code"))
        adjustment, reasons, warnings = _event_adjustment(context)
        if context:
            row["score"] = round((_to_float(row.get("score")) or 0.0) + adjustment, 2)
            row["reasons"] = _append_text(row.get("reasons"), reasons)
            row["warnings"] = _append_text(row.get("warnings"), warnings)
            row["event_display"] = _event_display(context)
            row["event_note"] = context.get("note")
            row["event_trade_note"] = _event_trade_note(context)
            row["event_risk_labels"] = "、".join(context.get("risk_labels") or [])
            row["event_level"] = (context.get("event_catalyst") or {}).get("level")
            row["event_primary"] = (context.get("event_catalyst") or {}).get("primary")
            row["event_truth_score"] = context.get("event_truth_score")
            row["event_truth_grade"] = context.get("event_truth_grade")
            row["event_truth_display"] = _event_truth_display(context)
            row["event_truth_reasons"] = "、".join(context.get("event_truth_reasons") or [])
            row["event_truth_warnings"] = "、".join(context.get("event_truth_warnings") or [])
            row["event_latest_title"] = (context.get("latest_item") or {}).get("title")
            row["event_latest_date"] = (context.get("latest_item") or {}).get("date")
            row["event_status"] = context.get("status")
        else:
            row.setdefault("event_display", "--")
            row.setdefault("event_note", "--")
            row.setdefault("event_trade_note", "--")
            row.setdefault("event_risk_labels", "")
            row.setdefault("event_level", None)
            row.setdefault("event_primary", None)
            row.setdefault("event_truth_score", None)
            row.setdefault("event_truth_grade", "未扫描")
            row.setdefault("event_truth_display", "--")
            row.setdefault("event_truth_reasons", "")
            row.setdefault("event_truth_warnings", "")
            row.setdefault("event_latest_title", None)
            row.setdefault("event_latest_date", None)
            row.setdefault("event_status", "not_scanned")
        updated.append(row)
    return updated


def _add_candidate(candidates, code, name=None):
    code = _normalize_code(code)
    if not code:
        return None
    item = candidates.setdefault(
        code,
        {
            "stock_code": code,
            "stock_name": name or code,
            "sources": [],
            "strategy": None,
            "strategies": [],
            "industry_leaders": [],
        },
    )
    if name and (not item.get("stock_name") or item.get("stock_name") == code):
        item["stock_name"] = name
    return item


def _merge_candidates(strategy_frame, leader_frame):
    candidates = {}
    if strategy_frame is not None and not strategy_frame.empty:
        for row in strategy_frame.to_dict("records"):
            item = _add_candidate(candidates, row.get("stock_code"), row.get("stock_name"))
            if not item:
                continue
            item["strategies"].append(row)
            if item.get("strategy") is None:
                item["strategy"] = row
            item["sources"].append("策略落盘")

    if leader_frame is not None and not leader_frame.empty:
        for row in leader_frame.to_dict("records"):
            item = _add_candidate(candidates, row.get("股票代码"), row.get("股票名称"))
            if not item:
                continue
            item["industry_leaders"].append(row)
            item["sources"].append("行业龙头")
    return list(candidates.values())


def _amount_ratio(quote, strategy, leader, board=None):
    amount = _to_float((quote or {}).get("amount"))
    base = _to_float((strategy or {}).get("amount_avg_5d"))
    if base is None and leader:
        amount_yi = _to_float(leader.get("今日成交额_亿"))
        ratio = _to_float(leader.get("成交额较5日均额倍数"))
        if amount_yi is not None and ratio:
            base = amount_yi * 100000000 / ratio
    if base is None and board:
        amount_yi = _to_float(board.get("成交额亿"))
        ratio = _to_float(board.get("额比5日"))
        if amount_yi is not None and ratio:
            base = amount_yi * 100000000 / ratio
    if amount is None or not base:
        return None
    return round(amount / base, 2)


def _best_leader(industry_leaders):
    if not industry_leaders:
        return None
    return sorted(
        industry_leaders,
        key=lambda row: (
            _to_float(row.get("热点排名")) if _to_float(row.get("热点排名")) is not None else 999,
            _to_float(row.get("板块内龙头排名")) if _to_float(row.get("板块内龙头排名")) is not None else 999,
        ),
    )[0]


def _board_from_candidate(strategy, leader, board_lookup):
    board_lookup = board_lookup or {}
    if leader:
        name = str(leader.get("板块名称") or "").strip()
        if name in board_lookup:
            return board_lookup[name]
    for name in str((strategy or {}).get("industry") or "").split(","):
        name = name.strip()
        if name in board_lookup:
            return board_lookup[name]
    return None


def _board_review_from_candidate(strategy, leader, board_review):
    board_review = board_review or {}
    if leader:
        name = str(leader.get("板块名称") or "").strip()
        if name in board_review:
            return board_review[name]
    for name in str((strategy or {}).get("industry") or "").split(","):
        name = name.strip()
        if name in board_review:
            return board_review[name]
    return None


def _leader_transition_from_candidate(leader, leader_transition_lookup):
    if not leader:
        return None
    board_name = str(leader.get("板块名称") or "").strip()
    stock_code = _normalize_code(leader.get("股票代码"))
    if not board_name or not stock_code:
        return None
    return (leader_transition_lookup or {}).get((board_name, stock_code))


def _leader_rank_text(transition):
    if not transition:
        return "--"
    latest_rank = _to_float(transition.get("latest_rank"))
    previous_rank = _to_float(transition.get("previous_rank"))
    if latest_rank is None:
        return "--"
    latest_text = f"第{int(latest_rank)}"
    if previous_rank is None:
        return f"新进{latest_text}"
    previous_text = f"第{int(previous_rank)}"
    if latest_rank < previous_rank:
        return f"{previous_text}->{latest_text}"
    if latest_rank > previous_rank:
        return f"{previous_text}->{latest_text}回落"
    return f"{latest_text}延续"


def _score_leader_transition(transition, board_review_item):
    if not transition:
        return {
            "score": 0.0,
            "label": "--",
            "advice": "--",
            "rank_text": "--",
            "reasons": [],
            "warnings": [],
        }

    board_conclusion = str((board_review_item or {}).get("conclusion") or "")
    latest_rank = _to_float(transition.get("latest_rank"))
    rank_change = _to_float(transition.get("rank_change"))
    appear_days = int(transition.get("appear_days") or 0)
    is_new = bool(transition.get("is_new"))
    is_top_switch = bool(transition.get("is_top_switch"))
    score = 0.0
    reasons = []
    warnings = []

    if is_new and board_conclusion == "过热不追":
        label = "新晋但过热"
        score -= 4
        warnings.append("新晋龙头但板块过热")
        advice = "只看分歧承接，不追加速"
    elif is_new and board_conclusion == "分歧回避":
        label = "新晋但分歧"
        score -= 8
        warnings.append("新晋龙头但板块分歧")
        advice = "先回避，等板块风险释放"
    elif is_new and latest_rank == 1:
        label = "新晋主攻龙头"
        score += 16
        reasons.append("新晋板块第一龙头")
        advice = "重点看开盘承接和换手后的二次确认"
    elif is_new:
        label = "新晋龙头"
        score += 10
        reasons.append("最近新进板块龙头")
        advice = "可关注，但必须跟随板块强度确认"
    elif is_top_switch:
        label = "新龙替旧龙"
        score += 12
        reasons.append("板块第一龙头发生切换")
        advice = "重点看能否带动板块扩散"
    elif rank_change is not None and rank_change > 0:
        label = "龙头排名上升"
        score += 8
        reasons.append("龙头排名上升")
        advice = "看排名上升后能否放量延续"
    elif appear_days >= 2:
        label = "老龙延续"
        score += 5
        reasons.append(f"龙头连续出现{appear_days}天")
        advice = "按趋势延续处理，弱转强才加分"
    else:
        label = "普通龙头"
        advice = "按普通板块龙头观察"

    if latest_rank is not None and latest_rank <= 1 and board_conclusion not in {"过热不追", "分歧回避"}:
        score += 3
    if board_conclusion == "持续稳定关注" and label in {"新晋主攻龙头", "新晋龙头", "新龙替旧龙"}:
        score += 5
        reasons.append("新龙匹配稳定主线")
    if board_conclusion == "过热不追" and label not in {"新晋但过热"}:
        warnings.append("板块过热，龙头信号降级")
    if board_conclusion == "分歧回避" and label not in {"新晋但分歧"}:
        warnings.append("板块分歧，龙头信号降级")

    return {
        "score": round(score, 2),
        "label": label,
        "advice": advice,
        "rank_text": _leader_rank_text(transition),
        "reasons": reasons,
        "warnings": warnings,
    }


def _score_strategy_context(strategy, strategies):
    if not strategy:
        return 0.0, [], []
    age = _to_float(strategy.get("strategy_age_trade_days"))
    repeat_count = len(strategies or [])
    score = 0.0
    reasons = []
    warnings = []

    if age is None:
        score += 12
        reasons.append("近20日策略信号")
    elif age <= 1:
        score += 28
        reasons.append("近1日策略信号")
    elif age <= 4:
        score += 22
        reasons.append(f"近{int(age) + 1}日策略信号")
    elif age <= 9:
        score += 16
        reasons.append(f"近{int(age) + 1}日策略信号")
    else:
        score += 10
        reasons.append("较早策略信号")

    strategy_type = str(strategy.get("strategy_type") or "")
    if "adaptive" in strategy_type:
        score += 6
        reasons.append("自适应策略")
    elif "long_runway" in strategy_type:
        score += 3
        reasons.append("长跑跟踪")

    if repeat_count >= 2:
        score += min(8, 3 + repeat_count)
        reasons.append(f"近20日多次出现{repeat_count}次")
    if age is not None and age >= 10:
        warnings.append("策略信号偏旧")
    return score, reasons, warnings


def _safe_price_text(label, value):
    value = _round(value, 2)
    return f"{label}{value}" if value is not None else None


def _price_zone_text(label, low, high=None):
    low = _round(low, 2)
    high = _round(high if high is not None else low, 2)
    if low is None and high is None:
        return None
    if low is None:
        return f"{label}{high}"
    if high is None:
        return f"{label}{low}"
    zone_low, zone_high = sorted([low, high])
    if zone_low == zone_high:
        return f"{label}{zone_low}"
    return f"{label}{zone_low}-{zone_high}"


def _build_trade_texts(state, level, strategy, quote):
    current = _to_float((quote or {}).get("current"))
    prev_close = _to_float((quote or {}).get("prev_close"))
    open_price = _to_float((quote or {}).get("open"))
    day_high = _to_float((quote or {}).get("high"))
    day_low = _to_float((quote or {}).get("low"))
    ref_low = _to_float((strategy or {}).get("ref_low"))
    stop_price = _to_float((strategy or {}).get("stop_price"))

    supports = [
        _safe_price_text("防守", stop_price),
        _safe_price_text("昨收", prev_close),
        _safe_price_text("开盘", open_price),
    ]
    supports = [item for item in supports if item]
    support_text = " / ".join(supports[:2]) or "--"
    support_values = [
        value for value in [stop_price, prev_close, open_price] if _to_float(value) is not None
    ]
    support_zone_text = (
        _price_zone_text("支撑区", min(support_values[:2]), max(support_values[:2]))
        if support_values
        else support_text
    )
    high_text = _safe_price_text("日高", day_high) or "--"
    chase_zone_text = (
        _price_zone_text("高位区间", day_high * 0.985, day_high)
        if day_high is not None
        else "--"
    )
    weak_lines = [
        _safe_price_text("防守", stop_price),
        _safe_price_text("昨低", ref_low),
        _safe_price_text("日低", day_low),
    ]
    weak_text = " / ".join([item for item in weak_lines if item][:2]) or "--"

    if state == "跌破防守":
        return "无", f"不能快速收回{support_text}就不看", f"已跌破{weak_text}，先降风险"
    if state == "触板回落":
        return "等承接", f"不追；等重新站回{high_text}，或回踩不破{support_text}后再看", f"跌回{weak_text}或反抽不过{high_text}"
    if level in {"核心盯盘", "重点关注"}:
        if current is not None and day_high is not None and current >= day_high * 0.985:
            trigger = f"{chase_zone_text}不追；等回踩{support_zone_text}不破，或放量突破{high_text}再看"
        else:
            trigger = f"站稳{support_text}，并放量重新上攻{high_text}"
        invalid = f"跌回{weak_text}或冲高回落不收回昨收"
        return "等确认", trigger, invalid
    if level == "二级观察":
        return "观察", f"先收回{support_text}，再看二次放量", f"跌破{weak_text}或午后仍弱于昨收"
    return "放弃", f"除非快速收回{support_text}", f"跌破{weak_text}继续回避"


def _limit_up_pct(stock_code, stock_name=None):
    code = _normalize_code(stock_code)
    name = str(stock_name or "")
    if "ST" in name.upper():
        return 5.0
    if code and code.startswith(("300", "301", "688", "689")):
        return 20.0
    if code and code.startswith(("920", "8", "4")):
        return 30.0
    return 10.0


def _quote_time(quote):
    text = str((quote or {}).get("quote_datetime") or "")
    matched = re.search(r"(\d{2}):(\d{2})(?::\d{2})?", text)
    if not matched:
        return None
    try:
        return dt.time(int(matched.group(1)), int(matched.group(2)))
    except ValueError:
        return None


def _is_tail_session(quote):
    quote_time = _quote_time(quote)
    return bool(quote_time and quote_time >= dt.time(14, 30))


def _score_intraday_quality(
    quote,
    touched_limit_up=False,
    limit_fade=False,
    stop_broken=False,
    day_position=None,
    amount_ratio=None,
    limit_up_pct=10.0,
):
    if not quote:
        return {
            "score": 0.0,
            "quality": "无实时行情",
            "limit_quality": "--",
            "closing_acceptance": "--",
            "reasons": [],
            "warnings": [],
            "is_hard_weak": False,
            "quality_note": "无实时行情，不能判断盘口质量",
        }

    current = _to_float(quote.get("current"))
    prev_close = _to_float(quote.get("prev_close"))
    open_price = _to_float(quote.get("open"))
    high = _to_float(quote.get("high"))
    change_pct = _to_float(quote.get("change_pct"))
    day_position = _to_float(day_position)
    amount_ratio = _to_float(amount_ratio)
    limit_up_pct = _to_float(limit_up_pct) or 10.0
    score = 0.0
    reasons = []
    warnings = []
    quality = "普通承接"
    limit_quality = "未触板"
    closing_acceptance = "非尾盘"
    hard_weak = False

    near_high = bool(current is not None and high is not None and current >= high * 0.985)
    near_limit = bool(change_pct is not None and change_pct >= limit_up_pct - 0.35)

    if stop_broken:
        quality = "跌破防守"
        limit_quality = "防守破位"
        score -= 28
        hard_weak = True
        warnings.append("跌破策略防守")
    elif limit_fade:
        quality = "炸板/触板回落"
        limit_quality = "触板回落"
        score -= 22
        hard_weak = True
        warnings.append("触板后明显回落")
        warnings.append("未接入逐笔数据，炸板次数不可判定")
    elif touched_limit_up:
        if near_high and near_limit:
            quality = "封板近似强"
            limit_quality = "近似封住"
            score += 12
            reasons.append("触板后仍贴近日高")
            warnings.append("未接入封单明细，封板质量为价格近似")
        elif near_high:
            quality = "高位承接"
            limit_quality = "触板后高位"
            score += 6
            reasons.append("触板后仍在高位承接")
        else:
            quality = "触板回落"
            limit_quality = "触板回落"
            score -= 14
            hard_weak = True
            warnings.append("触板后未能维持高位")
    elif day_position is not None and current is not None and prev_close is not None and open_price is not None:
        if day_position >= 80 and current >= prev_close and current >= open_price:
            quality = "承接强"
            score += 10
            reasons.append("日内高位且站上昨收/开盘")
        elif day_position >= 55 and current >= prev_close and current >= open_price:
            quality = "修复承接"
            score += 6
            reasons.append("站回昨收和开盘")
        elif day_position <= 30 or (current < prev_close and current < open_price):
            quality = "承接弱"
            score -= 10
            hard_weak = True
            warnings.append("日内承接偏弱")
        elif current < prev_close:
            quality = "水下观察"
            score -= 5
            warnings.append("未站回昨收")

    if amount_ratio is not None:
        if amount_ratio >= 1.2 and quality in {"承接强", "修复承接", "高位承接", "封板近似强"}:
            score += 4
            reasons.append("量能确认")
        elif amount_ratio < 0.45 and change_pct is not None and change_pct > 2:
            score -= 4
            warnings.append("上涨量能不足")

    if high is not None and current is not None and current > 0 and high / current >= 1.035:
        score -= 8
        hard_weak = True
        warnings.append("冲高回落幅度偏大")
        if quality not in {"炸板/触板回落", "触板回落", "跌破防守"}:
            quality = "冲高回落"

    if _is_tail_session(quote):
        if current is not None and high is not None and prev_close is not None and open_price is not None:
            if current >= high * 0.985 and current >= prev_close and current >= open_price:
                closing_acceptance = "尾盘强承接"
                score += 8
                reasons.append("尾盘贴近日高")
                if quality == "普通承接":
                    quality = "尾盘强承接"
            elif current >= prev_close and (day_position is None or day_position >= 55):
                closing_acceptance = "尾盘修复"
                score += 4
                reasons.append("尾盘仍在昨收上方")
            else:
                closing_acceptance = "尾盘承接弱"
                score -= 6
                hard_weak = True
                warnings.append("尾盘承接不足")

    note = "当前行情源无封单/逐笔明细，封板质量和炸板次数使用价格位置近似"
    return {
        "score": round(score, 2),
        "quality": quality,
        "limit_quality": limit_quality,
        "closing_acceptance": closing_acceptance,
        "reasons": list(dict.fromkeys(reasons)),
        "warnings": list(dict.fromkeys(warnings)),
        "is_hard_weak": bool(hard_weak),
        "quality_note": note,
    }


def _day_position_from_quote(quote):
    current = _to_float((quote or {}).get("current"))
    high = _to_float((quote or {}).get("high"))
    low = _to_float((quote or {}).get("low"))
    if current is None or high is None or low is None or high <= low:
        return None
    return round((current - low) / (high - low) * 100, 2)


def _enrich_emotion_watch_rows(rows):
    rows = [dict(row) for row in (rows or [])]
    if not rows:
        return []

    codes = [row.get("stock_code") for row in rows]
    quotes = _fetch_sina_quotes(codes)
    enriched = []
    for row in rows:
        code = _normalize_code(row.get("stock_code"))
        quote = quotes.get(code)
        row["stock_code"] = code
        if not quote:
            row.update(
                {
                    "current": None,
                    "change_pct": None,
                    "quote_datetime": None,
                    "day_position": None,
                    "intraday_quality": "无实时行情",
                    "intraday_quality_score": 0.0,
                    "limit_quality": "--",
                    "closing_acceptance": "--",
                    "touched_limit_up": False,
                    "limit_fade": False,
                }
            )
            enriched.append(row)
            continue

        current = _to_float(quote.get("current"))
        high = _to_float(quote.get("high"))
        high_from_prev_pct = _to_float(quote.get("high_from_prev_pct"))
        limit_up_pct = _limit_up_pct(code, quote.get("name") or row.get("stock_name"))
        touched_limit_up = bool(
            high_from_prev_pct is not None
            and high_from_prev_pct >= limit_up_pct - 0.35
        )
        limit_fade = bool(touched_limit_up and current is not None and high is not None and current < high * 0.97)
        day_position = _day_position_from_quote(quote)
        intraday_quality = _score_intraday_quality(
            quote,
            touched_limit_up=touched_limit_up,
            limit_fade=limit_fade,
            stop_broken=False,
            day_position=day_position,
            amount_ratio=row.get("amount_ratio_5d"),
            limit_up_pct=limit_up_pct,
        )
        row.update(
            {
                "stock_name": quote.get("name") or row.get("stock_name"),
                "current": quote.get("current"),
                "change_pct": quote.get("change_pct"),
                "open_gap_pct": quote.get("open_gap_pct"),
                "day_position": day_position,
                "quote_datetime": quote.get("quote_datetime"),
                "intraday_quality": intraday_quality.get("quality"),
                "intraday_quality_score": intraday_quality.get("score"),
                "limit_quality": intraday_quality.get("limit_quality"),
                "closing_acceptance": intraday_quality.get("closing_acceptance"),
                "touched_limit_up": touched_limit_up,
                "limit_fade": limit_fade,
                "intraday_quality_note": intraday_quality.get("quality_note"),
            }
        )
        enriched.append(row)
    return enriched


def _risk_overlay_labels(overlay):
    labels = str((overlay or {}).get("risk_overlay_labels") or "").strip()
    if labels in {"", "无明显特殊池风险", "nan", "None"}:
        return ""
    return labels


def _risk_overlay_adjustment(overlay):
    if not overlay:
        return {
            "score": 0.0,
            "reasons": [],
            "warnings": [],
            "block": False,
            "downgrade": False,
            "high": False,
            "level": None,
            "risk_score": None,
        }

    risk_score = _to_float(overlay.get("risk_overlay_score")) or 0.0
    level = str(overlay.get("risk_overlay_level") or "").strip()
    labels = _risk_overlay_labels(overlay) or "风险覆盖提示"
    block = _to_bool(overlay.get("risk_overlay_block_formal"))
    downgrade = _to_bool(overlay.get("risk_overlay_downgrade"))
    high = bool(block or level == "高" or risk_score >= 8)
    score = 0.0
    reasons = []
    warnings = []

    if block:
        score -= 38
        warnings.append(f"风险覆盖硬拦截:{labels}")
    elif high:
        score -= 28
        warnings.append(f"风险覆盖高风险:{labels}")
    elif downgrade or level == "中" or risk_score >= 4:
        score -= 16
        warnings.append(f"风险覆盖降级:{labels}")
    elif risk_score > 0:
        score -= 5
        warnings.append(f"风险覆盖提醒:{labels}")
    elif level == "低":
        reasons.append("风险覆盖低风险")

    special_pool = str(overlay.get("special_pool_label") or "").strip()
    if special_pool and special_pool not in {"普通股票池", "normal"} and special_pool not in labels:
        warnings.append(f"特殊池:{special_pool}")

    return {
        "score": round(score, 2),
        "reasons": reasons,
        "warnings": warnings,
        "block": block,
        "downgrade": bool(downgrade or level == "中" or risk_score >= 4),
        "high": high,
        "level": level or None,
        "risk_score": risk_score,
    }


def _cap_level_by_risk(level, action, risk_adjustment):
    if not risk_adjustment:
        return level, action
    if risk_adjustment.get("block"):
        return "回避", "风险覆盖硬拦截，不开新仓；持仓只看承接和风控线"
    if risk_adjustment.get("high"):
        return "回避", "风险覆盖高风险，先回避普通买点"
    if risk_adjustment.get("downgrade") and level in {"核心盯盘", "重点关注"}:
        return "二级观察", "风险覆盖降级，只观察承接确认，不主动追买"
    return level, action


def _risk_overlay_display(overlay):
    if not overlay:
        return "--"
    labels = _risk_overlay_labels(overlay)
    level = str(overlay.get("risk_overlay_level") or "").strip()
    if _to_bool(overlay.get("risk_overlay_block_formal")):
        return f"硬拦截:{labels or level or '--'}"
    if _to_bool(overlay.get("risk_overlay_downgrade")):
        return f"降级:{labels or level or '--'}"
    if labels:
        return f"{level or '风险'}:{labels}"
    return level or "--"


def _score_candidate(candidate, quote, board_lookup=None, board_review=None, leader_transition_lookup=None):
    strategy = candidate.get("strategy") or {}
    strategies = candidate.get("strategies") or []
    leader = _best_leader(candidate.get("industry_leaders") or [])
    board = _board_from_candidate(strategy, leader, board_lookup)
    board_review_item = _board_review_from_candidate(strategy, leader, board_review)
    leader_transition = _leader_transition_from_candidate(leader, leader_transition_lookup)
    leader_transition_score = _score_leader_transition(leader_transition, board_review_item)
    board_score = _score_board(board)
    score = 0.0
    reasons = []
    warnings = []
    risk_overlay = candidate.get("risk_overlay") or {}
    risk_adjustment = _risk_overlay_adjustment(risk_overlay)

    if strategy:
        strategy_score, strategy_reasons, strategy_warnings = _score_strategy_context(strategy, strategies)
        score += strategy_score
        reasons.extend(strategy_reasons)
        warnings.extend(strategy_warnings)

    if board:
        score += board_score["score"] * 0.45
        reasons.extend(board_score["reasons"])
        warnings.extend(board_score["warnings"])

    if board_review_item:
        conclusion = str(board_review_item.get("conclusion") or "")
        appear_days = board_review_item.get("appear_days") or 0
        total_days = board_review_item.get("total_days") or 0
        score += (board_review_item.get("score") or 0) * 0.3
        reasons.append(f"板块近几日{appear_days}/{total_days}日上榜")
        if conclusion == "持续稳定关注":
            score += 8
            reasons.append("板块持续稳定关注")
        elif conclusion == "可跟踪但等确认":
            score += 3
            reasons.append("板块可跟踪")
        elif conclusion == "过热不追":
            score -= 10
            warnings.append("板块过热不追")
        elif conclusion == "分歧回避":
            score -= 18
            warnings.append("板块昨日分歧回避")
        elif conclusion == "弱观察":
            score -= 5
            warnings.append("板块延续性一般")
        lifecycle = str(board_review_item.get("theme_lifecycle") or "")
        lifecycle_score = _to_float(board_review_item.get("theme_lifecycle_score")) or 0.0
        if lifecycle and lifecycle != "--":
            score += lifecycle_score
            if lifecycle_score > 0:
                reasons.append(f"题材生命周期:{lifecycle}")
            elif lifecycle_score < 0:
                warnings.append(f"题材生命周期:{lifecycle}")
        leader_tier = str(board_review_item.get("leader_tier_label") or "")
        leader_tier_score = _to_float(board_review_item.get("leader_tier_score")) or 0.0
        if leader_tier and leader_tier != "--":
            score += leader_tier_score
            if leader_tier_score > 0:
                reasons.append(f"板块梯队:{leader_tier}")
            elif leader_tier_score < 0:
                warnings.append(f"板块梯队:{leader_tier}")
        warnings.extend(board_review_item.get("warnings") or [])
        warnings.extend(board_review_item.get("theme_lifecycle_warnings") or [])

    if leader:
        score += leader_transition_score["score"]
        reasons.extend(leader_transition_score["reasons"])
        warnings.extend(leader_transition_score["warnings"])
        score += 10
        reasons.append(f"{leader.get('板块名称') or '--'}龙头")
        leader_type = str(leader.get("龙头类型") or "")
        for key, value in LEADER_TYPE_SCORE.items():
            if key in leader_type:
                score += value
                break
        score += _rank_score_by_rank(leader.get("热点排名"), max_score=9, floor=30)
        score += _rank_score_by_rank(leader.get("关注排名"), max_score=7, floor=30)
        risk = str(leader.get("风险提示") or "")
        if "成交过度集中" in risk:
            score -= 3
            warnings.append("成交过度集中")

    score += risk_adjustment["score"]
    reasons.extend(risk_adjustment["reasons"])
    warnings.extend(risk_adjustment["warnings"])

    if not quote:
        level = "二级观察" if score >= 55 else "暂不关注"
        action = "无法盘中确认，先不升级关注"
        level, action = _cap_level_by_risk(level, action, risk_adjustment)
        intention, trigger, invalid = _build_trade_texts("无实时行情", level, strategy, quote)
        return {
            "score": round(score, 2),
            "level": level,
            "state": "无实时行情",
            "action": action,
            "intention": intention,
            "trigger": trigger,
            "invalid": invalid,
            "reasons": reasons,
            "warnings": warnings,
            "key_levels": _key_levels(strategy),
            "amount_ratio": None,
            "board_score": board_score["score"],
            "board_review_score": (board_review_item or {}).get("score"),
            "board_conclusion": (board_review_item or {}).get("conclusion"),
            "board_leader_summary": (board_review_item or {}).get("leader_summary"),
            "theme_lifecycle": (board_review_item or {}).get("theme_lifecycle"),
            "theme_lifecycle_score": (board_review_item or {}).get("theme_lifecycle_score"),
            "theme_lifecycle_advice": (board_review_item or {}).get("theme_lifecycle_advice"),
            "leader_tier_label": (board_review_item or {}).get("leader_tier_label"),
            "leader_tier_score": (board_review_item or {}).get("leader_tier_score"),
            "leader_tier_advice": (board_review_item or {}).get("leader_tier_advice"),
            "leader_transition_label": leader_transition_score["label"],
            "leader_transition_advice": leader_transition_score["advice"],
            "leader_rank_text": leader_transition_score["rank_text"],
            "leader_appear_days": (leader_transition or {}).get("appear_days"),
            "intraday_quality": "无实时行情",
            "intraday_quality_score": 0.0,
            "intraday_quality_note": "无实时行情，不能判断盘口质量",
            "limit_quality": "--",
            "closing_acceptance": "--",
            "intraday_hard_weak": False,
            "touched_limit_up": False,
            "limit_fade": False,
            "risk_overlay_score": risk_overlay.get("risk_overlay_score"),
            "risk_overlay_level": risk_overlay.get("risk_overlay_level"),
            "risk_overlay_labels": risk_overlay.get("risk_overlay_labels"),
            "risk_overlay_action": risk_overlay.get("risk_overlay_action"),
            "risk_overlay_display": _risk_overlay_display(risk_overlay),
            "risk_overlay_block_formal": risk_adjustment.get("block"),
            "risk_overlay_downgrade": risk_adjustment.get("downgrade"),
        }

    current = _to_float(quote.get("current"))
    prev_close = _to_float(quote.get("prev_close"))
    open_price = _to_float(quote.get("open"))
    high = _to_float(quote.get("high"))
    low = _to_float(quote.get("low"))
    change_pct = _to_float(quote.get("change_pct"))
    high_from_prev_pct = _to_float(quote.get("high_from_prev_pct"))
    day_position = None
    if current is not None and high is not None and low is not None and high > low:
        day_position = (current - low) / (high - low) * 100

    ref_high = _to_float(strategy.get("ref_high"))
    ref_low = _to_float(strategy.get("ref_low"))
    stop_price = _to_float(strategy.get("stop_price"))
    amount_ratio = _amount_ratio(quote, strategy, leader, board=board)

    if change_pct is not None:
        if change_pct >= 5:
            score += 24
            reasons.append("盘中强攻")
        elif change_pct >= 3:
            score += 18
            reasons.append("盘中走强")
        elif change_pct >= 1:
            score += 10
            reasons.append("盘中红盘修复")
        elif change_pct >= 0:
            score += 5
            reasons.append("站在红盘")
        elif change_pct <= -3:
            score -= 16
            warnings.append("盘中明显走弱")
        elif change_pct <= -1:
            score -= 8
            warnings.append("盘中偏弱")

    if current is not None and prev_close is not None and current >= prev_close:
        score += 6
        reasons.append("站回昨收")
    if current is not None and open_price is not None and current >= open_price:
        score += 5
        reasons.append("站回开盘")
    if day_position is not None:
        if day_position >= 75:
            score += 8
            reasons.append("日内位置偏强")
        elif day_position <= 35:
            score -= 6
            warnings.append("日内位置偏低")
    if ref_high is not None and current is not None and current >= ref_high:
        score += 10
        reasons.append("突破昨日高点")
    if ref_low is not None and low is not None and low < ref_low:
        score -= 8
        warnings.append("盘中跌破昨日低点")
    if amount_ratio is not None:
        if amount_ratio >= 1.0:
            score += 8
            reasons.append("盘中成交已超5日均额")
        elif amount_ratio >= 0.55:
            score += 4
            reasons.append("盘中成交放大")
    limit_up_pct = _limit_up_pct(candidate.get("stock_code"), (quote or {}).get("name") or candidate.get("stock_name"))
    touched_limit_up = bool(
        high_from_prev_pct is not None
        and high_from_prev_pct >= limit_up_pct - 0.35
    )
    limit_fade = bool(
        touched_limit_up
        and current is not None
        and high is not None
        and current < high * 0.97
    )
    if touched_limit_up:
        reasons.append("盘中触及涨停")
    if limit_fade:
        score -= 10
        warnings.append("触板回落")
    stop_broken = bool(stop_price is not None and current is not None and current < stop_price)
    if stop_broken:
        score -= 35
        warnings.append("跌破策略防守")

    intraday_quality = _score_intraday_quality(
        quote,
        touched_limit_up=touched_limit_up,
        limit_fade=limit_fade,
        stop_broken=stop_broken,
        day_position=day_position,
        amount_ratio=amount_ratio,
        limit_up_pct=limit_up_pct,
    )
    score += intraday_quality["score"]
    reasons.extend(intraday_quality["reasons"])
    warnings.extend(intraday_quality["warnings"])

    if stop_broken:
        state = "跌破防守"
    elif limit_fade:
        state = "触板回落"
    elif intraday_quality.get("quality") in {"承接弱", "冲高回落", "水下观察"}:
        state = "承接偏弱"
    elif change_pct is not None and change_pct >= 3 and day_position and day_position >= 65:
        state = "盘中强势"
    elif current is not None and prev_close is not None and current >= prev_close:
        state = "修复走强"
    elif day_position is not None and day_position >= 55:
        state = "弱修复"
    else:
        state = "承接偏弱"

    if stop_broken:
        level = "回避"
        action = "回避新买点；持仓按策略先降风险"
    elif score >= 92:
        level = "核心盯盘"
        action = "只等分歧承接或二次确认，不追高一笔打满"
    elif score >= 72:
        level = "重点关注"
        action = "等回踩承接或重新突破日高后再考虑"
    elif score >= 55:
        level = "二级观察"
        action = "观察，不满足触发条件不动手"
    else:
        level = "暂不关注"
        action = "暂不关注，除非快速收回关键位"
    level, action = _cap_level_by_risk(level, action, risk_adjustment)
    intention, trigger, invalid = _build_trade_texts(state, level, strategy, quote)

    return {
        "score": round(score, 2),
        "level": level,
        "state": state,
        "action": action,
        "intention": intention,
        "trigger": trigger,
        "invalid": invalid,
        "reasons": reasons,
        "warnings": warnings,
        "key_levels": _key_levels(strategy, quote),
        "amount_ratio": amount_ratio,
        "day_position": _round(day_position, 2),
        "board_score": board_score["score"],
        "board_review_score": (board_review_item or {}).get("score"),
        "board_conclusion": (board_review_item or {}).get("conclusion"),
        "board_leader_summary": (board_review_item or {}).get("leader_summary"),
        "theme_lifecycle": (board_review_item or {}).get("theme_lifecycle"),
        "theme_lifecycle_score": (board_review_item or {}).get("theme_lifecycle_score"),
        "theme_lifecycle_advice": (board_review_item or {}).get("theme_lifecycle_advice"),
        "leader_tier_label": (board_review_item or {}).get("leader_tier_label"),
        "leader_tier_score": (board_review_item or {}).get("leader_tier_score"),
        "leader_tier_advice": (board_review_item or {}).get("leader_tier_advice"),
        "leader_transition_label": leader_transition_score["label"],
        "leader_transition_advice": leader_transition_score["advice"],
        "leader_rank_text": leader_transition_score["rank_text"],
        "leader_appear_days": (leader_transition or {}).get("appear_days"),
        "intraday_quality": intraday_quality.get("quality"),
        "intraday_quality_score": intraday_quality.get("score"),
        "intraday_quality_note": intraday_quality.get("quality_note"),
        "limit_quality": intraday_quality.get("limit_quality"),
        "closing_acceptance": intraday_quality.get("closing_acceptance"),
        "intraday_hard_weak": intraday_quality.get("is_hard_weak"),
        "touched_limit_up": touched_limit_up,
        "limit_fade": limit_fade,
        "risk_overlay_score": risk_overlay.get("risk_overlay_score"),
        "risk_overlay_level": risk_overlay.get("risk_overlay_level"),
        "risk_overlay_labels": risk_overlay.get("risk_overlay_labels"),
        "risk_overlay_action": risk_overlay.get("risk_overlay_action"),
        "risk_overlay_display": _risk_overlay_display(risk_overlay),
        "risk_overlay_block_formal": risk_adjustment.get("block"),
        "risk_overlay_downgrade": risk_adjustment.get("downgrade"),
    }


def _key_levels(strategy, quote=None):
    levels = []
    stop_price = _round((strategy or {}).get("stop_price"), 2)
    if stop_price is not None:
        levels.append(f"防守{stop_price}")
    for label, key in [("昨低", "ref_low"), ("昨收", "ref_close"), ("昨高", "ref_high"), ("MA5", "ma5")]:
        value = _round((strategy or {}).get(key), 2)
        if value is not None:
            levels.append(f"{label}{value}")
    if quote:
        prev_close = _round(quote.get("prev_close"), 2)
        high = _round(quote.get("high"), 2)
        if prev_close is not None and not any(f"昨收{prev_close}" == item for item in levels):
            levels.append(f"昨收{prev_close}")
        if high is not None:
            levels.append(f"日高{high}")
    return " / ".join(levels[:5]) or "--"


def _format_money_yi(value):
    value = _to_float(value)
    if value is None:
        return "--"
    return f"{round(value / 100000000, 2)}亿"


def _industry_label(leader, board=None):
    if not leader and not board:
        return "--"
    if not leader:
        return (
            f"{_board_name(board) or '--'}"
            f"/{board.get('热点版本') or '--'}"
            f"/热{board.get('热点排名') or '--'}"
            f"/关{board.get('关注排名') or '--'}"
        )
    return (
        f"{leader.get('板块名称') or '--'}"
        f"/{(board or leader).get('热点版本') or '--'}"
        f"/热{(board or leader).get('热点排名') or '--'}"
        f"/关{(board or leader).get('关注排名') or '--'}"
    )


def _short_industry_label(value, limit=3):
    parts = [
        part.strip()
        for part in re.split(r"[,，/、;；]+", str(value or ""))
        if part and part.strip()
    ]
    parts = list(dict.fromkeys(parts))
    return " / ".join(parts[: int(limit or 3)]) or "--"


def _candidate_board_display(strategy=None, leader=None, board=None):
    board_name = str((leader or {}).get("板块名称") or _board_name(board) or "").strip()
    if board_name:
        return board_name
    return _short_industry_label((strategy or {}).get("industry"))


def build_intraday_focus(
    trade_date=None,
    strategy_date=None,
    industry_date=None,
    use_latest_strategy=False,
    top=20,
    board_limit=25,
    leaders_per_board=3,
    strategy_lookback_trade_days=DEFAULT_STRATEGY_LOOKBACK_TRADE_DAYS,
    industry_review_days=DEFAULT_INDUSTRY_REVIEW_DAYS,
    event_scan_limit=DEFAULT_EVENT_SCAN_LIMIT,
    event_lookback_days=DEFAULT_EVENT_LOOKBACK_DAYS,
):
    warnings = []
    target_trade_date = _date_text(trade_date) or _latest_history_date()
    if not target_trade_date:
        raise RuntimeError("无法确定分析交易日")

    requested_strategy_date = _date_text(strategy_date)
    strategy_frame = _load_strategy_candidates(
        strategy_date=requested_strategy_date,
        target_trade_date=target_trade_date,
        lookback_trade_days=strategy_lookback_trade_days,
    )
    strategy_dates = sorted(set(strategy_frame["strategy_date_text"].dropna().tolist())) if not strategy_frame.empty else []
    resolved_strategy_date = requested_strategy_date or (
        f"{strategy_dates[0]}~{strategy_dates[-1]}" if strategy_dates else f"近{strategy_lookback_trade_days}个交易日"
    )
    if strategy_frame.empty and requested_strategy_date and use_latest_strategy:
        fallback_date = _latest_strategy_date(target_trade_date)
        if fallback_date and fallback_date != requested_strategy_date:
            strategy_frame = _load_strategy_candidates(strategy_date=fallback_date)
            resolved_strategy_date = fallback_date
            warnings.append(f"交易日 {requested_strategy_date} 无策略落盘，已回退到最近策略日 {fallback_date}")
    elif strategy_frame.empty:
        warnings.append(f"{resolved_strategy_date} 无策略落盘，本次只用行业热点龙头观察")

    resolved_industry_date = _date_text(industry_date) or target_trade_date
    industry_dir, boards, leaders = _load_industry_reports(resolved_industry_date)
    if industry_dir:
        resolved_industry_date = industry_dir.name
    else:
        warnings.append(f"未找到 {resolved_industry_date} 及以前的行业热点报告")

    review_dirs, review_boards, review_leaders = _load_industry_review(
        resolved_industry_date,
        review_days=industry_review_days,
    )
    board_review = _board_trend_review(
        review_boards,
        leaders=review_leaders,
        total_days=len(review_dirs),
    )
    leader_transition_lookup = _leader_transition_lookup(review_leaders)
    top_board_reviews = _top_board_reviews(board_review, limit=10)
    if not review_dirs:
        warnings.append("未找到可用于近几日比较的行业热点报告")

    board_lookup = _board_lookup(boards)
    leader_frame = _load_industry_leader_candidates(
        leaders,
        boards=boards,
        board_limit=board_limit,
        leaders_per_board=leaders_per_board,
    )
    candidates = _merge_candidates(strategy_frame, leader_frame)
    stock_codes = [item["stock_code"] for item in candidates]
    risk_overlay_date, risk_overlay_lookup = _load_risk_overlay_for_codes(stock_codes, target_trade_date)
    if candidates and not risk_overlay_date:
        warnings.append("未读取到风险覆盖模型结果，本次不做风险覆盖降级")
    elif risk_overlay_date and risk_overlay_date != target_trade_date:
        warnings.append(f"风险覆盖模型最新为 {risk_overlay_date}，早于分析交易日 {target_trade_date}")
    for candidate in candidates:
        candidate["risk_overlay"] = risk_overlay_lookup.get(candidate.get("stock_code"))
    risk_overlay_values = list(risk_overlay_lookup.values())
    risk_overlay_blocked_count = sum(
        1 for row in risk_overlay_values if _to_bool(row.get("risk_overlay_block_formal"))
    )
    risk_overlay_downgrade_count = sum(
        1 for row in risk_overlay_values if _to_bool(row.get("risk_overlay_downgrade"))
    )
    risk_overlay_high_count = sum(
        1 for row in risk_overlay_values if str(row.get("risk_overlay_level") or "") == "高"
    )
    quotes = _fetch_sina_quotes(stock_codes)
    market = _fetch_market_summary()

    rows = []
    for candidate in candidates:
        code = candidate["stock_code"]
        quote = quotes.get(code)
        leader = _best_leader(candidate.get("industry_leaders") or [])
        strategy = candidate.get("strategy") or {}
        board = _board_from_candidate(strategy, leader, board_lookup)
        scored = _score_candidate(
            candidate,
            quote,
            board_lookup=board_lookup,
            board_review=board_review,
            leader_transition_lookup=leader_transition_lookup,
        )
        rows.append(
            {
                "stock_code": code,
                "stock_name": (quote or {}).get("name") or candidate.get("stock_name"),
                "sources": "、".join(dict.fromkeys(candidate.get("sources") or [])),
                "score": scored["score"],
                "level": scored.get("level"),
                "state": scored["state"],
                "action": scored["action"],
                "intention": scored.get("intention"),
                "trigger": scored.get("trigger"),
                "invalid": scored.get("invalid"),
                "current": (quote or {}).get("current"),
                "change_pct": (quote or {}).get("change_pct"),
                "open_gap_pct": (quote or {}).get("open_gap_pct"),
                "day_position": scored.get("day_position"),
                "amount": (quote or {}).get("amount"),
                "amount_ratio": scored.get("amount_ratio"),
                "board_score": scored.get("board_score"),
                "board_review_score": scored.get("board_review_score"),
                "board_conclusion": scored.get("board_conclusion"),
                "board_leader_summary": scored.get("board_leader_summary"),
                "theme_lifecycle": scored.get("theme_lifecycle"),
                "theme_lifecycle_score": scored.get("theme_lifecycle_score"),
                "theme_lifecycle_advice": scored.get("theme_lifecycle_advice"),
                "leader_tier_label": scored.get("leader_tier_label"),
                "leader_tier_score": scored.get("leader_tier_score"),
                "leader_tier_advice": scored.get("leader_tier_advice"),
                "leader_transition_label": scored.get("leader_transition_label"),
                "leader_transition_advice": scored.get("leader_transition_advice"),
                "leader_rank_text": scored.get("leader_rank_text"),
                "leader_appear_days": scored.get("leader_appear_days"),
                "intraday_quality": scored.get("intraday_quality"),
                "intraday_quality_score": scored.get("intraday_quality_score"),
                "intraday_quality_note": scored.get("intraday_quality_note"),
                "limit_quality": scored.get("limit_quality"),
                "closing_acceptance": scored.get("closing_acceptance"),
                "intraday_hard_weak": scored.get("intraday_hard_weak"),
                "touched_limit_up": scored.get("touched_limit_up"),
                "limit_fade": scored.get("limit_fade"),
                "risk_overlay_date": risk_overlay_date,
                "risk_overlay_score": scored.get("risk_overlay_score"),
                "risk_overlay_level": scored.get("risk_overlay_level"),
                "risk_overlay_labels": scored.get("risk_overlay_labels"),
                "risk_overlay_action": scored.get("risk_overlay_action"),
                "risk_overlay_display": scored.get("risk_overlay_display"),
                "risk_overlay_block_formal": scored.get("risk_overlay_block_formal"),
                "risk_overlay_downgrade": scored.get("risk_overlay_downgrade"),
                "board_name": (leader or {}).get("板块名称") or _board_name(board),
                "board_display": _candidate_board_display(strategy, leader, board),
                "strategy_industry": strategy.get("industry") if strategy else None,
                "industry_focus": _industry_label(leader, board=board),
                "leader_type": (leader or {}).get("龙头类型"),
                "key_levels": scored["key_levels"],
                "reasons": "；".join(dict.fromkeys(scored["reasons"])) or "--",
                "warnings": "；".join(dict.fromkeys(scored["warnings"])) or "--",
                "quote_datetime": (quote or {}).get("quote_datetime"),
                "strategy_date": _date_text(strategy.get("trade_date")) if strategy else None,
                "strategy_type": strategy.get("strategy_type"),
                "strategy_age_trade_days": strategy.get("strategy_age_trade_days"),
                "strategy_hits": len(candidate.get("strategies") or []),
                "industry_date": resolved_industry_date if leader else None,
            }
        )

    event_contexts = {}
    if rows and event_scan_limit and int(event_scan_limit) > 0:
        scan_frame = pd.DataFrame(rows)
        scan_frame = scan_frame.sort_values(["score", "change_pct"], ascending=[False, False]).head(
            int(event_scan_limit)
        )
        for scan_row in scan_frame.to_dict("records"):
            code = scan_row.get("stock_code")
            if not code or code in event_contexts:
                continue
            event_contexts[code] = _build_event_context(
                code,
                stock_name=scan_row.get("stock_name"),
                lookback_days=event_lookback_days,
            )
        rows = _apply_event_context_to_rows(rows, event_contexts)

    result_frame = pd.DataFrame(rows)
    leader_focus_rows = []
    event_focus_rows = []
    risk_focus_rows = []
    action_sections = _build_action_sections(result_frame, limit=8)
    if not result_frame.empty:
        focus_labels = {"新晋主攻龙头", "新晋龙头", "新龙替旧龙", "龙头排名上升", "新晋但过热", "新晋但分歧"}
        leader_focus_frame = result_frame[result_frame["leader_transition_label"].isin(focus_labels)].copy()
        if not leader_focus_frame.empty:
            leader_focus_frame = leader_focus_frame.sort_values(["score", "change_pct"], ascending=[False, False]).head(10)
            leader_focus_rows = leader_focus_frame.to_dict("records")
        if "risk_overlay_display" in result_frame.columns:
            risk_focus_frame = result_frame[
                result_frame["risk_overlay_block_formal"].fillna(False).astype(bool)
                | result_frame["risk_overlay_downgrade"].fillna(False).astype(bool)
                | result_frame["risk_overlay_level"].fillna("").isin(["高", "中"])
            ].copy()
            if not risk_focus_frame.empty:
                risk_focus_frame["_risk_block_sort"] = risk_focus_frame["risk_overlay_block_formal"].fillna(False).astype(int)
                risk_focus_frame["_risk_downgrade_sort"] = risk_focus_frame["risk_overlay_downgrade"].fillna(False).astype(int)
                risk_focus_frame["_risk_score_sort"] = risk_focus_frame["risk_overlay_score"].apply(_to_float).fillna(0)
                risk_focus_frame = risk_focus_frame.sort_values(
                    ["_risk_block_sort", "_risk_downgrade_sort", "_risk_score_sort", "score"],
                    ascending=[False, False, False, False],
                ).head(10)
                risk_focus_rows = risk_focus_frame.drop(
                    columns=["_risk_block_sort", "_risk_downgrade_sort", "_risk_score_sort"],
                    errors="ignore",
                ).to_dict("records")
        if "event_display" in result_frame.columns:
            event_focus_frame = result_frame[
                (result_frame["event_display"].fillna("--") != "--")
                & (result_frame["event_display"].fillna("--") != "无明确催化")
            ].copy()
            if not event_focus_frame.empty:
                event_focus_frame = event_focus_frame.sort_values(["score", "change_pct"], ascending=[False, False]).head(10)
                event_focus_rows = event_focus_frame.to_dict("records")
        result_frame = result_frame.sort_values(["score", "change_pct"], ascending=[False, False]).head(int(top))

    emotion_watch_date, emotion_watch_rows, emotion_watch_total_count = _load_emotion_leader_watch(
        target_trade_date,
        limit=10,
    )
    emotion_watch_rows = _enrich_emotion_watch_rows(emotion_watch_rows)
    if not emotion_watch_date:
        warnings.append("未找到情绪龙头胚子池报告，盘中页暂不展示该模块")
    elif not emotion_watch_rows:
        warnings.append(f"情绪龙头胚子池 {emotion_watch_date} 无候选")

    return {
        "success": True,
        "run_time": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target_trade_date": target_trade_date,
        "strategy_date": resolved_strategy_date,
        "strategy_count": int(len(strategy_frame)),
        "strategy_lookback_trade_days": int(strategy_lookback_trade_days or DEFAULT_STRATEGY_LOOKBACK_TRADE_DAYS),
        "strategy_date_count": int(len(strategy_dates)),
        "industry_date": resolved_industry_date,
        "industry_dir": str(industry_dir) if industry_dir else None,
        "industry_review_days": int(industry_review_days or DEFAULT_INDUSTRY_REVIEW_DAYS),
        "industry_review_dates": [daily_dir.name for daily_dir in review_dirs],
        "board_review_count": int(len(board_review)),
        "top_board_reviews": top_board_reviews,
        "leader_transition_count": int(len(leader_transition_lookup)),
        "leader_focus_rows": leader_focus_rows,
        **action_sections,
        "risk_overlay_date": risk_overlay_date,
        "risk_overlay_count": int(len(risk_overlay_lookup)),
        "risk_overlay_blocked_count": int(risk_overlay_blocked_count),
        "risk_overlay_downgrade_count": int(risk_overlay_downgrade_count),
        "risk_overlay_high_count": int(risk_overlay_high_count),
        "risk_focus_rows": risk_focus_rows,
        "event_scan_limit": int(event_scan_limit or 0),
        "event_lookback_days": int(event_lookback_days or DEFAULT_EVENT_LOOKBACK_DAYS),
        "event_context_count": int(len(event_contexts)),
        "event_focus_rows": event_focus_rows,
        "board_count": int(len(board_lookup)),
        "leader_candidate_count": int(len(leader_frame)),
        "emotion_watch_date": emotion_watch_date,
        "emotion_watch_total_count": int(emotion_watch_total_count),
        "emotion_watch_count": int(len(emotion_watch_rows)),
        "emotion_watch_rows": emotion_watch_rows,
        "candidate_count": int(len(candidates)),
        "quote_count": int(len(quotes)),
        "market": market,
        "warnings": warnings,
        "rows": result_frame.to_dict("records") if not result_frame.empty else [],
    }


def _market_text(market):
    if not market:
        return "--"
    return "；".join(
        f"{item.get('name')} {item.get('current')}({item.get('change_pct')}%)"
        for item in market
    )


def _rank_text(value):
    value = _to_float(value)
    if value is None:
        return "--"
    if float(value).is_integer():
        return str(int(value))
    return str(_round(value, 1))


def _md_cell(value):
    text = str(value if value is not None else "--").strip() or "--"
    return text.replace("|", "/").replace("\n", " ")


def _md_pct(value):
    value = _to_float(value)
    return "--" if value is None else f"{round(value, 2)}%"


def _md_num(value, digits=2):
    value = _to_float(value)
    return "--" if value is None else str(round(value, digits))


def _board_review_advice(item):
    conclusion = str((item or {}).get("conclusion") or "")
    latest_advice = str((item or {}).get("latest_advice") or "")
    lifecycle_advice = str((item or {}).get("theme_lifecycle_advice") or "")
    tier_advice = str((item or {}).get("leader_tier_advice") or "")
    if lifecycle_advice and lifecycle_advice != "--":
        if tier_advice and tier_advice != "--":
            return f"{lifecycle_advice}；{tier_advice}"
        return lifecycle_advice
    if conclusion == "持续稳定关注":
        return "优先找分歧低吸或二次确认，龙头不弱才考虑跟随"
    if conclusion == "可跟踪但等确认":
        return "先看龙头延续和板块扩散，放量确认后再升级"
    if conclusion == "过热不追":
        return "只看龙头开板承接，不追加速段"
    if conclusion == "分歧回避":
        return "暂不参与，等风险释放后再看"
    if "先观察" in latest_advice:
        return "观察为主，等排名、广度、资金同步抬升"
    return "弱观察，不作为日内优先方向"


def _board_posture_text(conclusion):
    conclusion = str(conclusion or "").strip()
    mapping = {
        "持续稳定关注": "板块持续，等承接",
        "可跟踪但等确认": "板块可跟踪，等确认",
        "过热不追": "板块过热，等承接",
        "分歧回避": "板块分歧，先回避",
        "弱观察": "板块弱观察",
    }
    return mapping.get(conclusion, conclusion or "--")


def _row_get(row, key, default=None):
    if row is None:
        return default
    try:
        return row.get(key, default)
    except AttributeError:
        return default


def _row_text(row, *keys):
    return "；".join(str(_row_get(row, key) or "") for key in keys)


def _has_meaningful_event(row):
    event = str(_row_get(row, "event_display") or "").strip()
    return bool(event and event not in {"--", "无明确催化"})


def _has_event_risk(row):
    event = str(_row_get(row, "event_display") or "")
    warnings = str(_row_get(row, "warnings") or "")
    truth_grade = str(_row_get(row, "event_truth_grade") or "")
    return event.startswith("风险:") or truth_grade == "风险" or "事件风险" in warnings or "硬风险事件" in warnings


def _is_risk_blocked(row):
    risk_display = str(_row_get(row, "risk_overlay_display") or "")
    risk_level = str(_row_get(row, "risk_overlay_level") or "")
    return bool(
        _to_bool(_row_get(row, "risk_overlay_block_formal"))
        or risk_display.startswith("硬拦截")
        or risk_level == "高"
    )


def _is_risk_downgraded(row):
    risk_display = str(_row_get(row, "risk_overlay_display") or "")
    risk_level = str(_row_get(row, "risk_overlay_level") or "")
    return bool(
        _to_bool(_row_get(row, "risk_overlay_downgrade"))
        or risk_display.startswith("降级")
        or risk_level == "中"
    )


def _is_touch_fade(row):
    return str(_row_get(row, "state") or "") == "触板回落" or "触板回落" in str(_row_get(row, "warnings") or "")


def _has_intraday_quality_risk(row):
    quality = str(_row_get(row, "intraday_quality") or "")
    closing = str(_row_get(row, "closing_acceptance") or "")
    warnings = str(_row_get(row, "warnings") or "")
    return bool(
        _to_bool(_row_get(row, "intraday_hard_weak"))
        or quality in {"炸板/触板回落", "触板回落", "跌破防守", "承接弱", "冲高回落", "水下观察"}
        or closing == "尾盘承接弱"
        or "冲高回落幅度偏大" in warnings
    )


def _action_bucket(row):
    level = str(_row_get(row, "level") or "")
    state = str(_row_get(row, "state") or "")
    board_conclusion = str(_row_get(row, "board_conclusion") or "")
    quality_score = _to_float(_row_get(row, "intraday_quality_score")) or 0.0
    if _is_risk_blocked(row) or level == "回避":
        return "回避"
    if _is_risk_downgraded(row) or _has_event_risk(row) or _is_touch_fade(row) or _has_intraday_quality_risk(row):
        return "观察"
    if board_conclusion in {"弱观察", "过热不追", "分歧回避"}:
        return "观察"
    if state == "弱修复":
        return "观察"
    if level in {"核心盯盘", "重点关注"} and state in {"盘中强势", "修复走强"} and quality_score >= 5:
        return "可操作"
    if level == "二级观察":
        return "观察"
    return "放弃"


def _action_reason(row):
    bits = []
    board_text = _board_posture_text(_row_get(row, "board_conclusion"))
    if board_text and board_text != "--":
        bits.append(board_text)
    lifecycle = str(_row_get(row, "theme_lifecycle") or "")
    if lifecycle and lifecycle != "--":
        bits.append(f"周期:{lifecycle}")
    tier = str(_row_get(row, "leader_tier_label") or "")
    if tier and tier != "--":
        bits.append(f"梯队:{tier}")
    leader = str(_row_get(row, "leader_transition_label") or "")
    if leader and leader != "--":
        bits.append(leader)
    event = str(_row_get(row, "event_display") or "")
    if event and event not in {"--", "无明确催化"}:
        truth = str(_row_get(row, "event_truth_display") or "")
        bits.append(f"{truth}/{event}" if truth and truth != "--" else event)
    risk = str(_row_get(row, "risk_overlay_display") or "")
    if risk and risk not in {"--", "低"}:
        bits.append(risk)
    quality = str(_row_get(row, "intraday_quality") or "")
    if quality and quality != "--":
        bits.append(f"盘口:{quality}")
    state = str(_row_get(row, "state") or "")
    if state:
        bits.append(state)
    return "；".join(dict.fromkeys(bits[:6])) or "--"


def _avoid_reason(row):
    risk = str(_row_get(row, "risk_overlay_display") or "")
    event = str(_row_get(row, "event_display") or "")
    warnings = str(_row_get(row, "warnings") or "")
    bits = []
    if risk and risk != "--":
        bits.append(risk)
    if event and event not in {"--", "无明确催化"}:
        bits.append(event)
    quality = str(_row_get(row, "intraday_quality") or "")
    if quality and quality not in {"--", "普通承接"}:
        bits.append(f"盘口:{quality}")
    if warnings and warnings != "--":
        bits.append(warnings.split("；")[0])
    return "；".join(dict.fromkeys(bits[:3])) or "--"


def _build_action_sections(result_frame, limit=8):
    if result_frame is None or result_frame.empty:
        return {
            "action_group_counts": {},
            "actionable_rows": [],
            "observe_rows": [],
            "avoid_rows": [],
        }

    frame = result_frame.copy()
    frame["_action_bucket"] = frame.apply(_action_bucket, axis=1)
    frame["_sort_change_pct"] = frame["change_pct"].apply(_to_float).fillna(-999)
    frame["_sort_score"] = frame["score"].apply(_to_float).fillna(-999)
    frame = frame.sort_values(["_sort_score", "_sort_change_pct"], ascending=[False, False])
    counts = {
        bucket: int((frame["_action_bucket"] == bucket).sum())
        for bucket in ["可操作", "观察", "回避", "放弃"]
    }

    def rows_for(bucket, row_limit):
        rows = frame[frame["_action_bucket"] == bucket].head(int(row_limit or limit)).copy()
        rows = rows.drop(columns=["_action_bucket", "_sort_change_pct", "_sort_score"], errors="ignore")
        return rows.to_dict("records")

    return {
        "action_group_counts": counts,
        "actionable_rows": rows_for("可操作", limit),
        "observe_rows": rows_for("观察", limit),
        "avoid_rows": rows_for("回避", limit),
    }


def render_markdown(result):
    lines = [
        "# 盘中关注池",
        "",
        f"- 运行时间: {result.get('run_time')}",
        f"- 分析交易日: {result.get('target_trade_date')}",
        (
            f"- 策略范围: 最近{result.get('strategy_lookback_trade_days')}个交易日"
            f"({result.get('strategy_date')})，命中策略日{result.get('strategy_date_count')}个，"
            f"策略数: {result.get('strategy_count')}"
        ),
        (
            f"- 行业主报告: {result.get('industry_date')}，行业龙头候选: "
            f"{result.get('leader_candidate_count')}"
        ),
        (
            f"- 行业近几日比较: {', '.join(result.get('industry_review_dates') or []) or '--'}，"
            f"覆盖板块: {result.get('board_review_count')}"
        ),
        f"- 龙头变化跟踪: {result.get('leader_transition_count')}条板块龙头历史",
        (
            f"- 风险覆盖模型: {result.get('risk_overlay_date') or '--'}，"
            f"候选覆盖 {result.get('risk_overlay_count')}/{result.get('candidate_count')}，"
            f"高风险 {result.get('risk_overlay_high_count')}，"
            f"硬拦截 {result.get('risk_overlay_blocked_count')}，"
            f"降级 {result.get('risk_overlay_downgrade_count')}"
        ),
        (
            f"- 事件催化扫描: 近{result.get('event_lookback_days')}日，"
            f"{result.get('event_context_count')}/{result.get('event_scan_limit')}只"
        ),
        (
            f"- 情绪龙头胚子池: {result.get('emotion_watch_date') or '--'}，"
            f"核心展示 {result.get('emotion_watch_count') or 0}/全量{result.get('emotion_watch_total_count') or 0}"
        ),
        "- 新增研判口径: 事件真实性分级、题材生命周期、板块梯队强弱、盘口质量近似判断",
        f"- 实时行情覆盖: {result.get('quote_count')}/{result.get('candidate_count')}",
        f"- 大盘: {_market_text(result.get('market'))}",
    ]
    if result.get("warnings"):
        lines.append(f"- 提醒: {'；'.join(result.get('warnings') or [])}")
    lines.append("")

    action_counts = result.get("action_group_counts") or {}
    if action_counts:
        lines.append("## 操作分层")
        lines.append("")
        lines.append(
            f"- 可操作: {action_counts.get('可操作', 0)}；"
            f"观察: {action_counts.get('观察', 0)}；"
            f"回避: {action_counts.get('回避', 0)}；"
            f"放弃: {action_counts.get('放弃', 0)}"
        )
        lines.append("")

    if result.get("actionable_rows"):
        lines.append("## 今日可操作候选")
        lines.append("")
        lines.append("|代码|名称|板块|层级|理由|题材周期|盘口|现价|涨幅|触发|放弃|")
        lines.append("|---|---|---|---|---|---|---|---:|---:|---|---|")
        for row in result.get("actionable_rows") or []:
            lines.append(
                "|{code}|{name}|{board}|{level}|{reason}|{cycle}|{quality}|{current}|{pct}|{trigger}|{invalid}|".format(
                    code=_md_cell(row.get("stock_code")),
                    name=_md_cell(row.get("stock_name")),
                    board=_md_cell(row.get("board_display") or row.get("board_name")),
                    level=_md_cell(row.get("level")),
                    reason=_md_cell(_action_reason(row)),
                    cycle=_md_cell(row.get("theme_lifecycle")),
                    quality=_md_cell(row.get("intraday_quality")),
                    current=row.get("current") if row.get("current") is not None else "--",
                    pct=f"{row.get('change_pct')}%" if row.get("change_pct") is not None else "--",
                    trigger=_md_cell(row.get("trigger")),
                    invalid=_md_cell(row.get("invalid")),
                )
            )
        lines.append("")
    else:
        lines.append("## 今日可操作候选")
        lines.append("")
        lines.append("- 暂无干净的可操作候选，优先观察而不是强行出手。")
        lines.append("")

    if result.get("emotion_watch_rows"):
        lines.append("## 情绪龙头胚子池")
        lines.append("")
        lines.append("|阶段|代码|名称|板块|分数|5日|20日|额比|换手|现价|涨幅|盘口|热榜|触发|放弃|")
        lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|")
        for row in result.get("emotion_watch_rows") or []:
            quality = str(row.get("intraday_quality") or "--")
            limit_quality = str(row.get("limit_quality") or "")
            if limit_quality and limit_quality not in {"--", "未触板"} and limit_quality not in quality:
                quality = f"{quality}/{limit_quality}"
            if _to_bool(row.get("limit_fade")) and "回落" not in quality:
                quality = f"{quality}/回落"
            lines.append(
                "|{stage}|{code}|{name}|{industry}|{score}|{chg5}|{chg20}|{ratio}|{turnover}|{current}|{pct}|{quality}|{leader}|{trigger}|{invalid}|".format(
                    stage=_md_cell(row.get("stage")),
                    code=_md_cell(row.get("stock_code")),
                    name=_md_cell(row.get("stock_name")),
                    industry=_md_cell(_short_industry_label(row.get("industry"), limit=2)),
                    score=_md_num(row.get("score")),
                    chg5=_md_pct(row.get("change_5d")),
                    chg20=_md_pct(row.get("change_20d")),
                    ratio=_md_num(row.get("amount_ratio_5d")),
                    turnover=_md_pct(row.get("turnover_rate")),
                    current=_md_num(row.get("current")),
                    pct=_md_pct(row.get("change_pct")),
                    quality=_md_cell(quality),
                    leader=_md_cell(row.get("leader_context")),
                    trigger=_md_cell(row.get("trigger")),
                    invalid=_md_cell(row.get("invalid")),
                )
            )
        lines.append("")

    if result.get("observe_rows"):
        lines.append("## 只观察候选")
        lines.append("")
        lines.append("|代码|名称|板块|观察原因|盘口|现价|涨幅|操作口径|")
        lines.append("|---|---|---|---|---|---:|---:|---|")
        for row in result.get("observe_rows") or []:
            lines.append(
                "|{code}|{name}|{board}|{reason}|{quality}|{current}|{pct}|{action}|".format(
                    code=_md_cell(row.get("stock_code")),
                    name=_md_cell(row.get("stock_name")),
                    board=_md_cell(row.get("board_display") or row.get("board_name")),
                    reason=_md_cell(_action_reason(row)),
                    quality=_md_cell(row.get("intraday_quality")),
                    current=row.get("current") if row.get("current") is not None else "--",
                    pct=f"{row.get('change_pct')}%" if row.get("change_pct") is not None else "--",
                    action=_md_cell(row.get("action")),
                )
            )
        lines.append("")

    if result.get("avoid_rows"):
        lines.append("## 高风险/回避")
        lines.append("")
        lines.append("|代码|名称|板块|回避原因|现价|涨幅|口径|")
        lines.append("|---|---|---|---|---:|---:|---|")
        for row in result.get("avoid_rows") or []:
            lines.append(
                "|{code}|{name}|{board}|{reason}|{current}|{pct}|{action}|".format(
                    code=_md_cell(row.get("stock_code")),
                    name=_md_cell(row.get("stock_name")),
                    board=_md_cell(row.get("board_display") or row.get("board_name")),
                    reason=_md_cell(_avoid_reason(row)),
                    current=row.get("current") if row.get("current") is not None else "--",
                    pct=f"{row.get('change_pct')}%" if row.get("change_pct") is not None else "--",
                    action=_md_cell(row.get("action")),
                )
            )
        lines.append("")

    if result.get("top_board_reviews"):
        lines.append("## 板块研判")
        lines.append("")
        lines.append("|板块|生命周期|梯队|结论|稳定分|出现|排名|龙头延续|操作口径|")
        lines.append("|---|---|---|---|---:|---|---|---|---|")
        for item in result.get("top_board_reviews") or []:
            lines.append(
                "|{board}|{cycle}|{tier}|{conclusion}|{score}|{appear}|热点第{hot} / 关注第{attention}|{leaders}|{advice}|".format(
                    board=_md_cell(item.get("board_name")),
                    cycle=_md_cell(item.get("theme_lifecycle")),
                    tier=_md_cell(item.get("leader_tier_label")),
                    conclusion=_md_cell(item.get("conclusion")),
                    score=item.get("score") if item.get("score") is not None else "--",
                    appear=f"{item.get('appear_days')}/{item.get('total_days')}",
                    hot=_rank_text(item.get("latest_hot_rank")),
                    attention=_rank_text(item.get("latest_attention_rank")),
                    leaders=_md_cell(item.get("leader_summary")),
                    advice=_md_cell(_board_review_advice(item)),
                )
            )
        lines.append("")
    if result.get("leader_focus_rows"):
        lines.append("## 新晋龙头观察")
        lines.append("")
        lines.append("|代码|名称|板块|龙头变化|板块口径|现价|涨幅|操作口径|")
        lines.append("|---|---|---|---|---|---:|---:|---|")
        for row in result.get("leader_focus_rows") or []:
            lines.append(
                "|{code}|{name}|{board}|{change}|{conclusion}|{current}|{pct}|{advice}|".format(
                    code=_md_cell(row.get("stock_code")),
                    name=_md_cell(row.get("stock_name")),
                    board=_md_cell(row.get("board_name")),
                    change=_md_cell(
                        f"{row.get('leader_transition_label') or '--'}"
                        f"({row.get('leader_rank_text') or '--'})"
                    ),
                    conclusion=_md_cell(_board_posture_text(row.get("board_conclusion"))),
                    current=row.get("current") if row.get("current") is not None else "--",
                    pct=f"{row.get('change_pct')}%" if row.get("change_pct") is not None else "--",
                    advice=_md_cell(row.get("leader_transition_advice")),
                )
            )
        lines.append("")
    if result.get("risk_focus_rows"):
        lines.append("## 风险覆盖观察")
        lines.append("")
        lines.append("|代码|名称|板块|风控|分数|处理口径|盘中|")
        lines.append("|---|---|---|---|---:|---|---|")
        for row in result.get("risk_focus_rows") or []:
            lines.append(
                "|{code}|{name}|{board}|{risk}|{risk_score}|{action}|{state}|".format(
                    code=_md_cell(row.get("stock_code")),
                    name=_md_cell(row.get("stock_name")),
                    board=_md_cell(row.get("board_display") or row.get("board_name")),
                    risk=_md_cell(row.get("risk_overlay_display")),
                    risk_score=row.get("risk_overlay_score") if row.get("risk_overlay_score") is not None else "--",
                    action=_md_cell(row.get("risk_overlay_action") or row.get("action")),
                    state=_md_cell(row.get("state")),
                )
            )
        lines.append("")
    if result.get("event_focus_rows"):
        lines.append("## 事件催化观察")
        lines.append("")
        lines.append("|代码|名称|板块|真实性|事件|最新来源|交易口径|")
        lines.append("|---|---|---|---|---|---|---|")
        for row in result.get("event_focus_rows") or []:
            latest = " ".join(
                part for part in [row.get("event_latest_date"), row.get("event_latest_title")] if part
            )
            lines.append(
                "|{code}|{name}|{board}|{truth}|{event}|{latest}|{note}|".format(
                    code=_md_cell(row.get("stock_code")),
                    name=_md_cell(row.get("stock_name")),
                    board=_md_cell(row.get("board_display") or row.get("board_name")),
                    truth=_md_cell(row.get("event_truth_display")),
                    event=_md_cell(row.get("event_display")),
                    latest=_md_cell(latest or row.get("event_note")),
                    note=_md_cell(row.get("event_trade_note")),
                )
            )
        lines.append("")
    lines.append(
        "|排名|代码|名称|板块|层级|来源|分数|现价|涨幅|题材|梯队|盘口|龙头变化|事件|风控|建议|触发|放弃|提醒|"
    )
    lines.append("|---:|---|---|---|---|---|---:|---:|---:|---|---|---|---|---|---|---|---|---|---|")
    for idx, row in enumerate(result.get("rows") or [], start=1):
        lines.append(
            "|{idx}|{code}|{name}|{board}|{level}|{sources}|{score}|{current}|{change}|{cycle}|{tier}|{quality}|{leader_change}|{event}|{risk}|{action}|{trigger}|{invalid}|{warnings}|".format(
                idx=idx,
                code=_md_cell(row.get("stock_code")),
                name=_md_cell(row.get("stock_name")),
                board=_md_cell(row.get("board_display") or row.get("board_name")),
                level=_md_cell(row.get("level")),
                sources=_md_cell(row.get("sources")),
                score=row.get("score") if row.get("score") is not None else "--",
                current=row.get("current") if row.get("current") is not None else "--",
                change=f"{row.get('change_pct')}%" if row.get("change_pct") is not None else "--",
                cycle=_md_cell(row.get("theme_lifecycle") or _board_posture_text(row.get("board_conclusion"))),
                tier=_md_cell(row.get("leader_tier_label") or "--"),
                quality=_md_cell(row.get("intraday_quality") or row.get("state")),
                leader_change=_md_cell(
                    f"{row.get('leader_transition_label') or '--'}"
                    f"({row.get('leader_rank_text') or '--'})"
                    if row.get("leader_transition_label") and row.get("leader_transition_label") != "--"
                    else "--"
                ),
                event=_md_cell(
                    f"{row.get('event_truth_display')}/{row.get('event_display')}"
                    if row.get("event_truth_display") and row.get("event_truth_display") != "--"
                    else row.get("event_display") or "--"
                ),
                risk=_md_cell(row.get("risk_overlay_display") or "--"),
                action=_md_cell(row.get("action")),
                trigger=_md_cell(row.get("trigger")),
                invalid=_md_cell(row.get("invalid")),
                warnings=_md_cell(row.get("warnings")),
            )
        )
    return "\n".join(lines) + "\n"


def _split_markdown_table_row(line):
    text = str(line or "").strip()
    if text.startswith("|"):
        text = text[1:]
    if text.endswith("|"):
        text = text[:-1]
    return [part.strip() for part in text.split("|")]


def _is_markdown_table_align(line):
    cells = _split_markdown_table_row(line)
    if not cells:
        return False
    return all(re.match(r"^:?-{3,}:?$", cell or "") for cell in cells)


def _html_text(value):
    return html.escape(str(value if value is not None else "--"), quote=True)


def _html_cell_class(text, header=None):
    text = str(text or "")
    header = str(header or "")
    classes = []
    if header in {"涨幅", "今日均涨", "5日均涨", "5日超额", "5日", "20日", "换手"} and re.search(r"^-?\d+(?:\.\d+)?%$", text):
        value = _to_float(text)
        if value is not None and value > 0:
            classes.append("num-up")
        elif value is not None and value < 0:
            classes.append("num-down")
    positive_words = ["可操作", "核心盯盘", "重点关注", "主升延续", "扩散发酵", "梯队强", "承接强", "封板近似强", "A早期胚子", "B右侧确认"]
    observe_words = ["观察", "稳定观察", "修复承接", "C题材观察", "C龙头已成", "D弱相关", "触板回落", "冲高回落"]
    risk_words = ["回避", "硬拦截", "风险", "分歧退潮", "高潮加速", "炸板", "承接弱", "跌破", "D过热回避"]
    if any(word in text for word in risk_words):
        classes.append("tag-risk")
    elif any(word in text for word in positive_words):
        classes.append("tag-good")
    elif any(word in text for word in observe_words):
        classes.append("tag-watch")
    if header in {"代码", "现价", "涨幅", "分数", "稳定分"}:
        classes.append("nowrap")
    return " ".join(dict.fromkeys(classes))


def _markdown_table_row_hidden_by_code(row, headers, hide_code_prefixes=None):
    prefixes = tuple(str(prefix) for prefix in (hide_code_prefixes or ()) if str(prefix))
    if not prefixes or "代码" not in headers:
        return False
    code_index = headers.index("代码")
    if code_index >= len(row):
        return False
    code = _normalize_code(row[code_index])
    return bool(code and code.startswith(prefixes))


def _render_markdown_table(header_line, rows, hide_code_prefixes=None):
    headers = _split_markdown_table_row(header_line)
    body_rows = []
    for row_text in rows:
        if not row_text.strip().startswith("|") or _is_markdown_table_align(row_text):
            continue
        row = _split_markdown_table_row(row_text)
        if _markdown_table_row_hidden_by_code(row, headers, hide_code_prefixes=hide_code_prefixes):
            continue
        body_rows.append(row)
    output = ['<div class="table-wrap"><table>']
    output.append("<thead><tr>")
    for header in headers:
        output.append(f"<th>{_html_text(header)}</th>")
    output.append("</tr></thead><tbody>")
    for row in body_rows:
        output.append("<tr>")
        for index, cell in enumerate(row):
            header = headers[index] if index < len(headers) else ""
            class_text = _html_cell_class(cell, header=header)
            class_attr = f' class="{class_text}"' if class_text else ""
            output.append(f"<td{class_attr} data-label=\"{_html_text(header)}\">{_html_text(cell)}</td>")
        output.append("</tr>")
    output.append("</tbody></table></div>")
    return "\n".join(output)


def _markdown_to_html(markdown, hide_code_prefixes=None):
    lines = str(markdown or "").splitlines()
    output = []
    in_list = False
    index = 0

    def close_list():
        nonlocal in_list
        if in_list:
            output.append("</ul>")
            in_list = False

    while index < len(lines):
        line = lines[index].rstrip()
        stripped = line.strip()
        if not stripped:
            close_list()
            index += 1
            continue
        if stripped.startswith("|") and index + 1 < len(lines) and _is_markdown_table_align(lines[index + 1]):
            close_list()
            table_rows = []
            header_line = stripped
            index += 2
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_rows.append(lines[index].strip())
                index += 1
            output.append(
                _render_markdown_table(
                    header_line,
                    table_rows,
                    hide_code_prefixes=hide_code_prefixes,
                )
            )
            continue
        if stripped.startswith("# "):
            close_list()
            output.append(f"<h1>{_html_text(stripped[2:].strip())}</h1>")
        elif stripped.startswith("## "):
            close_list()
            title = stripped[3:].strip()
            anchor = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "-", title).strip("-")
            output.append(f'<section class="report-section" id="{_html_text(anchor)}">')
            output.append(f"<h2>{_html_text(title)}</h2>")
            output.append("</section>")
        elif stripped.startswith("- "):
            if not in_list:
                output.append('<ul class="summary-list">')
                in_list = True
            output.append(f"<li>{_html_text(stripped[2:].strip())}</li>")
        else:
            close_list()
            output.append(f"<p>{_html_text(stripped)}</p>")
        index += 1
    close_list()
    return "\n".join(output)


def render_html_report(markdown, title="盘中关注池", hide_code_prefixes=DEFAULT_HTML_HIDE_CODE_PREFIXES):
    body_html = _markdown_to_html(markdown, hide_code_prefixes=hide_code_prefixes)
    title_text = _html_text(title)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title_text}</title>
<style>
:root {{
  color-scheme: light;
  --bg: #f6f7f9;
  --panel: #ffffff;
  --text: #172033;
  --muted: #657083;
  --line: #d9dee8;
  --soft: #eef2f7;
  --red: #c53030;
  --red-bg: #fff1f1;
  --green: #087f5b;
  --green-bg: #ecfdf5;
  --amber: #946200;
  --amber-bg: #fff7db;
  --blue: #155e9f;
  --blue-bg: #eef6ff;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
  font-size: 14px;
  line-height: 1.55;
}}
.topbar {{
  position: sticky;
  top: 0;
  z-index: 20;
  display: flex;
  gap: 12px;
  align-items: center;
  justify-content: space-between;
  padding: 12px 22px;
  background: rgba(255, 255, 255, 0.94);
  border-bottom: 1px solid var(--line);
  backdrop-filter: blur(10px);
}}
.top-actions {{
  display: flex;
  gap: 12px;
  align-items: center;
}}
.brand {{
  font-weight: 700;
  letter-spacing: 0;
  white-space: nowrap;
}}
.refresh-status {{
  color: var(--muted);
  font-size: 12px;
  white-space: nowrap;
}}
.search {{
  width: min(420px, 48vw);
  height: 34px;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 0 12px;
  color: var(--text);
  background: #fff;
  outline: none;
}}
.search:focus {{
  border-color: #7aa7d9;
  box-shadow: 0 0 0 3px rgba(21, 94, 159, 0.12);
}}
main {{
  max-width: 1680px;
  margin: 0 auto;
  padding: 22px;
}}
h1 {{
  margin: 4px 0 14px;
  font-size: 26px;
  line-height: 1.25;
  letter-spacing: 0;
}}
.summary-list {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(310px, 1fr));
  gap: 8px;
  margin: 0 0 20px;
  padding: 0;
  list-style: none;
}}
.summary-list li {{
  min-height: 38px;
  padding: 8px 10px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--panel);
  color: var(--muted);
}}
.report-section {{
  margin-top: 24px;
  padding-top: 2px;
  border-top: 1px solid var(--line);
}}
h2 {{
  margin: 0 0 10px;
  padding-top: 10px;
  font-size: 18px;
  line-height: 1.3;
  letter-spacing: 0;
}}
p {{
  margin: 10px 0;
  color: var(--muted);
}}
.table-wrap {{
  width: 100%;
  overflow: auto;
  margin: 8px 0 22px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
}}
table {{
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  min-width: 980px;
}}
thead th {{
  background: #f0f3f8;
  color: #374151;
  font-weight: 700;
  text-align: left;
}}
th, td {{
  padding: 9px 10px;
  border-bottom: 1px solid var(--line);
  border-right: 1px solid #edf0f5;
  vertical-align: top;
}}
th:last-child, td:last-child {{ border-right: 0; }}
tbody tr:nth-child(even) td {{ background: #fbfcfe; }}
tbody tr:hover td {{ background: #eef6ff; }}
.nowrap {{ white-space: nowrap; }}
.num-up {{ color: var(--red); font-weight: 700; }}
.num-down {{ color: var(--green); font-weight: 700; }}
.tag-good, .tag-watch, .tag-risk {{
  font-weight: 650;
}}
.tag-good {{ color: var(--red); background: var(--red-bg); }}
.tag-watch {{ color: var(--amber); background: var(--amber-bg); }}
.tag-risk {{ color: var(--green); background: var(--green-bg); }}
tr.hidden {{ display: none; }}
@media (max-width: 760px) {{
  .topbar {{
    align-items: stretch;
    flex-direction: column;
    padding: 10px 14px;
  }}
  .top-actions {{
    align-items: stretch;
    flex-direction: column;
    gap: 8px;
  }}
  .refresh-status {{ white-space: normal; }}
  .search {{ width: 100%; }}
  main {{ padding: 14px; }}
  h1 {{ font-size: 22px; }}
  .summary-list {{ grid-template-columns: 1fr; }}
  th, td {{ padding: 8px; }}
}}
</style>
</head>
<body>
<header class="topbar">
  <div class="brand">{title_text}</div>
  <div class="top-actions">
    <span class="refresh-status" id="refreshStatus">检查更新中</span>
    <input class="search" id="tableSearch" type="search" placeholder="搜索代码、名称、板块、风险">
  </div>
</header>
<main id="reportRoot">
{body_html}
</main>
<script>
const input = document.getElementById('tableSearch');
input.addEventListener('input', () => {{
  const keyword = input.value.trim().toLowerCase();
  document.querySelectorAll('tbody tr').forEach(row => {{
    row.classList.toggle('hidden', keyword && !row.innerText.toLowerCase().includes(keyword));
  }});
}});
const refreshStatus = document.getElementById('refreshStatus');
let reportSignature = null;
async function checkReportUpdate() {{
  try {{
    const url = new URL(window.location.href);
    url.searchParams.set('_check', Date.now().toString());
    const response = await fetch(url.toString(), {{ method: 'HEAD', cache: 'no-store' }});
    const signature = response.headers.get('last-modified') || response.headers.get('etag') || '';
    const now = new Date().toLocaleTimeString('zh-CN', {{ hour12: false }});
    if (reportSignature && signature && signature !== reportSignature) {{
      refreshStatus.textContent = '报告已更新，刷新中';
      window.location.reload();
      return;
    }}
    if (!reportSignature && signature) {{
      reportSignature = signature;
    }}
    refreshStatus.textContent = `自动检查 ${{now}}`;
  }} catch (error) {{
    refreshStatus.textContent = '自动检查失败';
  }}
}}
checkReportUpdate();
setInterval(checkReportUpdate, 30000);
</script>
</body>
</html>
"""


def save_report(markdown, trade_date=None, run_time=None, hide_code_prefixes=DEFAULT_HTML_HIDE_CODE_PREFIXES):
    trade_date_text = _date_text(trade_date)
    now = run_time or dt.datetime.now()
    date_dir_name = trade_date_text or now.strftime("%Y-%m-%d")
    date_dir = INTRADAY_REPORT_DIR / date_dir_name
    date_dir.mkdir(parents=True, exist_ok=True)
    latest_path = date_dir / "intraday_focus_latest.md"
    latest_html_path = date_dir / "intraday_focus_latest.html"
    stable_latest_path = INTRADAY_REPORT_DIR / "intraday_focus_latest.md"
    stable_latest_html_path = INTRADAY_REPORT_DIR / "intraday_focus_latest.html"
    for old_path in date_dir.glob("intraday_focus_*.md"):
        if old_path != latest_path:
            old_path.unlink()
    for old_path in date_dir.glob("intraday_focus_*.html"):
        if old_path != latest_html_path:
            old_path.unlink()
    latest_path.write_text(markdown, encoding="utf-8")
    html_report = render_html_report(markdown, hide_code_prefixes=hide_code_prefixes)
    latest_html_path.write_text(html_report, encoding="utf-8")
    stable_latest_path.write_text(markdown, encoding="utf-8")
    stable_latest_html_path.write_text(html_report, encoding="utf-8")
    return latest_path


def parse_args():
    parser = argparse.ArgumentParser(description="盘中分析库里最新交易日的策略落盘和行业热点，输出今日关注池")
    parser.add_argument("--trade-date", default=None, help="分析交易日；默认取库里最新交易日")
    parser.add_argument("--strategy-date", default=None, help="指定单日策略落盘日期；不填则取近20个交易日策略池")
    parser.add_argument("--industry-date", default=None, help="行业热点主报告日期；默认同分析交易日，若无日志则回退到更早")
    parser.add_argument("--use-latest-strategy", action="store_true", help="策略日无落盘时回退到最近策略日")
    parser.add_argument("--top", type=int, default=20, help="输出前N只")
    parser.add_argument("--board-limit", type=int, default=25, help="最多纳入前N个热点板块")
    parser.add_argument("--leaders-per-board", type=int, default=3, help="每个热点板块最多纳入N个龙头")
    parser.add_argument(
        "--strategy-lookback-trade-days",
        type=int,
        default=DEFAULT_STRATEGY_LOOKBACK_TRADE_DAYS,
        help="策略池回看最近N个交易日",
    )
    parser.add_argument(
        "--industry-review-days",
        type=int,
        default=DEFAULT_INDUSTRY_REVIEW_DAYS,
        help="行业热点近几日比较最多回看N份日志",
    )
    parser.add_argument(
        "--event-scan-limit",
        type=int,
        default=DEFAULT_EVENT_SCAN_LIMIT,
        help="对初筛前N只补充公告/新闻事件催化；0表示关闭",
    )
    parser.add_argument(
        "--event-lookback-days",
        type=int,
        default=DEFAULT_EVENT_LOOKBACK_DAYS,
        help="公告/新闻事件回看天数",
    )
    parser.add_argument("--json", action="store_true", help="输出JSON")
    parser.add_argument("--no-save", action="store_true", help="不保存Markdown报告")
    parser.add_argument("--show-688-in-html", action="store_true", help="HTML报告显示688开头科创板股票")
    return parser.parse_args()


def main():
    args = parse_args()
    result = build_intraday_focus(
        trade_date=args.trade_date,
        strategy_date=args.strategy_date,
        industry_date=args.industry_date,
        use_latest_strategy=args.use_latest_strategy,
        top=args.top,
        board_limit=args.board_limit,
        leaders_per_board=args.leaders_per_board,
        strategy_lookback_trade_days=args.strategy_lookback_trade_days,
        industry_review_days=args.industry_review_days,
        event_scan_limit=args.event_scan_limit,
        event_lookback_days=args.event_lookback_days,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, default=str, indent=2))
        return

    markdown = render_markdown(result)
    print(markdown)
    if not args.no_save:
        hide_code_prefixes = () if args.show_688_in_html else DEFAULT_HTML_HIDE_CODE_PREFIXES
        output_path = save_report(
            markdown,
            trade_date=result.get("target_trade_date"),
            hide_code_prefixes=hide_code_prefixes,
        )
        print(f"报告已覆盖: {output_path}")


if __name__ == "__main__":
    main()
