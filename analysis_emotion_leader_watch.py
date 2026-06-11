"""情绪龙头胚子池。

这个脚本只做观察雷达，不写入 a_stock_strategy_result。

它从库里最新交易日的日线结构、风险覆盖和行业热点日志里，寻找类似
红星发展早期那种“平台右侧突破、放量换手、开始有板块辨识度”的短线
情绪龙头胚子，并按 A/B/C/D 分层输出到 log/emotion_leader_watch。
"""

import argparse
import datetime as dt
import json
import math
import re
from pathlib import Path

import pandas as pd
import pymysql


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "log" / "emotion_leader_watch"
INDUSTRY_HOTSPOT_DIR = PROJECT_ROOT / "log" / "industry_hotspot"
DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "rootroot",
    "database": "gu_piao",
    "charset": "utf8mb4",
}

MODEL_VERSION = "emotion_leader_watch_v1"
DEFAULT_TOP = 80
DEFAULT_STAGE_LIMIT = 6
DEFAULT_EXCLUDE_PREFIXES = ("688", "920", "8", "4")


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
    if text.endswith("亿"):
        multiplier = 100000000
        text = text[:-1]
    elif text.endswith("万"):
        multiplier = 10000
        text = text[:-1]
    else:
        multiplier = 1
    try:
        return float(text) * multiplier
    except ValueError:
        return None


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


def _amount_yi(value):
    value = _to_float(value)
    if value is None:
        return None
    return value / 100000000 if value > 1000000 else value


def _latest_history_date():
    frame = _query_frame(
        """
        SELECT MAX(last_data_date) AS trade_date
        FROM a_stock_analysis_history
        WHERE CHAR_LENGTH(last_data_date) = 10
          AND last_data_date BETWEEN '2000-01-01' AND '2100-12-31'
        """
    )
    if frame.empty:
        return None
    return _date_text(frame["trade_date"].iloc[0])


def _latest_risk_date(trade_date):
    frame = _query_frame(
        """
        SELECT MAX(trade_date) AS trade_date
        FROM a_stock_risk_overlay
        WHERE trade_date <= %s
        """,
        [_date_text(trade_date)],
    )
    if frame.empty:
        return None
    return _date_text(frame["trade_date"].iloc[0])


def _load_snapshot(trade_date):
    risk_date = _latest_risk_date(trade_date)
    frame = _query_frame(
        """
        SELECT
            h.stock_code,
            h.stock_name,
            h.industry,
            h.latest_price,
            h.today_change,
            h.change_3d,
            h.change_5d,
            h.change_10d,
            h.change_20d,
            h.change_30d,
            h.today_amount,
            h.amount_avg_5d,
            h.amount_avg_10d,
            h.turnover_rate,
            h.ma5,
            h.ma10,
            h.ma20,
            h.ma60,
            h.ma120,
            h.high_20d,
            h.low_20d,
            h.high_60d,
            h.low_60d,
            h.high_120d,
            h.low_120d,
            h.today_open,
            h.today_high,
            h.today_low,
            r.risk_overlay_level,
            r.risk_overlay_score,
            r.risk_overlay_labels,
            r.risk_overlay_block_formal,
            r.risk_overlay_downgrade,
            r.risk_overlay_action,
            r.special_pool,
            r.special_pool_tags,
            r.listing_trade_days,
            r.limit_up_days_5d,
            r.consecutive_limit_up_days
        FROM a_stock_analysis_history h
        LEFT JOIN a_stock_risk_overlay r
          ON r.trade_date = %s
         AND r.stock_code COLLATE utf8mb4_general_ci = h.stock_code COLLATE utf8mb4_general_ci
        WHERE h.last_data_date = %s
        """,
        [risk_date, _date_text(trade_date)],
    )
    if frame.empty:
        return frame, risk_date
    frame["stock_code"] = frame["stock_code"].apply(_normalize_code)
    return frame[frame["stock_code"].notna()].copy(), risk_date


def _industry_dir_for_date(trade_date):
    wanted = _parse_date(trade_date)
    if not INDUSTRY_HOTSPOT_DIR.exists():
        return None
    dirs = []
    for child in INDUSTRY_HOTSPOT_DIR.iterdir():
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


def _read_csv(path):
    if not path or not Path(path).exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def _split_industries(value):
    parts = [
        part.strip()
        for part in re.split(r"[,，/、;；]+", str(value or ""))
        if part and part.strip()
    ]
    return list(dict.fromkeys(parts))


def _board_name(row):
    return str((row or {}).get("板块名称") or (row or {}).get("industry_name") or "").strip()


def _board_sort_tuple(item):
    hot_rank = _to_float((item or {}).get("热点排名") or (item or {}).get("hot_rank"))
    attention_rank = _to_float((item or {}).get("关注排名") or (item or {}).get("attention_rank"))
    hot_score = _to_float((item or {}).get("热点分") or (item or {}).get("hot_score"))
    return (
        hot_rank if hot_rank is not None else 999,
        attention_rank if attention_rank is not None else 999,
        -(hot_score or 0),
    )


def _load_board_contexts(daily_dir):
    board_map = {}
    if not daily_dir:
        return board_map
    boards = _read_csv(daily_dir / "industry_hotspot_boards.csv")
    if boards.empty:
        return board_map
    for row in boards.to_dict("records"):
        name = _board_name(row)
        if not name:
            continue
        if name not in board_map or _board_sort_tuple(row) < _board_sort_tuple(board_map[name]):
            board_map[name] = row
    return board_map


def _best_board_context(row, leader_context=None, board_map=None):
    board_map = board_map or {}
    candidates = []
    for name in (leader_context or {}).get("boards") or []:
        if name in board_map:
            candidates.append(board_map[name])
    for name in _split_industries((row or {}).get("industry")):
        if name in board_map:
            candidates.append(board_map[name])
    if not candidates:
        return None
    return sorted(candidates, key=_board_sort_tuple)[0]


def _board_status(board_context):
    if not board_context:
        return {
            "confirmed": False,
            "bad": False,
            "label": "无板块确认",
            "reason": "未进入行业热点报告",
        }

    version = str(board_context.get("热点版本") or "")
    shape = str(board_context.get("参与形态") or "")
    advice = str(board_context.get("参与建议") or "")
    trend = str(board_context.get("关注趋势") or "")
    hot_rank = _to_float(board_context.get("热点排名"))
    attention_rank = _to_float(board_context.get("关注排名"))
    bad = bool(
        any(keyword in shape for keyword in ["加速高潮", "放量下跌", "高位分化"])
        or any(keyword in advice for keyword in ["回避", "谨慎追高"])
        or "分歧" in advice
    )
    confirmed = bool(
        version in {"主线热点", "强势热点"}
        or (hot_rank is not None and hot_rank <= 80)
        or (attention_rank is not None and attention_rank <= 80)
        or trend in {"持续升温", "快速升温"}
    )
    a_confirmed = bool(
        not bad
        and (
            version in {"主线热点", "强势热点", "关注上升"}
            or (
                trend in {"持续升温", "快速升温"}
                and hot_rank is not None
                and hot_rank <= 50
                and (attention_rank is None or attention_rank <= 80)
            )
            or (
                "排名快升" in trend
                and hot_rank is not None
                and hot_rank <= 20
                and attention_rank is not None
                and attention_rank <= 30
                and "一日脉冲" not in version
                and "一日脉冲" not in shape
            )
        )
    )
    label = f"{_board_name(board_context) or '--'}:{version or '--'}/热{int(hot_rank) if hot_rank and hot_rank < 999 else '--'}/关{int(attention_rank) if attention_rank and attention_rank < 999 else '--'}"
    if bad:
        reason = shape or advice or "板块高位/分歧"
    elif confirmed:
        reason = trend or version or "板块进入热点"
    else:
        reason = "板块热度未确认"
    return {
        "confirmed": confirmed,
        "a_confirmed": a_confirmed,
        "bad": bad,
        "label": label,
        "reason": reason,
        "shape": shape,
        "advice": advice,
        "version": version,
        "trend": trend,
    }


def _load_previous_watch_state(trade_date):
    target = _parse_date(trade_date)
    if not OUTPUT_DIR.exists():
        return {}
    state = {}
    for path in sorted(OUTPUT_DIR.glob("*/emotion_leader_watch_latest.csv")):
        day = _parse_date(path.parent.name)
        if not day or (target and day >= target):
            continue
        frame = _read_csv(path)
        if frame.empty or "stock_code" not in frame.columns:
            continue
        for _, row in frame.iterrows():
            code = _normalize_code(row.get("stock_code"))
            if not code:
                continue
            item = state.setdefault(
                code,
                {
                    "appear_days": 0,
                    "last_date": None,
                    "last_stage": None,
                    "max_score": None,
                },
            )
            item["appear_days"] += 1
            item["last_date"] = path.parent.name
            item["last_stage"] = row.get("stage")
            score = _to_float(row.get("score"))
            if score is not None:
                item["max_score"] = max(item["max_score"] or score, score)
    return state


def _strength_context(snapshot):
    if snapshot is None or snapshot.empty:
        return {"market_change_5d_median": None, "industry_change_20d_median": {}}
    frame = snapshot.copy()
    frame["change_5d"] = frame["change_5d"].apply(_to_float)
    frame["change_20d"] = frame["change_20d"].apply(_to_float)
    industry_values = {}
    for _, row in frame.iterrows():
        for industry in _split_industries(row.get("industry")):
            industry_values.setdefault(industry, []).append(row.get("change_20d"))
    industry_median = {}
    for industry, values in industry_values.items():
        cleaned = [value for value in values if value is not None and not pd.isna(value)]
        if cleaned:
            industry_median[industry] = float(pd.Series(cleaned).median())
    cleaned_5d = frame["change_5d"].dropna()
    return {
        "market_change_5d_median": float(cleaned_5d.median()) if not cleaned_5d.empty else None,
        "industry_change_20d_median": industry_median,
    }


def _load_leader_context(trade_date):
    daily_dir = _industry_dir_for_date(trade_date)
    leader_map = {}
    appear_map = {}
    board_map = _load_board_contexts(daily_dir)
    if daily_dir:
        leaders = _read_csv(daily_dir / "industry_hotspot_leaders.csv")
        if not leaders.empty and "股票代码" in leaders.columns:
            leaders = leaders.copy()
            leaders["股票代码"] = leaders["股票代码"].apply(_normalize_code)
            for _, row in leaders.dropna(subset=["股票代码"]).iterrows():
                code = row["股票代码"]
                item = leader_map.setdefault(
                    code,
                    {
                        "boards": set(),
                        "types": set(),
                        "best_hot_rank": 999,
                        "best_attention_rank": 999,
                        "best_leader_rank": 999,
                        "best_score": 0.0,
                    },
                )
                item["boards"].add(str(row.get("板块名称") or "").strip())
                item["types"].add(str(row.get("龙头类型") or "").strip())
                for key, column, default, mode in [
                    ("best_hot_rank", "热点排名", 999, "min"),
                    ("best_attention_rank", "关注排名", 999, "min"),
                    ("best_leader_rank", "板块内龙头排名", 999, "min"),
                    ("best_score", "龙头分", 0, "max"),
                ]:
                    value = _to_float(row.get(column))
                    if value is None:
                        value = default
                    if mode == "max":
                        item[key] = max(item[key], value)
                    else:
                        item[key] = min(item[key], value)

    if INDUSTRY_HOTSPOT_DIR.exists():
        for daily_leader in sorted(INDUSTRY_HOTSPOT_DIR.glob("*/industry_hotspot_leaders.csv")):
            if daily_dir and daily_leader.parent.name > daily_dir.name:
                continue
            leaders = _read_csv(daily_leader)
            if leaders.empty or "股票代码" not in leaders.columns:
                continue
            codes = leaders["股票代码"].apply(_normalize_code).dropna().unique().tolist()
            for code in codes:
                appear_map.setdefault(code, set()).add(daily_leader.parent.name)

    cleaned = {}
    for code, item in leader_map.items():
        cleaned[code] = {
            "boards": [value for value in item["boards"] if value],
            "types": [value for value in item["types"] if value],
            "best_hot_rank": item["best_hot_rank"],
            "best_attention_rank": item["best_attention_rank"],
            "best_leader_rank": item["best_leader_rank"],
            "best_score": item["best_score"],
        }
    return daily_dir.name if daily_dir else None, cleaned, appear_map, board_map


def _is_excluded(row, exclude_prefixes):
    code = str(row.get("stock_code") or "")
    name = str(row.get("stock_name") or "")
    if any(code.startswith(prefix) for prefix in exclude_prefixes):
        return True
    if "ST" in name.upper() or name.startswith("*"):
        return True
    listing_days = _to_float(row.get("listing_trade_days"))
    if listing_days is not None and listing_days <= 90:
        return True
    return False


def _metric(row, key):
    return _to_float(row.get(key))


def _leader_context_text(leader):
    if not leader:
        return "--"
    boards = " / ".join((leader.get("boards") or [])[:3]) or "--"
    types = " / ".join((leader.get("types") or [])[:3]) or "--"
    hot_rank = leader.get("best_hot_rank")
    attention_rank = leader.get("best_attention_rank")
    hot_text = int(hot_rank) if hot_rank and hot_rank < 999 else "--"
    attention_text = int(attention_rank) if attention_rank and attention_rank < 999 else "--"
    return f"{boards}; {types}; 热{hot_text}/关{attention_text}"


def _score_emotion_row(
    row,
    leader_context=None,
    board_context=None,
    appear_days=0,
    watch_state=None,
    strength_context=None,
):
    price = _metric(row, "latest_price")
    if not price:
        return None

    score = 0.0
    reasons = []
    warnings = []
    change_5d = _metric(row, "change_5d")
    change_20d = _metric(row, "change_20d")
    amount = _metric(row, "today_amount")
    amount_avg_5d = _metric(row, "amount_avg_5d")
    amount_ratio_5d = amount / amount_avg_5d if amount and amount_avg_5d else None
    amount_yi = _amount_yi(amount)
    turnover_rate = _metric(row, "turnover_rate")
    today_change = _metric(row, "today_change")
    open_price = _metric(row, "today_open")
    high_price = _metric(row, "today_high")
    low_price = _metric(row, "today_low")
    high_20d = _metric(row, "high_20d")
    low_60d = _metric(row, "low_60d")
    near_high_20d = price / high_20d * 100 if price and high_20d else None
    from_low_60d = (price / low_60d - 1) * 100 if price and low_60d else None
    close_position = (price - low_price) / (high_price - low_price) * 100 if high_price and low_price and high_price > low_price else None
    fade_from_high_pct = (high_price / price - 1) * 100 if high_price and price else None
    ma_values = [_metric(row, key) for key in ("ma5", "ma10", "ma20", "ma60")]
    above_ma_count = sum(1 for value in ma_values if value and price > value)
    ma_bull = bool(all(ma_values) and ma_values[0] > ma_values[1] > ma_values[2] > ma_values[3])
    board_status = _board_status(board_context)
    strength_context = strength_context or {}
    market_median_5d = _to_float(strength_context.get("market_change_5d_median"))
    industry_medians = strength_context.get("industry_change_20d_median") or {}
    market_relative_5d = change_5d - market_median_5d if change_5d is not None and market_median_5d is not None else None
    industry_relative_20d = None
    matched_industry = None
    for industry in _split_industries(row.get("industry")):
        if industry in industry_medians and change_20d is not None:
            industry_relative_20d = change_20d - industry_medians[industry]
            matched_industry = industry
            break
    watch_state = watch_state or {}
    watch_appear_days = int(watch_state.get("appear_days") or 0)

    if above_ma_count >= 4:
        score += 18
        reasons.append("站上MA5/10/20/60")
    elif above_ma_count == 3:
        score += 10
        reasons.append("站上3条均线")
    if ma_bull:
        score += 8
        reasons.append("均线多头")

    if near_high_20d is not None:
        if near_high_20d >= 98:
            score += 14
            reasons.append("贴近/突破20日高点")
        elif near_high_20d >= 94:
            score += 10
            reasons.append("接近20日高点")

    if change_5d is not None:
        if 5 <= change_5d <= 25:
            score += 18
            reasons.append("5日启动未高潮")
        elif 25 < change_5d <= 45:
            score += 8
            warnings.append("5日涨幅偏热")
        elif change_5d > 45:
            score -= 14
            warnings.append("5日涨幅过热")
        elif 0 <= change_5d < 5:
            score += 6
            reasons.append("5日温和走强")

    if change_20d is not None:
        if 8 <= change_20d <= 45:
            score += 20
            reasons.append("20日早期区间")
        elif 45 < change_20d <= 80:
            score += 6
            warnings.append("20日涨幅偏热")
        elif change_20d > 80:
            score -= 18
            warnings.append("20日涨幅过热")
        elif change_20d < 0:
            score -= 8
            warnings.append("20日仍弱")

    if amount_ratio_5d is not None:
        if amount_ratio_5d >= 1.5:
            score += 14
            reasons.append("成交显著放量")
        elif amount_ratio_5d >= 1.15:
            score += 8
            reasons.append("成交温和放量")
        elif amount_ratio_5d < 0.7 and change_5d and change_5d > 10:
            score -= 5
            warnings.append("缩量上涨")

    if turnover_rate is not None:
        if 6 <= turnover_rate <= 18:
            score += 14
            reasons.append("换手活跃")
        elif 18 < turnover_rate <= 28:
            score += 4
            warnings.append("高换手")
        elif turnover_rate > 28:
            score -= 8
            warnings.append("换手过热")

    if amount_yi is not None:
        if amount_yi >= 8:
            score += 8
            reasons.append("成交容量够")
        elif amount_yi >= 3:
            score += 4
        else:
            score -= 3
            warnings.append("成交容量偏小")

    if close_position is not None:
        if close_position >= 75:
            score += 8
            reasons.append("收盘位置强")
        elif close_position >= 55:
            score += 3
            reasons.append("收盘位置尚可")
        else:
            score -= 12
            warnings.append("收盘位置偏弱")
    if fade_from_high_pct is not None:
        if fade_from_high_pct >= 4:
            score -= 12
            warnings.append("冲高回落明显")
        elif fade_from_high_pct >= 3:
            score -= 6
            warnings.append("上影线偏长")

    if leader_context:
        if _to_float(leader_context.get("best_leader_rank")) is not None and leader_context["best_leader_rank"] <= 1:
            score += 10
            reasons.append("板块前排")
        hot_rank = _to_float(leader_context.get("best_hot_rank"))
        attention_rank = _to_float(leader_context.get("best_attention_rank"))
        if (hot_rank is not None and hot_rank <= 50) or (attention_rank is not None and attention_rank <= 80):
            score += 6
            reasons.append("行业热榜可见")

    if board_status["confirmed"] and not board_status["bad"]:
        score += 8
        reasons.append("板块热度确认")
    elif board_status["bad"]:
        score -= 14
        warnings.append(f"板块形态不适合追:{board_status['reason']}")
    else:
        warnings.append(board_status["reason"])

    if appear_days:
        if appear_days <= 2:
            score += 6
            reasons.append(f"热榜新进{appear_days}日")
        elif appear_days <= 5:
            score += 3
            reasons.append(f"热榜持续{appear_days}日")
        else:
            warnings.append(f"热榜已持续{appear_days}日")

    if market_relative_5d is not None:
        if market_relative_5d >= 8:
            score += 8
            reasons.append("5日显著跑赢市场")
        elif market_relative_5d >= 5:
            score += 4
            reasons.append("5日跑赢市场")
        elif market_relative_5d < 2:
            score -= 8
            warnings.append("5日相对市场不够强")
    if industry_relative_20d is not None:
        if industry_relative_20d >= 5:
            score += 6
            reasons.append("20日跑赢同行")
        elif industry_relative_20d < 0:
            score -= 6
            warnings.append(f"20日弱于同行:{matched_industry}")

    if watch_appear_days >= 4 and near_high_20d is not None and near_high_20d < 99:
        score -= 10
        warnings.append(f"胚子池重复{watch_appear_days}日未突破")
    elif watch_appear_days:
        warnings.append(f"胚子池已出现{watch_appear_days}日")

    risk_level = str(row.get("risk_overlay_level") or "")
    if risk_level == "低":
        score += 8
        reasons.append("风控低")
    elif risk_level == "中":
        score += 2
        warnings.append("风控中")
    elif risk_level == "高":
        score -= 20
        warnings.append("风控高")

    limit_up_days_5d = _metric(row, "limit_up_days_5d")
    if limit_up_days_5d is not None and limit_up_days_5d >= 2:
        score -= 10
        warnings.append("近期连板/过热")

    hard_overheated = bool(
        risk_level == "高"
        or (limit_up_days_5d is not None and limit_up_days_5d >= 2)
        or (change_20d is not None and change_20d > 80)
        or (change_5d is not None and change_5d > 45)
    )
    early_gain = bool(
        change_5d is not None
        and 5 <= change_5d <= 25
        and change_20d is not None
        and 8 <= change_20d <= 45
    )
    breakout_shape = bool(
        near_high_20d is not None
        and near_high_20d >= 96
        and above_ma_count >= 4
    )
    volume_ok = bool(amount_ratio_5d is not None and amount_ratio_5d >= 1.15)
    turnover_ok = bool(turnover_rate is not None and 4 <= turnover_rate <= 18)
    liquidity_ok = bool(amount_yi is not None and amount_yi >= 3)
    has_board_visibility = bool(leader_context or appear_days > 0)
    has_capacity_or_board = bool((amount_yi is not None and amount_yi >= 8) or has_board_visibility)
    close_quality_ok = bool(
        close_position is not None
        and close_position >= (70 if today_change is not None and today_change >= 5 else 55)
        and (fade_from_high_pct is None or fade_from_high_pct < 3.5)
    )
    board_gate_ok = bool(board_status.get("a_confirmed") and not board_status["bad"])
    relative_gate_ok = bool(
        market_relative_5d is None
        or market_relative_5d >= 5
    ) and bool(
        industry_relative_20d is None
        or industry_relative_20d >= 0
    )
    repeat_gate_ok = bool(watch_appear_days < 4 or (near_high_20d is not None and near_high_20d >= 99))

    if hard_overheated:
        stage = "D过热回避"
        trigger = "只做复盘样本；若继续强，只观察开板换手后的二次承接"
        invalid = "放量跌破5日线或高位长阴，视为退潮确认"
    elif (
        score >= 112
        and early_gain
        and breakout_shape
        and volume_ok
        and turnover_ok
        and liquidity_ok
        and has_capacity_or_board
        and close_quality_ok
        and board_gate_ok
        and relative_gate_ok
        and repeat_gate_ok
    ):
        stage = "A早期胚子"
        trigger = "等分歧不破5日线/昨收后，再放量突破日高"
        invalid = "跌回MA10或冲高回落超过3%且收不回"
    elif score >= 98 and breakout_shape and (change_20d is None or change_20d <= 70) and not board_status["bad"]:
        stage = "B右侧确认"
        trigger = "只看回踩承接或弱转强，不追急拉"
        invalid = "放量跌破MA5，或板块前排转弱"
    elif score >= 76:
        stage = "C龙头已成"
        trigger = "只看承接，不作为新开仓优先点"
        invalid = "高位放量长阴或连续跌破昨收"
    else:
        stage = "观察"
        trigger = "信号不足，只做跟踪"
        invalid = "跌破平台或量能消失"

    return {
        "score": round(score, 2),
        "stage": stage,
        "trigger": trigger,
        "invalid": invalid,
        "amount_ratio_5d": _round(amount_ratio_5d, 2),
        "today_amount_yi": _round(amount_yi, 2),
        "near_high_20d_pct": _round(near_high_20d, 2),
        "from_low_60d_pct": _round(from_low_60d, 2),
        "close_position_pct": _round(close_position, 2),
        "fade_from_high_pct": _round(fade_from_high_pct, 2),
        "above_ma_count": int(above_ma_count),
        "ma_bull": bool(ma_bull),
        "board_context": board_status["label"],
        "board_confirmed": bool(board_status["confirmed"]),
        "board_bad": bool(board_status["bad"]),
        "board_status_reason": board_status["reason"],
        "market_relative_5d": _round(market_relative_5d, 2),
        "industry_relative_20d": _round(industry_relative_20d, 2),
        "watch_appear_days": int(watch_appear_days),
        "watch_last_date": watch_state.get("last_date"),
        "a_gate_close_quality": bool(close_quality_ok),
        "a_gate_board": bool(board_gate_ok),
        "a_gate_relative_strength": bool(relative_gate_ok),
        "a_gate_repeat": bool(repeat_gate_ok),
        "reasons": "；".join(dict.fromkeys(reasons)) or "--",
        "warnings": "；".join(dict.fromkeys(warnings)) or "--",
    }


def build_emotion_leader_watch(
    trade_date=None,
    top=DEFAULT_TOP,
    include_688=False,
    include_bj=False,
):
    target_trade_date = _date_text(trade_date) or _latest_history_date()
    if not target_trade_date:
        raise RuntimeError("无法确定情绪龙头胚子池分析交易日")

    snapshot, risk_date = _load_snapshot(target_trade_date)
    if snapshot.empty:
        raise RuntimeError(f"{target_trade_date} 没有 a_stock_analysis_history 快照")

    industry_date, leader_map, appear_map, board_map = _load_leader_context(target_trade_date)
    strength = _strength_context(snapshot)
    previous_watch_state = _load_previous_watch_state(target_trade_date)
    exclude_prefixes = []
    for prefix in DEFAULT_EXCLUDE_PREFIXES:
        if prefix == "688" and include_688:
            continue
        if prefix in {"920", "8", "4"} and include_bj:
            continue
        exclude_prefixes.append(prefix)

    rows = []
    for raw in snapshot.to_dict("records"):
        code = _normalize_code(raw.get("stock_code"))
        if not code:
            continue
        raw["stock_code"] = code
        if _is_excluded(raw, tuple(exclude_prefixes)):
            continue
        leader_context = leader_map.get(code)
        board_context = _best_board_context(raw, leader_context=leader_context, board_map=board_map)
        appear_days = len(appear_map.get(code, set()))
        scored = _score_emotion_row(
            raw,
            leader_context=leader_context,
            board_context=board_context,
            appear_days=appear_days,
            watch_state=previous_watch_state.get(code),
            strength_context=strength,
        )
        if not scored:
            continue
        if scored["score"] < 45 and scored["stage"] == "观察":
            continue
        rows.append(
            {
                "trade_date": target_trade_date,
                "model_version": MODEL_VERSION,
                "stage": scored["stage"],
                "stock_code": code,
                "stock_name": raw.get("stock_name"),
                "industry": raw.get("industry"),
                "score": scored["score"],
                "latest_price": _round(raw.get("latest_price"), 2),
                "today_change": _round(raw.get("today_change"), 2),
                "change_3d": _round(raw.get("change_3d"), 2),
                "change_5d": _round(raw.get("change_5d"), 2),
                "change_10d": _round(raw.get("change_10d"), 2),
                "change_20d": _round(raw.get("change_20d"), 2),
                "change_30d": _round(raw.get("change_30d"), 2),
                "today_amount_yi": scored["today_amount_yi"],
                "amount_ratio_5d": scored["amount_ratio_5d"],
                "turnover_rate": _round(raw.get("turnover_rate"), 2),
                "near_high_20d_pct": scored["near_high_20d_pct"],
                "from_low_60d_pct": scored["from_low_60d_pct"],
                "close_position_pct": scored["close_position_pct"],
                "fade_from_high_pct": scored["fade_from_high_pct"],
                "above_ma_count": scored["above_ma_count"],
                "ma_bull": scored["ma_bull"],
                "board_context": scored["board_context"],
                "board_confirmed": scored["board_confirmed"],
                "board_bad": scored["board_bad"],
                "board_status_reason": scored["board_status_reason"],
                "market_relative_5d": scored["market_relative_5d"],
                "industry_relative_20d": scored["industry_relative_20d"],
                "watch_appear_days": scored["watch_appear_days"],
                "watch_last_date": scored["watch_last_date"],
                "a_gate_close_quality": scored["a_gate_close_quality"],
                "a_gate_board": scored["a_gate_board"],
                "a_gate_relative_strength": scored["a_gate_relative_strength"],
                "a_gate_repeat": scored["a_gate_repeat"],
                "risk_overlay_date": risk_date,
                "risk_overlay_level": raw.get("risk_overlay_level"),
                "risk_overlay_score": _round(raw.get("risk_overlay_score"), 2),
                "risk_overlay_labels": raw.get("risk_overlay_labels"),
                "limit_up_days_5d": _round(raw.get("limit_up_days_5d"), 0),
                "consecutive_limit_up_days": _round(raw.get("consecutive_limit_up_days"), 0),
                "industry_hotspot_date": industry_date,
                "leader_appear_days": int(appear_days),
                "leader_context": _leader_context_text(leader_context),
                "trigger": scored["trigger"],
                "invalid": scored["invalid"],
                "reasons": scored["reasons"],
                "warnings": scored["warnings"],
            }
        )

    result_frame = pd.DataFrame(rows)
    if not result_frame.empty:
        stage_order = {"A早期胚子": 1, "B右侧确认": 2, "C龙头已成": 3, "D过热回避": 4, "观察": 5}
        result_frame["_stage_order"] = result_frame["stage"].map(stage_order).fillna(9)
        result_frame = result_frame.sort_values(
            ["_stage_order", "score", "change_20d"],
            ascending=[True, False, False],
        ).drop(columns=["_stage_order"])
        if top:
            result_frame = result_frame.head(int(top)).copy()

    stage_counts = {}
    if not result_frame.empty:
        stage_counts = {key: int(value) for key, value in result_frame["stage"].value_counts().items()}

    return {
        "success": True,
        "run_time": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "trade_date": target_trade_date,
        "risk_overlay_date": risk_date,
        "industry_hotspot_date": industry_date,
        "model_version": MODEL_VERSION,
        "include_688": bool(include_688),
        "include_bj": bool(include_bj),
        "stage_counts": stage_counts,
        "candidate_count": int(len(result_frame)),
        "rows": result_frame.to_dict("records") if not result_frame.empty else [],
    }


def _md_cell(value):
    text = str(value if value is not None else "--").strip() or "--"
    return text.replace("|", "/").replace("\n", " ")


def _pct_text(value):
    value = _to_float(value)
    return "--" if value is None else f"{round(value, 2)}%"


def _num_text(value):
    value = _to_float(value)
    return "--" if value is None else str(round(value, 2))


def render_markdown(result, stage_limit=DEFAULT_STAGE_LIMIT):
    rows = result.get("rows") or []
    lines = [
        "# 情绪龙头胚子池",
        "",
        f"- 运行时间: {result.get('run_time')}",
        f"- 分析交易日: {result.get('trade_date')}",
        f"- 风险覆盖日期: {result.get('risk_overlay_date') or '--'}",
        f"- 行业热点日期: {result.get('industry_hotspot_date') or '--'}",
        f"- 模型版本: {result.get('model_version')}",
        f"- 候选数量: {result.get('candidate_count')}，分层: {result.get('stage_counts') or {}}",
        "- 口径: 只做观察雷达，不写入正式策略表；默认过滤688、北交所、ST、近端新股。",
        "- A类硬门槛: 收盘质量、板块确认、相对强度、重复入池未突破全部通过，才允许进A。",
        "",
    ]

    stage_names = ["A早期胚子", "B右侧确认", "C龙头已成", "D过热回避", "观察"]
    for stage in stage_names:
        stage_rows = [row for row in rows if row.get("stage") == stage][: int(stage_limit or DEFAULT_STAGE_LIMIT)]
        if not stage_rows:
            continue
        lines.append(f"## {stage}")
        lines.append("")
        lines.append("|代码|名称|板块|分数|5日|20日|收盘位|日高回落|相对5日|同行20日|板块确认|成交额亿|额比|换手|触发|放弃|提醒|")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---|---|")
        for row in stage_rows:
            lines.append(
                "|{code}|{name}|{industry}|{score}|{chg5}|{chg20}|{close_pos}|{fade}|{market_rel}|{industry_rel}|{board}|{amount}|{ratio}|{turnover}|{trigger}|{invalid}|{warnings}|".format(
                    code=_md_cell(row.get("stock_code")),
                    name=_md_cell(row.get("stock_name")),
                    industry=_md_cell(row.get("industry")),
                    score=_num_text(row.get("score")),
                    chg5=_pct_text(row.get("change_5d")),
                    chg20=_pct_text(row.get("change_20d")),
                    close_pos=_pct_text(row.get("close_position_pct")),
                    fade=_pct_text(row.get("fade_from_high_pct")),
                    market_rel=_pct_text(row.get("market_relative_5d")),
                    industry_rel=_pct_text(row.get("industry_relative_20d")),
                    board=_md_cell(row.get("board_context") or row.get("leader_context")),
                    amount=_num_text(row.get("today_amount_yi")),
                    ratio=_num_text(row.get("amount_ratio_5d")),
                    turnover=_pct_text(row.get("turnover_rate")),
                    trigger=_md_cell(row.get("trigger")),
                    invalid=_md_cell(row.get("invalid")),
                    warnings=_md_cell(row.get("warnings")),
                )
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def save_outputs(result, output_dir=None):
    trade_date = _date_text(result.get("trade_date")) or dt.datetime.now().strftime("%Y-%m-%d")
    root = Path(output_dir) if output_dir else OUTPUT_DIR
    date_dir = root / trade_date
    date_dir.mkdir(parents=True, exist_ok=True)
    stable_dir = root
    stable_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(result.get("rows") or [])
    markdown = render_markdown(result)
    json_text = json.dumps(result, ensure_ascii=False, default=str, indent=2)

    md_path = date_dir / "emotion_leader_watch_latest.md"
    csv_path = date_dir / "emotion_leader_watch_latest.csv"
    json_path = date_dir / "emotion_leader_watch_latest.json"
    stable_md_path = stable_dir / "emotion_leader_watch_latest.md"
    stable_csv_path = stable_dir / "emotion_leader_watch_latest.csv"
    stable_json_path = stable_dir / "emotion_leader_watch_latest.json"

    md_path.write_text(markdown, encoding="utf-8")
    stable_md_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(json_text, encoding="utf-8")
    stable_json_path.write_text(json_text, encoding="utf-8")
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    frame.to_csv(stable_csv_path, index=False, encoding="utf-8-sig")

    return {
        "daily_dir": str(date_dir),
        "md_path": str(md_path),
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "stable_md_path": str(stable_md_path),
        "stable_csv_path": str(stable_csv_path),
        "stable_json_path": str(stable_json_path),
    }


def run_emotion_leader_watch(
    trade_date=None,
    top=DEFAULT_TOP,
    include_688=False,
    include_bj=False,
    output_dir=None,
    print_report=True,
):
    result = build_emotion_leader_watch(
        trade_date=trade_date,
        top=top,
        include_688=include_688,
        include_bj=include_bj,
    )
    output_paths = save_outputs(result, output_dir=output_dir)
    result["output_paths"] = output_paths
    if print_report:
        print(render_markdown(result))
        print(f"情绪龙头胚子池已保存: {output_paths['md_path']}")
        print(f"CSV: {output_paths['csv_path']}")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="生成情绪龙头胚子观察池")
    parser.add_argument("--trade-date", default=None, help="分析交易日；默认取库里最新交易日")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP, help="最多输出候选数")
    parser.add_argument("--include-688", action="store_true", help="包含688科创板")
    parser.add_argument("--include-bj", action="store_true", help="包含北交所")
    parser.add_argument("--output-dir", default=None, help="输出根目录；默认 log/emotion_leader_watch")
    parser.add_argument("--json", action="store_true", help="输出JSON")
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_emotion_leader_watch(
        trade_date=args.trade_date,
        top=args.top,
        include_688=args.include_688,
        include_bj=args.include_bj,
        output_dir=args.output_dir,
        print_report=not args.json,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, default=str, indent=2))


if __name__ == "__main__":
    main()
