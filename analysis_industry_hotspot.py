"""行业热点、关注度上升和板块龙头分析。

从 a_stock_analysis_history 读取最近交易日数据，按 industry 拆分板块，
用金融视角拆成价格强度、资金确认、上涨广度、趋势质量、拥挤风险等维度，
输出当前热点版本、关注上升板块，以及每个板块的当前龙头股。
"""

from __future__ import annotations

import argparse
import math
import os
from datetime import datetime

import numpy as np
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

NUMERIC_COLUMNS = [
    "latest_price",
    "today_open",
    "today_high",
    "today_low",
    "today_change",
    "change_3d",
    "change_5d",
    "change_10d",
    "change_20d",
    "change_30d",
    "today_vol",
    "today_amount",
    "turnover_rate",
    "vol_avg_3d",
    "vol_avg_5d",
    "vol_avg_10d",
    "vol_avg_20d",
    "amount_avg_5d",
    "amount_avg_10d",
    "turnover_avg_5d",
    "turnover_avg_10d",
    "stock_rank",
    "today_amp",
    "amp_3d",
    "amp_5d",
    "amp_10d",
    "amp_20d",
    "vr_today",
    "vr_3d",
    "vr_5d",
    "vr_10d",
    "vr_20d",
    "ma5",
    "ma10",
    "ma20",
    "ma60",
    "ma120",
    "high_20d",
    "low_20d",
    "high_60d",
    "low_60d",
    "high_120d",
    "low_120d",
    "volatility_10d",
    "volatility_20d",
    "ma20_slope_5d",
    "ma60_slope_10d",
]

FORWARD_HORIZONS = (3, 5, 10)


def _connect():
    return pymysql.connect(**DB_CONFIG)


def _query_frame(connection, sql, params=None):
    with connection.cursor() as cursor:
        cursor.execute(sql, params or [])
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
    return pd.DataFrame.from_records(rows, columns=columns)


def _normalize_date_text(value):
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    text = str(value).strip()
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        return text[:10]
    return text or None


def _ensure_columns(frame, columns, default_value=np.nan):
    for column in columns:
        if column not in frame.columns:
            frame[column] = default_value


def _ordered_unique(values):
    result = []
    seen = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _parse_industry(value):
    if value is None or pd.isna(value):
        return []
    return _ordered_unique(str(value).replace("，", ",").split(","))


def _safe_divide(numerator, denominator):
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def _clip_factor(series, upper=5):
    return pd.to_numeric(series, errors="coerce").clip(lower=0, upper=upper).fillna(0)


def _round_value(value, digits=2):
    if value is None or pd.isna(value):
        return "--"
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return "--"


def _format_percent(value):
    rounded = _round_value(value, 2)
    return "--" if rounded == "--" else f"{rounded}%"


def _format_number(value, digits=2):
    rounded = _round_value(value, digits)
    if rounded == "--":
        return "--"
    if float(rounded).is_integer():
        return str(int(rounded))
    return str(rounded)


def load_recent_history(lookback_trade_days=30, end_date=None):
    connection = _connect()
    try:
        params = []
        date_sql = """
            SELECT DISTINCT last_data_date
            FROM a_stock_analysis_history
            WHERE last_data_date IS NOT NULL
        """
        if end_date:
            date_sql += " AND last_data_date <= %s"
            params.append(str(end_date))
        date_sql += " ORDER BY last_data_date DESC LIMIT %s"
        params.append(int(max(1, lookback_trade_days)))

        dates = _query_frame(connection, date_sql, params)
        if dates.empty:
            return pd.DataFrame(), []

        trade_dates = [_normalize_date_text(value) for value in dates["last_data_date"].tolist()]
        trade_dates = [value for value in trade_dates if value]
        if not trade_dates:
            return pd.DataFrame(), []

        placeholders = ",".join(["%s"] * len(trade_dates))
        history_sql = f"""
            SELECT *
            FROM a_stock_analysis_history
            WHERE last_data_date IN ({placeholders})
            ORDER BY last_data_date, stock_code
        """
        history = _query_frame(connection, history_sql, trade_dates)
    finally:
        connection.close()

    if history.empty:
        return history, trade_dates

    _ensure_columns(history, ["stock_code", "stock_name", "industry", "last_data_date"], "")
    _ensure_columns(history, NUMERIC_COLUMNS)

    history["last_data_date"] = pd.to_datetime(history["last_data_date"], errors="coerce")
    history = history.dropna(subset=["last_data_date", "stock_code"]).copy()
    history["stock_code"] = history["stock_code"].astype(str).str.strip()
    history["stock_name"] = history["stock_name"].fillna("").astype(str).str.strip()
    history["stock_name"] = history["stock_name"].replace("", np.nan).fillna(history["stock_code"])

    for column in NUMERIC_COLUMNS:
        history[column] = pd.to_numeric(history[column], errors="coerce")

    if "is_last_info" in history.columns:
        history["_is_last_info"] = pd.to_numeric(history["is_last_info"], errors="coerce").fillna(0)
    else:
        history["_is_last_info"] = 0

    history = history.sort_values(["last_data_date", "stock_code", "_is_last_info"])
    history = history.drop_duplicates(subset=["last_data_date", "stock_code"], keep="last")
    history = history.drop(columns=["_is_last_info"], errors="ignore")
    history = history[history["latest_price"].fillna(0).gt(0) | history["today_amount"].fillna(0).gt(0)].copy()
    return history.reset_index(drop=True), sorted(set(trade_dates))


def prepare_stock_industry(history):
    frame = history.copy()
    frame["industry_list"] = frame["industry"].apply(_parse_industry)
    frame = frame[frame["industry_list"].map(bool)].copy()
    if frame.empty:
        return frame

    frame["amount_ratio_5d"] = _safe_divide(frame["today_amount"], frame["amount_avg_5d"])
    frame["amount_ratio_10d"] = _safe_divide(frame["today_amount"], frame["amount_avg_10d"])
    frame["vol_ratio_5d"] = _safe_divide(frame["today_vol"], frame["vol_avg_5d"])
    frame["vol_ratio_10d"] = _safe_divide(frame["today_vol"], frame["vol_avg_10d"])
    frame["turnover_ratio_5d"] = _safe_divide(frame["turnover_rate"], frame["turnover_avg_5d"])
    frame["turnover_ratio_10d"] = _safe_divide(frame["turnover_rate"], frame["turnover_avg_10d"])
    frame["close_to_20d_high"] = _safe_divide(frame["latest_price"], frame["high_20d"])
    frame["close_to_60d_high"] = _safe_divide(frame["latest_price"], frame["high_60d"])
    frame["price_vs_ma20"] = _safe_divide(frame["latest_price"], frame["ma20"]) - 1
    frame["price_vs_ma60"] = _safe_divide(frame["latest_price"], frame["ma60"]) - 1

    frame["_today_up"] = np.where(frame["today_change"].notna(), frame["today_change"].gt(0).astype(float), np.nan)
    frame["_change_5d_up"] = np.where(frame["change_5d"].notna(), frame["change_5d"].gt(0).astype(float), np.nan)
    frame["_strong_today"] = np.where(frame["today_change"].notna(), frame["today_change"].ge(3).astype(float), np.nan)
    frame["_limit_up"] = np.where(frame["today_change"].notna(), frame["today_change"].ge(9.5).astype(float), 0)
    frame["_weak_today"] = np.where(frame["today_change"].notna(), frame["today_change"].le(-3).astype(float), np.nan)
    frame["_near_20d_high"] = np.where(frame["close_to_20d_high"].notna(), frame["close_to_20d_high"].ge(0.97).astype(float), np.nan)
    frame["_ma20_up"] = np.where(frame["ma20_slope_5d"].notna(), frame["ma20_slope_5d"].gt(0).astype(float), np.nan)
    frame["_above_ma20"] = np.where(frame["price_vs_ma20"].notna(), frame["price_vs_ma20"].gt(0).astype(float), np.nan)

    exploded = frame.explode("industry_list").rename(columns={"industry_list": "industry_name"})
    exploded["industry_name"] = exploded["industry_name"].astype(str).str.strip()
    exploded = exploded[exploded["industry_name"] != ""].copy()
    exploded = exploded.drop_duplicates(subset=["last_data_date", "stock_code", "industry_name"], keep="last")
    return exploded.reset_index(drop=True)


def build_market_context(history):
    market = (
        history.groupby("last_data_date", sort=False)
        .agg(
            market_today_change=("today_change", "mean"),
            market_change_5d=("change_5d", "mean"),
            market_change_10d=("change_10d", "mean"),
            market_breadth_today=("_today_up", "mean"),
            market_breadth_5d=("_change_5d_up", "mean"),
            market_total_amount=("today_amount", "sum"),
        )
        .reset_index()
    )
    market["market_breadth_today"] *= 100
    market["market_breadth_5d"] *= 100
    return market


def build_industry_daily(stock_industry, history):
    market = build_market_context(history)

    industry = (
        stock_industry.groupby(["last_data_date", "industry_name"], sort=False)
        .agg(
            stock_count=("stock_code", "nunique"),
            avg_today_change=("today_change", "mean"),
            median_today_change=("today_change", "median"),
            avg_change_3d=("change_3d", "mean"),
            avg_change_5d=("change_5d", "mean"),
            avg_change_10d=("change_10d", "mean"),
            avg_change_20d=("change_20d", "mean"),
            breadth_today_pct=("_today_up", "mean"),
            breadth_5d_pct=("_change_5d_up", "mean"),
            strong_stock_pct=("_strong_today", "mean"),
            weak_stock_pct=("_weak_today", "mean"),
            limit_up_count=("_limit_up", "sum"),
            total_amount=("today_amount", "sum"),
            avg_amount=("today_amount", "mean"),
            avg_turnover_rate=("turnover_rate", "mean"),
            avg_amount_ratio_5d=("amount_ratio_5d", "mean"),
            avg_vol_ratio_5d=("vol_ratio_5d", "mean"),
            avg_turnover_ratio_5d=("turnover_ratio_5d", "mean"),
            near_20d_high_pct=("_near_20d_high", "mean"),
            above_ma20_pct=("_above_ma20", "mean"),
            ma20_up_pct=("_ma20_up", "mean"),
            volatility_20d=("volatility_20d", "mean"),
        )
        .reset_index()
    )

    for column in [
        "breadth_today_pct",
        "breadth_5d_pct",
        "strong_stock_pct",
        "weak_stock_pct",
        "near_20d_high_pct",
        "above_ma20_pct",
        "ma20_up_pct",
    ]:
        industry[column] = industry[column] * 100

    amount_ranked = stock_industry.sort_values(
        ["last_data_date", "industry_name", "today_amount"],
        ascending=[True, True, False],
        na_position="last",
    )
    top3_amount = (
        amount_ranked.groupby(["last_data_date", "industry_name"], sort=False)
        .head(3)
        .groupby(["last_data_date", "industry_name"], sort=False)["today_amount"]
        .sum()
        .rename("top3_amount")
        .reset_index()
    )
    industry = industry.merge(top3_amount, on=["last_data_date", "industry_name"], how="left")
    industry["top3_amount_share"] = industry["top3_amount"] / industry["total_amount"].replace(0, np.nan)
    industry["top3_amount_share"] = industry["top3_amount_share"].clip(lower=0, upper=1)
    industry["total_amount_yi"] = industry["total_amount"] / 100000000
    industry["avg_amount_yi"] = industry["avg_amount"] / 100000000

    industry = industry.merge(market, on="last_data_date", how="left")
    industry["industry_alpha_today"] = industry["avg_today_change"] - industry["market_today_change"]
    industry["industry_alpha_5d"] = industry["avg_change_5d"] - industry["market_change_5d"]
    industry["industry_alpha_10d"] = industry["avg_change_10d"] - industry["market_change_10d"]
    return score_industries(industry)


def score_industries(industry):
    frame = industry.copy()

    amount_ratio = _clip_factor(frame["avg_amount_ratio_5d"], 5)
    vol_ratio = _clip_factor(frame["avg_vol_ratio_5d"], 5)
    turnover_ratio = _clip_factor(frame["avg_turnover_ratio_5d"], 5)
    total_amount_scale = np.log10(frame["total_amount_yi"].clip(lower=0).fillna(0) + 1)

    frame["价格强度分"] = (
        frame["avg_today_change"].fillna(0) * 0.9
        + frame["avg_change_3d"].fillna(0) * 0.7
        + frame["avg_change_5d"].fillna(0) * 0.55
        + frame["avg_change_10d"].fillna(0) * 0.25
        + frame["industry_alpha_5d"].fillna(0) * 0.75
        + frame["industry_alpha_10d"].fillna(0) * 0.35
    )
    frame["资金确认分"] = (
        amount_ratio * 12
        + vol_ratio * 8
        + turnover_ratio * 6
        + frame["avg_turnover_rate"].fillna(0).clip(lower=0, upper=20) * 0.9
        + total_amount_scale * 8
    )
    frame["扩散广度分"] = (
        frame["breadth_today_pct"].fillna(0) * 0.06
        + frame["breadth_5d_pct"].fillna(0) * 0.05
        + frame["strong_stock_pct"].fillna(0) * 0.08
        + frame["limit_up_count"].fillna(0) * 2.5
        - frame["weak_stock_pct"].fillna(0) * 0.05
    )
    frame["趋势质量分"] = (
        frame["near_20d_high_pct"].fillna(0) * 0.07
        + frame["above_ma20_pct"].fillna(0) * 0.05
        + frame["ma20_up_pct"].fillna(0) * 0.05
        + (1 - frame["top3_amount_share"].fillna(0.5)).clip(lower=0, upper=1) * 5
    )

    frame["hot_score"] = (
        frame["价格强度分"]
        + frame["扩散广度分"]
        + frame["资金确认分"] * 0.65
        + frame["趋势质量分"]
    ).round(4)
    frame["attention_score"] = (
        frame["资金确认分"]
        + amount_ratio * 10
        + turnover_ratio * 8
        + total_amount_scale * 6
        + frame["top3_amount_share"].fillna(0) * 6
    ).round(4)

    frame = frame.sort_values(["last_data_date", "industry_name"]).reset_index(drop=True)
    frame["hot_rank"] = frame.groupby("last_data_date")["hot_score"].rank(method="min", ascending=False)
    frame["attention_rank"] = frame.groupby("last_data_date")["attention_score"].rank(method="min", ascending=False)

    frame = frame.sort_values(["industry_name", "last_data_date"]).reset_index(drop=True)
    grouped = frame.groupby("industry_name", group_keys=False)
    frame["prev_attention_score"] = grouped["attention_score"].shift(1)
    frame["prev_hot_score"] = grouped["hot_score"].shift(1)
    frame["prev_attention_rank"] = grouped["attention_rank"].shift(1)
    frame["prev_hot_rank"] = grouped["hot_rank"].shift(1)
    frame["attention_score_prev5_mean"] = grouped["attention_score"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=2).mean()
    )
    frame["hot_score_prev5_mean"] = grouped["hot_score"].transform(lambda s: s.shift(1).rolling(5, min_periods=2).mean())
    frame["top10_days_5d"] = grouped["hot_rank"].transform(lambda s: s.le(10).rolling(5, min_periods=1).sum())

    frame["attention_score_1d_delta"] = frame["attention_score"] - frame["prev_attention_score"]
    frame["attention_score_vs_5d"] = frame["attention_score"] - frame["attention_score_prev5_mean"]
    frame["hot_score_1d_delta"] = frame["hot_score"] - frame["prev_hot_score"]
    frame["attention_rank_improve_1d"] = frame["prev_attention_rank"] - frame["attention_rank"]
    frame["hot_rank_improve_1d"] = frame["prev_hot_rank"] - frame["hot_rank"]

    frame["rising_score"] = (
        frame["attention_score_1d_delta"].fillna(0) * 0.45
        + frame["attention_score_vs_5d"].fillna(0) * 0.7
        + frame["hot_score_1d_delta"].fillna(0) * 0.25
        + frame["attention_rank_improve_1d"].fillna(0) * 0.8
        + frame["hot_rank_improve_1d"].fillna(0) * 0.5
        + np.where(frame["avg_today_change"].fillna(0) > 0, 3, 0)
        + np.where(frame["breadth_today_pct"].fillna(0) >= 55, 3, 0)
    ).round(4)

    frame["热点版本"] = frame.apply(classify_hot_version, axis=1)
    frame["关注趋势"] = frame.apply(classify_attention_trend, axis=1)
    frame["风险提示"] = frame.apply(build_risk_flags, axis=1)
    frame["金融解读"] = frame.apply(build_industry_interpretation, axis=1)
    return frame.sort_values(["last_data_date", "hot_rank", "attention_rank"]).reset_index(drop=True)


def classify_hot_version(row):
    if (
        row["hot_rank"] <= 5
        and row["attention_rank"] <= 10
        and row["breadth_5d_pct"] >= 55
        and row["avg_change_5d"] > 0
    ):
        return "主线热点"
    if row["hot_rank"] <= 10 and row["avg_today_change"] > 0 and row["breadth_today_pct"] >= 55:
        return "强势热点"
    if row["attention_score_vs_5d"] > 8 and row["attention_rank"] <= 25 and row["avg_today_change"] > 0:
        return "关注上升"
    if row["avg_today_change"] >= 2 and row["breadth_today_pct"] >= 60 and row["avg_change_5d"] <= 0:
        return "一日脉冲"
    if row["attention_score_vs_5d"] > 0 and row["avg_today_change"] > 0:
        return "资金试探"
    return "观察"


def classify_attention_trend(row):
    one_day = row.get("attention_score_1d_delta")
    five_day = row.get("attention_score_vs_5d")
    rank_improve = row.get("attention_rank_improve_1d")
    if pd.notna(one_day) and pd.notna(five_day) and one_day > 0 and five_day > 0:
        return "持续升温"
    if pd.notna(rank_improve) and rank_improve >= 8:
        return "排名快升"
    if pd.notna(one_day) and one_day > 0:
        return "短线升温"
    if pd.notna(five_day) and five_day > 0:
        return "较5日偏热"
    return "平稳/回落"


def build_risk_flags(row):
    flags = []
    if row["top3_amount_share"] >= 0.65:
        flags.append("成交过度集中")
    if row["avg_today_change"] > 1.5 and row["breadth_today_pct"] < 45:
        flags.append("少数股拉动")
    if row["avg_today_change"] > 0 and row["avg_amount_ratio_5d"] < 1:
        flags.append("缩量上涨")
    if row["avg_turnover_rate"] >= 12 and row["avg_change_5d"] >= 10:
        flags.append("短线拥挤")
    if row["weak_stock_pct"] >= 25:
        flags.append("内部分化")
    if row["near_20d_high_pct"] >= 75 and row["avg_today_change"] <= 0:
        flags.append("高位滞涨")
    return "；".join(flags) if flags else "暂无明显异常"


def build_industry_interpretation(row):
    return (
        f"{row['热点版本']}，热点排名{int(row['hot_rank'])}、关注排名{int(row['attention_rank'])}；"
        f"今日均涨{_format_percent(row['avg_today_change'])}，5日均涨{_format_percent(row['avg_change_5d'])}，"
        f"5日超额{_format_percent(row['industry_alpha_5d'])}；"
        f"上涨广度{_format_percent(row['breadth_today_pct'])}，成交额放大{_format_number(row['avg_amount_ratio_5d'])}倍。"
    )


def classify_participation_setup(row):
    avg_today = row.get("avg_today_change", 0) or 0
    avg_5d = row.get("avg_change_5d", 0) or 0
    breadth_today = row.get("breadth_today_pct", 0) or 0
    breadth_5d = row.get("breadth_5d_pct", 0) or 0
    amount_ratio = row.get("avg_amount_ratio_5d", 0) or 0
    turnover = row.get("avg_turnover_rate", 0) or 0
    attention_vs_5d = row.get("attention_score_vs_5d", 0) or 0
    hot_rank = row.get("hot_rank", 9999) or 9999
    attention_rank = row.get("attention_rank", 9999) or 9999
    top3_share = row.get("top3_amount_share", 0) or 0
    weak_pct = row.get("weak_stock_pct", 0) or 0

    if avg_today < 0 and amount_ratio >= 1.3:
        return "放量下跌"
    if avg_5d >= 8 and avg_today >= 3 and turnover >= 8:
        return "加速高潮"
    if top3_share >= 0.65 and breadth_today < 60:
        return "少数股拉动"
    if avg_today > 0 and amount_ratio < 1 and breadth_today < 60:
        return "缩量反弹"
    if weak_pct >= 25 and breadth_today < 50:
        return "高位分化"
    if (
        hot_rank <= 15
        and attention_rank <= 30
        and avg_5d > 0
        and breadth_5d >= 55
        and attention_vs_5d > 0
        and amount_ratio >= 1.1
    ):
        return "主线延续"
    if (
        attention_vs_5d >= 8
        and -5 <= avg_5d <= 4
        and avg_today >= 0
        and breadth_today >= 50
        and amount_ratio >= 1.25
    ):
        return "低位放量启动"
    if breadth_today >= 70 and breadth_5d >= 55 and 0 <= avg_today <= 4 and top3_share < 0.55 and amount_ratio >= 1:
        return "广度扩散"
    if attention_vs_5d > 0 and avg_today > 0:
        return "资金试探"
    return "普通观察"


def build_participation_advice(row):
    setup = row.get("参与形态") or classify_participation_setup(row)
    if setup == "主线延续":
        return "优先跟踪，适合等分歧回踩或龙头确认后参与"
    if setup == "低位放量启动":
        return "重点观察，等次日放量延续和板块扩散确认"
    if setup == "广度扩散":
        return "可跟踪板块内强股，优先选放量且不拥挤的龙头"
    if setup == "资金试探":
        return "先观察，等热点排名或上涨广度继续抬升"
    if setup == "加速高潮":
        return "谨慎追高，更适合等分歧后的二次确认"
    if setup == "少数股拉动":
        return "谨慎，先看是否从龙头扩散到跟涨股"
    if setup == "缩量反弹":
        return "参与价值低，缺少资金确认"
    if setup == "放量下跌":
        return "回避，资金分歧偏大"
    if setup == "高位分化":
        return "谨慎，强股和弱股分化后容易回撤"
    return "观察为主，等待放量、广度和龙头共振"


def prepare_history_for_market(history):
    frame = history.copy()
    frame["amount_ratio_5d"] = _safe_divide(frame["today_amount"], frame["amount_avg_5d"])
    frame["_today_up"] = np.where(frame["today_change"].notna(), frame["today_change"].gt(0).astype(float), np.nan)
    frame["_change_5d_up"] = np.where(frame["change_5d"].notna(), frame["change_5d"].gt(0).astype(float), np.nan)
    return frame


def build_leaders(stock_industry, latest_date, leader_count=3):
    latest = stock_industry[stock_industry["last_data_date"] == latest_date].copy()
    if latest.empty:
        return pd.DataFrame(), pd.DataFrame(columns=["industry_name", "龙头股"])

    related = (
        latest.groupby("stock_code", sort=False)
        .agg(
            stock_name=("stock_name", "first"),
            related_boards=("industry_name", lambda s: ",".join(_ordered_unique(s.tolist()))),
        )
        .reset_index()
    )

    amount_factor = _clip_factor(latest["amount_ratio_5d"], 5)
    vol_factor = _clip_factor(latest["vol_ratio_5d"], 5)
    turnover_factor = _clip_factor(latest["turnover_ratio_5d"], 5)
    amount_scale = np.log10((latest["today_amount"] / 100000000).clip(lower=0).fillna(0) + 1)
    near_high_bonus = np.where(latest["close_to_20d_high"].fillna(0) >= 0.97, 6, 0)
    limit_bonus = np.where(latest["today_change"].fillna(0) >= 9.5, 10, 0)
    rank_score = 80 / (latest["stock_rank"].fillna(9999).clip(lower=1) + 1)

    latest["leader_score"] = (
        latest["today_change"].fillna(0) * 1.2
        + latest["change_3d"].fillna(0) * 1.0
        + latest["change_5d"].fillna(0) * 0.85
        + latest["change_10d"].fillna(0) * 0.35
        + amount_factor * 8
        + vol_factor * 5
        + turnover_factor * 4
        + latest["turnover_rate"].fillna(0).clip(lower=0, upper=25) * 0.8
        + amount_scale * 6
        + near_high_bonus
        + limit_bonus
        + latest["ma20_slope_5d"].fillna(0) * 0.45
        + rank_score
    ).round(4)

    latest["leader_rank"] = latest.groupby("industry_name")["leader_score"].rank(method="first", ascending=False)
    leaders = latest[latest["leader_rank"] <= leader_count].copy()
    leaders = leaders.merge(related[["stock_code", "related_boards"]], on="stock_code", how="left")
    leaders["leader_style"] = leaders.apply(classify_leader_style, axis=1)

    leader_summary = (
        leaders.sort_values(["industry_name", "leader_rank"])
        .groupby("industry_name", sort=False)
        .apply(format_leader_group)
        .rename("龙头股")
        .reset_index()
    )
    return leaders.sort_values(["industry_name", "leader_rank"]).reset_index(drop=True), leader_summary


def classify_leader_style(row):
    if row["today_change"] >= 9.5:
        return "涨停龙头"
    if row["amount_ratio_5d"] >= 2 and row["change_5d"] >= 5:
        return "放量趋势"
    if row["today_amount"] >= 1500000000 and row["change_5d"] > 0:
        return "容量中军"
    if row["change_3d"] >= 6:
        return "弹性领涨"
    if row["close_to_20d_high"] >= 0.97:
        return "趋势核心"
    return "观察核心"


def format_leader_group(group):
    parts = []
    for row in group.sort_values("leader_rank").itertuples(index=False):
        parts.append(
            f"{int(row.leader_rank)}.{row.stock_name}({row.stock_code})"
            f"[{row.leader_style}, 分{_format_number(row.leader_score)}, "
            f"今{_format_percent(row.today_change)}, 5日{_format_percent(row.change_5d)}, "
            f"额比{_format_number(row.amount_ratio_5d)}x, 相关:{row.related_boards}]"
        )
    return " | ".join(parts)


def _escape_markdown_cell(value):
    if value is None or pd.isna(value):
        return "--"
    return str(value).replace("|", "/").replace("\n", " ")


def markdown_table(frame, columns, headers=None, limit=None):
    if frame.empty:
        return "_无数据_"

    view = frame.loc[:, columns].head(limit).copy()
    headers = headers or columns
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(_escape_markdown_cell(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def format_board_output(latest_boards, leader_summary, min_stock_count, limit=None):
    report = latest_boards[latest_boards["stock_count"] >= min_stock_count].copy()
    report = report.merge(leader_summary, on="industry_name", how="left")
    report["参与形态"] = report.apply(classify_participation_setup, axis=1)
    report["参与建议"] = report.apply(build_participation_advice, axis=1)
    report["当前交易日"] = report["last_data_date"].dt.strftime("%Y-%m-%d")
    report["板块名称"] = report["industry_name"]
    report["热点排名"] = report["hot_rank"].astype(int)
    report["关注排名"] = report["attention_rank"].astype(int)
    report["成分股数"] = report["stock_count"].astype(int)
    report["热点分"] = report["hot_score"].map(lambda value: _format_number(value))
    report["关注度分"] = report["attention_score"].map(lambda value: _format_number(value))
    report["关注较昨日"] = report["attention_score_1d_delta"].map(lambda value: _format_number(value))
    report["关注较5日"] = report["attention_score_vs_5d"].map(lambda value: _format_number(value))
    report["今日均涨"] = report["avg_today_change"].map(_format_percent)
    report["5日均涨"] = report["avg_change_5d"].map(_format_percent)
    report["5日超额"] = report["industry_alpha_5d"].map(_format_percent)
    report["上涨广度"] = report["breadth_today_pct"].map(_format_percent)
    report["5日广度"] = report["breadth_5d_pct"].map(_format_percent)
    report["成交额亿"] = report["total_amount_yi"].map(lambda value: _format_number(value))
    report["额比5日"] = report["avg_amount_ratio_5d"].map(lambda value: _format_number(value))
    report["量比5日"] = report["avg_vol_ratio_5d"].map(lambda value: _format_number(value))
    report["前三成交占比"] = (report["top3_amount_share"] * 100).map(_format_percent)
    report["龙头股"] = report["龙头股"].fillna("--")
    report = report.sort_values(["hot_rank", "attention_rank"])
    if limit:
        return report.head(limit)
    return report


def format_leader_output(leaders, board_report):
    if leaders.empty or board_report.empty:
        return pd.DataFrame()

    board_context_columns = [
        "industry_name",
        "当前交易日",
        "板块名称",
        "热点版本",
        "参与形态",
        "参与建议",
        "关注趋势",
        "热点排名",
        "关注排名",
        "热点分",
        "关注度分",
        "今日均涨",
        "5日均涨",
        "5日超额",
        "上涨广度",
        "风险提示",
        "hot_rank",
        "attention_rank",
    ]
    board_context = board_report[board_context_columns].drop_duplicates(subset=["industry_name"])
    report = leaders.merge(board_context, on="industry_name", how="inner")
    if report.empty:
        return report

    report["板块内龙头排名"] = report["leader_rank"].astype(int)
    report["股票代码"] = report["stock_code"]
    report["股票名称"] = report["stock_name"]
    report["龙头类型"] = report["leader_style"]
    report["龙头分"] = report["leader_score"].map(lambda value: _format_number(value))
    report["今日涨幅"] = report["today_change"].map(_format_percent)
    report["近3日涨幅"] = report["change_3d"].map(_format_percent)
    report["近5日涨幅"] = report["change_5d"].map(_format_percent)
    report["近10日涨幅"] = report["change_10d"].map(_format_percent)
    report["今日成交额_亿"] = (report["today_amount"] / 100000000).map(lambda value: _format_number(value))
    report["成交额较5日均额倍数"] = report["amount_ratio_5d"].map(lambda value: _format_number(value))
    report["成交量较5日均量倍数"] = report["vol_ratio_5d"].map(lambda value: _format_number(value))
    report["换手率"] = report["turnover_rate"].map(_format_percent)
    report["接近20日高点比例"] = (report["close_to_20d_high"] * 100).map(_format_percent)
    report["相关板块"] = report["related_boards"]

    output_columns = [
        "当前交易日",
        "板块名称",
        "热点版本",
        "参与形态",
        "参与建议",
        "关注趋势",
        "热点排名",
        "关注排名",
        "热点分",
        "关注度分",
        "板块内龙头排名",
        "股票代码",
        "股票名称",
        "龙头类型",
        "龙头分",
        "今日涨幅",
        "近3日涨幅",
        "近5日涨幅",
        "近10日涨幅",
        "今日成交额_亿",
        "成交额较5日均额倍数",
        "成交量较5日均量倍数",
        "换手率",
        "接近20日高点比例",
        "相关板块",
        "风险提示",
    ]
    return (
        report.sort_values(["hot_rank", "attention_rank", "leader_rank"])
        .loc[:, output_columns]
        .reset_index(drop=True)
    )


def build_forward_industry_returns(stock_industry, horizons=FORWARD_HORIZONS):
    data = stock_industry.dropna(subset=["last_data_date", "stock_code", "latest_price"]).copy()
    data = data[data["latest_price"].gt(0)].copy()
    if data.empty:
        return pd.DataFrame(columns=["last_data_date", "industry_name"])

    dates = sorted(data["last_data_date"].dropna().unique())
    if not dates:
        return pd.DataFrame(columns=["last_data_date", "industry_name"])

    industry_by_date = {
        date: group[["last_data_date", "industry_name", "stock_code", "latest_price"]]
        .drop_duplicates(subset=["industry_name", "stock_code"])
        .copy()
        for date, group in data.groupby("last_data_date", sort=False)
    }
    price_by_date = {
        date: group[["stock_code", "latest_price"]].drop_duplicates(subset=["stock_code"]).copy()
        for date, group in data.groupby("last_data_date", sort=False)
    }

    merged_result = None
    for horizon in horizons:
        horizon_frames = []
        for index, current_date in enumerate(dates):
            future_index = index + horizon
            if future_index >= len(dates):
                continue

            future_date = dates[future_index]
            base_members = industry_by_date.get(current_date)
            future_prices = price_by_date.get(future_date)
            current_prices = price_by_date.get(current_date)
            if base_members is None or future_prices is None or current_prices is None:
                continue

            future_prices = future_prices.rename(columns={"latest_price": "future_price"})
            merged = base_members.merge(future_prices, on="stock_code", how="inner")
            merged = merged[merged["future_price"].gt(0) & merged["latest_price"].gt(0)].copy()
            if merged.empty:
                continue

            market_base = current_prices.merge(future_prices, on="stock_code", how="inner")
            market_base = market_base[market_base["future_price"].gt(0) & market_base["latest_price"].gt(0)].copy()
            market_ret = np.nan
            if not market_base.empty:
                market_ret = ((market_base["future_price"] / market_base["latest_price"] - 1) * 100).mean()

            merged[f"stock_forward_ret_{horizon}d"] = (
                merged["future_price"] / merged["latest_price"] - 1
            ) * 100
            grouped = (
                merged.groupby(["last_data_date", "industry_name"], sort=False)
                .agg(
                    **{
                        f"forward_ret_{horizon}d": (f"stock_forward_ret_{horizon}d", "mean"),
                        f"forward_win_pct_{horizon}d": (
                            f"stock_forward_ret_{horizon}d",
                            lambda s: s.gt(0).mean() * 100,
                        ),
                        f"forward_sample_count_{horizon}d": ("stock_code", "nunique"),
                    }
                )
                .reset_index()
            )
            grouped[f"forward_market_ret_{horizon}d"] = market_ret
            grouped[f"forward_alpha_{horizon}d"] = (
                grouped[f"forward_ret_{horizon}d"] - grouped[f"forward_market_ret_{horizon}d"]
            )
            horizon_frames.append(grouped)

        if not horizon_frames:
            continue

        horizon_result = pd.concat(horizon_frames, ignore_index=True, sort=False)
        if merged_result is None:
            merged_result = horizon_result
        else:
            merged_result = merged_result.merge(horizon_result, on=["last_data_date", "industry_name"], how="outer")

    if merged_result is None:
        return pd.DataFrame(columns=["last_data_date", "industry_name"])
    return merged_result


def _history_conclusion(avg_5d, win_5d, alpha_5d):
    if avg_5d is None or pd.isna(avg_5d):
        return "样本不足"
    if avg_5d > 0 and win_5d >= 55 and alpha_5d > 0:
        return "历史延续较好，可优先跟踪"
    if avg_5d > 0 and alpha_5d > 0:
        return "能跑赢市场，但胜率一般，适合等确认"
    if avg_5d > 0 and alpha_5d <= 0:
        return "有反弹但不占优，谨慎追高"
    if avg_5d <= 0 or win_5d < 45:
        return "容易昙花一现，谨慎追高"
    return "中性，等待更多确认"


def build_historical_participation_review(industry_daily, stock_industry, min_samples=8):
    if industry_daily.empty or stock_industry.empty:
        return pd.DataFrame()

    scored = industry_daily.copy()
    scored["参与形态"] = scored.apply(classify_participation_setup, axis=1)
    forward = build_forward_industry_returns(stock_industry)
    if forward.empty or "forward_ret_5d" not in forward.columns:
        return pd.DataFrame()

    scored = scored.merge(forward, on=["last_data_date", "industry_name"], how="left")
    latest_date = scored["last_data_date"].max()
    scored = scored[scored["last_data_date"] < latest_date].copy()
    if scored.empty:
        return pd.DataFrame()

    rows = []
    for setup, group in scored.groupby("参与形态", sort=False):
        valid_5d = group.dropna(subset=["forward_ret_5d"])
        sample_count = len(valid_5d)
        if sample_count < min_samples:
            continue

        row = {"参与形态": setup, "样本数": sample_count}
        for horizon in FORWARD_HORIZONS:
            ret_col = f"forward_ret_{horizon}d"
            win_col = f"forward_win_pct_{horizon}d"
            alpha_col = f"forward_alpha_{horizon}d"
            valid = group.dropna(subset=[ret_col])
            row[f"{horizon}日平均收益"] = valid[ret_col].mean() if not valid.empty else np.nan
            row[f"{horizon}日胜率"] = valid[win_col].mean() if not valid.empty else np.nan
            row[f"{horizon}日平均超额"] = valid[alpha_col].mean() if not valid.empty else np.nan

        avg_5d = row.get("5日平均收益")
        win_5d = row.get("5日胜率")
        alpha_5d = row.get("5日平均超额")
        row["历史结论"] = _history_conclusion(avg_5d, win_5d, alpha_5d)
        row["参与价值分"] = (
            (avg_5d if pd.notna(avg_5d) else 0) * 0.7
            + (alpha_5d if pd.notna(alpha_5d) else 0) * 1.5
            + ((win_5d if pd.notna(win_5d) else 50) - 50) * 0.05
            + math.log10(sample_count + 1) * 0.05
        )
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    summary = pd.DataFrame(rows).sort_values("参与价值分", ascending=False).reset_index(drop=True)
    for horizon in FORWARD_HORIZONS:
        for column in [f"{horizon}日平均收益", f"{horizon}日胜率", f"{horizon}日平均超额"]:
            summary[column] = summary[column].map(
                lambda value: "--" if pd.isna(value) else f"{round(float(value), 2)}%"
            )
    summary["参与价值分"] = summary["参与价值分"].round(2)
    return summary


def build_report(industry_daily, stock_industry, args, historical_review=None):
    latest_date = industry_daily["last_data_date"].max()
    latest_boards = industry_daily[industry_daily["last_data_date"] == latest_date].copy()
    leaders, leader_summary = build_leaders(stock_industry, latest_date, args.leader_count)
    all_board_report = format_board_output(latest_boards, leader_summary, args.min_stock_count)
    board_report = all_board_report.head(args.limit)

    rising = all_board_report[
        all_board_report["关注趋势"].isin(["持续升温", "排名快升", "短线升温", "较5日偏热"])
    ].copy()
    rising = rising.sort_values(
        by=["rising_score", "attention_rank"],
        ascending=[False, True],
    ).head(args.rising_limit)

    mainline_names = board_report[board_report["热点版本"].isin(["主线热点", "强势热点"])]["板块名称"].head(8).tolist()
    rising_names = rising["板块名称"].head(8).tolist()
    risk_names = board_report[board_report["风险提示"] != "暂无明显异常"]["板块名称"].head(6).tolist()

    date_text = latest_date.strftime("%Y-%m-%d")
    lines = [
        f"# 行业热点资金分析报告",
        "",
        f"- 数据截至: {date_text}",
        f"- 样本窗口: 最近 {args.lookback_trade_days} 个交易日",
        f"- 历史复盘: 最近 {args.review_trade_days} 个交易日，观察后续 3/5/10 日板块收益和超额收益",
        f"- 板块过滤: 成分股不少于 {args.min_stock_count} 只",
        f"- 分析口径: 价格强度 + 资金确认 + 上涨广度 + 趋势质量 + 拥挤风险，仅用于盘面研究",
        "",
        "## 结论摘要",
        "",
        f"- 当前主线/强势热点: {', '.join(mainline_names) if mainline_names else '暂无'}",
        f"- 关注上升板块: {', '.join(rising_names) if rising_names else '暂无'}",
        f"- 需要留意风险的板块: {', '.join(risk_names) if risk_names else '暂无明显集中风险'}",
        "",
        "## 当前热点版本",
        "",
        markdown_table(
            board_report,
            [
                "板块名称",
                "热点版本",
                "参与形态",
                "关注趋势",
                "热点排名",
                "关注排名",
                "成分股数",
                "热点分",
                "关注度分",
                "今日均涨",
                "5日均涨",
                "5日超额",
                "上涨广度",
                "成交额亿",
                "额比5日",
                "参与建议",
                "风险提示",
            ],
        ),
        "",
        "## 历史复盘：哪些形态更值得参与",
        "",
        markdown_table(
            historical_review if historical_review is not None else pd.DataFrame(),
            [
                "参与形态",
                "样本数",
                "3日平均收益",
                "3日胜率",
                "5日平均收益",
                "5日胜率",
                "5日平均超额",
                "10日平均收益",
                "10日胜率",
                "历史结论",
            ],
        ),
        "",
        "## 关注上升板块",
        "",
        markdown_table(
            rising,
            [
                "板块名称",
                "热点版本",
                "关注趋势",
                "关注较昨日",
                "关注较5日",
                "关注排名",
                "今日均涨",
                "上涨广度",
                "额比5日",
                "量比5日",
                "金融解读",
            ],
        ),
        "",
        "## 板块当前龙头股",
        "",
        markdown_table(
            board_report,
            [
                "板块名称",
                "热点版本",
                "热点排名",
                "关注排名",
                "龙头股",
            ],
        ),
        "",
        "## 评分说明",
        "",
        "- 热点分: 价格强度、上涨广度、资金确认、趋势质量的综合得分，偏向判断赚钱效应和主线强度。",
        "- 关注度分: 成交额/成交量/换手率相对均值的放大、绝对成交规模和成交集中度，偏向判断资金注意力。",
        "- 关注较昨日/较5日: 当前关注度分相对上一交易日和前5个交易日均值的变化，用于识别关注上升。",
        "- 龙头股: 结合涨幅、放量、换手、成交容量、接近20日高点、均线斜率和市场排名选出。",
    ]

    leader_report = format_leader_output(leaders, all_board_report)
    output = "\n".join(lines)
    return output, all_board_report, leader_report, historical_review


def save_outputs(report, board_report, leaders, historical_review, args):
    log_path = func.getenv("LOG_PATH") or os.path.join(os.getcwd(), "log")
    date_value = board_report["当前交易日"].iloc[0] if not board_report.empty else datetime.now().strftime("%Y-%m-%d")
    daily_dir = os.path.join(log_path, "industry_hotspot", date_value)
    os.makedirs(daily_dir, exist_ok=True)

    output_path = args.output or os.path.join(daily_dir, "industry_hotspot_report.md")
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(report)

    output_dir = os.path.dirname(output_path) or daily_dir
    board_csv = os.path.join(output_dir, "industry_hotspot_boards.csv")
    leader_csv = os.path.join(output_dir, "industry_hotspot_leaders.csv")
    board_report.to_csv(board_csv, index=False, encoding="utf-8-sig")
    leaders.to_csv(leader_csv, index=False, encoding="utf-8-sig")
    review_csv = os.path.join(output_dir, "industry_hotspot_history_review.csv")
    if historical_review is not None and not historical_review.empty:
        historical_review.to_csv(review_csv, index=False, encoding="utf-8-sig")

    return {
        "daily_dir": daily_dir,
        "report_path": output_path,
        "board_csv_path": board_csv,
        "leader_csv_path": leader_csv,
        "review_csv_path": review_csv if historical_review is not None and not historical_review.empty else None,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="分析 a_stock_analysis_history 的行业热点、关注上升和龙头股")
    parser.add_argument("--lookback-trade-days", type=int, default=15, help="读取最近多少个交易日")
    parser.add_argument("--end-date", default=None, help="分析截止交易日，格式 YYYY-MM-DD；默认取表内最新交易日")
    parser.add_argument("--min-stock-count", type=int, default=3, help="板块最少成分股数量")
    parser.add_argument("--limit", type=int, default=30, help="当前热点输出数量")
    parser.add_argument("--rising-limit", type=int, default=15, help="关注上升板块输出数量")
    parser.add_argument("--leader-count", type=int, default=3, help="每个板块输出前几个龙头股")
    parser.add_argument("--review-trade-days", type=int, default=35, help="历史复盘读取最近多少个交易日")
    parser.add_argument("--review-min-samples", type=int, default=8, help="历史复盘每类形态最少样本数")
    parser.add_argument("--output", default=None, help="Markdown 报告输出路径")
    parser.add_argument("--save-csv", action="store_true", help="兼容旧参数；现在默认生成 CSV")
    return parser.parse_args()


def run_industry_hotspot_analysis(
    end_date=None,
    lookback_trade_days=15,
    min_stock_count=3,
    limit=30,
    rising_limit=15,
    leader_count=3,
    review_trade_days=35,
    review_min_samples=8,
    output=None,
    print_report=True,
):
    args = argparse.Namespace(
        end_date=end_date,
        lookback_trade_days=lookback_trade_days,
        min_stock_count=min_stock_count,
        limit=limit,
        rising_limit=rising_limit,
        leader_count=leader_count,
        review_trade_days=review_trade_days,
        review_min_samples=review_min_samples,
        output=output,
        save_csv=True,
    )
    history, trade_dates = load_recent_history(args.lookback_trade_days, args.end_date)
    if history.empty:
        raise RuntimeError("a_stock_analysis_history 没有可分析数据")

    stock_industry = prepare_stock_industry(history)
    if stock_industry.empty:
        raise RuntimeError("industry 字段为空，无法拆分板块")

    history_for_market = prepare_history_for_market(history)
    industry_daily = build_industry_daily(stock_industry, history_for_market)

    review_days = max(args.review_trade_days, args.lookback_trade_days + max(FORWARD_HORIZONS) + 5)
    review_history, _ = load_recent_history(review_days, args.end_date)
    review_stock_industry = prepare_stock_industry(review_history)
    review_market = prepare_history_for_market(review_history)
    review_industry_daily = build_industry_daily(review_stock_industry, review_market)
    historical_review = build_historical_participation_review(
        review_industry_daily,
        review_stock_industry,
        min_samples=args.review_min_samples,
    )

    report, board_report, leaders, historical_review = build_report(
        industry_daily,
        stock_industry,
        args,
        historical_review=historical_review,
    )
    output_paths = save_outputs(report, board_report, leaders, historical_review, args)

    if print_report:
        print(report)
        print("")
        print(f"输出目录: {output_paths['daily_dir']}")
        print(f"报告已保存: {output_paths['report_path']}")
        print(f"板块明细CSV: {output_paths['board_csv_path']}")
        print(f"龙头明细CSV: {output_paths['leader_csv_path']}")
        if output_paths.get("review_csv_path"):
            print(f"历史复盘CSV: {output_paths['review_csv_path']}")
        if trade_dates:
            print(f"实际读取交易日: {trade_dates[0]} 至 {trade_dates[-1]}，共 {len(trade_dates)} 个交易日")

    return {
        "success": True,
        "trade_dates": trade_dates,
        "output_paths": output_paths,
        "board_count": int(len(board_report)),
        "leader_count": int(len(leaders)),
        "history_review_count": int(len(historical_review)) if historical_review is not None else 0,
    }


def main():
    args = parse_args()
    return run_industry_hotspot_analysis(
        end_date=args.end_date,
        lookback_trade_days=args.lookback_trade_days,
        min_stock_count=args.min_stock_count,
        limit=args.limit,
        rising_limit=args.rising_limit,
        leader_count=args.leader_count,
        review_trade_days=args.review_trade_days,
        review_min_samples=args.review_min_samples,
        output=args.output,
        print_report=True,
    )


if __name__ == "__main__":
    main()
