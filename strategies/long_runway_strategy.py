"""Medium/long-term runway strategy public workflow."""

from . import common as core

for _name in dir(core):
    if not _name.startswith("__") and _name not in globals():
        globals()[_name] = getattr(core, _name)


def _prepare_long_runway_history(history):
    _emit_runtime_status(f"{LONG_RUNWAY_MODEL_DISPLAY}缓存: 长周期上下文构建开始, rows={len(history)}")
    runway_history = _build_long_runway_context(history)
    _emit_runtime_status(f"{LONG_RUNWAY_MODEL_DISPLAY}缓存: 长周期上下文构建完成, rows={len(runway_history)}")
    runway_history = _build_long_runway_forward_returns(runway_history, horizons=LONG_RUNWAY_HORIZONS)
    _emit_runtime_status(f"{LONG_RUNWAY_MODEL_DISPLAY}缓存: 远期收益构建完成, rows={len(runway_history)}")
    runway_history = _add_percentile_columns(runway_history, LONG_RUNWAY_FEATURE_COLUMNS)
    _emit_runtime_status(f"{LONG_RUNWAY_MODEL_DISPLAY}缓存: 横截面分位构建完成, rows={len(runway_history)}")
    return runway_history.reset_index(drop=True)


def _long_runway_cache_config():
    return {
        "schema_version": LONG_RUNWAY_CACHE_SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "long_runway_model_version": LONG_RUNWAY_MODEL_VERSION,
        "horizons": tuple(LONG_RUNWAY_HORIZONS),
        "half_life_days": LONG_RUNWAY_HALF_LIFE_DAYS,
        "winner_ratio": LONG_RUNWAY_WINNER_RATIO,
        "loser_ratio": LONG_RUNWAY_LOSER_RATIO,
        "min_daily_rows": LONG_RUNWAY_MIN_DAILY_ROWS,
        "feature_columns": tuple(LONG_RUNWAY_FEATURE_COLUMNS),
    }


def _long_runway_cache_metadata(runway_history):
    latest_trade_date = None
    sample_start = None
    sample_end = None
    trade_days = 0
    row_count = 0
    if runway_history is not None and not runway_history.empty:
        dates = pd.to_datetime(runway_history["last_data_date"], errors="coerce").dropna()
        if not dates.empty:
            sample_start = dates.min().strftime("%Y-%m-%d")
            sample_end = dates.max().strftime("%Y-%m-%d")
            latest_trade_date = sample_end
            trade_days = int(dates.nunique())
        row_count = int(len(runway_history))
    return {
        **_long_runway_cache_config(),
        "latest_trade_date": latest_trade_date,
        "sample_start": sample_start,
        "sample_end": sample_end,
        "trade_days": trade_days,
        "row_count": row_count,
        "cache_path": str(LONG_RUNWAY_CONTEXT_CACHE_PATH),
    }


def _long_runway_cache_is_compatible(metadata):
    metadata = metadata or {}
    expected = _long_runway_cache_config()
    for key, expected_value in expected.items():
        actual_value = metadata.get(key)
        if isinstance(expected_value, tuple):
            actual_value = tuple(actual_value or ())
        if actual_value != expected_value:
            return False
    return True


def _load_long_runway_cache_payload():
    path = LONG_RUNWAY_CONTEXT_CACHE_PATH
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as error:
        _emit_runtime_status(f"{LONG_RUNWAY_MODEL_DISPLAY}缓存读取失败，将全量重建: {error}")
        return None

    if not _long_runway_cache_is_compatible((payload or {}).get("metadata")):
        _emit_runtime_status(f"{LONG_RUNWAY_MODEL_DISPLAY}缓存版本或模型口径变化，将全量重建")
        return None
    runway_history = (payload or {}).get("runway_history")
    if runway_history is None or runway_history.empty:
        _emit_runtime_status(f"{LONG_RUNWAY_MODEL_DISPLAY}缓存为空，将全量重建")
        return None
    return payload


def _save_long_runway_cache(runway_history, horizon_profiles=None):
    if runway_history is None or runway_history.empty:
        return None

    LONG_RUNWAY_CONTEXT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": _long_runway_cache_metadata(runway_history),
        "runway_history": runway_history.reset_index(drop=True),
        "horizon_profiles": horizon_profiles,
    }
    temp_path = LONG_RUNWAY_CONTEXT_CACHE_PATH.with_suffix(".tmp")
    with temp_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp_path, LONG_RUNWAY_CONTEXT_CACHE_PATH)
    return payload["metadata"]


def _merge_runway_history_cache(cached_history, new_context):
    frames = []
    if cached_history is not None and not cached_history.empty:
        frames.append(cached_history)
    if new_context is not None and not new_context.empty:
        frames.append(new_context)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged = _prepare_common_frame(merged, dedupe_keys=("last_data_date", "stock_code"))
    return merged.sort_values(["last_data_date", "stock_code"]).reset_index(drop=True)


def _refresh_long_runway_forward_returns(frame, affected_start_date):
    if frame is None or frame.empty or affected_start_date is None:
        return frame

    prepared = frame.copy()
    prepared["last_data_date"] = pd.to_datetime(prepared["last_data_date"], errors="coerce")
    affected_start = pd.to_datetime(affected_start_date, errors="coerce")
    if pd.isna(affected_start):
        return prepared

    price_columns = ["last_data_date", "stock_code", "latest_price", "today_change"]
    for column in ["today_open", "today_high", "today_low"]:
        if column in prepared.columns:
            price_columns.append(column)

    tail_prices = prepared[prepared["last_data_date"] >= affected_start][price_columns].copy()
    if tail_prices.empty:
        return prepared

    refreshed = _build_long_runway_forward_returns(tail_prices, horizons=LONG_RUNWAY_HORIZONS)
    update_columns = ["entry_change"]
    for horizon_days in LONG_RUNWAY_HORIZONS:
        update_columns.extend(
            [
                f"exit_date_{horizon_days}d",
                f"exit_change_{horizon_days}d",
                f"gross_return_{horizon_days}d",
                f"return_{horizon_days}d",
                f"forward_return_{horizon_days}d",
                f"forward_gross_return_{horizon_days}d",
                f"forward_trade_date_{horizon_days}d",
            ]
        )
    update_columns = [column for column in update_columns if column in refreshed.columns]

    keys = ["last_data_date", "stock_code"]
    base = prepared.set_index(keys)
    updates = refreshed.set_index(keys)[update_columns]
    for column in update_columns:
        if column not in base.columns:
            base[column] = pd.NA
    common_index = base.index.intersection(updates.index)
    if len(common_index) > 0:
        base.loc[common_index, update_columns] = updates.loc[common_index, update_columns]
    return base.reset_index().sort_values(["last_data_date", "stock_code"]).reset_index(drop=True)


def _attach_long_runway_historical_memory(runway_history, snapshot, as_of_date):
    if runway_history is None or runway_history.empty or snapshot is None or snapshot.empty:
        return snapshot

    as_of = pd.to_datetime(as_of_date, errors="coerce")
    if pd.isna(as_of):
        return snapshot

    memories = []
    for horizon_days in LONG_RUNWAY_HORIZONS:
        return_col = f"forward_return_{horizon_days}d"
        exit_col = f"forward_trade_date_{horizon_days}d"
        if return_col not in runway_history.columns or exit_col not in runway_history.columns:
            continue

        history_slice = runway_history[["stock_code", "last_data_date", exit_col, return_col]].copy()
        history_slice[return_col] = pd.to_numeric(history_slice[return_col], errors="coerce")
        history_slice[exit_col] = pd.to_datetime(history_slice[exit_col], errors="coerce")
        history_slice["last_data_date"] = pd.to_datetime(history_slice["last_data_date"], errors="coerce")
        history_slice = history_slice[
            history_slice[return_col].notna()
            & history_slice[exit_col].notna()
            & (history_slice[exit_col] <= as_of)
        ].copy()
        if history_slice.empty:
            continue

        idx = history_slice.groupby("stock_code")[return_col].idxmax()
        memory = history_slice.loc[idx, ["stock_code", "last_data_date", exit_col, return_col]].copy()
        memory = memory.rename(
            columns={
                "last_data_date": f"historical_max_signal_date_{horizon_days}d",
                exit_col: f"historical_max_exit_date_{horizon_days}d",
                return_col: f"historical_max_return_{horizon_days}d",
            }
        )
        memories.append(memory)

    if not memories:
        return snapshot

    enriched = snapshot.copy()
    for memory in memories:
        stale_columns = [column for column in memory.columns if column != "stock_code" and column in enriched.columns]
        if stale_columns:
            enriched = enriched.drop(columns=stale_columns)
        enriched = enriched.merge(memory, on="stock_code", how="left")
    return enriched


def _build_long_runway_history_full_cache(end_date=None):
    history = _load_history(
        end_date=end_date,
        columns=LONG_RUNWAY_FRAME_COLUMNS,
        chunked=True,
        progress_label=f"{LONG_RUNWAY_MODEL_DISPLAY}历史读取",
    )
    if history.empty:
        return history, {"cache_mode": "full_rebuild_failed", "cache_path": str(LONG_RUNWAY_CONTEXT_CACHE_PATH)}

    _emit_runtime_status(f"{LONG_RUNWAY_MODEL_DISPLAY}缓存: 全量构建开始, end_date={_to_date_text(end_date) or 'latest'}")
    runway_history = _prepare_long_runway_history(history)
    metadata = _save_long_runway_cache(runway_history, horizon_profiles=None)
    _emit_runtime_status(
        f"{LONG_RUNWAY_MODEL_DISPLAY}缓存: 全量构建完成, "
        f"latest_trade_date={(metadata or {}).get('latest_trade_date')}, "
        f"trade_days={(metadata or {}).get('trade_days')}, rows={(metadata or {}).get('row_count')}"
    )
    return runway_history, {"cache_mode": "full_rebuild", "cache_path": str(LONG_RUNWAY_CONTEXT_CACHE_PATH)}


def _load_or_update_long_runway_history_cache(end_date=None, rebuild_cache=False):
    requested_end = _to_date_text(end_date)
    if rebuild_cache:
        return _build_long_runway_history_full_cache(end_date=end_date)

    payload = _load_long_runway_cache_payload()
    if payload is None:
        return _build_long_runway_history_full_cache(end_date=end_date)

    cached_history = payload["runway_history"].copy()
    cached_history["last_data_date"] = pd.to_datetime(cached_history["last_data_date"], errors="coerce")
    cache_end = _to_date_text((payload.get("metadata") or {}).get("latest_trade_date"))
    if not requested_end:
        trade_dates = _load_history_trade_dates()
        requested_end = pd.to_datetime(trade_dates[-1]).strftime("%Y-%m-%d") if trade_dates else cache_end

    if requested_end == cache_end:
        _emit_runtime_status(
            f"{LONG_RUNWAY_MODEL_DISPLAY}缓存: 命中, trade_date={cache_end}, "
            f"rows={len(cached_history)}, path={LONG_RUNWAY_CONTEXT_CACHE_PATH}"
        )
        return cached_history, {
            "cache_mode": "hit",
            "cache_path": str(LONG_RUNWAY_CONTEXT_CACHE_PATH),
            "horizon_profiles": payload.get("horizon_profiles"),
        }

    if cache_end and requested_end and pd.to_datetime(requested_end) < pd.to_datetime(cache_end):
        _emit_runtime_status(
            f"{LONG_RUNWAY_MODEL_DISPLAY}缓存: 请求日期早于缓存日期，为避免未来函数，按目标日期全量重建: "
            f"requested={requested_end}, cache={cache_end}"
        )
        history = _load_history(
            end_date=requested_end,
            columns=LONG_RUNWAY_FRAME_COLUMNS,
            chunked=True,
            progress_label=f"{LONG_RUNWAY_MODEL_DISPLAY}历史读取",
        )
        if history.empty:
            return history, {"cache_mode": "historical_rebuild_failed", "cache_path": str(LONG_RUNWAY_CONTEXT_CACHE_PATH)}
        return _prepare_long_runway_history(history), {
            "cache_mode": "historical_rebuild_no_save",
            "cache_path": str(LONG_RUNWAY_CONTEXT_CACHE_PATH),
        }

    trade_dates = _load_history_trade_dates(end_date=requested_end)
    trade_date_texts = [pd.to_datetime(date).strftime("%Y-%m-%d") for date in trade_dates]
    if not trade_date_texts or cache_end not in trade_date_texts:
        _emit_runtime_status(f"{LONG_RUNWAY_MODEL_DISPLAY}缓存: 交易日断点缺失，将全量重建")
        return _build_long_runway_history_full_cache(end_date=requested_end)

    cache_end_index = trade_date_texts.index(cache_end)
    new_trade_dates = trade_date_texts[cache_end_index + 1 :]
    if not new_trade_dates:
        return cached_history, {
            "cache_mode": "hit",
            "cache_path": str(LONG_RUNWAY_CONTEXT_CACHE_PATH),
            "horizon_profiles": payload.get("horizon_profiles"),
        }

    raw_start_index = max(0, cache_end_index - LONG_RUNWAY_ROLLING_CONTEXT_TRADE_DAYS)
    raw_start_date = trade_date_texts[raw_start_index]
    _emit_runtime_status(
        f"{LONG_RUNWAY_MODEL_DISPLAY}缓存: 增量刷新开始, cache_end={cache_end}, "
        f"target={requested_end}, new_trade_days={len(new_trade_dates)}, raw_start={raw_start_date}"
    )
    tail_history = _load_history(
        start_date=raw_start_date,
        end_date=requested_end,
        columns=LONG_RUNWAY_FRAME_COLUMNS,
        chunked=True,
        progress_label=f"{LONG_RUNWAY_MODEL_DISPLAY}增量历史读取",
    )
    if tail_history.empty:
        _emit_runtime_status(f"{LONG_RUNWAY_MODEL_DISPLAY}缓存: 增量数据为空，沿用旧缓存")
        return cached_history, {
            "cache_mode": "incremental_empty",
            "cache_path": str(LONG_RUNWAY_CONTEXT_CACHE_PATH),
            "horizon_profiles": payload.get("horizon_profiles"),
        }

    tail_context = _prepare_long_runway_history(tail_history)
    first_new_date = pd.to_datetime(new_trade_dates[0])
    new_context = tail_context[tail_context["last_data_date"] >= first_new_date].copy()
    merged = _merge_runway_history_cache(cached_history, new_context)

    affected_start_offset = max(LONG_RUNWAY_HORIZONS) + DEFAULT_ENTRY_OFFSET_DAYS + len(new_trade_dates) + LONG_RUNWAY_FORWARD_REFRESH_BUFFER_DAYS
    affected_start_index = max(0, cache_end_index - affected_start_offset)
    affected_start_date = trade_date_texts[affected_start_index]
    merged = _refresh_long_runway_forward_returns(merged, affected_start_date)
    metadata = _save_long_runway_cache(merged, horizon_profiles=None)
    _emit_runtime_status(
        f"{LONG_RUNWAY_MODEL_DISPLAY}缓存: 增量刷新完成, "
        f"latest_trade_date={(metadata or {}).get('latest_trade_date')}, "
        f"trade_days={(metadata or {}).get('trade_days')}, rows={(metadata or {}).get('row_count')}, "
        f"affected_start={affected_start_date}"
    )
    return merged, {"cache_mode": "incremental", "cache_path": str(LONG_RUNWAY_CONTEXT_CACHE_PATH)}


def _build_long_runway_context(frame):
    _assert_feature_unit_registry_complete()
    prepared = frame.sort_values(["stock_code", "last_data_date"]).copy()
    grouped = prepared.groupby("stock_code")

    for horizon_days in (20, 60, 120, 252):
        prepared[f"ret_{horizon_days}d"] = grouped["latest_price"].pct_change(horizon_days) * 100

    prepared["ma120"] = (
        grouped["latest_price"].rolling(120, min_periods=120).mean().reset_index(level=0, drop=True)
    )
    prepared["ma240"] = (
        grouped["latest_price"].rolling(240, min_periods=240).mean().reset_index(level=0, drop=True)
    )

    for horizon_days in (60, 120, 240):
        price_rolling = grouped["latest_price"].rolling(horizon_days, min_periods=horizon_days)
        prepared[f"high_{horizon_days}d"] = (
            price_rolling.max().reset_index(level=0, drop=True)
        )
        prepared[f"low_{horizon_days}d"] = (
            price_rolling.min().reset_index(level=0, drop=True)
        )
        volume_rolling = grouped["today_vol"].rolling(horizon_days, min_periods=horizon_days).mean()
        prepared[f"volume_avg_{horizon_days}d"] = (
            volume_rolling.reset_index(level=0, drop=True).groupby(prepared["stock_code"]).shift(1)
        )

    prepared["price_vs_ma20"] = _safe_ratio(prepared["latest_price"], prepared["ma20"]) - 1
    prepared["price_vs_ma60"] = _safe_ratio(prepared["latest_price"], prepared["ma60"]) - 1
    prepared["price_vs_ma120"] = _safe_ratio(prepared["latest_price"], prepared["ma120"]) - 1
    prepared["price_vs_ma240"] = _safe_ratio(prepared["latest_price"], prepared["ma240"]) - 1
    prepared["ma60_vs_ma120"] = _safe_ratio(prepared["ma60"], prepared["ma120"]) - 1
    prepared["ma120_vs_ma240"] = _safe_ratio(prepared["ma120"], prepared["ma240"]) - 1

    prepared["close_to_60d_high"] = _safe_ratio(prepared["latest_price"], prepared["high_60d"])
    prepared["close_to_120d_high"] = _safe_ratio(prepared["latest_price"], prepared["high_120d"])
    prepared["close_to_240d_high"] = _safe_ratio(prepared["latest_price"], prepared["high_240d"])
    prepared["close_to_60d_low"] = _safe_ratio(prepared["latest_price"], prepared["low_60d"])
    prepared["close_to_120d_low"] = _safe_ratio(prepared["latest_price"], prepared["low_120d"])
    prepared["close_to_240d_low"] = _safe_ratio(prepared["latest_price"], prepared["low_240d"])

    prepared["volume_vs_avg_20d"] = _safe_ratio(prepared["today_vol"], prepared["vol_avg_20d"])
    prepared["volume_vs_avg_60d"] = _safe_ratio(prepared["today_vol"], prepared["volume_avg_60d"])
    prepared["volume_vs_avg_120d"] = _safe_ratio(prepared["today_vol"], prepared["volume_avg_120d"])

    prepared["range_position_120d"] = (
        (prepared["latest_price"] - prepared["low_120d"]) / (prepared["high_120d"] - prepared["low_120d"])
    )
    prepared["range_position_240d"] = (
        (prepared["latest_price"] - prepared["low_240d"]) / (prepared["high_240d"] - prepared["low_240d"])
    )

    market_ctx = (
        prepared.groupby("last_data_date")
        .agg(
            market_ret_20d=("ret_20d", "mean"),
            market_ret_60d=("ret_60d", "mean"),
            market_ret_120d=("ret_120d", "mean"),
            market_breadth_20d=("ret_20d", lambda s: (s > 0).mean() * 100),
            market_breadth_60d=("ret_60d", lambda s: (s > 0).mean() * 100),
            market_breadth_120d=("ret_120d", lambda s: (s > 0).mean() * 100),
        )
        .reset_index()
    )

    industry_ctx = (
        prepared.dropna(subset=["industry_1"])
        .groupby(["last_data_date", "industry_1"])
        .agg(
            industry_ret_20d=("ret_20d", "mean"),
            industry_ret_60d=("ret_60d", "mean"),
            industry_ret_120d=("ret_120d", "mean"),
            industry_breadth_20d=("ret_20d", lambda s: (s > 0).mean() * 100),
            industry_breadth_60d=("ret_60d", lambda s: (s > 0).mean() * 100),
            industry_breadth_120d=("ret_120d", lambda s: (s > 0).mean() * 100),
            industry_count=("stock_code", "count"),
        )
        .reset_index()
    )

    if not industry_ctx.empty:
        industry_ctx = industry_ctx.merge(market_ctx, on="last_data_date", how="left")
        industry_ctx["industry_alpha_20d"] = industry_ctx["industry_ret_20d"] - industry_ctx["market_ret_20d"]
        industry_ctx["industry_alpha_60d"] = industry_ctx["industry_ret_60d"] - industry_ctx["market_ret_60d"]
        industry_ctx["industry_alpha_120d"] = industry_ctx["industry_ret_120d"] - industry_ctx["market_ret_120d"]
        industry_ctx["industry_strength_rank_20d"] = industry_ctx.groupby("last_data_date")["industry_ret_20d"].rank(
            method="average",
            pct=True,
            ascending=True,
        )
        industry_ctx["industry_strength_rank_60d"] = industry_ctx.groupby("last_data_date")["industry_ret_60d"].rank(
            method="average",
            pct=True,
            ascending=True,
        )
        industry_ctx["industry_strength_rank_120d"] = industry_ctx.groupby("last_data_date")["industry_ret_120d"].rank(
            method="average",
            pct=True,
            ascending=True,
        )

        prepared = prepared.merge(
            industry_ctx[
                [
                    "last_data_date",
                    "industry_1",
                    "industry_ret_20d",
                    "industry_ret_60d",
                    "industry_ret_120d",
                    "industry_breadth_20d",
                    "industry_breadth_60d",
                    "industry_breadth_120d",
                    "industry_alpha_20d",
                    "industry_alpha_60d",
                    "industry_alpha_120d",
                    "industry_strength_rank_20d",
                    "industry_strength_rank_60d",
                    "industry_strength_rank_120d",
                ]
            ],
            on=["last_data_date", "industry_1"],
            how="left",
        )
    else:
        for column in [
            "industry_ret_20d",
            "industry_ret_60d",
            "industry_ret_120d",
            "industry_breadth_20d",
            "industry_breadth_60d",
            "industry_breadth_120d",
            "industry_alpha_20d",
            "industry_alpha_60d",
            "industry_alpha_120d",
            "industry_strength_rank_20d",
            "industry_strength_rank_60d",
            "industry_strength_rank_120d",
        ]:
            prepared[column] = pd.NA

    prepared = prepared.merge(market_ctx, on="last_data_date", how="left")
    prepared["stock_rank_score"] = 1 / (pd.to_numeric(prepared["stock_rank"], errors="coerce") + 1)

    derived_columns = [
        "ret_20d",
        "ret_60d",
        "ret_120d",
        "ret_252d",
        "ma120",
        "ma240",
        "high_60d",
        "high_120d",
        "high_240d",
        "low_60d",
        "low_120d",
        "low_240d",
        "price_vs_ma20",
        "price_vs_ma60",
        "price_vs_ma120",
        "price_vs_ma240",
        "ma60_vs_ma120",
        "ma120_vs_ma240",
        "close_to_60d_high",
        "close_to_120d_high",
        "close_to_240d_high",
        "close_to_60d_low",
        "close_to_120d_low",
        "close_to_240d_low",
        "volume_avg_60d",
        "volume_avg_120d",
        "volume_vs_avg_20d",
        "volume_vs_avg_60d",
        "volume_vs_avg_120d",
        "range_position_120d",
        "range_position_240d",
        "industry_ret_20d",
        "industry_ret_60d",
        "industry_ret_120d",
        "industry_breadth_20d",
        "industry_breadth_60d",
        "industry_breadth_120d",
        "industry_alpha_20d",
        "industry_alpha_60d",
        "industry_alpha_120d",
        "industry_strength_rank_20d",
        "industry_strength_rank_60d",
        "industry_strength_rank_120d",
        "stock_rank_score",
    ]
    for column in derived_columns:
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    prepared["range_position_120d"] = prepared["range_position_120d"].replace([math.inf, -math.inf], pd.NA)
    prepared["range_position_240d"] = prepared["range_position_240d"].replace([math.inf, -math.inf], pd.NA)

    return prepared


def _build_long_runway_forward_returns(frame, horizons=LONG_RUNWAY_HORIZONS):
    if frame is None or frame.empty:
        return frame

    prepared = frame.sort_values(["stock_code", "last_data_date"]).reset_index(drop=True).copy()
    cost_pct = (DEFAULT_FEE_BPS + DEFAULT_SLIPPAGE_BPS) * 2 / 100
    prepared["latest_price"] = pd.to_numeric(prepared["latest_price"], errors="coerce")
    prepared["today_change"] = pd.to_numeric(prepared["today_change"], errors="coerce")
    grouped = prepared.groupby("stock_code", sort=False)

    entry_offset = int(DEFAULT_ENTRY_OFFSET_DAYS)
    entry_close = grouped["latest_price"].shift(-entry_offset)
    prepared["entry_change"] = grouped["today_change"].shift(-entry_offset)

    for horizon_days in horizons:
        exit_offset = entry_offset + int(horizon_days)
        exit_close = grouped["latest_price"].shift(-exit_offset)
        exit_change_col = f"exit_change_{horizon_days}d"
        gross_return_col = f"gross_return_{horizon_days}d"
        return_col = f"return_{horizon_days}d"

        prepared[f"exit_date_{horizon_days}d"] = grouped["last_data_date"].shift(-exit_offset)
        prepared[exit_change_col] = grouped["today_change"].shift(-exit_offset)
        prepared[gross_return_col] = (exit_close - entry_close) / entry_close * 100
        prepared[return_col] = prepared[gross_return_col] - cost_pct

        entry_abs = pd.to_numeric(prepared["entry_change"], errors="coerce").abs()
        exit_abs = pd.to_numeric(prepared[exit_change_col], errors="coerce").abs()
        tradable = (entry_abs.isna() | (entry_abs < DEFAULT_LIMIT_PCT)) & (
            exit_abs.isna() | (exit_abs < DEFAULT_LIMIT_PCT)
        )
        blocked_mask = prepared[return_col].notna() & (~tradable)
        prepared.loc[blocked_mask, [return_col, gross_return_col]] = pd.NA
        prepared[f"forward_return_{horizon_days}d"] = prepared[return_col]
        prepared[f"forward_gross_return_{horizon_days}d"] = prepared[gross_return_col]
        prepared[f"forward_trade_date_{horizon_days}d"] = prepared[f"exit_date_{horizon_days}d"]

    return prepared


def _build_long_runway_stage(record):
    price_vs_ma60 = _to_float(record.get("price_vs_ma60"))
    price_vs_ma120 = _to_float(record.get("price_vs_ma120"))
    price_vs_ma240 = _to_float(record.get("price_vs_ma240"))
    ma60_vs_ma120 = _to_float(record.get("ma60_vs_ma120"))
    ma120_vs_ma240 = _to_float(record.get("ma120_vs_ma240"))
    ret_20d = _to_float(record.get("ret_20d"))
    ret_60d = _to_float(record.get("ret_60d"))
    ret_120d = _to_float(record.get("ret_120d"))
    ret_252d = _to_float(record.get("ret_252d"))
    close_to_60d_high = _to_float(record.get("close_to_60d_high"))
    close_to_120d_high = _to_float(record.get("close_to_120d_high"))
    close_to_240d_high = _to_float(record.get("close_to_240d_high"))
    close_to_120d_low = _to_float(record.get("close_to_120d_low"))
    close_to_240d_low = _to_float(record.get("close_to_240d_low"))
    volume_vs_avg_20d = _to_float(record.get("volume_vs_avg_20d"))
    volume_vs_avg_60d = _to_float(record.get("volume_vs_avg_60d"))
    industry_alpha_20d = _to_float(record.get("industry_alpha_20d"))
    industry_alpha_60d = _to_float(record.get("industry_alpha_60d"))
    industry_alpha_120d = _to_float(record.get("industry_alpha_120d"))
    today_change = _to_float(record.get("today_change"))

    stage_scores = {
        "trend_broken": 0.0,
        "bottom_setup": 0.0,
        "breakout_start": 0.0,
        "main_run": 0.0,
        "pullback_reset": 0.0,
    }

    if price_vs_ma120 is not None and price_vs_ma120 < -0.06:
        stage_scores["trend_broken"] += 2
    if ma60_vs_ma120 is not None and ma60_vs_ma120 < -0.03:
        stage_scores["trend_broken"] += 2
    if ret_60d is not None and ret_60d < 0:
        stage_scores["trend_broken"] += 1
    if ret_120d is not None and ret_120d < 0:
        stage_scores["trend_broken"] += 1

    if price_vs_ma120 is not None and price_vs_ma120 <= 0.03:
        stage_scores["bottom_setup"] += 1
    if close_to_120d_low is not None and close_to_120d_low <= 1.12:
        stage_scores["bottom_setup"] += 2
    if ma60_vs_ma120 is not None and ma60_vs_ma120 >= -0.02:
        stage_scores["bottom_setup"] += 1
    if ret_20d is not None and ret_20d >= -5:
        stage_scores["bottom_setup"] += 1
    if industry_alpha_20d is not None and industry_alpha_20d >= 0:
        stage_scores["bottom_setup"] += 1

    if price_vs_ma120 is not None and price_vs_ma120 > 0:
        stage_scores["breakout_start"] += 1
    if close_to_120d_high is not None and close_to_120d_high >= 0.9:
        stage_scores["breakout_start"] += 2
    if ret_20d is not None and ret_20d > 0:
        stage_scores["breakout_start"] += 1
    if volume_vs_avg_20d is not None and volume_vs_avg_20d >= 0.9:
        stage_scores["breakout_start"] += 1
    if industry_alpha_60d is not None and industry_alpha_60d >= 0:
        stage_scores["breakout_start"] += 1

    if price_vs_ma120 is not None and price_vs_ma120 > 0:
        stage_scores["main_run"] += 1
    if close_to_120d_high is not None and close_to_120d_high >= 0.97:
        stage_scores["main_run"] += 2
    if ret_60d is not None and ret_60d >= 15:
        stage_scores["main_run"] += 2
    if ret_120d is not None and ret_120d >= 25:
        stage_scores["main_run"] += 2
    if volume_vs_avg_20d is not None and volume_vs_avg_20d >= 1:
        stage_scores["main_run"] += 1
    if industry_alpha_120d is not None and industry_alpha_120d >= 0:
        stage_scores["main_run"] += 1

    if price_vs_ma120 is not None and price_vs_ma120 > 0:
        stage_scores["pullback_reset"] += 1
    if ret_20d is not None and ret_20d < 0:
        stage_scores["pullback_reset"] += 2
    if ret_60d is not None and ret_60d > 0:
        stage_scores["pullback_reset"] += 1
    if ret_120d is not None and ret_120d > 0:
        stage_scores["pullback_reset"] += 1
    if close_to_120d_high is not None and close_to_120d_high < 0.88:
        stage_scores["pullback_reset"] += 1

    stage_name = max(stage_scores.items(), key=lambda item: (item[1], item[0]))[0]
    stage_labels = {
        "trend_broken": "趋势破坏",
        "bottom_setup": "底部蓄势",
        "breakout_start": "突破起涨",
        "main_run": "主升延续",
        "pullback_reset": "高位回撤",
    }
    stage_label = stage_labels.get(stage_name, "震荡整理")

    detail_bits = []
    if price_vs_ma60 is not None:
        detail_bits.append(f"价格相对60日均线{_round_or_none(price_vs_ma60 * 100, 2)}%")
    if price_vs_ma120 is not None:
        detail_bits.append(f"价格相对120日均线{_round_or_none(price_vs_ma120 * 100, 2)}%")
    if price_vs_ma240 is not None:
        detail_bits.append(f"价格相对240日均线{_round_or_none(price_vs_ma240 * 100, 2)}%")
    if ret_20d is not None:
        detail_bits.append(f"20日涨幅{_round_or_none(ret_20d, 2)}%")
    if ret_60d is not None:
        detail_bits.append(f"60日涨幅{_round_or_none(ret_60d, 2)}%")
    if ret_120d is not None:
        detail_bits.append(f"120日涨幅{_round_or_none(ret_120d, 2)}%")
    if ret_252d is not None:
        detail_bits.append(f"252日涨幅{_round_or_none(ret_252d, 2)}%")
    if close_to_120d_high is not None:
        detail_bits.append(f"收盘较120日高点{_round_or_none(close_to_120d_high * 100, 2)}%")
    if close_to_120d_low is not None:
        detail_bits.append(f"收盘较120日低点{_round_or_none(close_to_120d_low * 100, 2)}%")
    if volume_vs_avg_20d is not None:
        detail_bits.append(f"成交量相对20日均量{_round_or_none(volume_vs_avg_20d, 2)}倍")
    if industry_alpha_120d is not None:
        detail_bits.append(f"行业120日相对市场强度{_round_or_none(industry_alpha_120d, 2)}%")
    if today_change is not None:
        detail_bits.append(f"今日涨跌{_round_or_none(today_change, 2)}%")

    return {
        "stage_name": stage_name,
        "stage_label": stage_label,
        "stage_scores": {key: _round_or_none(value, 2) for key, value in stage_scores.items()},
        "stage_detail": "，".join(detail_bits) if detail_bits else "长跑阶段特征不足",
    }


def _build_long_runway_plan(record, market_env, stage_label=None, runway_score=None):
    latest_price = _to_float(record.get("latest_price"))
    ma60 = _to_float(record.get("ma60"))
    ma120 = _to_float(record.get("ma120"))
    ma240 = _to_float(record.get("ma240"))
    low_120d = _to_float(record.get("low_120d"))
    low_240d = _to_float(record.get("low_240d"))
    high_120d = _to_float(record.get("high_120d"))
    high_240d = _to_float(record.get("high_240d"))
    price_vs_ma120 = _to_float(record.get("price_vs_ma120"))
    ret_20d = _to_float(record.get("ret_20d"))
    ret_60d = _to_float(record.get("ret_60d"))
    ret_120d = _to_float(record.get("ret_120d"))
    volume_vs_avg_20d = _to_float(record.get("volume_vs_avg_20d"))
    risk_overlay_block = bool(record.get("risk_overlay_block_formal"))
    risk_overlay_downgrade = bool(record.get("risk_overlay_downgrade"))
    risk_overlay_labels = str(record.get("risk_overlay_labels") or "").strip()

    stage_label = stage_label or "震荡整理"
    market_env = market_env or "未知"

    action = "观望"
    hold_low, hold_high = 4, 12
    exp_low, exp_high = 8.0, 25.0
    stop_price = None
    stop_rule = "长周期趋势尚未确认，先等待更清晰的结构。"
    reenter_rule = "重新站上120日均线并维持强势后，再考虑跟随。"
    position_hint = "轻仓"

    if stage_label == "底部蓄势":
        action = "观察布局"
        hold_low, hold_high = 8, 20
        exp_low, exp_high = 8.0, 25.0
        support = _nearest_support_below(latest_price, [ma120, ma240, low_120d, low_240d])
        stop_price = _round_or_none((support * 0.98) if support else (latest_price * 0.94 if latest_price else None), 2)
        stop_rule = "跌破120日均线或底部平台后先撤。"
        reenter_rule = "重新收复120日均线并站稳，再考虑加回。"
        position_hint = "试探性轻仓"

    elif stage_label == "突破起涨":
        action = "小仓跟随" if market_env != "偏弱" else "观望"
        hold_low, hold_high = 12, 30
        exp_low, exp_high = 15.0, 40.0
        support = _nearest_support_below(latest_price, [ma60, ma120, low_120d, high_120d])
        stop_price = _round_or_none((support * 0.985) if support else (latest_price * 0.93 if latest_price else None), 2)
        stop_rule = "突破平台后若回落失守60/120日均线，说明突破失败，先撤。"
        reenter_rule = "再度放量站回突破位并确认强势后，再考虑继续。"
        position_hint = "中小仓位"

    elif stage_label == "主升延续":
        action = "持有跟随" if market_env != "偏弱" else "谨慎持有"
        hold_low, hold_high = 20, 60
        exp_low, exp_high = 20.0, 60.0
        support = _nearest_support_below(latest_price, [ma60, ma120, ma240, low_120d])
        stop_price = _round_or_none((support * 0.98) if support else (latest_price * 0.9 if latest_price else None), 2)
        stop_rule = "长周期趋势线被破坏，或连续跌破60/120日均线后不回收，先退出。"
        reenter_rule = "回踩后重新站回60/120日均线并转强，再考虑加仓。"
        position_hint = "中高仓位"

    elif stage_label == "高位回撤":
        action = "谨慎持有" if market_env in ("强势", "偏强") else "观望"
        hold_low, hold_high = 5, 15
        exp_low, exp_high = -5.0, 15.0
        support = _nearest_support_below(latest_price, [ma60, ma120, ma240, low_120d, low_240d])
        stop_price = _round_or_none((support * 0.975) if support else (latest_price * 0.92 if latest_price else None), 2)
        stop_rule = "如果回撤后连120日均线都守不住，说明阶段要切换，先撤。"
        reenter_rule = "重新回到120日均线之上，并再次放量，才考虑重新跟随。"
        position_hint = "轻仓观察"

    elif stage_label == "趋势破坏":
        action = "退出观察"
        hold_low, hold_high = 0, 5
        exp_low, exp_high = -10.0, 5.0
        support = _nearest_support_below(latest_price, [ma60, ma120, ma240, low_120d, low_240d])
        stop_price = _round_or_none((support * 0.97) if support else (latest_price * 0.9 if latest_price else None), 2)
        stop_rule = "长周期趋势已经破坏，先不追，等重新站回120日均线后再看。"
        reenter_rule = "至少重新收复120日均线并形成新的高点再考虑。"
        position_hint = "空仓观察"

    else:
        action = "观察"
        hold_low, hold_high = 4, 12
        exp_low, exp_high = 5.0, 20.0
        support = _nearest_support_below(latest_price, [ma60, ma120, ma240, low_120d])
        stop_price = _round_or_none((support * 0.98) if support else (latest_price * 0.94 if latest_price else None), 2)
        stop_rule = "趋势不明，先等更明确的方向。"
        reenter_rule = "重新站稳120日均线后再考虑。"
        position_hint = "轻仓"

    if runway_score is not None:
        if runway_score >= 85 and stage_label in ("突破起涨", "主升延续"):
            hold_high = max(hold_high, hold_low + 10)
            exp_high += 10.0
            if action in ("小仓跟随", "持有跟随", "谨慎持有"):
                action = "重点跟随"
                position_hint = "中高仓位"
        elif runway_score < 65:
            hold_high = max(hold_low, hold_high - 5)
            exp_high = max(exp_low + 1.0, exp_high - 10.0)
            if action in ("小仓跟随", "持有跟随", "重点跟随"):
                action = "观望"
                position_hint = "轻仓"

    if risk_overlay_block:
        action = "风险观察"
        position_hint = "空仓观察"
        exp_high = min(exp_high, exp_low + 5.0)
    elif risk_overlay_downgrade:
        if action in ("小仓跟随", "持有跟随", "重点跟随", "谨慎持有"):
            action = "谨慎观察"
        position_hint = "轻仓观察"
        exp_high = min(exp_high, exp_low + 10.0)

    hold_low = max(0, int(round(hold_low)))
    hold_high = max(hold_low, int(round(hold_high)))
    expected_low = _round_or_none(exp_low, 1)
    expected_high = _round_or_none(exp_high, 1)
    hold_period = f"{hold_low}-{hold_high}周" if hold_low != hold_high else f"{hold_low}周"
    expected_return = f"{expected_low}%~{expected_high}%"

    stage_piece = stage_label
    if ret_20d is not None and ret_60d is not None and ret_120d is not None:
        stage_piece = (
            f"{stage_label}（20日{_round_or_none(ret_20d, 2)}%，"
            f"60日{_round_or_none(ret_60d, 2)}%，120日{_round_or_none(ret_120d, 2)}%）"
        )

    conclusion = (
        f"当前更像{stage_piece}，建议{action}，跟随{hold_period}，预期{expected_return}，"
        f"跌破{stop_price}附近先撤。"
    )

    if market_env == "偏弱" and stage_label in ("突破起涨", "主升延续"):
        conclusion = f"市场偏弱时长跑票容易回撤，当前仍是{stage_piece}，动作应收敛为{action}。{conclusion}"
    if risk_overlay_block:
        conclusion = f"风险覆盖提示{risk_overlay_labels or '特殊池/事件风险'}，当前不按长跑买点处理，只保留观察。{conclusion}"
    elif risk_overlay_downgrade:
        conclusion = f"风险覆盖提示{risk_overlay_labels or '特殊池/事件风险'}，长跑信号降级观察。{conclusion}"

    return {
        "action": action,
        "hold_period": hold_period,
        "expected_return": expected_return,
        "stop_price": stop_price,
        "stop_rule": stop_rule,
        "reenter_rule": reenter_rule,
        "position_hint": position_hint,
        "conclusion": conclusion,
    }


def _resolve_long_runway_market_env(snapshot):
    market_ret_60d = _round_or_none(pd.to_numeric(snapshot["market_ret_60d"], errors="coerce").mean(), 4)
    market_breadth_60d = _round_or_none(pd.to_numeric(snapshot["market_breadth_60d"], errors="coerce").mean(), 2)
    market_ret_120d = _round_or_none(pd.to_numeric(snapshot["market_ret_120d"], errors="coerce").mean(), 4)
    market_breadth_120d = _round_or_none(pd.to_numeric(snapshot["market_breadth_120d"], errors="coerce").mean(), 2)

    if pd.isna(market_ret_60d):
        market_env = "未知"
    elif market_ret_60d > 3 and (market_breadth_60d or 0) >= 55:
        market_env = "强势"
    elif market_ret_60d > 0 and (market_breadth_60d or 0) >= 50:
        market_env = "偏强"
    elif market_ret_60d > -2:
        market_env = "震荡"
    else:
        market_env = "偏弱"

    return {
        "market_env": market_env,
        "market_ret_60d": market_ret_60d,
        "market_breadth_60d": market_breadth_60d,
        "market_ret_120d": market_ret_120d,
        "market_breadth_120d": market_breadth_120d,
    }


def _score_historical_runway_evidence(record):
    historical_max_60d = _to_float(record.get("historical_max_return_60d"))
    historical_max_120d = _to_float(record.get("historical_max_return_120d"))
    historical_max_252d = _to_float(record.get("historical_max_return_252d"))
    ret_60d = _to_float(record.get("ret_60d"))
    ret_120d = _to_float(record.get("ret_120d"))
    ret_252d = _to_float(record.get("ret_252d"))
    today_amount = _to_float(record.get("today_amount"))
    industry_alpha_120d = _to_float(record.get("industry_alpha_120d"))
    close_to_120d_high = _to_float(record.get("close_to_120d_high"))
    close_to_240d_high = _to_float(record.get("close_to_240d_high"))

    points = 0.0
    bits = []
    caution_bits = []

    def add_return_score(value, buckets, label):
        nonlocal points
        if value is None:
            return
        for threshold, score, note in buckets:
            if value >= threshold:
                points += score
                bits.append(f"{label}{_round_or_none(value, 2)}%，{note}")
                return

    long_252d = historical_max_252d if historical_max_252d is not None else ret_252d
    long_120d = historical_max_120d if historical_max_120d is not None else ret_120d
    long_60d = historical_max_60d if historical_max_60d is not None else ret_60d

    add_return_score(
        long_252d,
        [
            (180, 36, "一年级别超级长跑"),
            (120, 32, "一年级别长跑已验证"),
            (80, 24, "一年级别趋势较强"),
            (50, 16, "一年级别趋势有效"),
            (30, 8, "一年级别有趋势雏形"),
        ],
        "历史252日最高收益",
    )
    add_return_score(
        long_120d,
        [
            (120, 26, "半年级别主升已验证"),
            (80, 24, "半年级别强趋势"),
            (50, 18, "半年级别趋势有效"),
            (30, 12, "半年级别有弹性"),
            (15, 6, "半年级别初步转强"),
        ],
        "历史120日最高收益",
    )
    add_return_score(
        long_60d,
        [
            (100, 18, "季度级别强加速"),
            (50, 16, "季度级别主升"),
            (25, 10, "季度级别趋势延续"),
            (10, 5, "季度级别温和转强"),
        ],
        "历史60日最高收益",
    )

    if ret_252d is not None and ret_120d is not None and ret_252d >= 80 and ret_120d >= 30:
        points += 8
        bits.append("当前仍保留跨半年到一年趋势连续性")
    if ret_120d is not None and ret_60d is not None and ret_120d >= 50 and ret_60d >= 15:
        points += 6
        bits.append("当前半年趋势和季度趋势同向")
    if industry_alpha_120d is not None and industry_alpha_120d >= 0:
        points += 5
        bits.append(f"行业120日相对强度{_round_or_none(industry_alpha_120d, 2)}%")
    if today_amount is not None and today_amount >= 500000000:
        points += 5
        bits.append("成交额足以承载中长期跟踪")

    if close_to_120d_high is not None and close_to_120d_high < 0.78:
        caution_bits.append(f"已较120日高点回撤到{_round_or_none(close_to_120d_high * 100, 2)}%")
    if close_to_240d_high is not None and close_to_240d_high < 0.78:
        caution_bits.append(f"已较240日高点回撤到{_round_or_none(close_to_240d_high * 100, 2)}%")

    score = _round_or_none(max(0.0, min(points, 100.0)), 2)
    if score is None:
        score = 0.0

    if not bits:
        note = "历史长跑证据不足"
    else:
        note = "、".join(bits[:5])
        if caution_bits:
            note = f"{note}；当前提醒：{'、'.join(caution_bits[:2])}"

    return {
        "runway_historical_score": score,
        "runway_historical_note": note,
    }


def _score_long_runway_analyst_view(record, market_env):
    stage_label = record.get("runway_stage_label") or "震荡整理"
    adaptive_score = _to_float(record.get("adaptive_score")) or 0.0
    historical_evidence = _score_historical_runway_evidence(record)
    historical_score = _to_float(historical_evidence.get("runway_historical_score")) or 0.0
    price_vs_ma120 = _to_float(record.get("price_vs_ma120"))
    ma60_vs_ma120 = _to_float(record.get("ma60_vs_ma120"))
    ma120_vs_ma240 = _to_float(record.get("ma120_vs_ma240"))
    close_to_120d_high = _to_float(record.get("close_to_120d_high"))
    range_position_120d = _to_float(record.get("range_position_120d"))
    ret_20d = _to_float(record.get("ret_20d"))
    ret_60d = _to_float(record.get("ret_60d"))
    volume_vs_avg_20d = _to_float(record.get("volume_vs_avg_20d"))
    amount_vs_avg_10d = _to_float(record.get("amount_vs_avg_10d"))
    turnover_vs_avg_5d = _to_float(record.get("turnover_vs_avg_5d"))
    industry_alpha_60d = _to_float(record.get("industry_alpha_60d"))
    industry_alpha_120d = _to_float(record.get("industry_alpha_120d"))
    close_strength = _to_float(record.get("close_strength"))
    upper_shadow_ratio = _to_float(record.get("upper_shadow_ratio"))
    body_to_range_ratio = _to_float(record.get("body_to_range_ratio"))
    volatility_ratio_10_20 = _to_float(record.get("volatility_ratio_10_20"))
    ma20_slope_5d = _to_float(record.get("ma20_slope_5d"))
    ma60_slope_10d = _to_float(record.get("ma60_slope_10d"))
    risk_overlay_score = _to_float(record.get("risk_overlay_score")) or 0.0
    risk_overlay_block = bool(record.get("risk_overlay_block_formal"))
    risk_overlay_downgrade = bool(record.get("risk_overlay_downgrade"))
    risk_overlay_labels = str(record.get("risk_overlay_labels") or "").strip()

    quality_points = 0.0
    quality_max = 0.0
    risk_points = 0.0
    risk_max = 0.0
    quality_bits = []
    risk_bits = []

    def add_quality(condition, points, note):
        nonlocal quality_points, quality_max
        quality_max += points
        if condition:
            quality_points += points
            quality_bits.append(note)

    def add_risk(condition, points, note):
        nonlocal risk_points, risk_max
        risk_max += points
        if condition:
            risk_points += points
            risk_bits.append(note)

    add_quality(stage_label == "突破起涨", 14, "处于突破起涨")
    add_quality(stage_label == "主升延续", 12, "处于主升延续")
    add_quality(stage_label == "底部蓄势", 8, "处于底部蓄势")
    add_quality(price_vs_ma120 is not None and price_vs_ma120 > 0, 10, "站上120日均线")
    add_quality(ma60_vs_ma120 is not None and ma60_vs_ma120 > 0, 8, "60日均线强于120日均线")
    add_quality(ma120_vs_ma240 is not None and ma120_vs_ma240 >= 0, 6, "120日均线不弱于240日均线")
    add_quality(close_to_120d_high is not None and close_to_120d_high >= 0.92, 8, "接近120日高点")
    add_quality(range_position_120d is not None and range_position_120d >= 0.65, 6, "位于120日区间上沿")
    add_quality(volume_vs_avg_20d is not None and volume_vs_avg_20d >= 1.0, 6, "成交量高于20日均量")
    add_quality(amount_vs_avg_10d is not None and amount_vs_avg_10d >= 0.95, 6, "成交额不弱于10日均额")
    add_quality(turnover_vs_avg_5d is not None and turnover_vs_avg_5d >= 0.95, 5, "换手活跃度维持")
    add_quality(ma20_slope_5d is not None and ma20_slope_5d > 0, 5, "20日均线保持上斜")
    add_quality(ma60_slope_10d is not None and ma60_slope_10d > 0, 5, "60日均线保持上斜")
    add_quality(industry_alpha_60d is not None and industry_alpha_60d >= 0, 7, "行业60日相对强势")
    add_quality(industry_alpha_120d is not None and industry_alpha_120d >= 0, 8, "行业120日相对强势")
    add_quality(close_strength is not None and close_strength >= 0.6, 4, "收盘强度较好")
    add_quality(upper_shadow_ratio is not None and upper_shadow_ratio <= 0.35, 3, "上影压力可控")
    add_quality(body_to_range_ratio is not None and body_to_range_ratio >= 0.35, 3, "实体力度尚可")
    add_quality(volatility_ratio_10_20 is not None and volatility_ratio_10_20 <= 1.15, 4, "短波动未明显失控")

    add_risk(stage_label == "趋势破坏", 35, "长趋势已破坏")
    add_risk(stage_label == "高位回撤", 14, "处于高位回撤")
    add_risk(price_vs_ma120 is not None and price_vs_ma120 > 0.35, 10, "偏离120日均线过大")
    add_risk(price_vs_ma120 is not None and price_vs_ma120 > 0.5, 8, "远离长周期支撑")
    add_risk(ret_20d is not None and ret_20d > 25, 10, "20日涨幅过热")
    add_risk(ret_60d is not None and ret_60d > 80, 8, "60日涨幅偏热")
    add_risk(volume_vs_avg_20d is not None and volume_vs_avg_20d < 0.8, 5, "量能跟随不足")
    add_risk(industry_alpha_60d is not None and industry_alpha_60d < 0, 8, "行业60日转弱")
    add_risk(industry_alpha_120d is not None and industry_alpha_120d < 0, 8, "行业120日转弱")
    add_risk(close_strength is not None and close_strength < 0.45, 6, "收盘不够强")
    add_risk(upper_shadow_ratio is not None and upper_shadow_ratio > 0.45, 6, "上影偏重")
    add_risk(volatility_ratio_10_20 is not None and volatility_ratio_10_20 > 1.25, 6, "短波动明显放大")
    add_risk(ma20_slope_5d is not None and ma20_slope_5d < 0, 5, "20日均线转弱")
    add_risk(ma60_slope_10d is not None and ma60_slope_10d < 0, 6, "60日均线转弱")
    add_risk(risk_overlay_block, 28, f"风险覆盖硬拦截:{risk_overlay_labels or '特殊池/事件风险'}")
    add_risk(risk_overlay_downgrade, 14, f"风险覆盖降级:{risk_overlay_labels or '特殊池/事件风险'}")
    add_risk(risk_overlay_score >= 8, 10, f"风险覆盖分偏高:{_round_or_none(risk_overlay_score, 2)}")

    quality_score = _round_or_none(quality_points / quality_max * 100, 2) if quality_max > 0 else None
    risk_score = _round_or_none(risk_points / risk_max * 100, 2) if risk_max > 0 else None

    stage_adjustment = 0.0
    if stage_label == "突破起涨":
        stage_adjustment = 6.0
    elif stage_label == "主升延续":
        stage_adjustment = 4.0
    elif stage_label == "底部蓄势":
        stage_adjustment = 2.0
    elif stage_label == "高位回撤":
        stage_adjustment = -4.0
    elif stage_label == "趋势破坏":
        stage_adjustment = -18.0

    final_score = adaptive_score * 0.6
    if quality_score is not None:
        final_score += quality_score * 0.4
    if risk_score is not None:
        final_score -= risk_score * 0.25
    final_score += stage_adjustment

    if market_env == "偏弱":
        final_score -= 5.0
    elif market_env == "强势" and stage_label in ("突破起涨", "主升延续"):
        final_score += 2.0

    final_score = _round_or_none(max(0.0, min(100.0, final_score)), 2)

    conviction = "观察"
    if final_score is not None:
        if final_score >= 82:
            conviction = "高信念长跑"
        elif final_score >= 72:
            conviction = "重点跟踪"
        elif final_score >= 62:
            conviction = "候选观察"

    runway_eligible = (
        final_score is not None
        and final_score >= (72 if market_env == "偏弱" else 62)
        and stage_label != "趋势破坏"
        and not risk_overlay_block
    )
    if market_env == "偏弱" and stage_label == "高位回撤":
        runway_eligible = False

    overlay_note = []
    if quality_bits:
        overlay_note.append(f"质量加分：{'、'.join(quality_bits[:4])}")
    if risk_bits:
        overlay_note.append(f"风险提醒：{'、'.join(risk_bits[:3])}")
    overlay_summary = "；".join(overlay_note) if overlay_note else "长跑质量与风险信号一般"

    return {
        "runway_quality_score": quality_score,
        "runway_risk_score": risk_score,
        "runway_historical_score": historical_evidence.get("runway_historical_score"),
        "runway_historical_note": historical_evidence.get("runway_historical_note"),
        "runway_total_score": final_score,
        "runway_conviction": conviction,
        "runway_eligible": runway_eligible,
        "runway_overlay_note": overlay_summary,
    }


def _enrich_long_runway_plans(candidate_df, market_env):
    if candidate_df.empty:
        return candidate_df

    enriched_rows = []
    for row in candidate_df.to_dict("records"):
        stage_profile = _build_long_runway_stage(row)
        existing_reason = row.get("reason")
        stage_reason = f"{stage_profile.get('stage_label')}，{stage_profile.get('stage_detail')}"
        if existing_reason:
            row["reason"] = f"{stage_reason}；画像对齐：{existing_reason}"
        else:
            row["reason"] = stage_reason

        row.update(
            {
                "runway_stage_name": stage_profile.get("stage_name"),
                "runway_stage_label": stage_profile.get("stage_label"),
                "runway_stage_detail": stage_profile.get("stage_detail"),
                "runway_stage_scores": stage_profile.get("stage_scores"),
            }
        )
        overlay = _score_long_runway_analyst_view(row, market_env)
        row.update(overlay)
        row.update(
            _build_long_runway_plan(
                row,
                market_env,
                stage_label=stage_profile.get("stage_label"),
                runway_score=overlay.get("runway_total_score"),
            )
        )
        row.update(
            {
                "runway_action": row.pop("action"),
                "runway_hold_period": row.pop("hold_period"),
                "runway_expected_return": row.pop("expected_return"),
                "runway_stop_price": row.pop("stop_price"),
                "runway_stop_rule": row.pop("stop_rule"),
                "runway_reenter_rule": row.pop("reenter_rule"),
                "runway_position_hint": row.pop("position_hint"),
                "runway_conclusion": row.pop("conclusion"),
            }
        )
        history_note = overlay.get("runway_historical_note")
        note_parts = [row["reason"]]
        if history_note:
            note_parts.append(f"历史证据：{history_note}")
        note_parts.append(overlay["runway_overlay_note"])
        row["reason"] = "；".join(note_parts)
        enriched_rows.append(row)

    return pd.DataFrame(enriched_rows)


def _finalize_long_runway_candidates(candidate_df, market_env, top_candidate_count):
    if candidate_df.empty:
        return candidate_df, candidate_df

    enriched_df = _enrich_long_runway_plans(candidate_df, market_env)
    if enriched_df.empty:
        return enriched_df, enriched_df

    sort_columns = [
        "runway_total_score",
        "adaptive_score",
        "runway_quality_score",
        "score_252d",
        "score_120d",
        "score_60d",
    ]
    ranked_df = enriched_df[enriched_df["runway_eligible"].fillna(False)].copy()
    minimum_kept = max(3, min(int(top_candidate_count), 5))
    if len(ranked_df) < minimum_kept:
        ranked_df = enriched_df[enriched_df["runway_stage_label"] != "趋势破坏"].copy()
        if ranked_df.empty:
            ranked_df = enriched_df.copy()

    ranked_df = ranked_df.sort_values(sort_columns, ascending=[False] * len(sort_columns), na_position="last").reset_index(drop=True)
    top_df = ranked_df.head(int(top_candidate_count)).copy()
    return enriched_df, top_df


def _clear_long_runway_strategy_rows(trade_date):
    trade_date = _to_date_text(trade_date)
    if not trade_date:
        return
    func.executeDelete("a_stock_strategy_result", {"trade_date": trade_date, "strategy_type": LONG_RUNWAY_STRATEGY_TYPE})


def clear_long_runway_strategy_results(trade_date):
    _clear_long_runway_strategy_rows(trade_date)


def _json_compact(value):
    if value is None:
        return None
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def _build_long_runway_strategy_note(candidate, summary, rank, total):
    stage_scores = _json_compact(candidate.get("runway_stage_scores"))
    parts = [
        "层级:中长期跟踪",
        f"模型:{LONG_RUNWAY_MODEL_DISPLAY}({LONG_RUNWAY_MODEL_VERSION})",
        "性质:历史长期赢家画像匹配，不等同每日短线正式推荐",
        f"排名:{rank}/{total}",
        f"市场:{summary.get('market_env') or '--'}",
        f"样本:{summary.get('sample_start') or '--'}~{summary.get('sample_end') or '--'}",
        (
            f"分数:总分{candidate.get('runway_total_score')}, "
            f"画像{candidate.get('adaptive_score')}, "
            f"质量{candidate.get('runway_quality_score')}, "
            f"风险{candidate.get('runway_risk_score')}"
        ),
        (
            f"周期:60日{candidate.get('score_60d')}, "
            f"120日{candidate.get('score_120d')}, "
            f"252日{candidate.get('score_252d')}, "
            f"主导{candidate.get('dominant_horizon')}日/{candidate.get('dominant_family')}"
        ),
        (
            f"阶段:{candidate.get('runway_stage_label') or '--'}, "
            f"信念:{candidate.get('runway_conviction') or '--'}, "
            f"动作:{candidate.get('runway_action') or '--'}, "
            f"仓位:{candidate.get('runway_position_hint') or '--'}"
        ),
        (
            f"风险覆盖:{candidate.get('risk_overlay_labels') or '无明显特殊池风险'}, "
            f"覆盖动作:{candidate.get('risk_overlay_action') or '--'}"
        ),
        (
            f"跟踪:{candidate.get('runway_hold_period') or '--'}, "
            f"预期:{candidate.get('runway_expected_return') or '--'}, "
            f"防守:{candidate.get('runway_stop_price') or '--'}"
        ),
        f"结论:{candidate.get('runway_conclusion') or '--'}",
        f"放弃/退出:{candidate.get('runway_stop_rule') or '--'}",
        f"重新跟踪:{candidate.get('runway_reenter_rule') or '--'}",
        f"原因:{candidate.get('reason') or '--'}",
    ]
    if stage_scores:
        parts.append(f"阶段分:{stage_scores}")

    strategy_note = "；".join(str(part) for part in parts if part is not None)
    if len(strategy_note) > ADAPTIVE_NOTE_MAX_LENGTH:
        strategy_note = strategy_note[: ADAPTIVE_NOTE_MAX_LENGTH - 3] + "..."
    return strategy_note


def _build_long_runway_strategy_record(candidate, summary, trade_date, rank, total):
    change_30d_value = _normalize_scalar(candidate.get("ret_20d"))
    if change_30d_value is None:
        change_30d_value = _normalize_scalar(candidate.get("change_30d"))

    return {
        "trade_date": trade_date,
        "strategy_type": LONG_RUNWAY_STRATEGY_TYPE,
        "stock_code": _normalize_scalar(candidate.get("stock_code")),
        "stock_name": _normalize_scalar(candidate.get("stock_name")),
        "today_change": _normalize_scalar(candidate.get("today_change")),
        "industry": _normalize_scalar(candidate.get("industry_1")) or "",
        "change_30d": change_30d_value,
        "vr_today": _normalize_scalar(candidate.get("vr_today")),
        "vr_30d": _normalize_scalar(candidate.get("vr_30d")),
        "today_amp": _normalize_scalar(candidate.get("today_amp")),
        "amp_30d": _normalize_scalar(candidate.get("amp_30d")),
        "stock_rank": _normalize_scalar(candidate.get("stock_rank")),
        "today_amount": _normalize_scalar(candidate.get("today_amount")),
        "turnover_rate": _normalize_scalar(candidate.get("turnover_rate")),
        "strategy_note": _build_long_runway_strategy_note(candidate, summary, rank, total),
    }


def persist_long_runway_candidates_to_strategy_result(summary, trade_date=None):
    summary = summary or {}
    effective_trade_date = _to_date_text(trade_date or summary.get("latest_trade_date"))
    _clear_long_runway_strategy_rows(effective_trade_date)

    if not summary.get("success"):
        _emit_runtime_status(
            f"{LONG_RUNWAY_MODEL_DISPLAY}中长期跟踪落库跳过: "
            f"trade_date={effective_trade_date}, reason={summary.get('reason') or 'long_runway_failed'}"
        )
        return {
            "success": False,
            "reason": summary.get("reason") or "long_runway_failed",
            "saved_count": 0,
            "trade_date": effective_trade_date,
            "strategy_type": LONG_RUNWAY_STRATEGY_TYPE,
        }

    if not effective_trade_date:
        return {
            "success": False,
            "reason": "latest_trade_date_missing",
            "saved_count": 0,
            "strategy_type": LONG_RUNWAY_STRATEGY_TYPE,
        }

    top_candidates = summary.get("top_candidates") or []
    if not top_candidates:
        _emit_runtime_status(
            f"{LONG_RUNWAY_MODEL_DISPLAY}中长期跟踪落库完成: "
            f"trade_date={effective_trade_date}, saved_count=0, reason=candidate_empty"
        )
        return {
            "success": True,
            "reason": "candidate_empty",
            "saved_count": 0,
            "trade_date": effective_trade_date,
            "strategy_type": LONG_RUNWAY_STRATEGY_TYPE,
        }

    selected_candidates = []
    seen_codes = set()
    for candidate in top_candidates:
        stock_code = _normalize_scalar(candidate.get("stock_code"))
        if not stock_code or stock_code in seen_codes:
            continue
        seen_codes.add(stock_code)
        selected_candidates.append(candidate)

    saved_count = 0
    total_count = len(selected_candidates)
    for index, candidate in enumerate(selected_candidates, start=1):
        insert_result = func.executeInsert(
            "a_stock_strategy_result",
            _build_long_runway_strategy_record(
                candidate,
                summary,
                effective_trade_date,
                rank=index,
                total=total_count,
            ),
        )
        if insert_result.get("insertResult"):
            saved_count += 1

    _emit_runtime_status(
        f"{LONG_RUNWAY_MODEL_DISPLAY}中长期跟踪写入完成: "
        f"trade_date={effective_trade_date}, saved_count={saved_count}, "
        f"strategy_type={LONG_RUNWAY_STRATEGY_TYPE}, 层级=中长期跟踪"
    )
    return {
        "success": True,
        "reason": None if saved_count else "insert_failed_or_empty",
        "saved_count": saved_count,
        "trade_date": effective_trade_date,
        "strategy_type": LONG_RUNWAY_STRATEGY_TYPE,
        "tier": "中长期跟踪",
    }


def analysis_gu_piao_history_long_runway_model(
    history=None,
    start_date=None,
    end_date=None,
    top_candidate_count=TOP_CANDIDATE_COUNT,
    stock_code=None,
    use_cache=True,
    rebuild_cache=False,
):
    func.logInfo(f"开始分析历史长跑潜力与阶段（{LONG_RUNWAY_MODEL_DISPLAY}）")
    print(f"开始分析历史长跑潜力与阶段（{LONG_RUNWAY_MODEL_DISPLAY}）")

    target_stock_code = _normalize_stock_code(stock_code)

    cache_info = {"cache_mode": "disabled", "cache_path": str(LONG_RUNWAY_CONTEXT_CACHE_PATH)}
    if history is None and start_date is None and use_cache:
        runway_history, cache_info = _load_or_update_long_runway_history_cache(
            end_date=end_date,
            rebuild_cache=rebuild_cache,
        )
    else:
        if history is None:
            history = _load_history(
                start_date=start_date,
                end_date=end_date,
                columns=LONG_RUNWAY_FRAME_COLUMNS,
                chunked=True,
                progress_label=f"{LONG_RUNWAY_MODEL_DISPLAY}历史读取",
            )
        if history.empty:
            func.logInfo("a_stock_analysis_history 没有可用历史数据")
            return {
                "model_version": MODEL_VERSION,
                "long_runway_model_version": LONG_RUNWAY_MODEL_VERSION,
                "success": False,
                "reason": "history_empty",
                "top_candidates": [],
            }
        runway_history = _prepare_long_runway_history(history)

    if runway_history is None or runway_history.empty:
        func.logInfo("a_stock_analysis_history 没有可用历史数据")
        return {
            "model_version": MODEL_VERSION,
            "long_runway_model_version": LONG_RUNWAY_MODEL_VERSION,
            "success": False,
            "reason": "history_empty",
            "top_candidates": [],
        }

    if end_date is not None:
        end_date_text = _to_date_text(end_date)
        runway_history = runway_history[runway_history["last_data_date"] <= pd.to_datetime(end_date_text)].copy()

    if runway_history.empty:
        func.logInfo("a_stock_analysis_history 没有可用历史数据")
        return {
            "model_version": MODEL_VERSION,
            "long_runway_model_version": LONG_RUNWAY_MODEL_VERSION,
            "success": False,
            "reason": "history_empty",
            "top_candidates": [],
        }

    latest_trade_date = runway_history["last_data_date"].max()
    latest_snapshot = runway_history[runway_history["last_data_date"] == latest_trade_date].copy()
    latest_snapshot = _attach_long_runway_historical_memory(runway_history, latest_snapshot, latest_trade_date)
    if latest_snapshot.empty:
        func.logInfo(f"{LONG_RUNWAY_MODEL_DISPLAY}没有可用于评分的最新快照")
        return {
            "model_version": MODEL_VERSION,
            "long_runway_model_version": LONG_RUNWAY_MODEL_VERSION,
            "success": False,
            "reason": "snapshot_empty",
            "top_candidates": [],
        }

    cached_profiles = cache_info.get("horizon_profiles") if cache_info else None
    long_horizon_profiles = cached_profiles or {}
    if long_horizon_profiles:
        _emit_runtime_status(
            f"{LONG_RUNWAY_MODEL_DISPLAY}缓存: profile命中, trade_date={_to_date_text(latest_trade_date)}"
        )
    else:
        long_horizon_profiles = {}
        for horizon_days in LONG_RUNWAY_HORIZONS:
            profile = _build_horizon_profile(
                runway_history,
                horizon_days,
                family_features=LONG_RUNWAY_FAMILY_FEATURES,
                feature_labels=LONG_RUNWAY_FEATURE_LABELS,
                family_labels=LONG_RUNWAY_FAMILY_LABELS,
                half_life_days=LONG_RUNWAY_HALF_LIFE_DAYS,
                winner_ratio=LONG_RUNWAY_WINNER_RATIO,
                loser_ratio=LONG_RUNWAY_LOSER_RATIO,
                min_daily_rows=LONG_RUNWAY_MIN_DAILY_ROWS,
            )
            long_horizon_profiles[horizon_days] = profile
            func.logInfo(
                f"{LONG_RUNWAY_MODEL_DISPLAY}训练完成 horizon={horizon_days}d, sample_rows={profile['sample_rows']}, "
                f"sample_days={profile['sample_days']}, winner_rows={profile['winner_rows']}, "
                f"loser_rows={profile['loser_rows']}, positive_rate={profile['positive_rate']}%"
            )
        if cache_info.get("cache_mode") in {"hit", "incremental", "full_rebuild"}:
            _save_long_runway_cache(runway_history, horizon_profiles=long_horizon_profiles)
            _emit_runtime_status(
                f"{LONG_RUNWAY_MODEL_DISPLAY}缓存: profile已更新, trade_date={_to_date_text(latest_trade_date)}"
            )

    candidate_df = _score_candidates(latest_snapshot, long_horizon_profiles, None, apply_precision_filter=False)
    if candidate_df.empty:
        func.logInfo(f"{LONG_RUNWAY_MODEL_DISPLAY}候选评分为空")
        return {
            "model_version": MODEL_VERSION,
            "long_runway_model_version": LONG_RUNWAY_MODEL_VERSION,
            "success": False,
            "reason": "candidate_empty",
            "top_candidates": [],
        }
    latest_trade_date_text = None
    if pd.notna(latest_trade_date):
        latest_trade_date_text = pd.to_datetime(latest_trade_date).strftime("%Y-%m-%d")

    candidate_df = risk_overlay.apply_risk_overlay_to_candidates(
        candidate_df,
        history=runway_history,
        trade_date=latest_trade_date_text,
        include_external=False,
        filter_blocked=False,
        filter_downgraded=False,
    )
    if not candidate_df.empty:
        candidate_df["risk_labels"] = candidate_df.apply(
            lambda row: _merge_risk_label_text(row.get("risk_labels"), row.get("risk_overlay_labels")),
            axis=1,
        )

    top_families, top_features = _summarize_families(long_horizon_profiles)
    long_runway_note = _compose_model_note({"top_families": top_families, "top_features": top_features, "top_styles": []})
    market_ctx = _resolve_long_runway_market_env(latest_snapshot)
    market_env = market_ctx["market_env"]

    candidate_df, top_candidates_df = _finalize_long_runway_candidates(
        candidate_df,
        market_env,
        top_candidate_count=top_candidate_count,
    )
    focus_candidate = None
    if target_stock_code:
        focus_df = candidate_df[candidate_df["stock_code"] == target_stock_code].head(1)
        if not focus_df.empty:
            focus_candidate = focus_df.iloc[0].to_dict()

    stage_summary = {}
    if "runway_stage_label" in top_candidates_df.columns and not top_candidates_df.empty:
        stage_summary = dict(top_candidates_df["runway_stage_label"].fillna("未知").value_counts())

    summary = {
        "model_version": MODEL_VERSION,
        "long_runway_model_version": LONG_RUNWAY_MODEL_VERSION,
        "success": True,
        "sample_start": str(runway_history["last_data_date"].min().date()),
        "sample_end": str(runway_history["last_data_date"].max().date()),
        "trade_days": int(runway_history["last_data_date"].nunique()),
        "history_rows": int(len(runway_history)),
        "cache_mode": cache_info.get("cache_mode"),
        "cache_path": cache_info.get("cache_path"),
        "latest_trade_date": latest_trade_date_text,
        "requested_stock_code": target_stock_code,
        "market_env": market_env,
        "market_ret_60d": market_ctx["market_ret_60d"],
        "market_breadth_60d": market_ctx["market_breadth_60d"],
        "market_ret_120d": market_ctx["market_ret_120d"],
        "market_breadth_120d": market_ctx["market_breadth_120d"],
        "recency_half_life_days": LONG_RUNWAY_HALF_LIFE_DAYS,
        "winner_ratio": LONG_RUNWAY_WINNER_RATIO,
        "loser_ratio": LONG_RUNWAY_LOSER_RATIO,
        "model_definition": f"{LONG_RUNWAY_MODEL_DISPLAY}：基于历史大涨股学习长周期趋势、突破基座、量能、流动性、趋势质量与行业共振，并给出当前阶段判断。",
        "top_families": top_families,
        "top_features": top_features,
        "long_runway_note": long_runway_note,
        "stage_summary": stage_summary,
        "top_industries": _summarize_industries(candidate_df),
        "top_candidates": top_candidates_df.to_dict("records"),
        "focus_candidate": focus_candidate,
        "horizon_profiles": {
            f"{horizon_days}d": {
                "sample_rows": profile["sample_rows"],
                "sample_days": profile["sample_days"],
                "winner_rows": profile["winner_rows"],
                "loser_rows": profile["loser_rows"],
                "positive_rate": profile["positive_rate"],
                "top_features": profile["top_features"],
                "families": {
                    family_name: {
                        "label": LONG_RUNWAY_FAMILY_LABELS.get(family_name, family_name),
                        "importance": family_data["importance"],
                        "normalized_importance": family_data["normalized_importance"],
                        "top_features": family_data["items"][:3],
                    }
                    for family_name, family_data in profile["families"].items()
                },
            }
            for horizon_days, profile in long_horizon_profiles.items()
        },
    }

    func.logInfo(long_runway_note)
    func.logInfo(
        f"{LONG_RUNWAY_MODEL_DISPLAY}分析完成: trade_days={summary['trade_days']}, history_rows={summary['history_rows']}, "
        f"market_env={market_env}, top_candidates={len(summary['top_candidates'])}"
    )
    func.logInfo({
        "top_families": top_families[:3],
        "top_features": top_features[:5],
        "stage_summary": stage_summary,
    })
    print(f"{LONG_RUNWAY_MODEL_DISPLAY}分析完毕")
    return summary


def _backtest_long_runway_profiles(runway_history, top_candidate_count=TOP_CANDIDATE_COUNT):
    trade_dates = sorted(runway_history["last_data_date"].dropna().unique())
    eval_trade_dates = trade_dates[:: max(1, int(LONG_RUNWAY_REBALANCE_TRADE_DAYS))]
    metrics = {}

    for horizon_days in LONG_RUNWAY_HORIZONS:
        metrics[horizon_days] = {
            "evaluated_days": 0,
            "universe_return_sum": 0.0,
            "universe_win_sum": 0.0,
            "top_return_sum": 0.0,
            "top_win_sum": 0.0,
            "top_stage_counts": {},
            "top_conviction_counts": {},
            "stage_return_stats": {},
            "conviction_return_stats": {},
        }

    for eval_date in eval_trade_dates:
        snapshot = runway_history[runway_history["last_data_date"] == eval_date].copy()
        snapshot = _attach_long_runway_historical_memory(runway_history, snapshot, eval_date)
        if snapshot.empty or len(snapshot) < LONG_RUNWAY_MIN_DAILY_ROWS:
            continue

        eval_profiles = {}
        for horizon_days in LONG_RUNWAY_HORIZONS:
            horizon_mask = (
                runway_history[f"forward_trade_date_{horizon_days}d"].notna()
                & (runway_history[f"forward_trade_date_{horizon_days}d"] <= eval_date)
            )
            horizon_frame = runway_history[horizon_mask].copy()
            eval_profiles[horizon_days] = _build_horizon_profile(
                horizon_frame,
                horizon_days,
                family_features=LONG_RUNWAY_FAMILY_FEATURES,
                feature_labels=LONG_RUNWAY_FEATURE_LABELS,
                family_labels=LONG_RUNWAY_FAMILY_LABELS,
                half_life_days=LONG_RUNWAY_HALF_LIFE_DAYS,
                winner_ratio=LONG_RUNWAY_WINNER_RATIO,
                loser_ratio=LONG_RUNWAY_LOSER_RATIO,
                min_daily_rows=LONG_RUNWAY_MIN_DAILY_ROWS,
            )

        scored = _score_candidates(snapshot, eval_profiles, None, apply_precision_filter=False)
        if scored.empty:
            continue

        market_ctx = _resolve_long_runway_market_env(snapshot)
        scored = risk_overlay.apply_risk_overlay_to_candidates(
            scored,
            history=runway_history,
            trade_date=_to_date_text(eval_date),
            include_external=False,
            filter_blocked=False,
            filter_downgraded=False,
        )
        if not scored.empty:
            scored["risk_labels"] = scored.apply(
                lambda row: _merge_risk_label_text(row.get("risk_labels"), row.get("risk_overlay_labels")),
                axis=1,
            )
        _, top_candidates = _finalize_long_runway_candidates(
            scored,
            market_ctx["market_env"],
            top_candidate_count=top_candidate_count,
        )
        if top_candidates.empty:
            continue

        for horizon_days in LONG_RUNWAY_HORIZONS:
            future_col = f"forward_return_{horizon_days}d"
            if future_col not in snapshot.columns:
                continue

            universe_returns = pd.to_numeric(snapshot[future_col], errors="coerce").dropna()
            top_subset = top_candidates[
                ["stock_code", "runway_stage_label", "runway_conviction", "runway_total_score"]
            ].merge(
                snapshot[["stock_code", future_col]],
                on="stock_code",
                how="left",
            )
            top_returns = pd.to_numeric(top_subset[future_col], errors="coerce").dropna()

            metrics[horizon_days]["evaluated_days"] += 1
            metrics[horizon_days]["universe_return_sum"] += float(universe_returns.mean()) if not universe_returns.empty else 0.0
            metrics[horizon_days]["universe_win_sum"] += float((universe_returns > 0).mean()) if not universe_returns.empty else 0.0
            metrics[horizon_days]["top_return_sum"] += float(top_returns.mean()) if not top_returns.empty else 0.0
            metrics[horizon_days]["top_win_sum"] += float((top_returns > 0).mean()) if not top_returns.empty else 0.0

            for stage_label, count in top_candidates["runway_stage_label"].fillna("未知").value_counts().items():
                metrics[horizon_days]["top_stage_counts"][stage_label] = (
                    metrics[horizon_days]["top_stage_counts"].get(stage_label, 0) + int(count)
                )

            for conviction, count in top_candidates["runway_conviction"].fillna("观察").value_counts().items():
                metrics[horizon_days]["top_conviction_counts"][conviction] = (
                    metrics[horizon_days]["top_conviction_counts"].get(conviction, 0) + int(count)
                )

            for stage_label, group in top_subset.groupby(top_subset["runway_stage_label"].fillna("未知")):
                returns = pd.to_numeric(group[future_col], errors="coerce").dropna()
                if returns.empty:
                    continue
                stage_stat = metrics[horizon_days]["stage_return_stats"].setdefault(
                    stage_label,
                    {"count": 0, "return_sum": 0.0, "win_sum": 0.0},
                )
                stage_stat["count"] += int(len(returns))
                stage_stat["return_sum"] += float(returns.sum())
                stage_stat["win_sum"] += int((returns > 0).sum())

            for conviction, group in top_subset.groupby(top_subset["runway_conviction"].fillna("观察")):
                returns = pd.to_numeric(group[future_col], errors="coerce").dropna()
                if returns.empty:
                    continue
                conviction_stat = metrics[horizon_days]["conviction_return_stats"].setdefault(
                    conviction,
                    {"count": 0, "return_sum": 0.0, "win_sum": 0.0},
                )
                conviction_stat["count"] += int(len(returns))
                conviction_stat["return_sum"] += float(returns.sum())
                conviction_stat["win_sum"] += int((returns > 0).sum())

    result = {}
    for horizon_days, data in metrics.items():
        days = max(data["evaluated_days"], 1)
        result[horizon_days] = {
            "evaluated_days": data["evaluated_days"],
            "avg_universe_return": _round_or_none(data["universe_return_sum"] / days, 4) if data["evaluated_days"] else None,
            "avg_universe_win_rate": _round_or_none(data["universe_win_sum"] / days * 100, 2) if data["evaluated_days"] else None,
            "avg_top_return": _round_or_none(data["top_return_sum"] / days, 4) if data["evaluated_days"] else None,
            "avg_top_win_rate": _round_or_none(data["top_win_sum"] / days * 100, 2) if data["evaluated_days"] else None,
            "top_stage_counts": dict(sorted(data["top_stage_counts"].items(), key=lambda item: item[1], reverse=True)),
            "top_conviction_counts": dict(sorted(data["top_conviction_counts"].items(), key=lambda item: item[1], reverse=True)),
            "stage_return_stats": {
                stage_label: {
                    "count": stage_stat["count"],
                    "avg_return": _round_or_none(stage_stat["return_sum"] / stage_stat["count"], 4) if stage_stat["count"] else None,
                    "win_rate": _round_or_none(stage_stat["win_sum"] / stage_stat["count"] * 100, 2) if stage_stat["count"] else None,
                }
                for stage_label, stage_stat in sorted(
                    data["stage_return_stats"].items(),
                    key=lambda item: item[1]["return_sum"] / item[1]["count"] if item[1]["count"] else -999,
                    reverse=True,
                )
            },
            "conviction_return_stats": {
                conviction: {
                    "count": conviction_stat["count"],
                    "avg_return": _round_or_none(conviction_stat["return_sum"] / conviction_stat["count"], 4) if conviction_stat["count"] else None,
                    "win_rate": _round_or_none(conviction_stat["win_sum"] / conviction_stat["count"] * 100, 2) if conviction_stat["count"] else None,
                }
                for conviction, conviction_stat in sorted(
                    data["conviction_return_stats"].items(),
                    key=lambda item: item[1]["return_sum"] / item[1]["count"] if item[1]["count"] else -999,
                    reverse=True,
                )
            },
        }

    return result


def backtest_analysis_gu_piao_history_long_runway_model(
    start_date=None,
    end_date=None,
    top_candidate_count=TOP_CANDIDATE_COUNT,
    history=None,
):
    func.logInfo(f"开始回测{LONG_RUNWAY_MODEL_DISPLAY}（walk-forward）")
    print(f"开始回测{LONG_RUNWAY_MODEL_DISPLAY}（walk-forward）")

    if history is None:
        history = _load_history(
            start_date=start_date,
            end_date=end_date,
            columns=LONG_RUNWAY_FRAME_COLUMNS,
            chunked=True,
            progress_label=f"{LONG_RUNWAY_MODEL_DISPLAY}历史读取",
        )
    if history.empty:
        func.logInfo("a_stock_analysis_history 没有可用历史数据")
        return {
            "model_version": MODEL_VERSION,
            "long_runway_model_version": LONG_RUNWAY_MODEL_VERSION,
            "success": False,
            "reason": "history_empty",
            "backtest": {},
        }

    runway_history = _prepare_long_runway_history(history)

    trade_dates = runway_history["last_data_date"].dropna().nunique()
    backtest_result = _backtest_long_runway_profiles(runway_history, top_candidate_count=top_candidate_count)

    summary = {
        "model_version": MODEL_VERSION,
        "long_runway_model_version": LONG_RUNWAY_MODEL_VERSION,
        "success": True,
        "trade_days": int(trade_dates),
        "rebalance_trade_days": int(LONG_RUNWAY_REBALANCE_TRADE_DAYS),
        "backtest": {
            f"{horizon_days}d": metrics
            for horizon_days, metrics in backtest_result.items()
        },
    }

    func.logInfo({"long_runway_backtest": summary["backtest"]})
    print(f"{LONG_RUNWAY_MODEL_DISPLAY}回测完毕")
    return summary


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
