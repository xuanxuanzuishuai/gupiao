"""A股股票代码、名称和行业基础资料抓取。

作用:
    从 akshare 多个数据源抓取 A 股代码、股票名称、行业/板块信息，并
    做字段兼容、去重和兜底合并，为后续行情抓取和策略分析提供基础
    股票池。

流程:
    先按候选数据源依次请求代码名称和行业板块数据；
    再标准化代码、名称和行业字段，并合并多源结果；
    然后过滤异常记录、补齐缺失信息；
    最后把基础股票池写入本地数据库，供 get_gu_piao_info 使用。
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed

import akshare as ak
import pandas as pd

import func


CODE_COLUMN_CANDIDATES = ("代码", "证券代码", "股票代码", "stock_code", "symbol", "code")
NAME_COLUMN_CANDIDATES = ("名称", "股票简称", "简称", "证券简称", "stock_name", "name")
INDUSTRY_COLUMN_CANDIDATES = ("行业", "所属行业", "industry")
BOARD_NAME_COLUMN_CANDIDATES = ("板块名称", "行业名称", "行业", "name", "板块")
BOARD_ID_COLUMN_CANDIDATES = ("板块代码", "行业代码", "代码", "symbol", "code")

CODE_SOURCE_SPECS = [
    {
        "name": "code_name",
        "func_candidates": [("stock_info_a_code_name", {})],
    },
    {
        "name": "sina_spot",
        "func_candidates": [("stock_zh_a_spot_sina", {}), ("stock_zh_a_spot", {})],
    },
    {
        "name": "tencent_spot",
        "func_candidates": [("stock_zh_a_spot_tx", {}), ("stock_zh_a_spot_qq", {})],
    },
    {
        "name": "eastmoney_spot",
        "func_candidates": [("stock_zh_a_spot_em", {})],
    },
]

INDUSTRY_SOURCE_SPECS = [
    {
        "name": "eastmoney_industry",
        "board_func_candidates": [
            ("stock_board_industry_name_em", {}),
        ],
        "cons_func_candidates": [
            ("stock_board_industry_cons_em", "symbol"),
        ],
    },
    {
        "name": "sw_industry",
        # 申万行业分类
        "board_func_candidates": [
            ("sw_index_spot", {}),
        ],
        "cons_func_candidates": [
            ("sw_index_cons", "index_code"),
        ],
    },
    {
        "name": "csrc_industry",
        # 证监会行业分类
        "board_func_candidates": [
            ("stock_industry_category_cninfo", {}),
        ],
        # 没有单独成份股接口，但可以用分类表筛选
        "cons_func_candidates": [],
    },
    {
        "name": "ths_industry_flow",
        # 同花顺行业资金流（板块并非标准行业归属，但可统计）
        "board_func_candidates": [
            ("stock_fund_flow_industry", {"symbol": "即时"}),
        ],
        "cons_func_candidates": [],
    },
]

MAX_SOURCE_WORKERS = 4
MAX_BOARD_WORKERS = 1
PROGRESS_LOG_INTERVAL_SECONDS = 5
UPSERT_PROGRESS_INTERVAL = 200
MAX_ERROR_SAMPLES_PER_SOURCE = 10
CODE_SOURCE_STAGE_TIMEOUT_SECONDS = 90
INDUSTRY_SOURCE_STAGE_TIMEOUT_SECONDS = 6000
BOARD_FETCH_STAGE_TIMEOUT_SECONDS = 6000
RUNTIME_UNAVAILABLE_PROVIDERS = {}


def _normalize_stock_code(value):
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    matched = re.search(r"(\d{6})", text)
    return matched.group(1) if matched else None


def _choose_column(columns, candidates):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _ordered_unique(values):
    unique_values = []
    seen = set()

    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        unique_values.append(item)

    return unique_values


def _split_industry(value):
    if value is None or pd.isna(value):
        return []
    return _ordered_unique(str(value).split(","))


def _merge_industry(old_value, new_value):
    return ",".join(_ordered_unique(_split_industry(old_value) + _split_industry(new_value)))


def _call_ak_candidates(func_candidates, source_name, stage_name):
    errors = []

    for func_name, kwargs in func_candidates:
        fetcher = getattr(ak, func_name, None)
        if fetcher is None:
            errors.append(f"{func_name}:not_found")
            continue

        try:
            raw_df = fetcher(**kwargs)
            if not isinstance(raw_df, pd.DataFrame):
                errors.append(f"{func_name}:not_dataframe")
                continue
            return raw_df, func_name, errors
        except Exception as error:
            errors.append(f"{func_name}:{error.__class__.__name__}: {error}")

    raise RuntimeError(f"{source_name} {stage_name} 所有候选接口失败: {errors}")


def _provider_name(source_name):
    return str(source_name).split("_", 1)[0].strip().lower()


def _is_network_unstable_error(error_text):
    normalized = str(error_text).lower()
    keywords = (
        "remote end closed connection without response",
        "connection aborted",
        "remotedisconnected",
        "read timed out",
        "connect timeout",
        "connectionerror",
    )
    return any(keyword in normalized for keyword in keywords)


def _mark_provider_unavailable(source_name, error_text):
    provider = _provider_name(source_name)
    if not provider:
        return
    if not _is_network_unstable_error(error_text):
        return
    RUNTIME_UNAVAILABLE_PROVIDERS[provider] = str(error_text)
    func.logInfo(f"运行期标记数据源不可用: provider={provider}, reason={error_text}")


def _has_available_func(func_candidates):
    for func_name, _ in func_candidates:
        fetcher = getattr(ak, func_name, None)
        if callable(fetcher):
            return True
    return False


def _get_active_code_source_specs():
    active_specs = []
    skipped_sources = []

    for spec in CODE_SOURCE_SPECS:
        provider = _provider_name(spec["name"])
        if provider in RUNTIME_UNAVAILABLE_PROVIDERS:
            skipped_sources.append(
                f"{spec['name']}(runtime_disabled:{RUNTIME_UNAVAILABLE_PROVIDERS[provider]})"
            )
            continue
        if _has_available_func(spec["func_candidates"]):
            active_specs.append(spec)
        else:
            skipped_sources.append(spec["name"])

    if skipped_sources:
        func.logInfo(f"以下股票列表数据源在当前 akshare 版本不可用，自动跳过: {skipped_sources}")

    return active_specs


def _get_active_industry_source_specs():
    active_specs = []
    skipped_sources = []

    for spec in INDUSTRY_SOURCE_SPECS:
        available_board = _has_available_func(spec["board_func_candidates"])

        # 判断是否至少有一个可用的成分接口
        available_cons = False
        for func_name, _ in spec.get("cons_func_candidates", []):
            if callable(getattr(ak, func_name, None)):
                available_cons = True
                break

        # 如果没有成分股接口也允许，只要有板块接口就认为可用
        if available_board:
            active_specs.append(spec)
        else:
            skipped_sources.append(spec["name"])

    if skipped_sources:
        func.logInfo(
            f"以下行业分类源在当前 akshare 版本不可用，自动跳过: {skipped_sources}"
        )

    return active_specs


def _prepare_stock_universe_df(raw_df, source_name):
    code_col = _choose_column(raw_df.columns, CODE_COLUMN_CANDIDATES)
    if code_col is None:
        raise ValueError(f"{source_name} 缺少代码字段，当前字段: {list(raw_df.columns)}")

    name_col = _choose_column(raw_df.columns, NAME_COLUMN_CANDIDATES)

    stock_df = pd.DataFrame()
    stock_df["stock_code"] = raw_df[code_col].apply(_normalize_stock_code)

    if name_col is None:
        stock_df["stock_name"] = stock_df["stock_code"]
    else:
        stock_df["stock_name"] = raw_df[name_col].fillna("").astype(str).str.strip()

    stock_df = stock_df[stock_df["stock_code"].notna()].copy()
    stock_df["stock_name"] = stock_df["stock_name"].replace("", pd.NA).fillna(stock_df["stock_code"])
    stock_df = stock_df.drop_duplicates(subset=["stock_code"], keep="first").reset_index(drop=True)

    if stock_df.empty:
        raise ValueError(f"{source_name} 解析后股票列表为空")

    return stock_df


def _fetch_stock_universe_from_source(source_spec):
    source_name = source_spec["name"]
    raw_df, used_func, prev_errors = _call_ak_candidates(
        source_spec["func_candidates"],
        source_name=source_name,
        stage_name="股票列表",
    )
    prepared_df = _prepare_stock_universe_df(raw_df, source_name)
    return {
        "source": source_name,
        "used_func": used_func,
        "errors": prev_errors,
        "df": prepared_df,
    }


def _fetch_stock_universe_parallel():
    func.logInfo("开始并行抓取股票代码和名称（多数据源）")
    active_specs = _get_active_code_source_specs()
    if not active_specs:
        raise RuntimeError("当前 akshare 版本没有可用的股票列表接口")

    source_priority = {spec["name"]: index for index, spec in enumerate(active_specs)}
    source_results = []
    source_errors = {}

    with ThreadPoolExecutor(max_workers=min(MAX_SOURCE_WORKERS, len(active_specs))) as executor:
        future_map = {
            executor.submit(_fetch_stock_universe_from_source, spec): spec["name"] for spec in active_specs
        }

        try:
            for future in as_completed(future_map, timeout=CODE_SOURCE_STAGE_TIMEOUT_SECONDS):
                source_name = future_map[future]
                try:
                    result = future.result()
                    source_results.append(result)
                    func.logInfo(
                        f"[{source_name}] 股票列表抓取成功, source_func={result['used_func']}, count={len(result['df'])}"
                    )
                except Exception as error:
                    source_errors[source_name] = str(error)
                    _mark_provider_unavailable(source_name, source_errors[source_name])
                    func.logInfo(f"[{source_name}] 股票列表抓取失败: {error}")
        except FuturesTimeoutError:
            pass

        for future, source_name in future_map.items():
            if future.done():
                continue
            future.cancel()
            source_errors[source_name] = (
                f"timeout_after_{CODE_SOURCE_STAGE_TIMEOUT_SECONDS}s"
            )
            _mark_provider_unavailable(source_name, source_errors[source_name])
            func.logInfo(
                f"[{source_name}] 股票列表抓取超时，已取消等待（>{CODE_SOURCE_STAGE_TIMEOUT_SECONDS}s）"
            )

    if not source_results:
        raise RuntimeError(f"所有股票列表数据源都失败: {source_errors}")

    merged_frames = []
    for result in source_results:
        per_source_df = result["df"].copy()
        per_source_df["source_name"] = result["source"]
        per_source_df["source_priority"] = source_priority.get(result["source"], 9999)
        per_source_df["name_missing"] = per_source_df["stock_name"].fillna("").astype(str).str.strip().eq("")
        merged_frames.append(per_source_df)

    merged_df = pd.concat(merged_frames, ignore_index=True)
    merged_df = merged_df.sort_values(
        by=["stock_code", "name_missing", "source_priority", "stock_name"],
        ascending=[True, True, True, True],
        na_position="last",
    )
    merged_df = merged_df.drop_duplicates(subset=["stock_code"], keep="first").copy()
    merged_df = merged_df[["stock_code", "stock_name"]].reset_index(drop=True)

    func.logInfo(f"股票列表合并完成: {len(merged_df)} 只")
    return merged_df


def _prepare_board_df(raw_df, source_name):
    board_name_col = _choose_column(raw_df.columns, BOARD_NAME_COLUMN_CANDIDATES)
    if board_name_col is None:
        raise ValueError(f"{source_name} 缺少板块名称字段，当前字段: {list(raw_df.columns)}")

    board_id_col = _choose_column(raw_df.columns, BOARD_ID_COLUMN_CANDIDATES)

    board_df = pd.DataFrame()
    board_df["board_name"] = raw_df[board_name_col].fillna("").astype(str).str.strip()
    if board_id_col is None:
        board_df["board_id"] = board_df["board_name"]
    else:
        board_df["board_id"] = raw_df[board_id_col].fillna("").astype(str).str.strip()
        board_df["board_id"] = board_df["board_id"].where(
            board_df["board_id"] != "",
            board_df["board_name"],
        )

    board_df = board_df[(board_df["board_name"] != "") & (board_df["board_id"] != "")].copy()
    board_df = board_df.drop_duplicates(subset=["board_id", "board_name"]).reset_index(drop=True)

    if board_df.empty:
        raise ValueError(f"{source_name} 解析后板块列表为空")

    return board_df


def _extract_constituent_df(raw_df, board_name):
    code_col = _choose_column(raw_df.columns, CODE_COLUMN_CANDIDATES)
    if code_col is None:
        return pd.DataFrame(columns=["stock_code", "industry"])

    cons_df = pd.DataFrame()
    cons_df["stock_code"] = raw_df[code_col].apply(_normalize_stock_code)
    cons_df["industry"] = board_name
    cons_df = cons_df[cons_df["stock_code"].notna()].copy()
    return cons_df.drop_duplicates(subset=["stock_code", "industry"]).reset_index(drop=True)


def _fetch_board_constituents(board_id, board_name, source_spec):
    source_name = source_spec["name"]
    errors = []

    for func_name, symbol_param in source_spec["cons_func_candidates"]:
        fetcher = getattr(ak, func_name, None)
        if fetcher is None:
            errors.append(f"{func_name}:not_found")
            continue

        symbol_candidates = [board_id]
        if board_name != board_id:
            symbol_candidates.append(board_name)

        for symbol in symbol_candidates:
            try:
                raw_df = fetcher(**{symbol_param: symbol})
                if not isinstance(raw_df, pd.DataFrame):
                    errors.append(f"{func_name}({symbol}):not_dataframe")
                    continue

                cons_df = _extract_constituent_df(raw_df, board_name)
                if not cons_df.empty:
                    return cons_df, func_name, symbol, errors

                errors.append(f"{func_name}({symbol}):empty")
            except Exception as error:
                errors.append(f"{func_name}({symbol}):{error.__class__.__name__}: {error}")

    return pd.DataFrame(columns=["stock_code", "industry"]), None, None, errors


def _fetch_industry_map_from_source(source_spec):
    source_name = source_spec["name"]
    board_raw_df, used_board_func, board_errors = _call_ak_candidates(
        source_spec["board_func_candidates"],
        source_name=source_name,
        stage_name="行业板块列表",
    )
    board_df = _prepare_board_df(board_raw_df, source_name)
    total_boards = len(board_df)

    func.logInfo(f"[{source_name}] 板块列表抓取成功, board_func={used_board_func}, board_count={total_boards}")

    constituent_frames = []
    failed_board_count = 0
    error_samples = list(board_errors[:MAX_ERROR_SAMPLES_PER_SOURCE])
    last_log_ts = time.time()

    with ThreadPoolExecutor(max_workers=min(MAX_BOARD_WORKERS, total_boards)) as executor:
        future_map = {
            executor.submit(_fetch_board_constituents, row.board_id, row.board_name, source_spec): (
                row.board_id,
                row.board_name,
            )
            for row in board_df.itertuples(index=False)
        }

        completed = 0
        try:
            for future in as_completed(future_map, timeout=BOARD_FETCH_STAGE_TIMEOUT_SECONDS):
                completed += 1
                board_id, board_name = future_map[future]

                try:
                    cons_df, _, _, board_fetch_errors = future.result()
                    if not cons_df.empty:
                        constituent_frames.append(cons_df)
                    else:
                        failed_board_count += 1
                        if len(error_samples) < MAX_ERROR_SAMPLES_PER_SOURCE:
                            error_samples.append(f"{board_name}:{board_fetch_errors[:2]}")
                        func.logInfo(f"[{source_name}] 板块 {board_name} 抓取为空, board_id={board_id}")
                except Exception as error:
                    failed_board_count += 1
                    if len(error_samples) < MAX_ERROR_SAMPLES_PER_SOURCE:
                        error_samples.append(f"{board_name}:{error}")
                    func.logInfo(f"[{source_name}] 板块 {board_name} 抓取异常: {error}")

                now_ts = time.time()
                if completed == total_boards or now_ts - last_log_ts >= PROGRESS_LOG_INTERVAL_SECONDS:
                    func.logInfo(
                        f"[{source_name}] 行业成分抓取进度: {completed}/{total_boards}, "
                        f"成功板块={completed - failed_board_count}, 失败板块={failed_board_count}"
                    )
                    last_log_ts = now_ts
        except FuturesTimeoutError:
            pass

        for future, (board_id, board_name) in future_map.items():
            if future.done():
                continue
            future.cancel()
            failed_board_count += 1
            if len(error_samples) < MAX_ERROR_SAMPLES_PER_SOURCE:
                error_samples.append(f"{board_name}:timeout_after_{BOARD_FETCH_STAGE_TIMEOUT_SECONDS}s")
            func.logInfo(
                f"[{source_name}] 板块 {board_name} 抓取超时，已取消等待（>{BOARD_FETCH_STAGE_TIMEOUT_SECONDS}s）"
            )

    if not constituent_frames:
        raise RuntimeError(f"[{source_name}] 行业成分全部失败, samples={error_samples}")

    industry_df = pd.concat(constituent_frames, ignore_index=True)
    industry_df = industry_df.drop_duplicates(subset=["stock_code", "industry"]).reset_index(drop=True)

    func.logInfo(
        f"[{source_name}] 行业映射完成, stock_industry_pairs={len(industry_df)}, "
        f"failed_board_count={failed_board_count}, error_samples={error_samples}"
    )
    return {
        "source": source_name,
        "df": industry_df,
    }


def _fetch_industry_map_parallel(stock_df):
    func.logInfo("开始按股票逐个抓行业（稳定版）")

    results = []
    total = len(stock_df)

    MAX_WORKERS = 3
    RETRY_TIMES = 3
    SLEEP_SECONDS = 0.3

    def fetch_one(stock_code):
        for i in range(RETRY_TIMES):
            try:
                df = ak.stock_individual_info_em(symbol=stock_code)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    # 找行业字段
                    industry_row = df[df["item"].str.contains("行业", na=False)]
                    if not industry_row.empty:
                        return {
                            "stock_code": stock_code,
                            "industry": industry_row.iloc[0]["value"]
                        }
                return {"stock_code": stock_code, "industry": ""}
            except Exception as e:
                time.sleep(1.2 * (i + 1))
        return {"stock_code": stock_code, "industry": ""}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(fetch_one, row.stock_code): row.stock_code
            for row in stock_df.itertuples(index=False)
        }

        completed = 0
        last_log_ts = time.time()

        for future in as_completed(future_map):
            result = future.result()
            results.append(result)
            completed += 1

            now_ts = time.time()
            if completed % 100 == 0 or now_ts - last_log_ts > 5:
                func.logInfo(f"行业抓取进度: {completed}/{total}")
                last_log_ts = now_ts

            time.sleep(SLEEP_SECONDS)  # 限速防封

    industry_df = pd.DataFrame(results)
    industry_df = industry_df.drop_duplicates(subset=["stock_code"])

    func.logInfo(f"行业抓取完成: {len(industry_df)} 条")
    return industry_df


def _build_existing_stock_map():
    records = func.executeSelect("a_stock_analysis")["resultData"]
    if not records:
        return {}

    existing_df = pd.DataFrame(records)
    if existing_df.empty or "stock_code" not in existing_df.columns:
        return {}

    existing_df["stock_code"] = existing_df["stock_code"].apply(_normalize_stock_code)
    existing_df = existing_df[existing_df["stock_code"].notna()].copy()
    existing_df = existing_df.drop_duplicates(subset=["stock_code"], keep="last")
    return {
        row["stock_code"]: row.to_dict()
        for _, row in existing_df.iterrows()
    }


def _upsert_stocks(stock_df):
    existing_map = _build_existing_stock_map()
    stats = {
        "inserted": 0,
        "updated": 0,
        "unchanged": 0,
        "failed": 0,
    }

    total_count = len(stock_df)
    func.logInfo(f"开始写入股票主表, total={total_count}")

    for index, row in enumerate(stock_df.itertuples(index=False), start=1):
        stock_code = row.stock_code
        stock_name = row.stock_name
        new_industry = row.industry if isinstance(row.industry, str) else ""

        try:
            old_record = existing_map.get(stock_code)
            if old_record is None:
                insert_result = func.executeInsert(
                    "a_stock_analysis",
                    {
                        "stock_code": stock_code,
                        "stock_name": stock_name,
                        "industry": new_industry,
                    },
                )
                if insert_result.get("insertResult"):
                    stats["inserted"] += 1
                    existing_map[stock_code] = {
                        "stock_code": stock_code,
                        "stock_name": stock_name,
                        "industry": new_industry,
                    }
                else:
                    stats["failed"] += 1
            else:
                old_name = str(old_record.get("stock_name", "") or "").strip()
                old_industry = str(old_record.get("industry", "") or "").strip()
                merged_industry = _merge_industry(old_industry, new_industry)

                update_payload = {}
                if stock_name and stock_name != old_name:
                    update_payload["stock_name"] = stock_name
                if merged_industry != old_industry:
                    update_payload["industry"] = merged_industry

                if not update_payload:
                    stats["unchanged"] += 1
                else:
                    update_result = func.executeUpdate("a_stock_analysis", update_payload, {"stock_code": stock_code})
                    if update_result.get("updateResult"):
                        stats["updated"] += 1
                        old_record.update(update_payload)
                    else:
                        stats["failed"] += 1
        except Exception as error:
            stats["failed"] += 1
            func.logInfo(f"写入 {stock_code} 失败: {error}")

        if index % UPSERT_PROGRESS_INTERVAL == 0 or index == total_count:
            func.logInfo(
                f"主表写入进度: {index}/{total_count}, "
                f"inserted={stats['inserted']}, updated={stats['updated']}, "
                f"unchanged={stats['unchanged']}, failed={stats['failed']}"
            )

    return stats


def get_gu_piao_code_and_name():
    func.logInfo("开始抓取A股有哪些股票（多源并行）")
    stock_universe_df = _fetch_stock_universe_parallel()
    industry_grouped_df = _fetch_industry_map_parallel(stock_universe_df)

    if industry_grouped_df.empty:
        stock_universe_df["industry"] = ""
        func.logInfo("行业映射为空，本次仅更新股票代码和名称")
    else:
        stock_universe_df = stock_universe_df.merge(industry_grouped_df, how="left", on="stock_code")
        stock_universe_df["industry"] = stock_universe_df["industry"].fillna("").astype(str)

    stock_universe_df = stock_universe_df.drop_duplicates(subset=["stock_code"], keep="first").reset_index(drop=True)
    upsert_stats = _upsert_stocks(stock_universe_df)

    summary = {
        "universe_count": int(len(stock_universe_df)),
        "industry_coverage_count": int((stock_universe_df["industry"].str.strip() != "").sum()),
        "inserted": upsert_stats["inserted"],
        "updated": upsert_stats["updated"],
        "unchanged": upsert_stats["unchanged"],
        "failed": upsert_stats["failed"],
    }
    func.logInfo(f"抓取A股有哪些股票完毕: {summary}")
    return summary


if __name__ == "__main__":
    func.remove_path(func.getenv('LOG_PATH'))
    get_gu_piao_code_and_name()
