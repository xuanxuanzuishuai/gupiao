import os
import zipfile
import json
import re
import concurrent
import asyncio
from concurrent.futures import ThreadPoolExecutor
import heapq
import time
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from dotenv import load_dotenv
import logging
from requests.models import Response
import pymysql
from logging.handlers import RotatingFileHandler
import random
from urllib.parse import urlparse
import uuid
import shutil

A_STOCK_TABLES = {"a_stock_analysis", "a_stock_analysis_history"}
A_STOCK_STRING_NUMERIC_COLUMNS = {
    "latest_price",
    "today_change",
    "change_3d",
    "change_5d",
    "change_10d",
    "change_20d",
    "change_30d",
    "today_vol",
    "vol_avg_3d",
    "vol_avg_5d",
    "vol_avg_10d",
    "vol_avg_20d",
    "today_amp",
    "amp_3d",
    "amp_5d",
    "amp_10d",
    "amp_20d",
    "amp_30d",
    "vr_today",
    "vr_3d",
    "vr_5d",
    "vr_10d",
    "vr_20d",
    "vr_30d",
    "today_amount",
    "amount_avg_5d",
    "amount_avg_10d",
    "turnover_rate",
    "turnover_avg_5d",
    "turnover_avg_10d",
    "volatility_10d",
    "volatility_20d",
    "ma20_slope_5d",
    "ma60_slope_10d",
}
A_STOCK_DECIMAL_COLUMNS = {
    "today_open",
    "today_high",
    "today_low",
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
}
def _to_plain_numeric_string(value, scale=2):
    if value is None:
        return None

    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")

    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or stripped.lower() in {"none", "nan", "null"}:
            return None
        value = stripped

    try:
        decimal_value = Decimal(str(value))
        if decimal_value.is_nan():
            return None
        quant = Decimal("1").scaleb(-scale)
        decimal_value = decimal_value.quantize(quant, rounding=ROUND_HALF_UP)
        text = format(decimal_value, "f")
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text or "0"
    except (InvalidOperation, ValueError, TypeError):
        return str(value)


def _to_plain_decimal_number(value, scale=2):
    text = _to_plain_numeric_string(value, scale)
    if text is None:
        return None
    try:
        return float(text)
    except (ValueError, TypeError):
        return None


def _normalize_stock_table_payload(table_name, payload):
    if table_name not in A_STOCK_TABLES or not isinstance(payload, dict):
        return payload

    normalized = {}
    for key, value in payload.items():
        if key in A_STOCK_STRING_NUMERIC_COLUMNS:
            normalized[key] = _to_plain_numeric_string(value, 2)
        elif key in A_STOCK_DECIMAL_COLUMNS:
            normalized[key] = _to_plain_decimal_number(value, 2)
        else:
            normalized[key] = value
    return normalized


def _load_similarity_modules():
    from sentence_transformers import SentenceTransformer, util
    import torch

    return SentenceTransformer, util, torch


# def process_model(perModel, texts):
#     model = SentenceTransformer(perModel)
#     embeddings = model.encode(texts, convert_to_tensor=True)
#     cos_sim = util.pytorch_cos_sim(embeddings, embeddings)
#     return cos_sim
#
#
# def calculate_similarity(texts):
#     allModelArr = ['paraphrase-MiniLM-L6-v2',
#                    'paraphrase-distilroberta-base-v1', 'stsb-roberta-base-v2',
#                    'distiluse-base-multilingual-cased', 'bert-base-nli-mean-tokens', 'bert-base-nli-stsb-mean-tokens',
#                    'paraphrase-xlm-r-multilingual-v1'
#                    ]
#
#     # Initialize a zero tensor to store the cumulative cosine similarity
#     tensorSum = torch.zeros(len(texts), len(texts))
#
#     with ThreadPoolExecutor() as executor:
#         future_to_model = {executor.submit(process_model, model, texts): model for model in allModelArr}
#         for future in concurrent.futures.as_completed(future_to_model):
#             cos_sim = future.result()
#             tensorSum += cos_sim
#
#     # Calculate the mean similarity
#     tensorSum /= len(allModelArr)
#
#     return tensorSum.tolist()

async def process_model_async(perModel, texts):
    SentenceTransformer, util, _ = _load_similarity_modules()
    model = SentenceTransformer(perModel)
    embeddings = model.encode(texts, convert_to_tensor=True)
    cos_sim = util.pytorch_cos_sim(embeddings, embeddings)
    return cos_sim


async def calculate_similarity_async(texts):
    _, _, torch = _load_similarity_modules()
    allModelArr = ['paraphrase-MiniLM-L6-v2',
                   'paraphrase-distilroberta-base-v1', 'stsb-roberta-base-v2',
                   'distiluse-base-multilingual-cased', 'bert-base-nli-mean-tokens', 'bert-base-nli-stsb-mean-tokens',
                   'paraphrase-xlm-r-multilingual-v1'
                   ]
    tasks = [process_model_async(model, texts) for model in allModelArr]
    results = await asyncio.gather(*tasks)
    tensorSum = torch.sum(torch.stack(results), dim=0)
    tensorSum /= len(allModelArr)
    return tensorSum.tolist()


def calculate_similarity(texts):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(calculate_similarity_async(texts))


# 将目标目录打包成zip
def zip_directory(directory_path, output_path):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, directory_path))


# 目录不存在创建
def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


# 替换字符串
def replace_colon_semicolon(input_string):
    if input_string is None:
        return ''
    result_string = input_string.replace(':', '').replace(';', '')
    return result_string


def extract_and_decode_content(openai_object):
    # 将 OpenAIObject 转换为 JSON 字符串
    json_string = json.dumps(openai_object)

    # 解析 JSON 字符串
    data = json.loads(json_string)

    # 使用正则表达式匹配字段，并提取对应的值
    location = re.findall(r'产地：(.*?)[；\n]', data["content"])
    location = replace_colon_semicolon(location[0] if location else None)

    level = re.findall(r'等级：(.*?)[；\n]', data["content"])
    level = replace_colon_semicolon(level[0] if level else None)

    top_category = re.findall(r'一级分类：(.*?)[；\n]', data["content"])
    top_category = replace_colon_semicolon(top_category[0] if top_category else None)

    two_category = re.findall(r'二级分类：(.*?)[；\n]', data["content"])
    two_category = replace_colon_semicolon(two_category[0] if two_category else None)

    named = re.findall(r'日常人们称呼：(.*?)[；\n]', data["content"])
    named = replace_colon_semicolon(named[0] if named else None)

    three_category = re.findall(r'细分品种：(.*?)[；\n]', data["content"])
    three_category = replace_colon_semicolon(three_category[0] if three_category else None)

    packaging_form = re.search(r'包装形式：(.*?)$', data["content"], re.DOTALL)
    packaging_form = replace_colon_semicolon(packaging_form.group(1) if packaging_form else None)

    # 拼接字段为一个新的字符串
    result = f"location:{location};level:{level};top_category:{top_category};two_category:{two_category};named:{named};three_category:{three_category};packaging_form:{packaging_form}"

    return result


# 格式化字段
def extractInfo(combineInfo, attribute_weights):
    # 将combine_info字段按分号分割成列表
    combine_info_items = combineInfo.split(";")

    desired_values = {}

    # 遍历列表，提取"location"、"level"和"top_category"字段的值，并添加到字典中
    # print(combine_info_items)
    for item in combine_info_items:
        #   print(item)
        key, value = item.split(":")
        if key in attribute_weights:
            desired_values[key] = value

    return desired_values


# 处理数据得到结果
def process_data_thread(data_chunk, otherGoodsData, iznCompareResult, compareSetList, attribute_weights, similarity,
                        standardSimilarity, N):
    for data in data_chunk:
        recordId = data['id']
        dataExtractInfo = extractInfo(data['combine_info'], attribute_weights)
        if recordId not in iznCompareResult:
            iznCompareResult[recordId] = [None] * len(otherGoodsData)

        # 继续和每行的相似度
        for otherIndex, otherData in enumerate(otherGoodsData):
            otherDataExtractInfo = extractInfo(otherData['combine_info'], attribute_weights)

            if not iznCompareResult[recordId][otherIndex]:
                iznCompareResult[recordId][otherIndex] = [None] * len(attribute_weights)

            # 计算每个字段的值相似度后确定和当前记录行的整体相似度
            for weightIndex, proportion in enumerate(attribute_weights):
                dataFieldValue = dataExtractInfo[proportion]
                otherFieldValue = otherDataExtractInfo[proportion]
                similarityValue = similarity[compareSetList.index(dataFieldValue)][
                                      compareSetList.index(otherFieldValue)] * attribute_weights[proportion]

                iznCompareResult[recordId][otherIndex][weightIndex] = similarityValue

            # named和three_category 取最大的
            sum_of_first_4_elements = sum(iznCompareResult[recordId][otherIndex][0:3])
            # 这个判断是为了不是所有的都替换，需要前四个也满足
            if (sum_of_first_4_elements >= 0.20):
                max_value = max(iznCompareResult[recordId][otherIndex][4], iznCompareResult[recordId][otherIndex][5])
                iznCompareResult[recordId][otherIndex][4] = iznCompareResult[recordId][otherIndex][5] = max_value

            iznCompareResult[recordId][otherIndex] = sum(iznCompareResult[recordId][otherIndex])

        # 使用堆来获取大于阈值的前 N 个元素的原始索引
        result_indices = [idx for idx, value in enumerate(iznCompareResult[recordId]) if value > standardSimilarity]
        result_indices = heapq.nlargest(N, result_indices, key=lambda idx: iznCompareResult[recordId][idx])

        iznCompareResult[recordId] = result_indices


# 多线程计算相似度排序
def calculate_similarity_with_threads(data_chunks, otherGoodsData, compareSetList, attribute_weights, similarity,
                                      standardSimilarity, N, iznLen, remark):
    iznCompareResult = {}
    start_time = time.time()

    with ThreadPoolExecutor() as executor:
        for chunk in data_chunks:
            executor.submit(process_data_thread, chunk, otherGoodsData, iznCompareResult, compareSetList,
                            attribute_weights, similarity, standardSimilarity, N)

        # 增加一个进度信息
        while len(iznCompareResult) < iznLen:
            nowLen = len(iznCompareResult)
            remaining_percentage = ((iznLen - nowLen) / iznLen) * 100

            elapsed_time = time.time() - start_time
            avg_completion_rate = nowLen / elapsed_time
            remaining_count = iznLen - nowLen
            estimated_remaining_time = remaining_count / avg_completion_rate
            print(
                f"{remark}匹配最相似数据 剩余: {remaining_percentage:.2f}%, 预估处理时间: {estimated_remaining_time:.2f} 秒")
            logInfo(
                f"{remark}匹配最相似数据 剩余: {remaining_percentage:.2f}%, 预估处理时间: {estimated_remaining_time:.2f} 秒")
            time.sleep(1)

    return iznCompareResult


# 判断两个数字之间的差值是否超过了10%
def validate_difference_within_percentage(number1, number2, percentage_limit=10):
    try:
        number1 = float(number1)
        number2 = float(number2)
    except ValueError:
        raise ValueError("Both input values must be convertible to numbers.")

    difference = abs(number1 - number2)
    percentage_difference = (difference / number1) * 100
    return percentage_difference <= percentage_limit


# 得到市场的中文名称
def getMakdetZhName(marketId):
    # 要比较的市场id
    marketArr = {
        2: '高碑店',
        1: '新发地',
    }

    return marketArr[marketId]


# 得到当前时间,中文表示
def get_current_zh_time():
    current_datetime = datetime.now()
    formatted_date = current_datetime.strftime("%Y-%m-%d %H:%M:%S %A")
    return formatted_date


# 获取env名称
def getenv(envKey):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    # 指定 .env 文件的绝对路径
    dotenv_path = current_directory + '/.env'
    load_dotenv(dotenv_path=dotenv_path)
    return os.getenv(envKey)


def create_logger(logfile):
    handler = RotatingFileHandler(logfile, maxBytes=1024 * 1024, backupCount=0, encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger('custom_logger')
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger


def custom_encoder(obj):
    # 自定义JSON编码器，处理不支持直接JSON序列化的对象
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, Response):
        # 如果是Response对象，转换为字典
        return vars(obj)
    try:
        # 尝试将对象转换为字典，如果成功则返回字典
        return obj.__dict__
    except AttributeError:
        # 对于其他类型，返回其字符串表示
        return str(obj)


def logInfo(log_content):
    log_path = getenv('LOG_PATH')
    create_directory_if_not_exists(log_path)
    logfile = os.path.join(log_path, 'similar_log.txt')

    if not hasattr(logInfo, 'logger_instance'):
        logInfo.logger_instance = create_logger(logfile)

    try:
        if isinstance(log_content, Response):
            # 如果传入的是Response对象，获取其文本内容并记录
            log_content_str = log_content.text
        elif isinstance(log_content, (dict, list, tuple, str, int, float)):
            # 尝试将输入参数转换为JSON字符串
            log_content_str = json.dumps(log_content, ensure_ascii=False, default=custom_encoder)
        else:
            # 尝试将输入参数转换为字符串，如果是字符串，则解码为utf-8
            log_content_str = log_content.decode('utf-8') if isinstance(log_content, bytes) else str(log_content)

        logInfo.logger_instance.info('log info: %s', log_content_str, extra={'content': log_content_str})
    except Exception as e:
        # 如果无法转换，则记录异常信息
        logging.error(f"An error occurred: {str(e)}")


def executeFetchOne(table_name, params=None):
    connection = None
    executed_sql = None
    try:
        connection = pymysql.connect(
            host='127.0.0.1',
            user='root',
            password='rootroot',
            database='gu_piao',
            charset='utf8mb4'
        )
        with connection.cursor() as cursor:
            # 构造 SQL 查询语句
            if params:
                where_clause = " AND ".join(f"{key}=%s" for key in params.keys())
                sql = f"SELECT * FROM {table_name} WHERE {where_clause}"
                executed_sql = cursor.mogrify(sql, list(params.values()))
                cursor.execute(sql, list(params.values()))
            else:
                sql = f"SELECT * FROM {table_name}"
                executed_sql = sql
                cursor.execute(sql)

            records = cursor.fetchall()
            field_names = [desc[0] for desc in cursor.description]

            result_data = []

            for row in records:
                per_data = dict(zip(field_names, row))
                result_data.append(per_data)

            # 返回查询结果和实际执行的 SQL 语句
            logInfo(executed_sql)

            # 如果只有一行数据，返回第一行数据，否则返回空字典
            if len(result_data) == 1:
                return {'resultData': result_data[0], 'executeSql': executed_sql}
            else:
                return {'resultData': {}, 'executeSql': executed_sql}

    except Exception as e:
        logInfo('SQL 查询出错: ' + str(e))
        # 返回空字典和实际执行的 SQL 语句
        return {'resultData': {}, 'executeSql': executed_sql}

    finally:
        if connection:
            connection.close()


# 执行数据库select
def executeSelect(table_name, params=None):
    connection = None
    executed_sql = None
    try:
        connection = pymysql.connect(
            host='127.0.0.1',
            user='root',
            password='rootroot',
            database='gu_piao',
            charset='utf8mb4'
        )
        with connection.cursor() as cursor:
            # 构造 SQL 查询语句
            sql = f"SELECT * FROM {table_name}"
            if params:
                # 如果有参数字典，则构造 WHERE 子句
                where_clause = " AND ".join(f"{key}=%s" for key in params.keys())
                sql += f" WHERE {where_clause}"

            # 获取实际执行的 SQL 语句
            executed_sql = cursor.mogrify(sql, list(params.values()) if params else None)
            # 执行 SQL 查询
            cursor.execute(sql, list(params.values()) if params else None)
            records = cursor.fetchall()
            field_names = [desc[0] for desc in cursor.description]

        result_data = []

        for row in records:
            per_data = dict(zip(field_names, row))
            result_data.append(per_data)

        # 返回查询结果和实际执行的 SQL 语句
        logInfo(executed_sql)
        return {'resultData': result_data, 'executeSql': executed_sql}

    except Exception as e:
        logInfo('SQL 查询出错: ' + str(e))
        # 返回空列表和实际执行的 SQL 语句
        return {'resultData': [], 'executeSql': executed_sql}

    finally:
        if connection:
            connection.close()


# 获取当前时间戳
def getTimeStamp():
    return int(time.time())


# 执行插入
def executeInsert(table_name, data):
    connection = None
    executed_sql = None
    try:
        data = _normalize_stock_table_payload(table_name, data)
        connection = pymysql.connect(
            host='127.0.0.1',
            user='root',
            password='rootroot',
            database='gu_piao',
            charset='utf8mb4'
        )

        with connection.cursor() as cursor:
            # 构建SQL插入语句
            columns = ', '.join(data.keys())
            placeholders = ', '.join(['%s'] * len(data))
            sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"

            # 获取实际执行的 SQL 语句
            executed_sql = cursor.mogrify(sql, tuple(data.values()))

            # 执行插入操作
            cursor.execute(sql, tuple(data.values()))
            connection.commit()

            logInfo("数据插入成功！" + executed_sql)

            # 返回执行结果和实际执行的 SQL 语句
            return {'insertResult': True, 'executeSql': executed_sql}

    except Exception as e:
        logInfo('插入数据出错:' + str(e))
        return {'insertResult': False, 'executeSql': executed_sql}

    finally:
        if connection:
            connection.close()

def batchInsert(table_name, data_list):
    connection = None
    try:
        data_list = [_normalize_stock_table_payload(table_name, per_data) for per_data in data_list]
        connection = pymysql.connect(
            host='127.0.0.1',
            user='root',
            password='rootroot',
            database='gu_piao',
            charset='utf8mb4'
        )

        with connection.cursor() as cursor:
            if not data_list:
                raise ValueError("数据列表为空")

            # 构建SQL插入语句
            columns = ', '.join(data_list[0].keys())
            placeholders = ', '.join(['%s'] * len(data_list[0]))
            sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"

            # 获取实际执行的 SQL 语句
            executed_sql = cursor.mogrify(sql, tuple(data_list[0].values()))

            # 批量插入操作
            values_list = [tuple(data.values()) for data in data_list]
            cursor.executemany(sql, values_list)
            connection.commit()

            logInfo(f"批量插入数据成功！执行的 SQL: {executed_sql}")

            # 返回执行结果和实际执行的 SQL 语句
            return {'insertResult': True, 'executeSql': executed_sql}

    except Exception as e:
        logInfo(f'插入数据出错: {str(e)}')
        return {'insertResult': False, 'executeSql': executed_sql if 'executed_sql' in locals() else None}

    finally:
        if connection:
            connection.close()

# 数据库更新
def executeUpdate(table_name, update_data, where_condition):
    connection = None
    executed_sql = None
    try:
        update_data = _normalize_stock_table_payload(table_name, update_data)
        connection = pymysql.connect(
            host='127.0.0.1',
            user='root',
            password='rootroot',
            database='gu_piao',
            charset='utf8mb4'
        )

        with connection.cursor() as cursor:
            # 构建SQL更新语句
            set_values = ', '.join([f"{key}=%s" for key in update_data.keys()])
            where_values = ' AND '.join([f"{key}=%s" for key in where_condition.keys()])
            sql = f"UPDATE {table_name} SET {set_values} WHERE {where_values}"

            # 获取实际执行的 SQL 语句
            executed_sql = cursor.mogrify(sql, list(update_data.values()) + list(where_condition.values()))

            # 执行更新操作
            cursor.execute(sql, list(update_data.values()) + list(where_condition.values()))
            connection.commit()

            # 获取受影响的行数
            affected_rows = cursor.rowcount

            logInfo(f"数据更新成功！受影响的行数: {affected_rows}. SQL: {executed_sql}")
            # 返回执行结果和实际执行的 SQL 语句
            return {'updateResult': True, 'executeSql': executed_sql}

    except Exception as e:
        logInfo('更新数据出错:' + str(e))
        return {'updateResult': False, 'executeSql': executed_sql}

    finally:
        if connection:
            connection.close()

def executeDelete(table_name, where_condition):
    connection = None
    executed_sql = None
    try:
        connection = pymysql.connect(
            host='127.0.0.1',
            user='root',
            password='rootroot',
            database='gu_piao',
            charset='utf8mb4'
        )

        with connection.cursor() as cursor:
            if not where_condition:
                raise ValueError("删除条件不能为空！为了安全，必须指定 WHERE 条件。")

            # 构建 WHERE 子句
            where_values = ' AND '.join([f"{key}=%s" for key in where_condition.keys()])
            sql = f"DELETE FROM {table_name} WHERE {where_values}"

            # 获取实际执行 SQL
            executed_sql = cursor.mogrify(sql, list(where_condition.values()))

            # 执行删除
            cursor.execute(sql, list(where_condition.values()))
            connection.commit()
            affected_rows = cursor.rowcount

            logInfo(f"删除数据成功！受影响行数: {affected_rows}. SQL: {executed_sql}")
            return {'deleteResult': True, 'executeSql': executed_sql}

    except Exception as e:
        logInfo('删除数据出错:' + str(e))
        return {'deleteResult': False, 'executeSql': executed_sql}

    finally:
        if connection:
            connection.close()

def generateRandomIp():
    # 生成随机的IP地址，格式为 xxx.xxx.xxx.xxx
    ip = '.'.join(str(random.randint(0, 255)) for _ in range(4))
    return ip


def decrypt_hqsku_id(hqskuid, strict=True):
    if not strict:
        if hqskuid.isnumeric():
            return hqskuid

    return hqskuid[-8:].lstrip('0')


def getSkuNo(skuId):
    decrypted_sku = decrypt_hqsku_id(skuId, False)
    return int(decrypted_sku) + 10000


def extractDomain(url):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    return domain


def getCompanyname(code):
    company_names = {
        1: "启果果",
        2: "爱助农",
        3: "般果",
        4: "首衡",
        5: "联农",
        6: "新发地"
    }

    return company_names.get(code, "未知")


def getNeedGenerateCompany():
    return (1, 4, 5)

def getNeedCompareCompany():
    return (4,1)

def generate_token():
    return str(uuid.uuid4()).replace("-", "")

def remove_path(path):
    try:
        if os.path.isfile(path):
            os.remove(path)
            print(f"文件 {path} 已成功删除。")
            logInfo(f"文件 {path} 已成功删除。")
        elif os.path.isdir(path):
            shutil.rmtree(path)
            print(f"目录 {path} 及其内容已成功删除。")
            logInfo(f"目录 {path} 及其内容已成功删除。")
        else:
            print(f"路径 {path} 既不是文件也不是目录。")
            logInfo(f"路径 {path} 既不是文件也不是目录。")
    except FileNotFoundError:
        print(f"路径 {path} 未找到。")
        logInfo(f"路径 {path} 未找到。")
    except PermissionError:
        print(f"没有权限删除路径 {path}。")
        logInfo(f"没有权限删除路径 {path}。")
    except Exception as e:
        print(f"删除路径时发生错误: {e}")
        logInfo(f"删除路径时发生错误: {e}")

def clean_price(price_str):
    # 清理字符串，移除多余的点和非数字字符
    clean_str = ''.join(c for c in price_str if c.isdigit() or c == '.')
    if clean_str.count('.') > 1:
        # 如果仍然有多个点，则保留第一个点，移除后续的点
        parts = clean_str.split('.')
        clean_str = parts[0] + '.' + ''.join(parts[1:])
    return clean_str

def multiply_price_by_100(price_str):
    price_str = clean_price(price_str)
    # Step 1: Convert the string to a float
    price_float = float(price_str)

    # Step 2: Multiply the float by 100
    multiplied_price = price_float * 100

    # Step 3: Convert the result to an integer
    result_int = int(multiplied_price)

    return result_int
