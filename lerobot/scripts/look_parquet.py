import pandas as pd
import pyarrow.parquet as pq
import time
import polars as pl

def parquet_to_list_polars(file_path: str) -> list[dict]:
    """使用 Polars 将 Parquet 文件转换为字典列表（自动并行化）"""
    # 读取 Parquet 文件
    start_read = time.time()
    df = pl.read_parquet(file_path)
    read_time = time.time() - start_read

    # 转换为字典列表
    start_convert = time.time()
    list_of_dicts = df.rows(named=True)  # 关键优化步骤
    convert_time = time.time() - start_convert

    # 输出耗时统计
    print(f"[Polars] 读取耗时: {read_time:.2f}s")
    print(f"[Polars] 转换耗时: {convert_time:.2f}s")
    print(f"[Polars] 总行数: {len(list_of_dicts)}")
    
    return list_of_dicts

def parquet_to_list(file_path, batch_size=10000):
    # 打开Parquet文件并分批次读取
    time_a = time.time()
    table = pq.read_table(
                        file_path,
                        use_threads=True,  # 启用多线程
                        memory_map=True    # 内存映射文件加速
                    )
    time_b = time.time()
    full_list = []
    
    # column_data = {col: table[col].to_pylist() for col in table.column_names}
    # 转置为行式字典列表
    list_parquet = table.to_pylist()
    # list_parquet = [dict(zip(column_data.keys(), row)) for row in zip(*column_data.values())]
    
    time_c = time.time()
    print(f"读取Parquet文件耗时: {time_b - time_a:.2f}s")
    print(f"转换为列表耗时: {time_c - time_b:.2f}s")
    print(f"总计耗时: {time_c - time_a:.2f}s")
    return list_parquet

path = "/data_16T/lerobot_openx/furniture_bench_dataset_lerobot/merged.parquet"
path = "/data_16T/lerobot_openx/bridge_orig_lerobot/merged.parquet"
# path = "/data_16T/lerobot_openx/furniture_bench_dataset_lerobot/data/chunk-005/episode_005099.parquet"

# df = pd.read_parquet(path, engine="pyarrow")
# print(df.head())
# for index, row in df.iterrows():
#     print(index, row)

# time_start = time.time()
# full_list = parquet_to_list(path)
# time_end = time.time()
# print("time cost", time_end - time_start, "s")
# print(len(full_list))
# print(full_list[0])

full_list = parquet_to_list_polars(path)
print(len(full_list))
# print(type(full_list))
key_list = []
for k in full_list[0].keys():
    key_list.append(k)
print(key_list)
# print(k for k in full_list[0].keys())

    