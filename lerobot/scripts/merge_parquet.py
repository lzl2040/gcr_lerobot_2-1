import os
import glob
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

# 设置输入目录列表（支持多个根目录）和输出文件路径
format_dir = "/data_16T/lerobot_openx/fractal20220817_data_lerobot"
input_dirs = [
    os.path.join(format_dir, "data"),  # 替换为你的第二个目录路径
]
output_file = os.path.join(format_dir, "merged.parquet")  # 替换为输出文件路径

# 1️⃣ 递归收集所有Parquet文件路径
parquet_files = []
for dir in sorted(input_dirs):
    subdir_list = sorted(os.listdir(dir))
    for subdir in subdir_list:
        subdir_path = os.path.join(dir, subdir)
        subdir_parquet_list = sorted(os.listdir(subdir_path))
        for parquet_file in subdir_parquet_list:
            if parquet_file.endswith(".parquet"):
                parquet_files.append(os.path.join(subdir_path, parquet_file))

if not parquet_files:
    raise ValueError("未找到任何Parquet文件！请检查输入目录路径")


print(f"找到 {len(parquet_files)} 个Parquet文件")
print("准备合并...")
# 2️⃣ 读取所有文件并合并为Dataset
dataset = ds.dataset(
    parquet_files,
    format="parquet",
    partitioning="hive"  # 如果文件有Hive分区则启用，否则删除此参数
)

print("合并完成, 正在写入...")
# 3️⃣ 转换为内存表（自动处理Schema兼容性）
table = dataset.to_table()

# 4️⃣ 写入合并后的Parquet文件
pq.write_table(
    table,
    output_file,
    compression="snappy",  # 可选：'gzip', 'brotli', 'zstd'
    row_group_size=50 * 1024 * 1024  # 控制行组大小（默认1MB）
)

print(f"成功合并 {len(parquet_files)} 个文件 → 输出路径: {output_file}")