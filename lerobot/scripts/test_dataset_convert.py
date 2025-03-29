from datasets import Dataset
import polars as pl
import time
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--parquet_root", type=str, default="/data_16T/lerobot_openx/bridge_orig_lerobot/merged.parquet")
args = parser.parse_args()
parquet_file = args.parquet_root

time_start = time.time()

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Start reading parquet file: {parquet_file}")
parquet_pl = pl.read_parquet(parquet_file)

time_read = time.time()

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Start converting parquet to list")
parquet_list = parquet_pl.to_dicts()

time_convert = time.time()

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Start creating dataset")
dataset = Dataset.from_list(parquet_list, split="train")

time_end = time.time()
print(f"Dataset overview: \n{dataset}")
print(f"Dataset example: \n{dataset[0]}")

print(f"\nDataset length: {len(dataset)}")
print(f"Read parquet: {time_read - time_start:.2f}s")
print(f"Convert to list: {time_convert - time_read:.2f}s")
print(f"Create dataset: {time_end - time_convert:.2f}s")
print(f"Total time: {time_end - time_start:.2f}s")

