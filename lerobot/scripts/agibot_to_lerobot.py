import os
from lerobot.common.datasets.utils import (
    STATS_PATH,
    check_timestamps_sync,
    get_episode_data_index,
    serialize_dict,
    write_json,
)
from convert_to_lerobot import get_task_instruction, load_local_dataset, AgiBotDataset, FEATURES
from utils import split_chunk_list
from pathlib import Path
from functools import partial
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
import queue
import threading

episode_index_dict = {}
lock = threading.Lock()
def worker(q, chunk_size):
    while True:
        batch = q.get()
        if batch == None:
            break
        
        raw_data = batch["raw_data"]
        dataset = batch["dataset"]
        chunk = batch["chunk"]
        task_name = batch["task_name"]
        task_id = batch["task_id"]
        for i in range(chunk):
            episode = raw_data[i]
            lock.acquire()
            current_ep_idx = episode_index_dict[task_id]
            episode_buffer = {
                    "size": 0,
                    **{key: current_ep_idx if key == "episode_index" else [] for key in dataset.features},
            }
            episode_index_dict[task_id] += 1
            lock.release()
            for frame in episode[0]:
                episode_buffer = dataset.add_frame_with_out_buffer(frame, episode_buffer)
            
            dataset.save_episode_with_out_buffer(task=task_name, videos=episode[1], episode_buffer=episode_buffer) # videos is the path of videos

            


def main():
    global episode_index_dict
    aigbot_root = "/Data/rlds_raw/sample_dataset"
    obs_root = os.path.join(aigbot_root, "observations")
    lerobot_save_path = "/Data/lerobot_data/agibot"
    os.makedirs(lerobot_save_path, exist_ok=True)
    task_list = sorted(os.listdir(obs_root))
    chunk_size = 1000
    worker_num = 50
    data_queue = queue.Queue()
    worker_threads = []
    for i in range(worker_num):
        t = threading.Thread(target=worker, args=(data_queue, chunk_size))
        t.start()
        worker_threads.append(t)

    for task_id in task_list:
        episode_index_dict[task_id] = 0
        json_file = f"{aigbot_root}/task_info/task_{task_id}.json"
        repo_id = f"agibotworld/task_{task_id}"
        task_name = get_task_instruction(json_file)
        episode_list = sorted(
            [
                f.as_posix()
                for f in Path(aigbot_root).glob(f"observations/{task_id}/*")
                if f.is_dir()
            ]
        )
        episode_ids = [int(Path(path).name) for path in episode_list]
        raw_datasets_before_filter = process_map(
            partial(load_local_dataset, src_path=aigbot_root, task_id=task_id),
            episode_ids,
            max_workers=5,
            desc="Generating local dataset",
        )

        print(f"Episode number:{len(raw_datasets_before_filter)}")


        dataset = AgiBotDataset.create(
            repo_id=repo_id,
            root=f"{lerobot_save_path}/{repo_id}",
            fps=30,
            robot_type="a2d",
            features=FEATURES,
        )

        episode_num = len(raw_datasets_before_filter)
        chunk_list = split_chunk_list(range(episode_num), chunk_size)
        for chunk in chunk_list:
            batch = {
                "raw_data" : raw_datasets_before_filter,
                "dataset" : dataset,
                "chunk":chunk,
                "task_name" : task_name,
                "task_id" : task_id
            }
            data_queue.put(batch)

        # for episode in tqdm(raw_datasets_before_filter, desc="Generating dataset from raw datasets"):
        #     for frame in episode[0]:
        #         dataset.add_frame(frame)
        #     dataset.save_episode(task=task_name, videos=episode[1])
        # dataset.consolidate(
        #     run_compute_stats=True,
        #     stat_kwargs={"batch_size": 32, "num_workers": 8}
        # )
    
    for _ in range(worker_num):
        batch.put(None) # stop worker

    for t in worker_threads:
        t.join()


if __name__ == "__main__":
    main()