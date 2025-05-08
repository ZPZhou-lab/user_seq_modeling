import random
import torch
import pandas as pd
from datetime import datetime, timedelta

# generate fake data
def generate_event_seq(max_seq_len: int=64):


    event_len = random.randint(1, max_seq_len)

    events = []
    for i in range(event_len):
        # generate a random time during 2022 ~ now
        start_time = datetime(2022, 1, 1)
        end_time = datetime.now()
        time_diff = end_time - start_time
        random_seconds = random.randint(0, int(time_diff.total_seconds()))
        random_time = start_time + timedelta(seconds=random_seconds)
        # format the time as a string
        time_str = random_time.strftime("%Y-%m-%d %H:%M:%S")

        # generate a random event
        event = random.choice([
            "用户点击了主页按钮",
            "用户点击了消息按钮",
            "用户点击了设置按钮",
            "用户点击了个人中心按钮",
            "用户点击了搜索按钮",
            "用户点击了搜索结果按钮"
        ])

        events.append([time_str, event])
    
    return {
        'observe_time': '2025-05-01 00:00:00',
        'event': events,
        'label': random.choice([0, 1])
    }

if __name__ == '__main__':
    shards = 4
    # generate mock data
    for i in range(shards):
        shard = []
        for _ in range(1000):
            shard.append(generate_event_seq(max_seq_len=30))
        shard = pd.DataFrame(shard)
        torch.save(shard, f"data/shard_{i:06d}.pkl")