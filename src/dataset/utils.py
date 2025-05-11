import os
import pandas as pd
from datetime import datetime
import torch
import torch.distributed as dist

def _set_world_size_and_rank(obj, world_size: int, rank: int):
    """
    Set the world size and rank for distributed training.
    """
    if world_size is None:
        obj.world_size = dist.get_world_size() if dist.is_initialized() else 1
    else:
        obj.world_size = world_size
    
    if rank is None:
        obj.rank = dist.get_rank() if dist.is_initialized() else 0
    else:
        obj.rank = rank


class TableReader:
    def __init__(self, table: str):
        shards = os.listdir(table)
        shards = sorted([shard for shard in shards if shard.endswith('.pkl')])
        assert len(shards) > 0, f"no shards found in {table}"        
        shards = [os.path.join(table, shard) for shard in shards]
        self.reader = pd.concat([torch.load(shard, weights_only=False) for shard in shards], ignore_index=True)

    def to_pandas(self, start: int=0, limit: int=-1):
        if limit == -1:
            limit = len(self.reader)
        return self.reader.iloc[start:(start + limit)].copy()
    
    @property
    def table_size(self):
        return len(self.reader)
    
    def __len__(self):
        return len(self.reader)
    

def get_action_time_diff(
    action_time: str, 
    observe_time: str,
    mode: str='relative',
    max_diff_day: int=720,
    max_year_ago: int=10
):
    """
    locate the `action_time` as diff from `observe_time`.

    Parameters
    ----------
    action_time: str
        the time when the action happened.
    observe_time: str
        the time when do the observation.
    mode: str
        the mode of time diff, one of `relative`, `absolute`, default is `relative`.\n
        - `relative`: locate the time as diff in tuple `(days, hours, minutes and seconds)`.\n
        - `absolute`: locate the time as tuple `(-year, month, day, hour, minute, second)`.\n
    max_diff_day: int
        the max diff day, default is 720 days, time_diff > 720 days will be truncated to 720 days,\
        only used when mode is `relative`
    max_year_age: int
        the max year age, default is 10 years, year > 10 years will be truncated to 10 years,\
        only used when mode is `absolute`
    """
    # calculate the date diff from now
    # get the day, hour, minute, second diff
    obs_time = datetime.strptime(observe_time, '%Y-%m-%d %H:%M:%S')
    act_time = datetime.strptime(action_time, '%Y-%m-%d %H:%M:%S')
    diff = obs_time - act_time
    # Extract total days
    day_diff = diff.days

    if mode == 'relative':
        if day_diff >= max_diff_day:
            return (max_diff_day - 1, 23, 59, 59)
        # Calculate remaining seconds
        remainder = diff.seconds
        # Extract hours, minutes, seconds
        hour_diff = remainder // 3600
        remainder = remainder % 3600
        minute_diff = remainder // 60
        second_diff = remainder % 60
        return (day_diff, hour_diff, minute_diff, second_diff)
    elif mode == 'absolute':
        year = max(obs_time.year - act_time.year, 0)
        year = min(year, max_year_ago)
        return (year, act_time.month, act_time.day, act_time.hour, act_time.minute, act_time.second)
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'relative' or 'absolute'.")
