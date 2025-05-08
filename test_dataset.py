from src.dataset import (
    TextEventSequencePairDataset, 
    sequential_event_collate_fn,
    build_dataloader
)
from src.arguments import TrainingConfig, TimeEmbeddingConfig, ModelPath

ts_config = TimeEmbeddingConfig(
    use_time_embedding=True,
    mode='absolute',
    time_hiddens=256,
    max_diff_day=720,
    max_year_ago=10,
    mixup_activation='silu'
)
config = TrainingConfig(
    # dataset args
    train_data_dir='./data',
    valid_data_dir='./data',
    model_path=ModelPath.Qwen3_1B,
    shard_size=128,
    batch_size=2,
    max_seq_len=32,
    max_text_len=32,
    num_negatives=64,
    # training args
    name='test',
    log_dir='./logs',
    save_dir='./ckpt',
    learning_rate=1e-5,
    top_warmup_steps=-1,
    warmup_steps=100,
    grad_accum_steps=1,
    max_steps=10,
    log_freq=1,
    eval_steps=10,
    max_evel_iter=100,
    temprature=0.05,
    nce_threshold=0.99,
    nce_loss_lambda=0.5
)

if __name__ == "__main__":
    train_set = TextEventSequencePairDataset(config, ts_config, split='train', rank=0, world_size=4)
    train_loader = build_dataloader(
        train_set, config,
        collate_fn=sequential_event_collate_fn,
        rank=0,
        num_workers=2,
        pin_memory=True
    )
    print("Total batches:", len(train_loader))
    print("First 5 batches of data:")
    for step in range(5):
        batch = next(train_loader)
        print(f"Step {step}: {batch['labels']}")
    print("Next 5 batches of data:")
    for step in range(5):
        batch = next(train_loader)
        print(f"Step {step}: {batch['labels']}")