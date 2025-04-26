from src.arguments import TrainingConfig
from src.v2.dataset import TextEventSequencePairDataLoader
from src.v2.hllm import EventEncoder
import time

config = TrainingConfig(
    data_dir='./data',
    model_path='../../models/TinyLlama-1.1B-3T/',
    batch_size=4,
    max_seq_len=64,
    max_text_len=32,
    num_negatives=128
)


if __name__ == '__main__':
    # build data loader
    train_loader = TextEventSequencePairDataLoader(config, rank=0)

    # build event encoder
    event_encoder = EventEncoder(
        model_path=config.model_path,
        max_seq_len=config.max_seq_len,
        use_flash_attention=False
    )

    s = time.time()
    for i in range(10):
        batch = train_loader.next_batch()
        hidden_states, attention_mask = event_encoder(
            input_ids=batch['pos_input_ids'],
            event_len=batch['pos_event_len'],
            padding=True
        )
        hidden_states, _ = event_encoder(
            input_ids=batch['neg_input_ids'],
            event_len=batch['neg_event_len'],
            padding=False
        )
    e = time.time()
    print(f"Time taken for 5 iterations: {e - s:.2f} seconds")
    print(batch['pos_input_ids']['input_ids'].shape)
    print(batch['neg_input_ids']['input_ids'].shape)