from v1.hllm import EventEncoder
from v1.dataset import (
    TrainingConfig,
    TextEventSequencePairDataLoader
)

config = TrainingConfig(
    data_dir='./data',
    model_path='/Users/zhouzhipeng/llm_models/TinyLlama-1.1B-3T',
    batch_size=2,
    max_seq_len=16,
    num_negatives=256
)


if __name__ == '__main__':
    train_loader = TextEventSequencePairDataLoader(config, rank=0)

    # build event encoder
    event_encoder = EventEncoder(
        model_path=config.model_path,
        max_seq_len=config.max_seq_len
    )

    for i in range(5):
        batch = train_loader.next_batch()
        hidden_states = event_encoder(
            input_ids=batch['pos_input_ids'],
            position_ids=batch['pos_position_ids'],
            seq_varlen=batch['pos_varlen'],
        )
        print(hidden_states.shape)