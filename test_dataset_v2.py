import torch
from src.arguments import ModelPath, TrainingConfig
from src.user_encoder import UserEncoder
from src.v2 import TextEventSequencePairDataLoader
from src.v2 import EventEncoder
import time

config = TrainingConfig(
    data_dir='./data',
    model_path=ModelPath.Qwen3_1B,
    batch_size=2,
    max_seq_len=32,
    max_text_len=32,
    num_negatives=64
)


if __name__ == '__main__':
    # build data loader
    train_loader = TextEventSequencePairDataLoader(config, rank=0)
    # fix random seed
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # build event encoder
    event_encoder = EventEncoder(
        model_path=config.model_path,
        max_seq_len=config.max_seq_len,
        use_flash_attention=True
    )
    # build user encoder
    user_encoder = UserEncoder(model_path=config.model_path)

    s = time.time()
    for i in range(5):
        batch = train_loader.next_batch()
        pos_hidden_states, attention_mask = event_encoder(
            input_ids=batch['pos_input_ids'],
            event_len=batch['pos_event_len'],
            padding=True
        )
        neg_hidden_states, _ = event_encoder(
            input_ids=batch['neg_input_ids'],
            event_len=batch['neg_event_len'],
            padding=False
        )

        # call user encoder
        predictions = user_encoder(
            event_embeddings=pos_hidden_states,
            attention_mask=attention_mask
        )
        print(predictions[0][-1])
    
    e = time.time()
    print(f"Time taken for 5 iterations: {e - s:.2f} seconds")