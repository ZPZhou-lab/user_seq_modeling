import torch
from src.arguments import TrainingConfig
from src.user_encoder import UserEncoder
from src.v1 import TextEventSequencePairDataLoader
from src.v1 import EventEncoder
import time

config = TrainingConfig(
    data_dir='./data',
    model_path='../../models/TinyLlama-1.1B-3T/',
    batch_size=4,
    max_seq_len=32,
    max_text_len=32,
    num_negatives=128
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
    user_encoder = UserEncoder(
        model_path=config.model_path,
    )

    s = time.time()
    for i in range(1):
        batch = train_loader.next_batch()
        pos_hidden_states = event_encoder(
            input_ids=batch['pos_input_ids'],
            position_ids=batch['pos_position_ids'],
            seq_varlen=batch['pos_varlen']
        )
        print(batch['attention_mask'][0])
        print(pos_hidden_states.shape)
        print(pos_hidden_states[0][1:5])

        # call user encoder
        predictions = user_encoder(
            event_embeddings=pos_hidden_states,
            attention_mask=batch['attention_mask']
        )
        print(predictions.shape)
    
    e = time.time()
    print(f"Time taken for 5 iterations: {e - s:.2f} seconds")