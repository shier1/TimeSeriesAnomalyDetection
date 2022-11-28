python3 train.py \
--name=Conv-Fc-8-4-0-0-0 \
--train_batch_size=256 \
--eval_batch_size=16 \
--learning_rate=5e-3 \
--weight_decay=0.05 \
--num_steps=800 \
--decay_type='cosine' \
--warmup_steps=20