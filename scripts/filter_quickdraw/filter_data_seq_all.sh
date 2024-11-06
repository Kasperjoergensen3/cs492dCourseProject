python scripts/filter_quickdraw/filter_data_seq.py \
    --data_dir data/quickdraw/raw \
    --save_dir data/quickdraw/processed \
    --category "cat" \
    --json train_test_indices.json

python scripts/filter_quickdraw/filter_data_seq.py \
    --data_dir data/quickdraw/raw \
    --save_dir data/quickdraw/processed \
    --category "garden" \
    --json train_test_indices.json

python scripts/filter_quickdraw/filter_data_seq.py \
    --data_dir data/quickdraw/raw \
    --save_dir data/quickdraw/processed \
    --category "helicopter" \
    --json train_test_indices.json
