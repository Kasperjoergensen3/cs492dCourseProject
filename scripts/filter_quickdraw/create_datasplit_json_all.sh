python scripts/filter_quickdraw/create_datasplit_json.py \
    --data_dir data/quickdraw/processed \
    --category "cat" \
    --val_size 0.1 \
    --seed 0

python scripts/filter_quickdraw/create_datasplit_json.py \
    --data_dir data/quickdraw/processed \
    --category "garden" \
    --val_size 0.1 \
    --seed 1

python scripts/filter_quickdraw/create_datasplit_json.py \
    --data_dir data/quickdraw/processed \
    --category "helicopter" \
    --val_size 0.1 \
    --seed 2
