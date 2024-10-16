python scripts/filter_quickdraw/filter_data.py \
    --data_dir data/quickdraw/raw \
    --save_dir data/quickdraw/processed \
    --category "cat" \
    --num_train 10000 \
    --num_test 2000 \
    --seed 0

python scripts/filter_quickdraw/filter_data.py \
    --data_dir data/quickdraw/raw \
    --save_dir data/quickdraw/processed \
    --category "garden" \
    --num_train 10000 \
    --num_test 2000 \
    --seed 1

python scripts/filter_quickdraw/filter_data.py \
    --data_dir data/quickdraw/raw \
    --save_dir data/quickdraw/processed \
    --category "helicopter" \
    --num_train 10000 \
    --num_test 2000 \
    --seed 2
