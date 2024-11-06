from pathlib import Path
import json
import argparse
from PIL import Image, ImageDraw
import random
import numpy as np
import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="./data")
parser.add_argument("--category", type=str, default="cat")
parser.add_argument("--val_size", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

data_dir = Path(args.data_dir).joinpath(args.category)
train_test_indices_json = data_dir.joinpath("train_test_indices.json")

# read train_test_indices.json
with open(train_test_indices_json, "r") as f:
    indices = json.load(f)

train_indices = indices["train"]
test_indices = indices["test"]

# split train into train and val
# set random seed
random.seed(args.seed)
np.random.seed(args.seed)

# randomly sample indices
num_val = int(len(train_indices) * args.val_size)
val_indices = train_indices[:num_val]
train_indices = train_indices[num_val:]

print(
    f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}"
)

data = {"train": [], "val": []}
for split, split_name in [
    (train_indices, "train"),
    (val_indices, "val"),
]:
    image_dir = data_dir.joinpath(f"images_train")
    for idx in tqdm.tqdm(split, desc=f"Processing {split_name}"):
        image_files = sorted([f.name for f in image_dir.glob(f"{idx:06d}_*.png")])
        # append duplicate of last image to make condition for generating no extra strokes
        image_files.append(image_files[-1])
        for i in range(1, len(image_files)):
            sample = {
                "image": f"{image_dir.name}/{image_files[i]}",
                "condition_image": f"{image_dir.name}/{image_files[0]}",
            }
            data[split_name].append(sample)

data_file = data_dir.joinpath(f"train_split.json")
with open(data_file, "w") as f:
    json.dump(data, f)
