import os
import json
import ndjson
import argparse
from PIL import Image, ImageDraw
import random
import numpy as np
from tqdm import tqdm


def draw_strokes_and_save(
    strokes, idx, save_dir, height=256, width=256, save_intermediate=False
):
    image = Image.new("RGB", (width, height), "white")
    if save_intermediate:
        image.save(save_dir + f"/{idx:06d}_00.png")
    image_draw = ImageDraw.Draw(image)
    for i, stroke in enumerate(strokes):
        points = list(zip(stroke[0], stroke[1]))
        image_draw.line(points, fill=0)
        if save_intermediate:
            image.save(save_dir + f"/{idx:06d}_{i+1:02d}.png")
    if not save_intermediate:
        image.save(save_dir + f"/{idx:06d}.png")
    return image


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="./data")
parser.add_argument("--save_dir", type=str, default="./sketch_data")
parser.add_argument("--category", type=str, default="cat")
parser.add_argument("--json", type=str, default="train_test_indices.json")
args = parser.parse_args()

save_dir = os.path.join(args.save_dir, args.category)
os.makedirs(save_dir, exist_ok=True)
os.makedirs(os.path.join(save_dir, "images_train"), exist_ok=True)
os.makedirs(os.path.join(save_dir, "images_test"), exist_ok=True)


# load data
with open(os.path.join(args.data_dir, f"{args.category}.ndjson"), "r") as f:
    data = ndjson.load(f)

# load indices
with open(os.path.join(args.save_dir, args.json), "r") as f:
    indices = json.load(f)

train_indices = indices["train"]
test_indices = indices["test"]


# iterate over data
for idx in tqdm(train_indices, desc="Processing train"):
    item = data[idx]
    strokes = item["drawing"]
    draw_strokes_and_save(
        strokes, idx, os.path.join(save_dir, "images_train"), save_intermediate=True
    )

for idx in tqdm(test_indices, desc="Processing test"):
    item = data[idx]
    strokes = item["drawing"]
    draw_strokes_and_save(strokes, idx, os.path.join(save_dir, "images_train"))


print("Done!")
