import pandas
import os

raw = pandas.read_csv('../data/raw/HAM10000_metadata.csv')
deduplicated = raw.drop_duplicates(subset=['lesion_id'],ignore_index=True)
part1 = os.listdir('../data/raw/HAM10000_images_part_1')
part2 = os.listdir('../data/raw/HAM10000_images_part_2')
image_id = deduplicated['image_id']
dx = deduplicated['dx']
for id,label in zip(image_id,dx):
    if id + '.jpg' in part1:
        os.replace(f"../data/raw/HAM10000_images_part_1/{id}.jpg", f"../data/labeled/{label}/{id}.jpg")
    elif id + '.jpg' in part2:
        os.replace(f"../data/raw/HAM10000_images_part_2/{id}.jpg", f"../data/labeled/{label}/{id}.jpg")