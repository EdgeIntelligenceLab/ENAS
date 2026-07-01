import os
import json
from pycocotools.coco import COCO
from PIL import Image
from tqdm import tqdm

# ===== CONFIGURATION =====
COCO_ROOT = os.path.join(os.getcwd(), "coco")
OUTPUT_ROOT = os.path.join(os.getcwd(), "vww_dataset")

#COCO_ROOT = os.path.expanduser("~/datasets/coco")
#OUTPUT_ROOT = os.path.expanduser("~/datasets/vww_dataset")
# IMAGE_SIZE = (50, 50)
AREA_THRESHOLD = 0.005  # 0.5%

# ===========================

def create_dirs(split):
    for label in ["person", "not_person"]:
        os.makedirs(os.path.join(OUTPUT_ROOT, split, label), exist_ok=True)

def process_split(split):
    print(f"\nProcessing {split} set...")

    ann_file = os.path.join(COCO_ROOT, "annotations",
                            f"instances_{split}2014.json")

    img_dir = os.path.join(COCO_ROOT, f"{split}2014")

    coco = COCO(ann_file)

    # Get category ID for 'person'
    person_cat_id = coco.getCatIds(catNms=["person"])[0]

    img_ids = coco.getImgIds()

    for img_id in tqdm(img_ids):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info['file_name'])

        try:
            image = Image.open(img_path).convert("RGB")
        except:
            continue

        width, height = image.size
        image_area = width * height

        # Get annotations for this image
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[person_cat_id])
        anns = coco.loadAnns(ann_ids)

        label = "not_person"

        for ann in anns:
            bbox = ann['bbox']  # [x, y, w, h]
            bbox_area = bbox[2] * bbox[3]

            if bbox_area / image_area > AREA_THRESHOLD:
                label = "person"
                break

        # Resize image
        # image = image.resize(IMAGE_SIZE)

        save_path = os.path.join(
            OUTPUT_ROOT,
            split,
            label,
            img_info['file_name']
        )

        image.save(save_path)

def main():
    for split in ["train", "val"]:
        create_dirs(split)
        process_split(split)

    print("\nVisual Wake Words dataset generation complete.")

if __name__ == "__main__":
    main()
