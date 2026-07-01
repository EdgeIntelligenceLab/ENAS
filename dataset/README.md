# Datasets

ENAS is evaluated on two TinyML benchmarks. Both must live under `dataset/` with the
folder layout below before running any experiment (the paths are set in
`configs/datasets/vww.yaml` and `configs/datasets/melanoma.yaml`).

```
dataset/
├── vww_dataset/                 # Visual Wake Words (generated from COCO 2014)
│   ├── train/{person,not_person}
│   ├── val/{person,not_person}
│   └── test/{person,not_person}
└── melanoma_cancer_dataset/     # Melanoma (downloaded from Kaggle)
    ├── train/{benign,malignant}
    └── test/{benign,malignant}
```

---

## 1. Visual Wake Words (VWW)

VWW is derived from **MS-COCO 2014** following Chowdhery et al. (2019): an image is
labelled `person` if any person bounding box covers more than 0.5% of the image area,
else `not_person`. Images are kept at original resolution; resizing happens in the
training pipeline.

### Step 1 — download COCO 2014 (~19 GB)

```bash
cd dataset
mkdir -p coco && cd coco
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip train2014.zip && unzip val2014.zip && unzip annotations_trainval2014.zip
cd ..
```

Expected layout: `dataset/coco/{train2014,val2014,annotations}`.

### Step 2 — generate the VWW train/val splits

```bash
# from the dataset/ directory
python generate_vww_dataset.py
```

This writes `dataset/vww_dataset/{train,val}/{person,not_person}` (COCO `train2014` →
`train`, `val2014` → `val`). Requires `pycocotools`, `Pillow`, `tqdm`.

### Step 3 — carve out a fixed test split

The paper evaluates on a held-out test set sampled from the val split (seeded, 2000
images per class):

```bash
python create_test_dataset.py
```

This writes `dataset/vww_dataset/test/{person,not_person}`. The seed (42) makes the test
split reproducible.

---

## 2. Melanoma Cancer

Binary classification (benign / malignant) of dermoscopic images. Download the
**Melanoma Skin Cancer Dataset of 10000 Images** from Kaggle:

<https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images>

```bash
cd dataset
# requires a Kaggle API token (~/.kaggle/kaggle.json)
kaggle datasets download -d hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images
unzip melanoma-skin-cancer-dataset-of-10000-images.zip -d melanoma_cancer_dataset_raw
```

The Kaggle archive ships `train/` and `test/` folders, each with `benign/` and
`malignant/` subfolders. Move/rename the extracted top-level folder to
`dataset/melanoma_cancer_dataset/` so the paths match `configs/datasets/melanoma.yaml`.

---

## Verifying the datasets

```bash
python scripts/run_smoke_test.py        # checks both datasets load and one tiny search runs
```

## Notes

- **No resizing on disk.** Both datasets are stored at native resolution; each experiment
  resizes to its target `P_x × P_x` at load time, which is why the same data serves all
  nine resolutions.
- **VWW area threshold** (0.5%) follows the MLPerf Tiny / original VWW definition and is
  configurable in `generate_vww_dataset.py` (`AREA_THRESHOLD`).
- **Licensing.** COCO images are subject to the COCO terms of use; the VWW labelling is a
  derivative. The Melanoma dataset is subject to the Kaggle/ISIC dataset terms.
