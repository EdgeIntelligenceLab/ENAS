import os
import random
import shutil

random.seed(42)

base = "vww_dataset"
classes = ["person", "not_person"]

for cls in classes:
    src = os.path.join(base, "val", cls)
    dst = os.path.join(base, "test", cls)

    os.makedirs(dst, exist_ok=True)

    files = os.listdir(src)
    sampled = random.sample(files, 2000)

    for f in sampled:
        shutil.copy(os.path.join(src, f), os.path.join(dst, f))
