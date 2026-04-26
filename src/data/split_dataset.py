import json
import random
from pathlib import Path
from collections import defaultdict, Counter


# =========================
# Configuración
# =========================
LEVEL = "hard"  # "easy", "medium", "hard"
TRAIN_RATIO = 0.8
SEED = 42


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def filter_by_image_ids(images, annotations, image_id_set):
    split_images = [img for img in images if img["id"] in image_id_set]
    split_annotations = [ann for ann in annotations if ann["image_id"] in image_id_set]
    return split_images, split_annotations


# =========================
# Supercategorías
# =========================
def build_category_to_supercategory(categories):
    return {cat["id"]: cat["supercategory"] for cat in categories}


def assign_image_supercategory(annotations, category_to_supercategory):
    """
    Asigna a cada imagen una supercategoría dominante
    (la más frecuente dentro de sus anotaciones)
    """
    image_to_supercats = defaultdict(list)

    for ann in annotations:
        sc = category_to_supercategory.get(ann["category_id"])
        if sc is not None:
            image_to_supercats[ann["image_id"]].append(sc)

    image_main_supercat = {}
    for img_id, supercats in image_to_supercats.items():
        most_common = Counter(supercats).most_common(1)[0][0]
        image_main_supercat[img_id] = most_common

    return image_main_supercat


def stratified_split(image_ids, image_main_supercat, train_ratio, seed):
    """
    Divide manteniendo distribución por supercategoría
    """
    rng = random.Random(seed)

    groups = defaultdict(list)
    for img_id in image_ids:
        sc = image_main_supercat.get(img_id, "unknown")
        groups[sc].append(img_id)

    train_ids = set()
    val_ids = set()

    for sc, ids in groups.items():
        rng.shuffle(ids)
        cut = int(len(ids) * train_ratio)

        train_ids.update(ids[:cut])
        val_ids.update(ids[cut:])

    return train_ids, val_ids


def ensure_all_supercategories_present(train_ids, val_ids, image_main_supercat):
    """
    Asegura que TODAS las supercategorías estén en train y val
    moviendo imágenes si es necesario
    """
    def get_supercats(ids):
        return {image_main_supercat[i] for i in ids if i in image_main_supercat}

    train_sc = get_supercats(train_ids)
    val_sc = get_supercats(val_ids)

    all_sc = set(image_main_supercat.values())

    missing_train = all_sc - train_sc
    missing_val = all_sc - val_sc

    # mover de val → train
    for sc in missing_train:
        candidates = [i for i in val_ids if image_main_supercat.get(i) == sc]
        if candidates:
            chosen = candidates[0]
            val_ids.remove(chosen)
            train_ids.add(chosen)

    # mover de train → val
    for sc in missing_val:
        candidates = [i for i in train_ids if image_main_supercat.get(i) == sc]
        if candidates:
            chosen = candidates[0]
            train_ids.remove(chosen)
            val_ids.add(chosen)

    return train_ids, val_ids


# =========================
# MAIN
# =========================
def main():
    project_root = Path(__file__).resolve().parents[2]
    annotations_root = project_root / "dataset" / "annotations"

    categories_path = annotations_root / "categories.json"
    level_dir = annotations_root / LEVEL

    test_images_path = level_dir / f"test_images_{LEVEL}.json"
    test_annotations_path = level_dir / f"test_annotations_{LEVEL}.json"
    val_images_path = level_dir / f"val_images_{LEVEL}.json"
    val_annotations_path = level_dir / f"val_annotations_{LEVEL}.json"

    # Cargar datos
    categories_data = load_json(categories_path)
    test_images_data = load_json(test_images_path)
    test_annotations_data = load_json(test_annotations_path)
    val_images_data = load_json(val_images_path)
    val_annotations_data = load_json(val_annotations_path)

    categories = categories_data["categories"]
    test_images = test_images_data["images"]
    test_annotations = test_annotations_data["annotations"]

    old_val_images = val_images_data["images"]
    old_val_annotations = val_annotations_data["annotations"]

    # =========================
    # Preparar supercategorías
    # =========================
    category_to_supercategory = build_category_to_supercategory(categories)
    image_main_supercat = assign_image_supercategory(
        test_annotations, category_to_supercategory
    )

    # =========================
    # Split estratificado
    # =========================
    all_test_image_ids = [img["id"] for img in test_images]

    train_ids, val_ids = stratified_split(
        all_test_image_ids,
        image_main_supercat,
        TRAIN_RATIO,
        SEED,
    )

    # asegurar cobertura completa
    train_ids, val_ids = ensure_all_supercategories_present(
        train_ids, val_ids, image_main_supercat
    )

    # =========================
    # Construir splits
    # =========================
    train_images, train_annotations = filter_by_image_ids(
        test_images, test_annotations, train_ids
    )

    val_images, val_annotations = filter_by_image_ids(
        test_images, test_annotations, val_ids
    )

    test_final_images = old_val_images
    test_final_annotations = old_val_annotations

    # =========================
    # Guardar JSON
    # =========================
    train_coco = {
        "categories": categories,
        "images": train_images,
        "annotations": train_annotations,
    }

    val_coco = {
        "categories": categories,
        "images": val_images,
        "annotations": val_annotations,
    }

    test_coco = {
        "categories": categories,
        "images": test_final_images,
        "annotations": test_final_annotations,
    }

    out_dir = annotations_root / "splits" / LEVEL

    save_json(out_dir / "train_annotations.json", train_coco)
    save_json(out_dir / "val_annotations.json", val_coco)
    save_json(out_dir / "test_annotations.json", test_coco)

    # =========================
    # Logs
    # =========================
    print(f"[OK] LEVEL={LEVEL}")
    print(f"[OK] train images: {len(train_images)} | annotations: {len(train_annotations)}")
    print(f"[OK] val images:   {len(val_images)} | annotations: {len(val_annotations)}")
    print(f"[OK] test images:  {len(test_final_images)} | annotations: {len(test_final_annotations)}")
    print(f"[OK] output folder: {out_dir}")


if __name__ == "__main__":
    main()