import numpy as np

from ..training.training_utils import predict_bboxes
from ..utils.io import load_json
from ..data.data_utils import build_image_mapping, build_category_mapping, build_supercategory_name_mapping
from .plots import draw_bboxes

from ..paths import CATEGORIES_PATH, SUPERCATEGORIES_PATH

def get_best_worst_from_cm(cm):
    correct_per_class = np.diag(cm)
    errors_per_class = cm.sum(axis=0) - correct_per_class

    best_class = int(np.argmax(correct_per_class))
    best_correct = int(correct_per_class[best_class])

    worst_class = int(np.argmax(errors_per_class))
    worst_errors = int(errors_per_class[worst_class])

    return best_class, best_correct, worst_class, worst_errors


def find_image_by_class(
    model,
    device,
    ann_data,
    target_class,
    label_encoder,
    cat_map,
    images_base_path,
    image_map,
    preprocess,
    clip_model=None,
    prediction=True
):
    image_ids = [img["id"] for img in ann_data["images"]]

    for image_id in image_ids:
        image_path = images_base_path / image_map[image_id]

        image, results = predict_bboxes(
            model,
            device,
            image_path,
            ann_data,
            image_id,
            label_encoder,
            cat_map,
            preprocess,
            clip_model
        )

        for _, true_label, pred_label in results:
            if true_label != target_class:
                continue

            if prediction and pred_label == true_label:
                return image_id

            if not prediction and pred_label != true_label:
                return image_id

    return None


def get_bboxes_by_image_id(ann_data, image_id):
    bboxes = []

    for ann in ann_data["annotations"]:
        if ann["image_id"] == image_id:
            bboxes.append(ann["bbox"])  # formato COCO: (x, y, w, h)

    return bboxes
    


def show_predictions_on_image(
    model,
    device,
    cm,
    ann_path,
    images_base_path,
    metrics_path,
    label_encoder,
    clip_model,
    preprocess
):
    best_class, _, worst_class, _ = get_best_worst_from_cm(cm)
    
    annotations = load_json(ann_path)
    categories = load_json(CATEGORIES_PATH)
    supercategories = load_json(SUPERCATEGORIES_PATH)
    
    image_map = build_image_mapping(annotations["images"])
    cat_map = build_category_mapping(categories)
    supercat_map = build_supercategory_name_mapping(supercategories)
    best_image_id = find_image_by_class(
        model,
        device,
        annotations,
        best_class,
        label_encoder,
        cat_map,
        images_base_path,
        image_map,
        preprocess,
        clip_model,
        prediction=True
    )

    worst_image_id = find_image_by_class(
        model,
        device,
        annotations,
        worst_class,
        label_encoder,
        cat_map,
        images_base_path,
        image_map,
        preprocess,
        clip_model,
        prediction=False
    )

    if best_image_id is not None:
        best_path = images_base_path / image_map[best_image_id]
        image, results = predict_bboxes(
            model,
            device,
            best_path,
            annotations,
            best_image_id,
            label_encoder,
            cat_map,
            preprocess,
            clip_model
        )
        draw_bboxes(image, results, label_encoder, supercat_map, metrics_path / "best_pred_img.png")

    if worst_image_id is not None:
        worst_path = images_base_path / image_map[worst_image_id]
        image, results = predict_bboxes(
            model,
            device,
            worst_path,
            annotations,
            worst_image_id,
            label_encoder,
            cat_map,
            preprocess,
            clip_model
        )
        draw_bboxes(image, results, label_encoder, supercat_map, metrics_path / "worst_pred_img.png")