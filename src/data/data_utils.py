def build_category_mapping(cats_json):
    return {
        c["id"]: c["supercat_id"]
        for c in cats_json["categories"]
    }


def build_image_mapping(images):
    return {
        img["id"]: img["file_name"]
        for img in images
    }