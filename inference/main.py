import os
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from models.model import Model
from inference.save_predictions import SaveNPZFormat


def parse_arguments():
    """
    Parse the command line arguments
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--models", required=False, default="mdef_detr",
                    help="The models to be used for performing inference. Available options are,"
                         "['mdef_detr','mdef_detr_minus_language']")
    ap.add_argument("-i", "--input_images_dir_path", required=True,
                    help="The path to input images directory on which to run inference.")
    ap.add_argument("-c", "--model_checkpoints_path", required=False, default=None,
                    help="The path to models checkpoints. Required for all models except mdetr.")
    ap.add_argument("-tq", "--text_query", required=False, default="all objects",
                    help="The text query to be used in case of MViTs.")
    ap.add_argument("--multi_crop", action='store_true', help="Either to perform multi-crop inference or not. "
                                                              "Multi-crop inference is used only for DOTA dataset.")

    args = vars(ap.parse_args())

    return args


def run_inference(model, images_dir, output_path, caption=None, multi_crop=False):
    images_already_inferenced = []
    max_counter = 0
    for file in Path(output_path).parent.iterdir():
        if file.is_file() and file.name.endswith('.npz'):
            with open(file, "rb") as f:
                images_already_inferenced += list(np.load(f).keys())
            max_counter = max(max_counter, int(file.name.rsplit("_", 1)[-1].split(".", 1)[0]))

    images_already_inferenced = set(images_already_inferenced)
    images = os.listdir(images_dir)
    images = [image for image in images if image.split('.', 1)[0] not in images_already_inferenced]
    del images_already_inferenced

    dumper = SaveNPZFormat()
    dumper.counter = max_counter + 1
    detections = {}
    for i, image_name in enumerate(tqdm(images)):
        if i > 0 and i % 1000 == 0:
            dumper.update(detections)
            dumper.save(output_path)
            detections = {}
        image_path = f"{images_dir}/{image_name}"
        try:
            if multi_crop:
                detections[image_name.split('.')[0]] = model.infer_image_multi_crop(image_path, caption=caption)
            else:
                detections[image_name.split('.')[0]] = model.infer_image(image_path, caption=caption)
        except Exception as e:
            print(e, image_name)
    dumper.update(detections)
    dumper.save(output_path)


def main():
    # Parse arguments
    args = parse_arguments()
    model_name = args["models"]
    images_dir = args["input_images_dir_path"]
    checkpoints_path = args["model_checkpoints_path"]
    text_query = args["text_query"]
    multi_crop = args["multi_crop"]
    model = Model(model_name, checkpoints_path).get_model()
    output_dir = f"result/{model_name}/"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/result"
    run_inference(model, images_dir, output_path, caption=text_query, multi_crop=multi_crop)


if __name__ == "__main__":
    main()
