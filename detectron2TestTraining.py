import os
import numpy as np
import json
from detectron2.structures import BoxMode
import itertools
import cv2
import random
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog

# write a function that loads the dataset into detectron2's standard format
def get_balloon_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for _, v in imgs_anns.items():
        record = {}

        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["height"] = height
        record["width"] = width

        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = list(itertools.chain.from_iterable(poly))

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

data_dir = "./balloon"

for d in ["train", "val"]:
    dataset_path = f"{data_dir}/{d}"
    #print(f"Registered dataset: balloon/{d} with path: {dataset_path}")
    DatasetCatalog.register("balloon/" + d, lambda d=d: get_balloon_dicts(dataset_path))
    MetadataCatalog.get("balloon/" + d).set(thing_classes=["balloon"])

balloon_metadata = MetadataCatalog.get("balloon/train")

# Load custom config
cfg = get_cfg()
cfg.merge_from_file("mycfg.yaml")
# Load weights
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01   # set the testing threshold for this model
# Set testing data-set path
cfg.DATASETS.TEST = ("balloon/val", )
cfg.MODEL.DEVICE = "cpu"
# Create predictor (model for inference)
predictor = DefaultPredictor(cfg)

# Load and visualize a few random samples from the validation set
dataset_dicts = get_balloon_dicts(data_dir + '/val/')
for d in dataset_dicts:
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)

    # Imprimez les prédictions dans le terminal
    print(outputs["instances"].pred_classes)  # classes prédites
    print(outputs["instances"].pred_boxes)  # boîtes englobantes prédites
    print(outputs)

    v = Visualizer(im[:, :, ::-1],
                   metadata=balloon_metadata,
                   scale=0.8,
                   instance_mode=ColorMode.IMAGE_BW  # supprimer les couleurs des pixels non segmentés
                   )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # Afficher l'image en utilisant cv2.imshow
    cv2.imshow("Image", v.get_image()[:, :, ::-1])
    cv2.waitKey(0)  # Attendre qu'une touche soit pressée
    cv2.destroyAllWindows()  # Fermer la fenêtre lorsqu'une touche est pressée

#image de votre choix pour tester l'algorithme
im = cv2.imread('/Users/Kevin/Downloads/dcdw1.jpg')
outputs = predictor(im)

# Imprimez les prédictions dans le terminal
print(outputs["instances"].pred_classes)  # classes prédites
print(outputs["instances"].pred_boxes)  # boîtes englobantes prédites
print(outputs)

v = Visualizer(im[:, :, ::-1],
                metadata=balloon_metadata,
                scale=0.8,
                instance_mode=ColorMode.IMAGE_BW  # supprimer les couleurs des pixels non segmentés
                )
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

# Afficher l'image en utilisant cv2.imshow
cv2.imshow("Image", v.get_image()[:, :, ::-1])
cv2.waitKey(0)  # Attendre qu'une touche soit pressée
cv2.destroyAllWindows()  # Fermer la fenêtre lorsqu'une touche est pressée