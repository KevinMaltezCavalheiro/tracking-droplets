# import some common detectron2 utilities
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2

#ce script utilise le modèle pré entrainé de facebook sur les images de notre choix /!\ les objects détectés sont déjà définis.

# get image
im = cv2.imread("/Users/Kevin/Downloads/IMG_1918.jpg")

# Create config
cfg = get_cfg()
cfg.merge_from_file("/Users/Kevin/Downloads/mask_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1  # set threshold for this model
cfg.MODEL.WEIGHTS = "/Users/Kevin/Downloads/model_final_a3ec72.pkl"
# Set the device to CPU
cfg.MODEL.DEVICE = "cpu"

# Create predictor
predictor = DefaultPredictor(cfg)

# Make prediction
outputs = predictor(im)

# Visualize the prediction
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow("Prediction", out.get_image()[:, :, ::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()
