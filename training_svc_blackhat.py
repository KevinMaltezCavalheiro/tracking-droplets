# Importer les bibliothèques nécessaires
import matplotlib.pyplot as plt  # Pour afficher l'image
import numpy as np
import cv2
import time
import os
import statistics
import json
import pandas as pd
import math
import skimage  # Pour traiter l'image
from skimage.morphology import binary_dilation,remove_small_objects,binary_closing,label, closing, square, disk, binary_erosion
from skimage.measure import regionprops, find_contours
from skimage.draw import circle_perimeter
from skimage import io, color, morphology
import skimage.io
import scipy
from scipy import ndimage, stats
from scipy.stats import mode
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import functools
import trackpy as tp
import mplcursors
from matplotlib.widgets import Button
from joblib import Parallel, delayed
import threading

# Création du dictionnaire
files = {
    "video1": {
        "path": "/Volumes/Expansion/Kevin/trier/1_manip/deeper_chamber/droplet_generation/GenerationNoSurfactant1000fps.avi",
    },
    "video2": {
        "path": "/Volumes/Expansion/Kevin/trier/1_manip/chambre_normale/droplet_generation/FonctionalChip102522.avi",
    },
    "video3": {
        "path": "/Volumes/Expansion/doctorat/experiences/MaryLou/po150/pw120/pw120_1000fps.avi",
    },
    "video4": {
        "path": "/Volumes/Expansion/doctorat/experiences/MaryLou/po150/pw150/pw150_1000fps.avi",
    },
    "video5": {
        "path": "/Volumes/Expansion/doctorat/experiences/MaryLou/po150/pw163/pw163_1000fps.avi",
    },
    "video6": {
        "path": "/Volumes/Expansion/doctorat/experiences/fusion_in_chip/2023_10_10/6/2023_10_10/076A5213.MOV",
    },
    "video7": {
        "path": "/Volumes/Expansion/doctorat/experiences/fusion_in_chip/2023_10_10/7/2023_10_10/076A5219.MOV",
    },
    "video8": {
        "path": "/Volumes/Expansion/doctorat/experiences/fusion_in_chip/2023_10_10/8/2023_10_10/076A5232.MOV",
    }
}

class DetectedObject:
    def __init__(self, label, area, centroid, contour, touches_borders, circularity, centroid_validity, arearatio,
                 max_min_radius_ratio, perimeter, equivalent_radius, min_radius,
                 max_radius, cost_function_area, hierarchy, modal_radius, centroid_color,cost_function_perimeter,cost_function_radius):
        self.label = label
        self.area = area
        self.centroid = centroid
        self.contour = contour
        self.touches_borders = touches_borders
        self.circularity = circularity
        self.centroid_validity = centroid_validity
        self.arearatio = arearatio
        self.max_min_radius_ratio = max_min_radius_ratio
        self.perimeter = perimeter
        self.equivalent_radius = equivalent_radius
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.cost_function_area = cost_function_area
        self.hierarchy = hierarchy
        self.modal_radius = modal_radius
        self.centroid_color = centroid_color
        self.cost_function_perimeter = cost_function_perimeter
        self.cost_function_radius = cost_function_radius
        self.is_droplet = None

    def __str__(self):
        droplet_str = "Droplet: Yes" if self.is_droplet else "Droplet: No" if self.is_droplet is False else "Droplet: Undecided"
        return f"Label: {self.label}, Area: {self.area}, Centroid: {self.centroid}, " \
               f"Touches Borders: {self.touches_borders}, Circularity: {self.circularity}, " \
               f"Centroid Validity: {self.centroid_validity}, Area ratio hull: {self.arearatio}, " \
               f"Radii ratio: {self.max_min_radius_ratio}, " \
               f"perimeter: {self.perimeter}, Equivalent Radius: {self.equivalent_radius}, " \
               f"min radius: {self.min_radius}, max radius: {self.max_radius}, cost function area: {self.cost_function_area}," \
               f"hierarchy: {self.hierarchy}, modal radius: {self.modal_radius}, centroid color: {self.centroid_color}, " \
               f"cost function perimeter: {self.cost_function_perimeter}, cost function radius: {self.cost_function_radius}, {droplet_str}\n"

class ContourObject(DetectedObject):
    def __init__(self, label, area, centroid, contour, touches_borders, circularity, centroid_validity, arearatio,
                 max_min_radius_ratio, perimeter, equivalent_radius, min_radius,
                 max_radius, cost_function_area, hierarchy, modal_radius, centroid_color, cost_function_perimeter, cost_function_radius):
        super().__init__(label, area, centroid, contour, touches_borders, circularity, centroid_validity, arearatio,
                 max_min_radius_ratio, perimeter, equivalent_radius, min_radius,
                 max_radius, cost_function_area, hierarchy, modal_radius, centroid_color, cost_function_perimeter, cost_function_radius)

    def __str__(self):
        return super().__str__()

def save_model(model, filename='./svc/svm_model.joblib'):
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

def load_model(filename='./svc/svm_model.joblib'):
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model

def edge_detection_algorithm(images_gray, images_color, minimum_size, video, detectron2training, start_frame, entrainement, output_file):
    def process_frame(image, minimum_size, image_color, frame_index):
        def process_regions_detections(binary_image, ContourObject, image_color, frame_index):
            def calculate_circularity(area, perimeter):
                if perimeter == 0:
                    return 0  # éviter une division par zéro

                circularity = (4 * np.pi * area) / (perimeter ** 2)
                return circularity

            internal_contour_objects = []
            external_contour_objects = []

            binary_image = delete_small_spots_function(binary_image,minimum_size)
            binary_image = cv2.normalize(binary_image, None, 0, 255, cv2.NORM_MINMAX)
            binary_image = binary_image.astype(np.uint8)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image)

            for i in range(1, num_labels):
                mask = (labels == i).astype(np.uint8)
                regions_contour, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,
                                                              cv2.CHAIN_APPROX_SIMPLE)
                for j, region_contour in enumerate(regions_contour):

                    # Filtrer les formes en fonction de leur aire
                    area = cv2.contourArea(region_contour)
                    cost_function_area = -1
                    cost_function_perimeter = -1
                    cost_function_radius = -1
                    # Extraire le contour de la région
                    perimeter = cv2.arcLength(region_contour, True)
                    circularity = calculate_circularity(area, perimeter)
                    equivalent_radius = np.sqrt(area / np.pi)

                    # Vérifier si le contour touche les bords
                    touches_borders = any(
                        point[0][0] == 0 or point[0][1] == 0 or point[0][0] == binary_image.shape[1] - 1 or
                        point[0][1] == binary_image.shape[0] - 1 for point in region_contour)

                    # make convex hull around hand
                    hull = cv2.convexHull(region_contour)

                    # define area of hull and area of hand
                    areahull = cv2.contourArea(hull)

                    # find the percentage of area not covered by hand in convex hull
                    arearatio = -1
                    if areahull != 0:
                        arearatio = (1 - ((areahull - area) / areahull))

                    max_min_radius_ratio = -1
                    min_distance = -1
                    max_distance = -1
                    centroid_validity = -1
                    cx = -1
                    cy = -1
                    mode_distance = -1
                    centroid_color = [-1,-1,-1]
                    # Calculer le centroïde du contour
                    M = cv2.moments(region_contour)
                    if M["m00"] != 0:  # Éviter une division par zéro
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        inside = cv2.pointPolygonTest(region_contour, (cx, cy), measureDist=False)
                        if inside >= 0:
                            centroid_validity = 1
                            distances = [abs(cv2.pointPolygonTest(contour, (int(cx), int(cy)), True)) for contour in region_contour]
                            centroid_color = image_color[int(cy)][int(cx)]
                            min_distance = min(distances)
                            max_distance = max(distances)
                            if min_distance != 0:
                                max_min_radius_ratio = max_distance / min_distance
                                mode_distance = float(np.argmax(np.bincount(distances)))
                        else:
                            centroid_validity = 0

                    if touches_borders:
                        external_contour_object = ContourObject(label=[frame_index,i,j], area=area, centroid=[int(cx), int(cy)],
                                                                    contour=region_contour,
                                                                    touches_borders=touches_borders,
                                                                    circularity=circularity,
                                                                    centroid_validity=centroid_validity,
                                                                    arearatio=arearatio,
                                                                    max_min_radius_ratio=max_min_radius_ratio,
                                                                    perimeter=perimeter,
                                                                    equivalent_radius=equivalent_radius,
                                                                    min_radius=min_distance,
                                                                    max_radius=max_distance,
                                                                    cost_function_area = cost_function_area,
                                                                    cost_function_perimeter = cost_function_perimeter,
                                                                    cost_function_radius = cost_function_radius,
                                                                    hierarchy=hierarchy[0][j],
                                                                    modal_radius = mode_distance,
                                                                    centroid_color = centroid_color)
                        external_contour_object.is_droplet = False
                        external_contour_objects.append(external_contour_object)

                    else:
                        internal_contour_object = ContourObject(label=[frame_index,i,j], area=area, centroid=[int(cx), int(cy)],
                                                                    contour=region_contour,
                                                                    touches_borders=touches_borders,
                                                                    circularity=circularity,
                                                                    centroid_validity=centroid_validity,
                                                                    arearatio=arearatio,
                                                                    max_min_radius_ratio=max_min_radius_ratio,
                                                                    perimeter=perimeter,
                                                                    equivalent_radius=equivalent_radius,
                                                                    min_radius=min_distance,
                                                                    max_radius=max_distance,
                                                                    cost_function_area = cost_function_area,
                                                                    cost_function_perimeter = cost_function_perimeter,
                                                                    cost_function_radius = cost_function_radius,
                                                                    hierarchy=hierarchy[0][j],
                                                                    modal_radius = mode_distance,
                                                                    centroid_color = centroid_color)
                        internal_contour_objects.append(internal_contour_object)
            return internal_contour_objects, external_contour_objects

        _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

        internal_contour_objects, external_contour_objects = process_regions_detections(binary_image, ContourObject, image_color, frame_index)

        return internal_contour_objects, external_contour_objects

    def predict_droplets(final_objects):
        # Function to extract features from a DetectedObject, including internal contours
        def extract_features(droplet_objects):
            features = []
            for frame in droplet_objects:
                for droplet_object in frame:
                    if (droplet_object.is_droplet == True) or (droplet_object.is_droplet == False):
                        features.append([
                                droplet_object.area,  # 0
                                droplet_object.circularity,  # 1
                                droplet_object.centroid_validity,  # 2
                                droplet_object.arearatio,  # 3
                                droplet_object.max_min_radius_ratio,  # 4
                                droplet_object.perimeter,  # 8
                                droplet_object.equivalent_radius,  # 9
                                droplet_object.min_radius,  # 10
                                droplet_object.max_radius,  # 11
                                droplet_object.cost_function_area,  # 12
                                droplet_object.cost_function_perimeter,
                                droplet_object.cost_function_radius,
                                droplet_object.modal_radius,
                                droplet_object.centroid_color[0],
                                droplet_object.centroid_color[1],
                                droplet_object.centroid_color[2]
                                ])

            return features

        def extract_is_droplet(droplet_objects):
            features = []
            for frame in droplet_objects:
                for droplet_object in frame:
                    if (droplet_object.is_droplet == True) or (droplet_object.is_droplet == False):
                        features.append(droplet_object.is_droplet)
            return features

        # Vérifiez si les fichiers existent
        if os.path.exists('./svc/X_train_combined.joblib'):
            X_train = joblib.load('./svc/X_train_combined.joblib')
        else:
            X_train = []

        if os.path.exists('./svc/y_train_combined.joblib'):
            y_train = joblib.load('./svc/y_train_combined.joblib')
        else:
            y_train = []

        if os.path.exists('./svc/X_test_combined.joblib'):
            X_test = joblib.load('./svc/X_test_combined.joblib')
        else:
            X_test = []

        if os.path.exists('./svc/y_test_combined.joblib'):
            y_test = joblib.load('./svc/y_test_combined.joblib')
        else:
            y_test = []

        # Extract features and labels
        X = extract_features(final_objects)
        y = extract_is_droplet(final_objects)

        # Divisez les données en ensembles d'entraînement et de test
        X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X, y, test_size=0.2, random_state=42)

        # Ajoutez les nouvelles données à l'ensemble d'entraînement existant
        X_train = X_train + X_train_new
        y_train = y_train + y_train_new
        X_test = X_test + X_test_new
        y_test = y_test + y_test_new

        if entrainement == True:
            joblib.dump(X_train, './svc/X_train_combined.joblib')
            joblib.dump(y_train, './svc/y_train_combined.joblib')
            joblib.dump(X_test, './svc/X_test_combined.joblib')
            joblib.dump(y_test, './svc/y_test_combined.joblib')

            # Normalisez les caractéristiques
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Entraînez le modèle SVM
            model = SVC(kernel='linear')
            model.fit(X_train, y_train)

            # Faites des prédictions sur l'ensemble de test
            y_pred = model.predict(X_test)

            # Évaluez la performance du modèle
            accuracy = accuracy_score(y_test, y_pred)
            print(X_test)
            print(y_test)
            print(y_pred)
            print(f'Accuracy: {accuracy}')

            # Enregistrez le modèle
            save_model(model, './svc/svm_model.joblib')
            save_model(scaler, './svc/scaler.joblib')

    results = Parallel(n_jobs=-1)(delayed(process_frame)(image_gray, minimum_size, images_color[i], i)
                                  for i, image_gray in enumerate(images_gray))

    final_objects = [internal_contour_objects for internal_contour_objects, external_contour_objects in results]

    def calculate_cost_function(final_objects, minimum_size):
        areas = []
        perimeters = []
        radii = []
        for frame in final_objects:
            for droplet_object in frame:
                if droplet_object.area > minimum_size:
                    areas.append(droplet_object.area)
                    perimeters.append(droplet_object.perimeter)
                    radii.append(droplet_object.modal_radius)

        third_quantile_area = np.percentile(areas, 75)
        third_quantile_perimeter = np.percentile(perimeters, 75)
        third_quantile_radii = np.percentile(radii, 75)

        for frame in final_objects:
            for droplet_object in frame:
                droplet_object.cost_function_area = ((third_quantile_area - droplet_object.area) / third_quantile_area)
                droplet_object.cost_function_perimeter = ((third_quantile_perimeter - droplet_object.perimeter) / third_quantile_perimeter)
                droplet_object.cost_function_radius = ((third_quantile_radii - droplet_object.modal_radius) / third_quantile_radii)

    calculate_cost_function(final_objects, minimum_size)

    def display_droplet_contours(image_color, droplet_objects):
        for droplet_object in droplet_objects:
            # abc
            # if droplet_object.cost_function < -1 and droplet_object.circularity < 0.6:
            # if droplet_object.cost_function_area < -1:
            if True:
                image_draw = image_color.copy()
                cv2.drawContours(image_draw, [droplet_object.contour], 0, (0, 255, 0), 2)
                centroid = tuple(map(int, droplet_object.centroid))
                cv2.circle(image_draw, centroid, 5, (0, 255, 0), -1)

                # Afficher l'image avec les contours et le centroïde
                cv2.imshow("Droplets", image_draw)

                # Attendre l'entrée du clavier (0 signifie une attente indéfinie)
                key = cv2.waitKey(0)

                # Vérifier la touche pressée
                if key == ord('y'):  # Appuyez sur 'y' pour définir is_droplet à True
                    droplet_object.is_droplet = True
                elif key == ord('n'):  # Appuyez sur 'n' pour définir is_droplet à False
                    droplet_object.is_droplet = False
            else:
                continue

        # Fermer la fenêtre après la boucle
        cv2.destroyAllWindows()

    for i, frame in enumerate(final_objects):
        display_droplet_contours(images_color[i],frame)

    predict_droplets(final_objects)

    if detectron2training == True:
        def load_existing_data(json_file):
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    existing_data = json.load(f)
                return existing_data
            else:
                return {}

        def write_training_data_to_json(data, json_file):
            with open(json_file, 'w') as f:
                json.dump(data, f)

        # Chargez les données existantes depuis le fichier JSON
        output_json_file = output_file+"via_region_data.json"
        existing_data = load_existing_data(output_json_file)

        training_data = {}

        # Boucle sur les images colorées
        for i, image_color in enumerate(images_color):
            image_filename = video + "image" + str(start_frame+i) + ".jpg"

            # Placeholder pour les régions de chaque image
            regions = {}

            # Boucle sur les objets de gouttes (droplets)
            for j, droplet_object in enumerate(final_objects[i]):
                if droplet_object.is_droplet == True:
                    region_key = str(j)

                    # Utilisez directement les coordonnées du contour
                    x_coords, y_coords = droplet_object.contour[:, :, 0], droplet_object.contour[:, :, 1]

                    # Convertissez les coordonnées en listes Python
                    x_coords_list = x_coords.flatten().tolist()
                    y_coords_list = y_coords.flatten().tolist()

                    region_data = {
                        "shape_attributes": {
                            "name": "polygon",
                            "all_points_x": x_coords_list,
                            "all_points_y": y_coords_list
                        },
                        "region_attributes": {}
                    }

                    # Ajoutez la région au dictionnaire des régions
                    regions[region_key] = region_data

            image_size = image_color.shape[0] * image_color.shape[1]

            # Ajoutez les données d'entraînement pour cette image au dictionnaire global
            training_data[image_filename] = {
                "fileref": "",
                "size": image_size,  # Remplacez ceci par la vraie valeur de la taille
                "filename": image_filename,
                "base64_img_data": "",
                "file_attributes": {},
                "regions": regions
            }
            cv2.imwrite(output_file + image_filename, image_color)

        # Mettez à jour les données existantes avec les nouvelles données
        existing_data.update(training_data)

        # Écrivez les données d'entraînement mises à jour dans le fichier JSON
        write_training_data_to_json(existing_data, output_json_file)

def delete_small_spots_function(image_gray, minimum_size):
    # Vérifier si l'image est déjà binaire (0 ou 255)
    check_binary = np.all(np.logical_or(image_gray == 0, image_gray == 1))

    # Si l'image n'est pas déjà binaire, alors la binariser
    if not check_binary:
        image_gray = image_gray.astype(bool)
    # Supprimer les petits objets
    image_gray = remove_small_objects(image_gray, min_size=minimum_size, connectivity=1)

    image_gray = image_gray.astype(np.uint8)
    image_gray = image_gray * 255

    return image_gray

# Mettre à jour les limites lorsque l'utilisateur utilise la molette de la souris
def plot_with_zoom(image):
    def on_scroll(event):
        # Calculer le taux de zoom
        zoom_factor = 1.1 if event.button == 'up' else 1 / 1.1

        # Calculer les nouvelles limites de l'image
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
        x_width, y_height = (x_max - x_min) * zoom_factor, (y_max - y_min) * zoom_factor
        ax.set_xlim(x_center - x_width / 2, x_center + x_width / 2)
        ax.set_ylim(y_center - y_height / 2, y_center + y_height / 2)

        # Redessiner la figure
        fig.canvas.draw_idle()

    # Créer une figure avec un sous-plot
    fig, ax = plt.subplots()

    # Afficher l'image dans le sous-plot
    ax.imshow(image, cmap="gray")

    # Obtenir les dimensions de l'image
    image_width, image_height = image.shape[1], image.shape[0]

    # Définir les limites de zoom et de panoramique
    ax.set_xlim(0, image_width)
    ax.set_ylim(0, image_height)

    # Connecter la fonction on_scroll() à l'événement "scroll_event" du sous-plot
    fig.canvas.mpl_connect('scroll_event', on_scroll)

    # Afficher l'image et bloquer l'exécution du code jusqu'à ce que l'utilisateur ferme la fenêtre
    plt.imshow(image, cmap="gray")
    plt.show()

def open_file_and_load_images(path_video, video_sampling, start_frame, end_frame):
    # Ouvrir le fichier vidéo
    video_capture = cv2.VideoCapture(path_video)

    # Vérifier que la vidéo a été ouverte correctement
    if not video_capture.isOpened():
        print("Erreur lors de l'ouverture de la vidéo")
        exit()

    # Créer un tableau vide pour stocker les images
    images = []

    # Boucle tant que la vidéo est ouverte
    counter = 0
    while video_capture.isOpened():
        counter += 1
        # Si le compteur est supérieur à la dernière image souhaitée, sortir de la boucle
        if counter > end_frame:
            break

        # Si le compteur est inférieur au premier frame souhaité, passer à l'itération suivante
        if (counter <= start_frame) or (counter % video_sampling != 0):
            video_capture.grab()  # on lit le frame sans le stocker dans "frame"
            continue

        # Si le compteur est dans l'intervalle souhaité, lire et stocker l'image
        ret, frame = video_capture.read()
        if not ret:
            break
        images.append(frame)

    # Fermer la vidéo
    video_capture.release()

    # Afficher le nombre d'éléments de la liste
    print("le nombre d'images chargées depuis la vidéo est de: ", len(images))
    images_color = images.copy()
    images = skimage.color.rgb2gray(images)
    return images, images_color

def main():
    """variables principales du programme"""
    #chemins d'accès (voir dictionnaire)
    video = "video6"

    #nombre d'image prise en compte dans la vidéo, si video_sampling == 100, une image sur 100 est traitée dans le code (les vidéos sont trop grosses)
    #dans le cadre de l'entrainement, faire attention de ne pas avoir le même numéro de vidéo ET d'image
    #donc si ma video comporte 100 images et que je choisi un échantillonage à 1, on a 100 images numéroté 0,1,2,3,...
    #si je choisi un échantillonage à 10, on a 10 images numérotées 0,1,2,3,... mais correspondent en réalité à l'image 0,10,20,30,... de la video
    video_sampling = 1
    start_frame = 12
    end_frame = start_frame+1 #exemple type: np.infty == toute la video, start_frame+1 == image spécifique, nombre de votre choix

    #sauvegarde les données et entraine le modèle svm
    #rechercher "abc" dans le code, à cet endroit se trouve un if == TRUE, celui-ci peut être remplacé par les autres donnés en exemple
    #l'idée est de spécifier certaines données d'entrainement pour augmenter leurs poids, exemple ne considérer que les gouttes plus grosses que la moyenne
    entrainement = True

    #soit on enregistre les objects détectés pour l'entrainement de l'algorithme
    #soit on les enregistre pour sa phase de test et ainsi évaluer ses performances
    detectron2training = True
    output_file = "./balloon/train/"
    #output_file = "./balloon/val/"

    """début du code"""
    # Enregistrer le temps actuel avant de démarrer le code
    start_time = time.time()

    # Appel de la fonction avec le dictionnaire en argument
    path_video = files[video]["path"]

    images_gray, images_color = open_file_and_load_images(path_video,video_sampling, start_frame, end_frame)

    global compteur
    compteur_lock = threading.Lock()
    compteur = 0

    def process_image(image_gray):
        global compteur
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(image_gray.shape[0]/8), int(image_gray.shape[1]/8)))
        #rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(21, 21))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        image_gray = cv2.morphologyEx(image_gray, cv2.MORPH_BLACKHAT, rectKernel)
        image_gray = cv2.normalize(image_gray, None, 0, 255, cv2.NORM_MINMAX)
        image_gray = image_gray.astype(np.uint8)
        image_gray = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 51, -10)
        image_gray = delete_small_spots_function(image_gray, 200)
        image_gray = cv2.morphologyEx(image_gray, cv2.MORPH_CLOSE, kernel)

        # Utiliser un verrou pour incrémenter de manière sûre le compteur
        with compteur_lock:
            compteur += 1
            current_compteur = compteur
        print(f"Processing image {current_compteur}/{len(images_gray)}")

        return image_gray

    images_gray = Parallel(n_jobs=-1, require='sharedmem')(
        delayed(process_image)(image_gray) for image_gray in images_gray)

    """temps d'exécution du programme"""
    # Enregistrer le temps actuel après l'exécution du code
    end_time = time.time()
    # Calculer la durée
    duration = end_time - start_time
    # Afficher la durée
    print("temps d'exécution du code")
    print(duration)

    minimum_size = 200
    edge_detection_algorithm(images_gray, images_color, minimum_size, video, detectron2training, start_frame, entrainement, output_file)

if __name__ == "__main__":
    main()
