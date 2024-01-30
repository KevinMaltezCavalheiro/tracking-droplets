# Importer les bibliothèques nécessaires
import matplotlib.pyplot as plt  # Pour afficher l'image
import scipy
import numpy as np
import skimage  # Pour traiter l'image
from skimage import data, filters, feature, measure, draw  # Pour utiliser les fonctions de prefiltrage et de détection de contours de skimage
from skimage.morphology import binary_dilation,remove_small_objects,binary_closing,label, closing, square, disk, binary_erosion
from skimage.measure import regionprops, find_contours
from skimage import exposure
from skimage.draw import circle_perimeter
from skimage.feature import peak_local_max
import skimage.io
from scipy import ndimage, stats
from scipy.stats import mode
from scipy.signal import argrelextrema, find_peaks, peak_prominences, savgol_filter
import cv2
import time
import os
import statistics
import ffmpeg
from sklearn.preprocessing import normalize
from scipy.optimize import curve_fit
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MeanShift
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import OPTICS
import hdbscan
import json
from scipy.stats import zscore
import ruptures as rpt
import pywt
import pandas as pd
from scipy.ndimage import generic_filter1d
import matplotlib.patches as patches
import math
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle
import joblib
import functools

# Création du dictionnaire
files = {
    "video1": {
        "path": "/Volumes/Expansion/Kevin/trier/1_manip/deeper_chamber/droplet_generation/GenerationNoSurfactant1000fps.avi",
        "median_image": "/Volumes/Expansion/Kevin/trier/1_manip/deeper_chamber/droplet_generation/image_mediane.jpg",
        "mode_image": "/Volumes/Expansion/Kevin/trier/1_manip/deeper_chamber/droplet_generation/mode.jpg",
        "path_save": "/Volumes/Expansion/Kevin/trier/1_manip/deeper_chamber/droplet_generation/"
    },
    "video2": {
        "path": "/Volumes/Expansion/Kevin/trier/1_manip/chambre_normale/droplet_generation/FonctionalChip102522.avi",
        "median_image": "/Volumes/Expansion/Kevin/trier/1_manip/chambre_normale/droplet_generation/image_mediane.jpg",
        "mode_image": "/Volumes/Expansion/Kevin/trier/1_manip/chambre_normale/droplet_generation/mode.jpg",
        "path_save": "/Volumes/Expansion/Kevin/trier/1_manip/chambre_normale/droplet_generation/"
    },
    "video3": {
        "path": "/Volumes/Expansion/doctorat/experiences/MaryLou/po150/pw120/pw120_1000fps.avi",
        "median_image": "/Volumes/Expansion/doctorat/experiences/MaryLou/po150/pw120/image_mediane.jpg",
        "mode_image": "/Volumes/Expansion/doctorat/experiences/MaryLou/po150/pw120/mode.jpg",
        "path_save": "/Volumes/Expansion/doctorat/experiences/MaryLou/po150/pw120/"
    },
    "video4": {
        "path": "/Volumes/Expansion/doctorat/experiences/MaryLou/po150/pw150/pw150_1000fps.avi",
        "median_image": "/Volumes/Expansion/doctorat/experiences/MaryLou/po150/pw150/image_mediane.jpg",
        "mode_image": "/Volumes/Expansion/doctorat/experiences/MaryLou/po150/pw150/mode.jpg",
        "path_save": "/Volumes/Expansion/doctorat/experiences/MaryLou/po150/pw150/"
    },
    "video5": {
        "path": "/Volumes/Expansion/doctorat/experiences/MaryLou/po150/pw163/pw163_1000fps.avi",
        "median_image": "/Volumes/Expansion/doctorat/experiences/MaryLou/po150/pw163/image_mediane.jpg",
        "mode_image": "/Volumes/Expansion/doctorat/experiences/MaryLou/po150/pw163/mode.jpg",
        "path_save": "/Volumes/Expansion/doctorat/experiences/MaryLou/po150/pw163/"
    },
    "video6": {
        "path": "/Volumes/Expansion/doctorat/experiences/fusion_in_chip/2023_10_10/6/2023_10_10/076A5213.MOV",
        "median_image": "/Volumes/Expansion/doctorat/experiences/fusion_in_chip/2023_10_10/6/2023_10_10/image_mediane.jpg",
        "mode_image": "/Volumes/Expansion/doctorat/experiences/fusion_in_chip/2023_10_10/6/2023_10_10/mode.jpg",
        "path_save": "/Volumes/Expansion/doctorat/experiences/fusion_in_chip/2023_10_10/6/2023_10_10/"
    },
    "video7": {
        "path": "/Volumes/Expansion/doctorat/experiences/fusion_in_chip/2023_10_10/7/2023_10_10/076A5219.MOV",
        "median_image": "/Volumes/Expansion/doctorat/experiences/fusion_in_chip/2023_10_10/7/2023_10_10/image_mediane.jpg",
        "mode_image": "/Volumes/Expansion/doctorat/experiences/fusion_in_chip/2023_10_10/7/2023_10_10/mode.jpg",
        "path_save": "/Volumes/Expansion/doctorat/experiences/fusion_in_chip/2023_10_10/7/2023_10_10/"
    },
    "video8": {
        "path": "/Users/Kevin/Downloads/puits7min_5X_9_10.MOV",
        "median_image": "/Users/Kevin/Downloads/image_mediane.jpg",
        "mode_image": "/Users/Kevin/Downloads/mode.jpg",
        "path_save": "/Users/Kevin/Downloads/"
    },
    "video9": {
        "path": "/Users/Kevin/Downloads/puits7min_5X_9_8.MOV",
        "median_image": "/Users/Kevin/Downloads/image_mediane.jpg",
        "mode_image": "/Users/Kevin/Downloads/mode.jpg",
        "path_save": "/Users/Kevin/Downloads/"
    },
    "video10": {
        "path": "/Volumes/Expansion/doctorat/experiences/fusion_in_chip/2023_10_10/8/2023_10_10/076A5232.MOV",
        "median_image": "/Volumes/Expansion/doctorat/experiences/fusion_in_chip/2023_10_10/8/2023_10_10/image_mediane.jpg",
        "mode_image": "/Volumes/Expansion/doctorat/experiences/fusion_in_chip/2023_10_10/8/2023_10_10/mode.jpg",
        "path_save": "/Volumes/Expansion/doctorat/experiences/fusion_in_chip/2023_10_10/8/2023_10_10/"
    }
}

class DetectedObject:
    def __init__(self, label, area, centroid, contour, touches_borders, circularity, centroid_validity, arearatio,
                 max_min_radius_ratio, perimeter, equivalent_radius, min_radius,
                 max_radius, cost_function):
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
        self.cost_function = cost_function
        self.is_droplet = None

    def __str__(self):
        droplet_str = "Droplet: Yes" if self.is_droplet else "Droplet: No" if self.is_droplet is False else "Droplet: Undecided"
        return f"Label: {self.label}, Area: {self.area}, Centroid: {self.centroid}, " \
               f"Touches Borders: {self.touches_borders}, Circularity: {self.circularity}, " \
               f"Centroid Validity: {self.centroid_validity}, Area ratio hull: {self.arearatio}, " \
               f"Radii ratio: {self.max_min_radius_ratio}, " \
               f"perimeter: {self.perimeter}, Equivalent Radius: {self.equivalent_radius}, " \
               f"min radius: {self.min_radius}, max radius: {self.max_radius}, cost function: {self.cost_function}," \
               f" {droplet_str}\n"

class ContourObject(DetectedObject):
    def __init__(self, label, area, centroid, contour, touches_borders, circularity, centroid_validity, arearatio,
                 max_min_radius_ratio, perimeter, equivalent_radius, min_radius,
                 max_radius, cost_function):
        super().__init__(label, area, centroid, contour, touches_borders, circularity, centroid_validity, arearatio,
                 max_min_radius_ratio, perimeter, equivalent_radius, min_radius,
                 max_radius, cost_function)

    def __str__(self):
        return super().__str__()

def save_model(model, filename='./svm_model.joblib'):
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

def load_model(filename='./svm_model.joblib'):
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model

def gradient_background(image, kernel_size):
    # Appliquer un flou gaussien avec un noyau de 50x50
    kernel = (kernel_size, kernel_size)
    sigma = kernel_size * np.exp(-0.5)  # Vous pouvez ajuster sigma selon vos besoins

    image = cv2.GaussianBlur(image, kernel, sigma)

    return image

def edge_detection_algorithm(images, images_color, background_image, minimum_size, images_gray, background_image_gray):
    def process_regions_detections(binary_image, ContourObject, inernal_contour_objects, external_contour_objects, image_color, image, image_gray):
        def calculate_circularity(area,perimeter):
            if perimeter == 0:
                return 0  # éviter une division par zéro

            circularity = (4 * np.pi * area) / (perimeter ** 2)
            return circularity

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image)

        for i in range(1, num_labels):
            mask = (labels == i).astype(np.uint8)
            regions_contour, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,
                                                          cv2.CHAIN_APPROX_SIMPLE)
            for region_contour in regions_contour:

                # Filtrer les formes en fonction de leur aire
                area = cv2.contourArea(region_contour)
                cost_function = -1
                # Extraire le contour de la région
                perimeter = cv2.arcLength(region_contour, True)
                circularity = calculate_circularity(area,perimeter)
                equivalent_radius = np.sqrt(area/np.pi)

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
                    arearatio = (1-((areahull - area) / areahull))

                    max_min_radius_ratio = -1
                    min_distance = -1
                    max_distance = -1
                    centroid_validity = -1
                    cx=-1
                    cy=-1
                    # Calculer le centroïde du contour
                    M = cv2.moments(region_contour)
                    if M["m00"] != 0:  # Éviter une division par zéro
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        inside = cv2.pointPolygonTest(region_contour, (cx, cy), measureDist=False)
                        if inside >= 0:
                            centroid_validity = 1
                            distances = [abs(cv2.pointPolygonTest(contour, (int(cx), int(cy)), True)) for contour in region_contour]
                            min_distance = min(distances)
                            max_distance = max(distances)
                            if min_distance != 0:
                                max_min_radius_ratio = max_distance/min_distance
                        else:
                            centroid_validity = 0

                    if touches_borders:
                        external_contour_object = ContourObject(label=i, area=area, centroid=[int(cx),int(cy)], contour=region_contour,
                                               touches_borders=touches_borders, circularity=circularity,
                                               centroid_validity=centroid_validity, arearatio=arearatio,
                                               max_min_radius_ratio=max_min_radius_ratio,
                                               perimeter=perimeter, equivalent_radius=equivalent_radius,
                                               min_radius=min_distance,
                                               max_radius=max_distance, cost_function=cost_function)
                        external_contour_object.is_droplet = False
                        external_contour_objects.append(external_contour_object)

                    else:
                        internal_contour_object = ContourObject(label=i, area=area, centroid=[int(cx),int(cy)],
                                                   contour=region_contour,
                                                   touches_borders=touches_borders, circularity=circularity,
                                                   centroid_validity=centroid_validity, arearatio=arearatio,
                                                   max_min_radius_ratio=max_min_radius_ratio,
                                                   perimeter=perimeter, equivalent_radius=equivalent_radius,
                                                   min_radius=min_distance,
                                                   max_radius=max_distance, cost_function=cost_function)
                        internal_contour_objects.append(internal_contour_object)

        # image_draw = image_color.copy()
        #
        # # Dessinez le contour
        # for internal_contour_object in internal_contour_objects:
        #     cv2.drawContours(image_draw, [internal_contour_object.contour], 0, (0, 255, 0), 2)
        #
        # for external_contour_object in external_contour_objects:
        #     cv2.drawContours(image_draw, [external_contour_object.contour], 0, (0, 0, 255), 2)
        #
        # # Affichez l'image
        # cv2.imshow("Droplets", image_draw)
        #
        # cv2.waitKey(0)
        #
        # # Fermer la fenêtre après la boucle
        # cv2.destroyAllWindows()

    def display_droplet_contours(image_color, droplet_objects):
        for droplet_object in droplet_objects:
            # if droplet_object.cost_function < -1 and droplet_object.circularity < 0.6:
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

    final_objects_edges = []
    final_objects = []

    for i, image in enumerate(images):
        internal_contour_objects = []
        external_contour_objects = []

        image = image + background_image

        # Convertir en image binaire inversée
        _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

        process_regions_detections(binary_image, ContourObject, internal_contour_objects, external_contour_objects, images_color[i], image, images_gray[i])

        final_objects_edges.append(external_contour_objects)
        final_objects.append(internal_contour_objects)

    areas = []
    for frame in final_objects:
        for droplet_object in frame:
            if(droplet_object.area > minimum_size):
                areas.append(droplet_object.area)

    # Calcul du troisième quantile (quartile supérieur) des aires
    third_quantile_area = np.percentile(areas, 75)

    for frame in final_objects:
        for droplet_object in frame:
            droplet_object.cost_function = ((third_quantile_area - droplet_object.area) / (third_quantile_area))

    for i, frame in enumerate(final_objects):
        display_droplet_contours(images_color[i],frame)

    for frame in final_objects:
        for droplet_object in frame:
            print(droplet_object)

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
                        droplet_object.cost_function  # 12
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
    if os.path.exists('./X_train_combined.joblib'):
        X_train = joblib.load('./X_train_combined.joblib')
    else:
        X_train = []

    if os.path.exists('./y_train_combined.joblib'):
        y_train = joblib.load('./y_train_combined.joblib')
    else:
        y_train = []

    if os.path.exists('./X_test_combined.joblib'):
        X_test = joblib.load('./X_test_combined.joblib')
    else:
        X_test = []

    if os.path.exists('./y_test_combined.joblib'):
        y_test = joblib.load('./y_test_combined.joblib')
    else:
        y_test = []

    # Extract features and labels
    X = extract_features(final_objects)
    y = extract_is_droplet(final_objects)

    #Divisez les données en ensembles d'entraînement et de test
    X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X, y, test_size=0.2, random_state=42)

    #Ajoutez les nouvelles données à l'ensemble d'entraînement existant
    X_train = X_train + X_train_new
    y_train = y_train + y_train_new
    X_test = X_test + X_test_new
    y_test = y_test + y_test_new

    # joblib.dump(X_train, './X_train_combined.joblib')
    # joblib.dump(y_train, './y_train_combined.joblib')
    # joblib.dump(X_test, './X_test_combined.joblib')
    # joblib.dump(y_test, './y_test_combined.joblib')

    entrainement = False
    if entrainement == True:
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
        save_model(model, './svm_model.joblib')
        save_model(scaler, './scaler.joblib')

    return images, images_color, final_objects

def fill_holes(binary_image):
    # Trouver les contours des objets
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Créer une image noire pour le remplissage
    filled_image = np.zeros_like(binary_image)

    # Remplir les contours
    cv2.drawContours(filled_image, contours, -1, 255, thickness=cv2.FILLED)

    return filled_image

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

def plot_distribution(data, title, xlabel):
    plt.figure()
    plt.hist(data, bins=20, color='blue', edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.show()

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

def open_file_and_load_images(path_video, video_echantillonage, debut_frame, fin_frame):
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
        if counter > fin_frame:
            break

        # Si le compteur est inférieur au premier frame souhaité, passer à l'itération suivante
        if (counter <= debut_frame) or (counter % video_echantillonage != 0):
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

def tracking_cercle_show(images):
    """
               affiches les cercles sélectionnées sur une vidéo et affiches la trajectoire
    """
    min_radius = 5
    max_radius = 1000
    area_max = 1000000

    # Initialiser la liste pour les positions des contours circulaires
    positions = []

    for image in images:

        # Applique un seuil pour masquer les petits changements
        #thresh = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)[1]

        thresh = image

        #plot_with_zoom(thresh)

        # Trouve les contours de l'image thresholdée
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Boucle sur les contours
        for c in contours:
            # Calcule le cercle englobant
            (x, y), radius = cv2.minEnclosingCircle(c)
            # ignore les cercles trop petits ou trop grands
            if radius < min_radius or radius > max_radius:
                continue

            # ignore les cercles trop grands
            if cv2.contourArea(c) > area_max:
                continue

            # Dessine le cercle englobant autour de la bille
            cv2.circle(image, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            positions.append((int(x), int(y)))

        # Affiche l'image
        plot_with_zoom(image)

def background_operation (images, substract_background, median_or_mode_or_gaussianblur, path_median, path_mode, diff_or_absdiff_or_inverse, kernel_size):
    def load_background_image(path_background):
        # Lire l'image médiane à partir du disque
        # median_image = skimage.io.imread(path_median)
        background_image = cv2.imread(path_background)
        background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY)
        return background_image

    def evaluate_and_save_median(images, path_median=None):
        if path_median and os.path.exists(path_median):
            # Si le chemin vers l'image du mode existe, chargez-la
            median_image_uint8 = cv2.imread(path_median)
        else:
            # Calculer l'image médiane en utilisant la fonction median() de numpy
            median_image = np.median(images, axis=0)

            # Normaliser l'image médiane en utilisant la fonction normalize de OpenCV
            median_image_uint8 = cv2.normalize(median_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                               dtype=cv2.CV_8U)

            # Sauvegardez l'image du mediane si un chemin est spécifié
            if path_median:
                skimage.io.imsave(path_median, median_image_uint8)

        return median_image_uint8

    def evaluate_and_save_mode(images, path_mode=None):
        if path_mode and os.path.exists(path_mode):
            # Si le chemin vers l'image du mode existe, chargez-la
            background_mode = cv2.imread(path_mode)
        else:
            # Sinon, calculez le mode
            # Empilez les images en un tableau 4D (nombre d'images x hauteur x largeur x canaux)
            image_stack = np.stack(images, axis=0)

            # Calculez le mode pour chaque canal de couleur (R, G, B)
            background_mode = mode(image_stack, axis=0, keepdims=True)

            # Convertissez le résultat en un tableau 3D (hauteur x largeur x canaux)
            background_mode = background_mode.mode[0]

            # Sauvegardez l'image du mode si un chemin est spécifié
            if path_mode:
                skimage.io.imsave(path_mode, background_mode)

        return background_mode.astype(np.uint8)

    def background_evaluation_and_substraction(substract_background, path, images,
                                               diff_or_absdiff_or_inverse):
        def substract_background_func(path_background, images, diff_or_absdiff_or_inverse):
            # Vérifier l'existence du fichier
            if os.path.exists(path_background):
                # Le fichier existe, on charge l'image de la médiane
                print("Le fichier background existe")
                background_image = load_background_image(path_background)
                background_image = (background_image - np.min(background_image)) / (np.max(background_image) - np.min(background_image))
                background_image = np.uint8(background_image * 255)
            else:
                # Le fichier n'existe pas, on crée le ficher de la médiane et on le charge
                print("Le fichier background n'existe pas, création et chargement du fichier")
                if median_or_mode_or_gaussianblur == 0:
                    evaluate_and_save_median(images, path_background)
                elif median_or_mode_or_gaussianblur == 1:
                    evaluate_and_save_mode(images, path_background)
                background_image = load_background_image(path_background)
                background_image = (background_image - np.min(background_image)) / (np.max(background_image) - np.min(background_image))
                background_image = np.uint8(background_image * 255)

            # Calculer la différence entre chaque image et l'image médiane
            if diff_or_absdiff_or_inverse == 2:
                # première possibilité
                for i, image in enumerate(images):
                    images[i] = cv2.absdiff(image, background_image)
            elif diff_or_absdiff_or_inverse == 1:
                # deuxième possibilité
                for i, image in enumerate(images):
                    images[i] = np.subtract(image, background_image)
            elif diff_or_absdiff_or_inverse == 3:
                # deuxième possibilité
                for i, image in enumerate(images):
                    images[i] = np.subtract(image, background_image)
                    images[i] = cv2.bitwise_not(images[i])
            return images, background_image

        if substract_background == 1:
            print("on applique la soustraction du background")
            images, background_image = substract_background_func(path, images, diff_or_absdiff_or_inverse)
            return images, background_image
        else:
            print("on n'effectue pas la soustraction du background")
            background_image = None
            return images, background_image

    if substract_background == 1:
        if median_or_mode_or_gaussianblur == 0:
            path = path_median
            images, background_image = background_evaluation_and_substraction(substract_background, path, images, diff_or_absdiff_or_inverse)

        elif median_or_mode_or_gaussianblur == 1:
            path = path_mode
            images, background_image = background_evaluation_and_substraction(substract_background, path, images, diff_or_absdiff_or_inverse)
        else:
            background_image = np.uint8(gradient_background(images[0], kernel_size))
            for i, image in enumerate(images):
                images[i] = cv2.absdiff(image, background_image)
    else:
        background_image = None
    return images, background_image

def saving_image_gray (images):
    # Convertir l'image en uint8
    image_uint8 = (images[0] * 255).astype(np.uint8)
    # Enregistrer l'image en tant que fichier PNG
    cv2.imwrite(path_save + "grey_image.png", image_uint8)

def process_images(images, binary_threshold, threshold, prefiltrage, postfiltrage,
                   dilatation_and_contraction, fill_holes, delete_small_spots, rayon_pinceau, minimum_size):
    def preprocessing_image(image_gray, prefiltrage, postfiltrage, dilatation_and_contraction,
                            fill_holes, delete_small_spots, rayon_pinceau, minimum_size):
        def filtrage(image_gray, pre_or_post_filtrage):
            if pre_or_post_filtrage == 1:
                kernel = np.array([[1, 1, 1],
                                   [1, 1, 1],
                                   [1, 1, 1]])

                # Normaliser le noyau de convolution
                kernel = kernel / np.sum(kernel)

                # Appliquer le filtre de convolution
                image_gray = scipy.signal.convolve2d(image_gray, kernel, mode='same')
            elif pre_or_post_filtrage == 2:
                image_gray = cv2.GaussianBlur(image_gray, (9, 9), 0)
            elif pre_or_post_filtrage == 3:
                image_gray = cv2.convertScaleAbs(image_gray, alpha=255)
                image_gray = cv2.medianBlur(image_gray, 7)
            elif pre_or_post_filtrage == 4:
                image_gray = cv2.convertScaleAbs(image_gray, alpha=255)
                image_gray = cv2.bilateralFilter(image_gray, 11, 75, 75)
            elif pre_or_post_filtrage == 5:
                # Appliquer le seuillage par égalisation des histogrammes
                image_gray = skimage.exposure.equalize_hist(image_gray)
                # Convertir l'image en binaire
                threshold = skimage.filters.threshold_otsu(image_gray)
                image_gray = image_gray > threshold
                image_gray = image_gray.astype(np.uint8) * 255
            elif pre_or_post_filtrage == 6:
                # Appliquer le seuillage par Otsu
                threshold = skimage.filters.threshold_otsu(image_gray)
                image_gray = image_gray > threshold
                image_gray = image_gray.astype(np.uint8) * 255
            elif pre_or_post_filtrage == 7:
                # Calculer la moyenne des valeurs de pixel de l'image
                mean_value = np.mean(image_gray)
                # Appliquer le seuillage par moyenne
                image_gray = image_gray > mean_value
                image_gray = image_gray.astype(np.uint8) * 255

            return image_gray

        """partie traitement d'image"""
        image_gray = filtrage(image_gray, prefiltrage)

        if delete_small_spots == 1:

            image_gray = delete_small_spots_function(image_gray,minimum_size)

        if dilatation_and_contraction == 1:
            # Opération de dilatation
            # rayon du pinceau pour la dilatation et l'érosion, représente l'épaisseur du trait
            selem = disk(rayon_pinceau)
            dilated_image = binary_dilation(image_gray, selem)

            # Opération d'érosion
            image_gray = binary_erosion(dilated_image, selem)

        if fill_holes == 1:
            # Remplir les trous en utilisant binary_fill_holes
            image_gray = ndimage.binary_fill_holes(image_gray)

        image_gray = image_gray.astype(np.uint8)
        image_gray = image_gray * 255
        image_gray = filtrage(image_gray, postfiltrage)
        image_gray = image_gray.astype(np.uint8)

        return image_gray

    # Initialiser une liste vide
    processed_images = []
    for image in images:

        if binary_threshold == 1:
            # Convertir en niveaux de gris
            _, image_gray = cv2.threshold(image, 255 * threshold, 255, cv2.THRESH_BINARY)
        else:
            image_gray = image

        # détecte les contours de l'image, rempli les gouttes en blanc et le reste en noir, trouve les cercles sur l'image
        image_gray = preprocessing_image(image_gray, prefiltrage, postfiltrage, dilatation_and_contraction,
                                         fill_holes, delete_small_spots, rayon_pinceau,
                                         minimum_size)

        # Ajouter l'image traitée à la liste
        processed_images.append(image_gray)
    return processed_images

def sub_plot_img(ax, title, img):
    ax.imshow(img, cmap='gray')
    ax.set_title(title)

def main():
    """variables principales du programme"""
    #chemins d'accès (voir dictionnaire)
    video = "video6"

    # lire ou écrire les valeurs propres à la vidéo dans un fichier
    load_value = 0
    write_value = 0
    save_plots = 0
    assert not (load_value == 1 and write_value == 1), "load_value and write_value should not be both equal to 1"

    debug_image = 0

    #nombre d'image prise en compte dans la vidéo, si video_echantillonage == 100, une image sur 100 est traitée dans le code (les vidéos sont trop grosses)
    video_echantillonage = 1
    debut_frame = 200
    fin_frame = 252
    #scale
    pixel_per_micrometer = 0.5
    #pour l'analyse, on veut des contours de droplet blanc sur fond noir. On s'assure d'avoir cette image en évaluant la moyenne des pixels de l'image[0], si mean > 127 => image composée majoritairement de blanc => inverse toutes les images de la video
    auto_invert = 1

    #paramètres pour le traitement d'image
    #soustraction de lu backgroung 0: non, 1: oui
    substract_background = 1

    binary_threshold = 1
    dilatation_and_contraction = 1
    fill_holes = 0
    #suppression des taches si elles ont un nombre de pixel inférieur à minimum size
    delete_small_spots = 1
    #check si le cercle est rempli à 80% de blanc (à modifier dans process_images), si oui goutte, si non pas goutte
    circle_fill_check = 0

    """début du code"""
    # Enregistrer le temps actuel avant de démarrer le code
    if load_value == 0:
        start_time = time.time()

        # fig, axes = plt.subplots(2, 3, figsize=(15, 7))
        # ax1 = axes[0, 0]
        # ax2 = axes[0, 1]
        # ax3 = axes[0, 2]
        # ax4 = axes[1, 0]
        # ax5 = axes[1, 1]
        # ax6 = axes[1, 2]

        # Appel de la fonction avec le dictionnaire en argument
        path_video = files[video]["path"]
        path_median = files[video]["median_image"]
        path_mode = files[video]["mode_image"]
        path_save = files[video]["path_save"]

        images, images_color = open_file_and_load_images(path_video,video_echantillonage, debut_frame, fin_frame)
        # sub_plot_img(ax1,"First image in color",images_color[0])

        if save_plots == 1:
            saving_image_gray(images)

        """soustraction du background"""
        #utilisation de la différence ou de la difference absolue, la soustraction avec la médiane se fait de manière classique (1): 200 - 230 = -30 -> 255 - 30 = 225  ou en valeur absolue(2): 200 - 230 = -30 -> 30 ou inversé (3): inverse classique
        diff_or_absdiff_or_inverse = 2
        # 0: median, 1: mode, 2: gaussian blur
        median_or_mode_or_gaussianblur = 1
        # la taille du noyau gaussien, doit être impaire
        kernel_size = 301

        images = [(image - np.min(image)) / (np.max(image) - np.min(image)) for image in images]
        images = [np.uint8(image * 255) for image in images]
        images_gray = images.copy()
        images, background_image = background_operation(images, substract_background, median_or_mode_or_gaussianblur, path_median, path_mode,
                                                    diff_or_absdiff_or_inverse, kernel_size)
        background_image_gray = None
        if background_image is not None:
            # sub_plot_img(ax2,"Background",background_image)
            background_image_gray = background_image
            background_image_gray = (background_image_gray - np.min(background_image_gray)) / (
                        np.max(background_image_gray) - np.min(background_image_gray))
            background_image_gray = np.uint8(background_image_gray * 255)
            background_image_gaussian_kernel = np.uint8(gradient_background(background_image, kernel_size))
            background_image = cv2.absdiff(background_image, background_image_gaussian_kernel)
            background_image = (background_image - np.min(background_image)) / (np.max(background_image) - np.min(background_image))
            background_image = np.uint8(background_image * 255)
        # sub_plot_img(ax3,"image after background substraction",images[0])

        """image processing pour mettre en évidence les droplets"""
        # différents type de pre et post filtrage possible
        # 0 == rien
        # 1 == par convolution d'un noyau gaussien
        # 2 == GaussianBlur
        # 3 == medianBlur
        # 4 == bilateralFilter
        # 5 == Appliquer le seuillage par égalisation des histogrammes
        # 6 == Appliquer le seuillage par Otsu
        # 7 == Calculer la moyenne des valeurs de pixel de l'image
        prefiltrage = 1
        postfiltrage = 0
        # pinceau avec lequel on dialte et erode en pixel
        rayon_pinceau = 3
        # delete_small_spots: Définir le seuil de taille de connectivité à utiliser (en pixels), il s'agit du seuil permettant de supprimer les taches trop petites pour être des gouttes
        minimum_size = 500
        #pourcentage de pixels blanc dans un cercle qui a été fill pour qu'il soit considéré comme valide
        white_ratio_threshold = 0.9
        # Appliquez un seuil de 0.1 lors de la binarisation (utiliser binary_threshold)
        threshold = 0.1
        if auto_invert == 1:
            mean_value = np.mean(images[0])
            if mean_value > 127:
                images = [cv2.bitwise_not(image) for image in images]
                background_image = cv2.bitwise_not(background_image)

        processed_images = process_images(images, binary_threshold, threshold, prefiltrage, postfiltrage, dilatation_and_contraction,
                           fill_holes, delete_small_spots, rayon_pinceau, minimum_size)

        background_image = process_images([background_image], binary_threshold, threshold, prefiltrage, postfiltrage, dilatation_and_contraction,
                           fill_holes, delete_small_spots, rayon_pinceau, minimum_size)
        background_image = background_image[0]

        #processed_images = [processed_image + background_image for processed_image in processed_images]

        # sub_plot_img(ax4,"processed image",processed_images[0])
        processed_images, images_color, final_objects = edge_detection_algorithm(processed_images, images_color, background_image, minimum_size, images_gray, background_image_gray)
        # sub_plot_img(ax5,"hough algorithm on gray image",processed_images[0])
        # sub_plot_img(ax6,"hough algorithm on color image",images_color[0])

        """temps d'exécution du programme"""
        # Enregistrer le temps actuel après l'exécution du code
        end_time = time.time()
        # Calculer la durée
        duration = end_time - start_time
        # Afficher la durée
        print("temps d'exécution du code")
        print(duration)

        """partie plot"""
        # À la fin de votre fonction main, vous n'avez pas besoin de plt.show() car vous avez déjà créé la figure
        plt.tight_layout()
        plt.show()

        if debug_image == 1:
            for i, img in enumerate(processed_images):
                cv2.imshow(f"Image {i}", img)
                cv2.waitKey(0)  # Attendez que l'utilisateur appuie sur une touche
                cv2.destroyAllWindows()  # Fermez toutes les fenêtres OpenCV

                # Sauvegardez l'image dans le répertoire spécifié par path_save
                save_path = os.path.join(files["video7"]["path_save"], f"image_{i}.jpg")
                cv2.imwrite(save_path, img)

            for img in processed_images:
                plot_with_zoom(img)

        if save_plots == 1:
            # Enregistrer l'image en tant que fichier PNG
            cv2.imwrite(path_save + "grey_image_with_circles.png", processed_images[0])
            for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]):
                if ax.get_images():
                    fig = plt.figure(figsize=(8, 8))  # Créez une nouvelle figure
                    ax_img = ax.get_images()[0]
                    if ax_img is not None:
                        ax_new = fig.add_subplot(111)  # Utilisez 111 pour occuper toute la figure
                        ax_new.imshow(ax_img.get_array(), cmap='gray')
                        ax_new.set_title(ax.get_title())
                        ax_new.axis('off')  # Désactivez les axes pour éviter toute superposition
                        save_path = path_save + f'subplot_{i + 1}.png'
                        fig.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
                        plt.close(fig)

        params = {
            "video": video,
            "load_value": load_value,
            "write_value": write_value,
            "save_plots": save_plots,
            "video_echantillonage": video_echantillonage,
            "pixel_per_micrometer": pixel_per_micrometer,
            "substract_background": substract_background,
            "binary_threshold": binary_threshold,
            "dilatation_and_contraction": dilatation_and_contraction,
            "fill_holes": fill_holes,
            "delete_small_spots": delete_small_spots,
            "circle_fill_check": circle_fill_check,
            "median_or_mode_or_gaussianblur": median_or_mode_or_gaussianblur,
            "diff_or_absdiff_or_inverse": diff_or_absdiff_or_inverse,
            "kernel_size": kernel_size,
            "prefiltrage": prefiltrage,
            "postfiltrage": postfiltrage,
            "rayon_pinceau": rayon_pinceau,
            "minimum_size": minimum_size,
            "white_ratio_threshold": white_ratio_threshold,
            "threshold": threshold,
        }

        # Enregistrez le dictionnaire dans un fichier JSON
        if write_value == 1:
            with open(files[video]["path_save"] + "parametres.json", "w") as file:
                json.dump(params, file)

if __name__ == "__main__":
    main()

