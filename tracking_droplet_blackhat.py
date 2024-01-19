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

def save_model(model, filename='./svm_model.joblib'):
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

def load_model(filename='./svm_model.joblib'):
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model

def edge_detection_algorithm(images_gray, images_color, minimum_size):
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

    def predict_droplets(final_objects, final_objects_edges, images_color):
        # Function to extract features from a DetectedObject, including internal contours
        def extract_features(droplet_object):
            features = []
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
                droplet_object.cost_function_area  # 12
            ])

            return features
        loaded_model = load_model('./svm_model.joblib')
        loaded_scaler = load_model('./scaler.joblib')

        for i, frame in enumerate(final_objects):

            for droplet_object in frame:
                new_data = extract_features(droplet_object)
                new_data = loaded_scaler.transform(new_data)
                prediction = loaded_model.predict(new_data)
                droplet_object.is_droplet = bool(prediction)

                if droplet_object.is_droplet:
                    if not (np.all(droplet_object.hierarchy == np.array([-1, -1, -1, -1]))):
                        droplet_object.is_droplet = False

        for i, frame in enumerate(final_objects):
            image_draw = images_color[i].copy()
            for droplet_object in frame:
                if droplet_object.is_droplet:
                        cv2.drawContours(image_draw, [droplet_object.contour], 0, (0, 255, 0), 2)
                        centroid = tuple(map(int, droplet_object.centroid))
                        cv2.circle(image_draw, centroid, 5, (0, 255, 0), -1)
                else:
                    cv2.drawContours(image_draw, [droplet_object.contour], 0, (0, 0, 255), 2)
                    centroid = tuple(map(int, droplet_object.centroid))
                    cv2.circle(image_draw, centroid, 5, (0, 0, 255), -1)
                # print(droplet_object)

            frame = final_objects_edges[i]
            for edge_object in frame:
                cv2.drawContours(image_draw, [edge_object.contour], 0, (0, 0, 0), 2)
                centroid = tuple(map(int, edge_object.centroid))
                cv2.circle(image_draw, centroid, 5, (0, 0, 0), -1)

            cv2.imshow(f"Droplets - Frame {i}", image_draw)
            cv2.waitKey(0)
        cv2.destroyAllWindows()

        return final_objects, final_objects_edges

    results = Parallel(n_jobs=-1)(delayed(process_frame)(image_gray, minimum_size, images_color[i], i)
                                  for i, image_gray in enumerate(images_gray))

    final_objects_edges = [external_contour_objects for internal_contour_objects, external_contour_objects in results]
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

    final_objects, final_objects_edges = predict_droplets(final_objects, final_objects_edges, images_color)

    return final_objects, final_objects_edges

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

def tracking_algorithm(final_objects, images_color, Fps, pixel_per_micrometer):
    #
    # Supposons que droplet_objects contient les informations des droplets, y compris les centroids.
    # Vous devez créer un DataFrame pandas à partir de ces informations.
    data = []
    for frame_index, frame_objects in enumerate(final_objects):
        for droplet_object in frame_objects:
            if droplet_object.is_droplet:
                data.append({'frame': frame_index, 'x': droplet_object.centroid[0], 'y': droplet_object.centroid[1]})

    df = pd.DataFrame(data)

    traj = tp.link_df(df, search_range=100, memory=1)

    # Affichez les trajectoires résultantes
    tp.plot_traj(traj)

    # Utilisez cette boucle pour afficher les trajectoires superposées à chaque image
    # for i in range(len(images_color)):
    #     plt.figure(figsize=(8, 8))
    #     tp.plot_traj(traj, label=True, superimpose=images_color[i])

    data_list = []

    v_p_vector = []
    for item in set(traj.particle):
        sub = traj[traj.particle == item]
        dx = np.diff(sub.x) / 1.
        dy = np.diff(sub.y) / 1.
        dt = np.diff(sub.frame) / Fps
        v = (((np.sqrt(dy ** 2 + dx ** 2))) / dt) / pixel_per_micrometer
        for i in range(len(sub)):
            if i < len(sub) - 1:
                x, y, dx_val, dy_val, v_val, dt_val, frame_val = (
                    sub.x.iloc[i],
                    sub.y.iloc[i],
                    abs(dx[i]),
                    abs(dy[i]),
                    v[i],
                    dt[i],
                    sub.frame.iloc[i],
                )
                data_list.append(
                    {
                        "dx": dx_val,
                        "dy": dy_val,
                        "x": x,
                        "y": y,
                        "frame": frame_val,
                        "particle": item,
                        "dt": dt_val,
                        "v": v_val,
                    }
                )
                v_p_vector.append(v_val)
            else:
                x, y, dx_val, dy_val, v_val, dt_val, frame_val = (
                    sub.x.iloc[i],
                    sub.y.iloc[i],
                    abs(0),
                    abs(0),
                    0,
                    0,
                    sub.frame.iloc[i],
                )
                data_list.append(
                    {
                        "dx": dx_val,
                        "dy": dy_val,
                        "x": x,
                        "y": y,
                        "frame": frame_val,
                        "particle": item,
                        "dt": dt_val,
                        "v": v_val,
                    }
                )

    data = pd.DataFrame(data_list)

    mydict = {}

    for p in np.unique(data.particle):
        d = {}
        d['v'] = data.v[data.particle == p]
        d['frame'] = data.frame[data.particle == p]
        d['x'] = data.x[data.particle == p]
        d['y'] = data.y[data.particle == p]
        d['dx'] = data.dx[data.particle == p]
        d['dy'] = data.dy[data.particle == p]
        d['dt'] = data.dt[data.particle == p]
        mydict[p] = d

    # Initialize a DataFrame to store all the trajectories
    df_all_traj = pd.DataFrame()

    # Loop through each particle trajectory and add it to the DataFrame
    for particle_id, data_dict in mydict.items():
        df_particle_traj = pd.DataFrame(data_dict)
        df_particle_traj['particle_id'] = particle_id
        df_all_traj = pd.concat([df_all_traj, df_particle_traj], ignore_index=True)

    # Sort the DataFrame based on frame and particle_id
    df_all_traj = df_all_traj.sort_values(by=['frame', 'particle_id'])

    # Initialisation des variables
    state = {'frame_index': 0}

    # Créez une figure
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.2)  # Ajustez la position du bouton

    def update_image(event):
        if event.inaxes == axnext and state['frame_index'] < len(images_color) - 1:
            state['frame_index'] += 1
        elif event.inaxes == axprev and state['frame_index'] > 0:
            state['frame_index'] -= 1

        ax.clear()
        ax.imshow(images_color[state['frame_index']], cmap='gray')

        # Ajoutez votre logique de trajectoire ici
        for particle_id, df_particle_traj in df_all_traj.groupby('particle_id'):
            # Vérifiez si la particule est présente sur la frame courante
            if state['frame_index'] in df_particle_traj['frame'].values:
                df_frame = df_particle_traj[df_particle_traj['frame'] <= state['frame_index']]
                if not df_frame.empty:
                    ax.plot(df_frame['x'], df_frame['y'], label=f'Particle {particle_id}')

        ax.set_title(f'Frame {state["frame_index"]}')
        ax.legend()
        plt.draw()

    # Créez le bouton de navigation
    axprev = plt.axes([0.7, 0.01, 0.1, 0.05])
    axnext = plt.axes([0.81, 0.01, 0.1, 0.05])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(update_image)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(update_image)

    # Affichez la première image
    ax.imshow(images_color[state['frame_index']], cmap='gray')
    ax.set_title(f'Frame {state["frame_index"]}')

    plt.show()

    # Plot histogram
    fig, ax = plt.subplots(figsize=(8, 8))
    v_p_vector = np.array(v_p_vector)
    v_p_vector = v_p_vector[np.abs(v_p_vector) > 100]
    hist, bins = np.histogram(v_p_vector, bins='auto')
    ax.bar(bins[:-1], hist.astype(np.float32) / hist.sum(), width=(bins[1] - bins[0]), color='Blue')
    plt.legend()
    plt.ylabel('Frequency')
    plt.xlabel(r'velocity [$\mu$m/s]')
    plt.title('Histogram of velocity')
    plt.show()

    return data

def main():
    """variables principales du programme"""
    #chemins d'accès (voir dictionnaire)
    video = "video1"

    #nombre d'image prise en compte dans la vidéo, si video_echantillonage == 100, une image sur 100 est traitée dans le code (les vidéos sont trop grosses)
    video_echantillonage = 1
    debut_frame = 0
    fin_frame = 10

    #scale
    pixel_per_micrometer = 0.5
    fps = 60  # set the Number of frame per second at the video was recording.

    """début du code"""
    # Enregistrer le temps actuel avant de démarrer le code
    if load_value == 0:
        start_time = time.time()

        # Appel de la fonction avec le dictionnaire en argument
        path_video = files[video]["path"]

        images_gray, images_color = open_file_and_load_images(path_video,video_echantillonage, debut_frame, fin_frame)

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
        droplet_objects, edge_objects = edge_detection_algorithm(images_gray, images_color, minimum_size)

        tracking_data = tracking_algorithm(droplet_objects, images_color, fps, pixel_per_micrometer)

        for frame in droplet_objects:
            for droplet_object in frame:
                if droplet_object.is_droplet == True:
                    print(droplet_object)

        print(tracking_data)

if __name__ == "__main__":
    main()
