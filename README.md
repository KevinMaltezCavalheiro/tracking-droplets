Algorithme de détection de gouttes. Il se compose de deux codes distincts. Le premier est celui que j'ai développé à partir d'analyse d'images conventionnelle. Le deuxième se base sur l'algorithme de deep learning du groupe meta nommé detectron2.

Pour l'utilisation de mon algorithme. Voici les principales étapes d'installation.

1) Créer un environement pycharm et vérifier que le dossier de l'environnement virtuel (venv) a bien été créé

2) Copier le projet à partir de gitlab(https://gitlab.uliege.be/K.Maltez/tracking_droplets)
2.1) utiliser clone ssh ou clone https pour copier le projet sur votre pc. Sur batch (terminal sur mac), taper sans les guillemets:

'cd /chemin vers votre projet python et le répertoire venv/'

exemple: 
"cd /Users/Kevin/PycharmProjects/tracking_droplets/venv"

ensuite copier le projet avec la commande suivante toujours dans le terminal (si vous utilisez clone ssh):

'git clone git@gitlab.uliege.be:K.Maltez/tracking_droplets.git'

3)Vous pouvez maintenant ouvrir le projet dans pycharm. Les fichier relatifs à ce modèle sont "training_svc_blackhat.py", "tracking_droplet_blackhat.py" et le dossier "svc"

4) procéder à l'installation des différentes librairies nécessaires. Pour ce faire, taper la commande bash /!\ dans le terminal de pycharm /!\ (il faut installer les librairies dans l'environnement virtuel):

'pip install matplotlib numpy opencv-python time pandas scikit-image scipy scikit-learn joblib trackpy mplcursors'

Pour utiliser le code, voici les étapes principales:

1) Commencer par vous rendre dans le dictionnaire "files" situé après l'import des librairies dans le fichier tracking_droplet_blackhat.py. Modifier le chemin d'accès vers la vidéo de votre choix que vous aimeriez tester. Afin d'éviter de retapper le chemin à chaque changement de vidéo, vous pouvez sauvegarder chaque chemin d'accès comme "video1", "video2",...

2) Rendez-vous dans le main où vous pouvez spécifier votre vidéo au travers 'video = "video7"' par exemple. Celle-ci correspond au chemin d'accès défini plus haut. 

3) Toujours au début du main, modifier les paramètres principaux du programmes, taille de pixels et fps relatifs à la vidéo. Choisissez également les images à charger au travers les variables "video_echantillonage", "debut_frame" et "fin_frame". /!\ Il est évident que les fps sont modifiés si on choisi de prendre 1 image sur 10 par exemple (voir "video_echantillonage").

4) Lancer le code! Vous devriez voir les gouttes détectées par l'algorithme en vert

(En tant que première version, je suis très friant de vos retours, que ce soit sur l'optimisation ou sur les bugs potentiels. Je m'éforcerai de les corriger au fur et à mesure)




























