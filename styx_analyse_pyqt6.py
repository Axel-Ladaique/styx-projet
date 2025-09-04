#si probleme : bash reset_venv.sh
# rentrer dans la venv : 
'''   
cd /Users/axel.ldq/11_STYX_ANALYSE
source venv/bin/activate
python styx_analyse_pyqt6.py
'''
#si probleme : bash reset_venv.sh
# rentrer dans la venv : 
#   cd /Users/axel.ldq/Documents/Cours/3A/S6/Vélo_Condo/11_STYX_ANALYSE
#   source venv/bin/activate
#   python styx_analyse_pyqt6.py

import sys
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QLabel, QFileDialog, QListWidget, QListWidgetItem,
    QStackedWidget, QMessageBox, QSizePolicy, QTextEdit,
    QComboBox, QHBoxLayout, QFrame, QSplitter, QGridLayout,
    QScrollArea, QSlider
)
from PyQt6.QtGui import QPixmap
from datetime import datetime
import locale
import math
from PyQt6.QtGui import QPainter, QPixmap
from PyQt6.QtCore import Qt
import shutil  

# Configuration locale pour les dates en français
try:
    locale.setlocale(locale.LC_TIME, 'fr_FR.UTF-8')  # Pour Linux/macOS
except:
    try:
        locale.setlocale(locale.LC_TIME, 'French_France')  # Pour Windows
    except:
        pass  # Garder la locale par défaut si aucune ne fonctionne

DATA_FOLDER = "data_sessions"
RECENT_FILE_PATH = os.path.join(DATA_FOLDER, "recent_files.json")

os.makedirs(DATA_FOLDER, exist_ok=True)


# --- Fonctions utilitaires ---

class BackgroundWidget(QWidget):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.pixmap = QPixmap(image_path)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # Calculer le ratio pour "cover"
        widget_ratio = self.width() / self.height()
        pixmap_ratio = self.pixmap.width() / self.pixmap.height()

        if widget_ratio > pixmap_ratio:
            scaled_height = self.height()
            scaled_width = int(scaled_height * pixmap_ratio)
        else:
            scaled_width = self.width()
            scaled_height = int(scaled_width / pixmap_ratio)

        scaled_pixmap = self.pixmap.scaled(
            scaled_width, scaled_height,
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation
        )

        # Centrer l’image
        x = (self.width() - scaled_pixmap.width()) // 2
        y = (self.height() - scaled_pixmap.height()) // 2
        painter.drawPixmap(x, y, scaled_pixmap)

    def resizeEvent(self, event):
        self.update()

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calcule la distance entre deux points GPS en mètres"""
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return 0
    
    # Rayon de la Terre en mètres
    R = 6371000
    
    # Conversion en radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Différences
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    # Formule de Haversine
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c

def calculate_gps_distance(df):
    """Calcule la distance cumulée à partir des coordonnées GPS"""
    if "Lat" not in df.columns or "Lon" not in df.columns:
        return df
    
    distances = [0]  # Premier point à distance 0
    total_distance = 0
    
    for i in range(1, len(df)):
        dist = haversine_distance(
            df.iloc[i-1]["Lat"], df.iloc[i-1]["Lon"],
            df.iloc[i]["Lat"], df.iloc[i]["Lon"]
        )
        total_distance += dist
        distances.append(total_distance)
    
    df["Distance_GPS"] = distances
    return df

def clean_data(df):
    """Nettoie les données en supprimant les valeurs aberrantes"""
    df_clean = df.copy()
    
    # Vitesse : valeurs négatives et > 80 km/h considérées comme aberrantes
    if "Vitesse" in df_clean.columns:
        df_clean.loc[df_clean["Vitesse"] < 0, "Vitesse"] = 0
        df_clean.loc[df_clean["Vitesse"] > 80, "Vitesse"] = df_clean["Vitesse"].median()
    
    # Tension : valeurs < 0 ou > 50V considérées comme aberrantes
    if "Tension" in df_clean.columns:
        df_clean.loc[df_clean["Tension"] < 0, "Tension"] = 0
        df_clean.loc[df_clean["Tension"] > 50, "Tension"] = df_clean["Tension"].median()
    
    # Altitude : variations trop importantes
    if "Alt" in df_clean.columns:
        # Supprimer les sauts d'altitude > 100m d'un point à l'autre
        alt_diff = df_clean["Alt"].diff().abs()
        outliers = alt_diff > 100
        if outliers.any():
            df_clean.loc[outliers, "Alt"] = df_clean["Alt"].interpolate()
    
    # Coordonnées GPS : valeurs hors limites
    if "Lat" in df_clean.columns:
        df_clean.loc[~df_clean["Lat"].between(-90, 90), "Lat"] = np.nan
    if "Lon" in df_clean.columns:
        df_clean.loc[~df_clean["Lon"].between(-180, 180), "Lon"] = np.nan
    
    # Courants : valeurs extrêmes
    if "CurrentIn" in df_clean.columns:
        df_clean.loc[df_clean["CurrentIn"].abs() > 100, "CurrentIn"] = 0
    if "MotorCurrent" in df_clean.columns:
        df_clean.loc[df_clean["MotorCurrent"].abs() > 100, "MotorCurrent"] = 0
    
    return df_clean

def format_session_name(filename):
    """Formate le nom de session pour l'affichage"""
    try:
        # Extraire la partie timestamp du nom de fichier
        base_name = filename.replace(".csv", "").replace("session_", "")
        # Parser la date et l'heure
        dt = datetime.strptime(base_name, "%Y-%m-%d_%H-%M-%S")
        # Retourner au format français
        return dt.strftime("Trajet du %d %B %Y, à %Hh%M")
    except:
        return filename


# --- Fonctions de gestion des statistiques globales ---
def load_global_stats():
    """Charge les statistiques globales depuis un fichier JSON"""
    stats_file = os.path.join(DATA_FOLDER, "global_stats.json")
    if os.path.exists(stats_file):
        try:
            with open(stats_file, "r") as f:
                return json.load(f)
        except:
            pass
    
    # Valeurs par défaut
    return {
        "total_distance": 0,
        "total_duration": 0,
        "total_trips": 0,
        "total_energy_charged": 0,
        "total_energy_discharged": 0,
        "last_updated": datetime.now().isoformat()
    }

def save_global_stats(stats):
    """Sauvegarde les statistiques globales"""
    stats_file = os.path.join(DATA_FOLDER, "global_stats.json")
    stats["last_updated"] = datetime.now().isoformat()
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

def update_global_stats_from_file(csv_path, operation="add"):
    """Met à jour les statistiques globales à partir d'un fichier"""
    try:
        df = pd.read_csv(csv_path)
        df_clean = clean_data(df)
        df_final = calculate_gps_distance(df_clean)
        
        # Calculer les métriques du trajet
        if "Distance_GPS" in df_final.columns:
            distance = df_final["Distance_GPS"].iloc[-1]
        elif "Distance" in df_final.columns:
            distance = df_final["Distance"].iloc[-1]
        else:
            distance = 0
            
        duration = df_final["Temps"].iloc[-1] if "Temps" in df_final.columns else 0
        energy_charged = df_final["WHCharged"].sum() if "WHCharged" in df_final.columns else 0
        energy_discharged = df_final["WHDischarged"].sum() if "WHDischarged" in df_final.columns else 0
        
        # Charger les stats actuelles
        stats = load_global_stats()
        
        # Mettre à jour selon l'opération
        if operation == "add":
            stats["total_distance"] += distance
            stats["total_duration"] += duration
            stats["total_trips"] += 1
            stats["total_energy_charged"] += energy_charged
            stats["total_energy_discharged"] += energy_discharged
        elif operation == "remove":
            stats["total_distance"] = max(0, stats["total_distance"] - distance)
            stats["total_duration"] = max(0, stats["total_duration"] - duration)
            stats["total_trips"] = max(0, stats["total_trips"] - 1)
            stats["total_energy_charged"] = max(0, stats["total_energy_charged"] - energy_charged)
            stats["total_energy_discharged"] = max(0, stats["total_energy_discharged"] - energy_discharged)
        
        save_global_stats(stats)
        return stats
        
    except Exception as e:
        print(f"Erreur mise à jour stats: {e}")
        return load_global_stats()

def recalculate_all_stats():
    """Recalcule toutes les statistiques en parcourant tous les fichiers"""
    stats = {
        "total_distance": 0,
        "total_duration": 0,
        "total_trips": 0,
        "total_energy_charged": 0,
        "total_energy_discharged": 0,
        "last_updated": datetime.now().isoformat()
    }
    
    recent_files = load_recent_files()
    for filename in recent_files:
        file_path = os.path.join(DATA_FOLDER, filename)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                df_clean = clean_data(df)
                df_final = calculate_gps_distance(df_clean)
                
                if "Distance_GPS" in df_final.columns:
                    distance = df_final["Distance_GPS"].iloc[-1]
                elif "Distance" in df_final.columns:
                    distance = df_final["Distance"].iloc[-1]
                else:
                    distance = 0
                    
                duration = df_final["Temps"].iloc[-1] if "Temps" in df_final.columns else 0
                energy_charged = df_final["WHCharged"].sum() if "WHCharged" in df_final.columns else 0
                energy_discharged = df_final["WHDischarged"].sum() if "WHDischarged" in df_final.columns else 0
                
                stats["total_distance"] += distance
                stats["total_duration"] += duration
                stats["total_trips"] += 1
                stats["total_energy_charged"] += energy_charged
                stats["total_energy_discharged"] += energy_discharged
                
            except Exception as e:
                print(f"Erreur traitement {filename}: {e}")
    
    save_global_stats(stats)
    return stats
# --- Fonctions de gestion des fichiers récents ---
def load_recent_files():
    if os.path.exists(RECENT_FILE_PATH):
        with open(RECENT_FILE_PATH, "r") as f:
            return json.load(f)
    return []

def save_recent_file(new_file):
    recent = load_recent_files()
    if new_file not in recent:
        recent.insert(0, new_file)
    recent = recent[:20]  # Augmenter à 20 fichiers récents
    with open(RECENT_FILE_PATH, "w") as f:
        json.dump(recent, f)

def remove_recent_file(filename):
    """Supprime un fichier de la liste des récents"""
    recent = load_recent_files()
    if filename in recent:
        recent.remove(filename)
        with open(RECENT_FILE_PATH, "w") as f:
            json.dump(recent, f)

def delete_session_file(filename):
    """Supprime définitivement un fichier de session"""
    file_path = os.path.join(DATA_FOLDER, filename)
    if os.path.exists(file_path):
        # Mettre à jour les stats avant suppression
        update_global_stats_from_file(file_path, operation="remove")
        # Supprimer le fichier
        os.remove(file_path)
        # Supprimer de la liste des récents
        remove_recent_file(filename)
        return True
    return False

def handle_new_csv(path, parent=None):
    import pandas as pd
    import datetime
    import pytz
    from dateutil import parser
    import os
    from PyQt6.QtWidgets import QMessageBox

    df = pd.read_csv(path)

    # Trouver colonnes Date et Heure
    date_col = None
    heure_col = None
    for col in df.columns:
        c = col.lower()
        if "date" in c:
            date_col = col
        elif "heure" in c or "time" in c:
            heure_col = col

    dt = None
    if date_col and heure_col:
        for i in range(min(10, len(df))):
            date_val = str(df[date_col].iloc[i])
            heure_val = str(df[heure_col].iloc[i])

            try:
                # Parse date (ex: 2023-07-20)
                date_parsed = parser.parse(date_val).date()
                # Parse heure (ex: 13523500 -> 13:52:35)
                heure_str = f"{int(heure_val[:2]):02d}:{int(heure_val[2:4]):02d}:{int(heure_val[4:6]):02d}"
                time_parsed = parser.parse(heure_str).time()

                dt = datetime.datetime.combine(date_parsed, time_parsed, tzinfo=datetime.timezone.utc)

                # Convertir heure UTC en heure de Paris
                paris_tz = pytz.timezone("Europe/Paris")
                dt = dt.astimezone(paris_tz)
                break
            except Exception:
                continue

    if not dt:
        dt = datetime.datetime.now(pytz.timezone("Europe/Paris"))

    filename = dt.strftime("session_%Y-%m-%d_%H-%M-%S.csv")
    destination = os.path.join(DATA_FOLDER, filename)

    # Test si le fichier existe déjà
    if os.path.exists(destination):
        QMessageBox.information(
            QApplication.activeWindow(),
            "Fichier déjà importé",
            f"Le fichier '{format_session_name(filename)}' a déjà été importé."
        )
        return destination

    # Nettoyer les données et calculer la distance GPS
    df_clean = clean_data(df)
    df_final = calculate_gps_distance(df_clean)
    
    df_final.to_csv(destination, index=False)
    save_recent_file(os.path.basename(destination))
    
    # Mettre à jour les statistiques globales
    update_global_stats_from_file(destination, operation="add")
    
    return destination


# --- Page d'accueil avec interface moderne ---
class HomePage(QWidget):
    def __init__(self, switch_to_analysis):
        super().__init__()
        self.switch_to_analysis = switch_to_analysis
        self.init_ui()
        self.refresh_stats()



    class BackgroundWidget(QWidget):
        def __init__(self, image_path, parent=None):
            super().__init__(parent)
            self.pixmap = QPixmap(image_path)

        def paintEvent(self, event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

            # Calculer le ratio pour "cover"
            widget_ratio = self.width() / self.height()
            pixmap_ratio = self.pixmap.width() / self.pixmap.height()

            if widget_ratio > pixmap_ratio:
                # Adapter la hauteur, crop sur la largeur
                scaled_height = self.height()
                scaled_width = int(scaled_height * pixmap_ratio)
            else:
                # Adapter la largeur, crop sur la hauteur
                scaled_width = self.width()
                scaled_height = int(scaled_width / pixmap_ratio)

            scaled_pixmap = self.pixmap.scaled(
                scaled_width, scaled_height, Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation
            )

            # Centrer l’image
            x = (self.width() - scaled_pixmap.width()) // 2
            y = (self.height() - scaled_pixmap.height()) // 2

            painter.drawPixmap(x, y, scaled_pixmap)

        def resizeEvent(self, event):
            self.update()


    def init_ui(self):
        # Layout principal
        main_layout = QHBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # === PANNEAU GAUCHE (Stats + Header) ===
        left_panel = QWidget()
        left_panel.setFixedWidth(350)
        left_panel.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(52, 152, 219, 0.9),
                    stop:1 rgba(41, 128, 185, 0.9));
                border-radius: 0;
            }
        """)
        left_layout = QVBoxLayout()
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(10, 30, 10, 20)

        # === HEADER ===
        header_layout = QVBoxLayout()
        header_layout.setSpacing(10)

        
        
        
        # Remplacer le titre STYX par le logo
        logo_title = QLabel()
        logo_pixmap = QPixmap("images/logo_styx_blanc_2.png")
        if not logo_pixmap.isNull():
            # Redimensionner le logo pour qu'il s'adapte bien
            scaled_logo = logo_pixmap.scaled(
                350, 140,  # Largeur, Hauteur maximales
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            logo_title.setPixmap(scaled_logo)
        else:
            # Fallback si l'image n'est pas trouvée
            logo_title.setText("STYX")
            logo_title.setStyleSheet("""
                font-size: 48px; 
                font-weight: 900; 
                color: white; 
                letter-spacing: 3px;
            """)
        
        logo_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        subtitle = QLabel("ANALYSE AVANCÉE")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("""
            font-size: 14px; 
            color: rgba(255,255,255,0.9); 
            font-weight: 600;
            letter-spacing: 2px;
            margin-bottom: 10px;
        """)
        
        description = QLabel("Explorez vos performances\nen vélo STYX")
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description.setStyleSheet("""
            font-size: 16px; 
            color: rgba(255,255,255,0.8); 
            line-height: 1.4;
            font-style: italic;
        """)

        logo_title.setStyleSheet("""
            QLabel {
                background: transparent;
                border-radius: 0;
            }
        """)

        subtitle.setStyleSheet("""
            QLabel {
                background: transparent;
                border-radius: 0;
            }
        """)

        description.setStyleSheet("""
            QLabel {
                background: transparent;
                border-radius: 0;
            }
        """)

        header_layout.addWidget(logo_title)  # Utiliser logo_title au lieu de title
        header_layout.addWidget(subtitle)
        header_layout.addWidget(description)
        left_layout.addLayout(header_layout)


        # === STATISTIQUES GLOBALES ===
        stats_layout = QVBoxLayout()
        stats_layout.setSpacing(15)
        stats_title = QLabel("📊 VOS PERFORMANCES")
        stats_title.setStyleSheet("""
            font-size: 16px; 
            background: transparent;
            font-weight: bold; 
            color: white; 
            margin-bottom: 10px;
            letter-spacing: 1px;
        """)
        stats_layout.addWidget(stats_title)
        
        self.stats_widgets = {
            'trips': self.create_vertical_stat_widget("Trajets effectués", "0", "🚲"),
            'distance': self.create_vertical_stat_widget("Distance parcourue", "0 km", "📏"),
            'duration': self.create_vertical_stat_widget("Temps de route", "0h 0min", "⏱️"),
            'energy_charged': self.create_vertical_stat_widget("Énergie récupérée", "0 Wh", "🔋")
        }
        stats_layout.addWidget(self.stats_widgets['trips'])
        stats_layout.addWidget(self.stats_widgets['distance'])
        stats_layout.addWidget(self.stats_widgets['duration'])
        stats_layout.addWidget(self.stats_widgets['energy_charged'])
        
        recalc_button = QPushButton("🔄 ACTUALISER")
        recalc_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(255,255,255,0.2);
                color: white;
                border: 2px solid rgba(255,255,255,0.3);
                padding: 12px 20px;
                border-radius: 8px;
                font-size: 12px;
                font-weight: bold;
                letter-spacing: 1px;
                margin-top: 20px;
            }
            QPushButton:hover {
                background-color: rgba(255,255,255,0.3);
                border-color: rgba(255,255,255,0.5);
            }
        """)
        recalc_button.clicked.connect(self.recalculate_stats)
        stats_layout.addWidget(recalc_button)
        left_layout.addLayout(stats_layout)
        left_layout.addStretch()
        left_panel.setLayout(left_layout)

        # === PANNEAU DROIT avec image de fond ===
        right_panel = BackgroundWidget("images/Image_Fond.png")
        right_layout = QVBoxLayout()
        right_layout.setSpacing(20)
        right_layout.setContentsMargins(40, 40, 40, 40)

        # === LISTE DES TRAJETS (TRANSPARENT) ===
        trips_frame = QFrame()
        trips_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 0.1);
                border-radius: 15px;
                border: 1px solid rgba(189, 195, 199, 0.5);
            }
        """)
        trips_layout = QVBoxLayout()
        trips_layout.setContentsMargins(30, 25, 30, 25)
        
        trips_header = QHBoxLayout()
        trips_title = QLabel("📋 Historique des trajets")
        trips_title.setStyleSheet("""
            font-size: 22px; 
            font-weight: bold; 
            color: #2c3e50;
            margin-bottom: 5px;
        """)
        
        open_button = QPushButton("📂 NOUVEAU TRAJET")
        open_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #27ae60, stop:1 #229954);
                color: white;
                border: none;
                padding: 12px 25px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 13px;
                letter-spacing: 1px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2ecc71, stop:1 #27ae60);
            }
        """)
        open_button.clicked.connect(self.open_new_file)
        
        trips_header.addWidget(trips_title)
        trips_header.addStretch()
        trips_header.addWidget(open_button)
        trips_layout.addLayout(trips_header)
        
        self.trips_list = QListWidget()
        self.trips_list.setStyleSheet("""
            QListWidget {
                border: none;
                background-color: rgba(255, 255, 255, 5);
                font-size: 14px;
            }
            QListWidget::item {
                padding: 15px;
                border: 1px solid rgba(236, 240, 241, 0.5);
                margin: 3px;
                border-radius: 8px;
                background-color: rgba(255, 255, 255, 0.3);
            }
            QListWidget::item:hover {
                background-color: rgba(52, 152, 219, 0.1);
                border: 1px solid rgba(52, 152, 219, 0.3);
            }
            QListWidget::item:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3498db, stop:1 #2980b9);
                color: white;
                border: 1px solid #2980b9;
            }
        """)
        self.trips_list.itemDoubleClicked.connect(self.open_trip)
        trips_layout.addWidget(self.trips_list)
        
        trips_frame.setLayout(trips_layout)
        right_layout.addWidget(trips_frame)

        # === BOUTONS D'ACTION ===
        actions_frame = QFrame()
        actions_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 0.3);
                border-radius: 10px;
                padding: 5px;
            }
        """)
        actions_layout = QHBoxLayout()
        
        view_button = QPushButton("👁️ ANALYSER")
        view_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3498db, stop:1 #2980b9);
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 14px;
                letter-spacing: 1px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #5dade2, stop:1 #3498db);
            }
        """)
        view_button.clicked.connect(self.view_selected_trip)
        
        delete_button = QPushButton("🗑️ SUPPRIMER")
        delete_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #e74c3c, stop:1 #c0392b);
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 14px;
                letter-spacing: 1px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ec7063, stop:1 #e74c3c);
            }
        """)
        delete_button.clicked.connect(self.delete_selected_trip)
        
        actions_layout.addWidget(view_button)
        actions_layout.addWidget(delete_button)
        actions_layout.addStretch()
        
        actions_frame.setLayout(actions_layout)
        right_layout.addWidget(actions_frame)
        right_panel.setLayout(right_layout)

        # Ajouter les panneaux au layout principal
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)
        
        self.setLayout(main_layout)
        self.refresh_list()



    def create_vertical_stat_widget(self, title, value, icon):
        """Crée un widget de statistique vertical"""
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 0.15);
                border-radius: 10px;
                border: 1px solid rgba(255, 255, 255, 0.2);
                padding: 3px;
                margin: 2px;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setSpacing(4)
        
        # Icon et titre
        header = QHBoxLayout()
        icon_label = QLabel(icon)
        icon_label.setStyleSheet("font-size: 18px;")
        title_label = QLabel(title)
        title_label.setStyleSheet("""
            font-size: 20px; 
            color: rgba(255,255,255,0.9); 
            font-weight: 600;
        """)
        
        header.addWidget(icon_label)
        header.addWidget(title_label)
        header.addStretch()
        
        # Valeur
        value_label = QLabel(value)
        value_label.setStyleSheet("""
            font-size: 24px; 
            font-weight: bold; 
            color: white;
            margin-top: 5px;
        """)
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        layout.addLayout(header)
        layout.addWidget(value_label)
        frame.setLayout(layout)
        
        # Stocker le label de valeur pour la mise à jour
        frame.value_label = value_label
        
        return frame

    def refresh_stats(self):
        """Met à jour les statistiques affichées"""
        stats = load_global_stats()
        
        # Conversion des unités
        distance_km = stats['total_distance'] / 1000
        duration_hours = stats['total_duration'] / 3600
        hours = int(duration_hours)
        minutes = int((duration_hours - hours) * 60)
        
        # Mise à jour des widgets (seulement 4 maintenant)
        self.stats_widgets['trips'].value_label.setText(str(stats['total_trips']))
        self.stats_widgets['distance'].value_label.setText(f"{distance_km:.1f} km")
        self.stats_widgets['duration'].value_label.setText(f"{hours}h {minutes}min")
        self.stats_widgets['energy_charged'].value_label.setText(f"{stats['total_energy_charged']:.0f} Wh")

    def recalculate_stats(self):
        """Recalcule toutes les statistiques"""
        recalculate_all_stats()
        self.refresh_stats()
        QMessageBox.information(self, "Statistiques", "Les statistiques ont été recalculées avec succès !")

    def refresh_list(self):
        """Met à jour la liste des trajets"""
        self.trips_list.clear()
        for filename in load_recent_files():
            display_name = format_session_name(filename)
            
            # Créer un widget personnalisé pour chaque trajet
            item = QListWidgetItem()
            item.setText(display_name)
            item.setData(Qt.ItemDataRole.UserRole, filename)
            
            # Ajouter des infos supplémentaires si possible
            try:
                file_path = os.path.join(DATA_FOLDER, filename)
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    df_clean = clean_data(df)
                    df_final = calculate_gps_distance(df_clean)
                    
                    if "Distance_GPS" in df_final.columns:
                        distance = df_final["Distance_GPS"].iloc[-1] / 1000  # en km
                    elif "Distance" in df_final.columns:
                        distance = df_final["Distance"].iloc[-1] / 1000
                    else:
                        distance = 0
                    
                    duration = df_final["Temps"].iloc[-1] / 60 if "Temps" in df_final.columns else 0  # en minutes
                    
                    item.setText(f"{display_name}\n📏 {distance:.1f} km • ⏱️ {duration:.0f} min")
            except:
                pass
            
            self.trips_list.addItem(item)

    def open_new_file(self):
        """Ouvre un nouveau fichier CSV"""
        path, _ = QFileDialog.getOpenFileName(self, "Choisir un fichier CSV", "", "Fichiers CSV (*.csv)")
        if path:
            try:
                copied_path = handle_new_csv(path)
                self.refresh_list()
                self.refresh_stats()
                self.switch_to_analysis(copied_path)
            except Exception as e:
                QMessageBox.critical(self, "Erreur", f"Impossible de charger le fichier : {e}")

    def open_trip(self, item):
        """Ouvre un trajet en double-cliquant"""
        self.view_selected_trip()

    def view_selected_trip(self):
        """Voir le trajet sélectionné"""
        current_item = self.trips_list.currentItem()
        if current_item:
            filename = current_item.data(Qt.ItemDataRole.UserRole)
            full_path = os.path.join(DATA_FOLDER, filename)
            if os.path.exists(full_path):
                self.switch_to_analysis(full_path)
            else:
                QMessageBox.warning(self, "Fichier manquant", "Ce fichier n'existe plus.")
                self.refresh_list()
                self.refresh_stats()

    def delete_selected_trip(self):
        """Supprime le trajet sélectionné"""
        current_item = self.trips_list.currentItem()
        if not current_item:
            QMessageBox.information(self, "Aucune sélection", "Veuillez sélectionner un trajet à supprimer.")
            return
        
        filename = current_item.data(Qt.ItemDataRole.UserRole)
        display_name = format_session_name(filename)
        
        # Demande de confirmation
        reply = QMessageBox.question(
            self, 
            "Confirmer la suppression",
            f"Êtes-vous sûr de vouloir supprimer définitivement le trajet :\n\n{display_name}\n\n"
            "⚠️ Cette action est irréversible !",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            if delete_session_file(filename):
                QMessageBox.information(self, "Suppression", "Le trajet a été supprimé avec succès.")
                self.refresh_list()
                self.refresh_stats()
            else:
                QMessageBox.warning(self, "Erreur", "Impossible de supprimer le fichier.")


# --- Widget graphique individuel ---
class GraphWidget(QWidget):
    def __init__(self, graph_id, on_cursor_change, on_zoom_change=None, advanced_mode_callback=None):
        super().__init__()
        self.graph_id = graph_id
        self.on_cursor_change = on_cursor_change
        self.on_zoom_change = on_zoom_change
        self.advanced_mode_callback = advanced_mode_callback
        self.df = None
        self.cursor_line = None
        self.locked = False
        self.zoom_active = False
        self.toolbar = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(2)
        
        # Sélecteur de graphique
        self.graph_selector = QComboBox()
        self.graph_selector.currentTextChanged.connect(self.update_graph)
        layout.addWidget(self.graph_selector)
        
        # Zone de graphique
        self.figure = plt.figure(figsize=(5, 3))
        self.ax = self.figure.add_subplot(111)
        self.figure_canvas = FigureCanvas(self.figure)
        
        # Créer la toolbar mais masquée par défaut
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
        self.toolbar = NavigationToolbar2QT(self.figure_canvas, self)
        self.toolbar.setMaximumHeight(25)
        self.toolbar.hide()
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.figure_canvas)
        
        self.setLayout(layout)

    def set_advanced_mode(self, advanced_mode):
        """Affiche/masque la toolbar selon le mode"""
        if advanced_mode:
            self.toolbar.show()
        else:
            self.toolbar.hide()
            if self.df is not None:
                self.reset_zoom()

    def reset_zoom(self):
        """Remet le graphique à l'échelle par défaut"""
        if self.df is not None and len(self.df) > 0:
            self.ax.set_xlim(self.df["Temps"].min(), self.df["Temps"].max())
            self.figure_canvas.draw()

    def set_data(self, df):
        self.df = df

    def get_available_options(self, advanced_mode=False):
        """Retourne les options disponibles selon le mode"""
        if self.df is None:
            return ["Aucun"]
            
        options = ["Aucun"]
        
        # Mode normal
        if "Tension" in self.df.columns:
            options.append("Tension")
        if "Vitesse" in self.df.columns:
            options.append("Vitesse")
        if "GazFrein" in self.df.columns:
            options.append("GazFrein")
        if "WHCharged" in self.df.columns and "WHDischarged" in self.df.columns:
            options.append("Énergie (bilan)")
        if "Tension" in self.df.columns and "CurrentIn" in self.df.columns:
            options.append("Puissance électrique")
        if "Alt" in self.df.columns:
            options.append("Altitude")
        
        # Mode avancé
        if advanced_mode:
            if "WHCharged" in self.df.columns:
                options.append("Énergie chargée")
            if "WHDischarged" in self.df.columns:
                options.append("Énergie déchargée")
            if "Distance" in self.df.columns:
                options.append("Distance (capteur)")
            if "Distance_GPS" in self.df.columns:
                options.append("Distance (GPS)")
            if "CurrentIn" in self.df.columns:
                options.append("Courant entrant")
            if "MotorCurrent" in self.df.columns:
                options.append("Courant moteur")
            if "Lat" in self.df.columns:
                options.append("Latitude")
            if "Lon" in self.df.columns:
                options.append("Longitude")
            if "Vsat" in self.df.columns:
                options.append("Vitesse satellite")
            if "Cap" in self.df.columns:
                options.append("Cap")
            if "Sat" in self.df.columns:
                options.append("Satellites")
            if "HDOP" in self.df.columns:
                options.append("HDOP")
        
        return options

    def update_options(self, advanced_mode=False):
        """Met à jour les options de la liste déroulante"""
        if self.df is None:
            return
            
        current_selection = self.graph_selector.currentText()
        options = self.get_available_options(advanced_mode)
        
        self.graph_selector.clear()
        self.graph_selector.addItems(options)
        
        if current_selection in options:
            self.graph_selector.setCurrentText(current_selection)
        elif current_selection != "Aucun" and not advanced_mode:
            self.graph_selector.setCurrentText("Aucun")
        
        self.set_advanced_mode(advanced_mode)

    def update_graph(self, graph_type):
        if self.df is None or graph_type == "Aucun":
            self.ax.clear()
            self.ax.set_title("Aucun graphique sélectionné")
            self.figure.tight_layout()
            self.figure_canvas.draw()
            return
            
        self.ax.clear()
        
        if graph_type == "Vitesse":
            self.ax.plot(self.df["Temps"], self.df["Vitesse"], 'b-', linewidth=2)
            self.ax.set_ylabel("Vitesse (km/h)")
            self.ax.fill_between(self.df["Temps"], self.df["Vitesse"], alpha=0.3)
            
        elif graph_type == "Altitude":
            self.ax.plot(self.df["Temps"], self.df["Alt"], 'g-', linewidth=2)
            self.ax.set_ylabel("Altitude (m)")
            self.ax.fill_between(self.df["Temps"], self.df["Alt"], alpha=0.3)
            
        elif graph_type == "Tension":
            self.ax.plot(self.df["Temps"], self.df["Tension"], 'orange', linewidth=2)
            self.ax.set_ylabel("Tension (V)")
            
        elif graph_type == "GazFrein":
            self.ax.plot(self.df["Temps"], self.df["GazFrein"], 'purple', linewidth=2)
            self.ax.set_ylabel("Gaz/Frein")
            
        elif graph_type == "Énergie (bilan)":
            # Calculer le bilan énergétique : énergie chargée - énergie déchargée
            wh_charged = self.df["WHCharged"]
            wh_discharged = self.df["WHDischarged"]
            
            # Déterminer si les données sont déjà cumulatives
            if wh_charged.is_monotonic_increasing and wh_charged.iloc[-1] > wh_charged.iloc[0]:
                charged_data = wh_charged
            else:
                charged_data = wh_charged.cumsum()
                
            if wh_discharged.is_monotonic_increasing and wh_discharged.iloc[-1] > wh_discharged.iloc[0]:
                discharged_data = wh_discharged
            else:
                discharged_data = wh_discharged.cumsum()
            
            # Calcul du bilan
            energy_balance = charged_data - discharged_data
            
            # Graphique
            colors = ['green' if x >= 0 else 'red' for x in energy_balance]
            self.ax.plot(self.df["Temps"], energy_balance, 'blue', linewidth=2)
            self.ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
            self.ax.fill_between(self.df["Temps"], energy_balance, 0, 
                               where=(energy_balance >= 0), color='green', alpha=0.3, label='Excédent')
            self.ax.fill_between(self.df["Temps"], energy_balance, 0, 
                               where=(energy_balance < 0), color='red', alpha=0.3, label='Déficit')
            self.ax.set_ylabel("Bilan énergétique (Wh)")
            self.ax.legend()
            
        elif graph_type == "Puissance électrique":
            power = self.df["Tension"] * self.df["CurrentIn"]
            self.ax.plot(self.df["Temps"], power, 'orange', linewidth=2)
            self.ax.set_ylabel("Puissance électrique (W)")
            
        # Mode avancé uniquement
        elif graph_type == "Énergie chargée":
            # Vérifier si les données sont déjà cumulatives ou instantanées
            wh_data = self.df["WHCharged"]
            
            # Si les valeurs augmentent de façon monotone, c'est déjà cumulatif
            if wh_data.is_monotonic_increasing and wh_data.iloc[-1] > wh_data.iloc[0]:
                energy_data = wh_data  # Déjà cumulatif
            else:
                energy_data = wh_data.cumsum()  # Calculer le cumulatif
            
            self.ax.plot(self.df["Temps"], energy_data, 'g-', linewidth=2)
            self.ax.set_ylabel("Énergie chargée (Wh)")
            self.ax.fill_between(self.df["Temps"], energy_data, alpha=0.3, color='green')
            
            # Améliorer l'échelle Y
            y_min = energy_data.min()
            y_max = energy_data.max()
            if y_max > y_min:
                margin = (y_max - y_min) * 0.05  # 5% de marge
                self.ax.set_ylim(max(0, y_min - margin), y_max + margin)
            else:
                self.ax.set_ylim(bottom=0)
            
        elif graph_type == "Énergie déchargée":
            # Vérifier si les données sont déjà cumulatives ou instantanées
            wh_data = self.df["WHDischarged"]
            
            # Si les valeurs augmentent de façon monotone, c'est déjà cumulatif
            if wh_data.is_monotonic_increasing and wh_data.iloc[-1] > wh_data.iloc[0]:
                energy_data = wh_data  # Déjà cumulatif
            else:
                energy_data = wh_data.cumsum()  # Calculer le cumulatif
            
            self.ax.plot(self.df["Temps"], energy_data, 'r-', linewidth=2)
            self.ax.set_ylabel("Énergie déchargée (Wh)")
            self.ax.fill_between(self.df["Temps"], energy_data, alpha=0.3, color='red')
            
            # Améliorer l'échelle Y
            y_min = energy_data.min()
            y_max = energy_data.max()
            if y_max > y_min:
                margin = (y_max - y_min) * 0.05  # 5% de marge
                self.ax.set_ylim(max(0, y_min - margin), y_max + margin)
            else:
                self.ax.set_ylim(bottom=0)
            
        elif graph_type == "Distance (capteur)":
            self.ax.plot(self.df["Temps"], self.df["Distance"], 'brown', linewidth=2)
            self.ax.set_ylabel("Distance capteur (m)")
            
        elif graph_type == "Distance (GPS)":
            self.ax.plot(self.df["Temps"], self.df["Distance_GPS"], 'darkred', linewidth=2)
            self.ax.set_ylabel("Distance GPS (m)")
            
        elif graph_type == "Courant entrant":
            self.ax.plot(self.df["Temps"], self.df["CurrentIn"], 'g-', linewidth=2)
            self.ax.set_ylabel("Courant entrant (A)")
            
        elif graph_type == "Courant moteur":
            self.ax.plot(self.df["Temps"], self.df["MotorCurrent"], 'r-', linewidth=2)
            self.ax.set_ylabel("Courant moteur (A)")
            
        elif graph_type == "Latitude":
            self.ax.plot(self.df["Temps"], self.df["Lat"], 'navy', linewidth=2)
            self.ax.set_ylabel("Latitude")
            
        elif graph_type == "Longitude":
            self.ax.plot(self.df["Temps"], self.df["Lon"], 'teal', linewidth=2)
            self.ax.set_ylabel("Longitude")
            
        elif graph_type == "Vitesse satellite":
            self.ax.plot(self.df["Temps"], self.df["Vsat"], 'cyan', linewidth=2)
            self.ax.set_ylabel("Vitesse satellite")
            
        elif graph_type == "Cap":
            self.ax.plot(self.df["Temps"], self.df["Cap"], 'magenta', linewidth=2)
            self.ax.set_ylabel("Cap")
            
        elif graph_type == "Satellites":
            self.ax.plot(self.df["Temps"], self.df["Sat"], 'lime', linewidth=2)
            self.ax.set_ylabel("Nombre de satellites")
            
        elif graph_type == "HDOP":
            self.ax.plot(self.df["Temps"], self.df["HDOP"], 'coral', linewidth=2)
            self.ax.set_ylabel("HDOP")
        
        # Ligne de curseur
        self.cursor_line = self.ax.axvline(x=0, color='red', linestyle='--', linewidth=3, alpha=0.8)
        
        self.ax.set_xlabel("Temps (s)")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title(graph_type)
        
        # Événements souris
        self.figure_canvas.mpl_connect('motion_notify_event', self.on_graph_hover)
        self.figure_canvas.mpl_connect('button_press_event', self.on_graph_click)
        
        # Zoom synchronisé en mode avancé
        is_advanced = self.advanced_mode_callback() if self.advanced_mode_callback else False
        if is_advanced:
            self.ax.callbacks.connect('xlim_changed', self.on_xlim_changed)
        
        self.figure.tight_layout()
        self.figure_canvas.draw()

    def on_xlim_changed(self, ax):
        """Callback zoom"""
        if self.on_zoom_change and not self.zoom_active:
            xlims = ax.get_xlim()
            self.on_zoom_change(self.graph_id, xlims)

    def set_xlim(self, xlims):
        """Applique le zoom"""
        self.zoom_active = True
        self.ax.set_xlim(xlims)
        self.figure_canvas.draw_idle()
        self.zoom_active = False

    def on_graph_hover(self, event):
        if self.locked or event.inaxes != self.ax or self.df is None:
            return
        time_value = event.xdata
        if time_value is None:
            return
        closest_index = self.find_closest_index(time_value)
        self.on_cursor_change(closest_index, lock=False)

    def on_graph_click(self, event):
        if event.inaxes != self.ax or self.df is None:
            return
        time_value = event.xdata
        if time_value is None:
            return
        closest_index = self.find_closest_index(time_value)
        self.locked = not self.locked
        self.on_cursor_change(closest_index, lock=self.locked)

    def find_closest_index(self, time_value):
        time_diff = abs(self.df["Temps"] - time_value)
        return time_diff.idxmin()

    def update_cursor_position(self, index):
        if self.df is None or index >= len(self.df) or self.cursor_line is None:
            return
        row = self.df.iloc[index]
        self.cursor_line.set_xdata([row["Temps"], row["Temps"]])
        color = 'orange' if self.locked else 'red'
        self.cursor_line.set_color(color)
        self.figure_canvas.draw_idle()

    def unlock(self):
        self.locked = False
        if self.cursor_line:
            self.cursor_line.set_color('red')
            self.figure_canvas.draw_idle()


from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

class DualHandleSlider(QWidget):
    """Curseur personnalisé avec deux poignées pour sélectionner une plage"""
    
    rangeChanged = pyqtSignal(int, int)  # Signal émis quand la plage change
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.minimum = 0
        self.maximum = 100
        self.left_value = 0
        self.right_value = 100
        self.handle_radius = 10
        self.groove_height = 6
        self.dragging = None  # 'left', 'right', ou None
        self.setMinimumHeight(30)
        self.setMinimumWidth(200)
        
    def set_range(self, minimum, maximum):
        """Définit la plage de valeurs du curseur"""
        self.minimum = minimum
        self.maximum = maximum
        self.left_value = minimum
        self.right_value = maximum
        self.update()
        
    def set_values(self, left, right):
        """Définit les valeurs des deux poignées"""
        self.left_value = max(self.minimum, min(left, self.maximum))
        self.right_value = max(self.minimum, min(right, self.maximum))
        if self.left_value >= self.right_value:
            self.left_value = self.right_value - 1
        self.update()
        
    def get_values(self):
        """Retourne les valeurs actuelles des deux poignées"""
        return self.left_value, self.right_value
        
    def value_to_pixel(self, value):
        """Convertit une valeur en position pixel"""
        if self.maximum == self.minimum:
            return self.handle_radius
        ratio = (value - self.minimum) / (self.maximum - self.minimum)
        return self.handle_radius + ratio * (self.width() - 2 * self.handle_radius)
        
    def pixel_to_value(self, pixel):
        """Convertit une position pixel en valeur"""
        if self.width() <= 2 * self.handle_radius:
            return self.minimum
        ratio = (pixel - self.handle_radius) / (self.width() - 2 * self.handle_radius)
        ratio = max(0, min(1, ratio))
        return int(self.minimum + ratio * (self.maximum - self.minimum))
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Dessiner la rainure
        groove_rect = QRect(
            self.handle_radius, 
            (self.height() - self.groove_height) // 2,
            self.width() - 2 * self.handle_radius, 
            self.groove_height
        )
        painter.setBrush(QBrush(QColor(180, 180, 180)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(groove_rect, self.groove_height // 2, self.groove_height // 2)
        
        # Dessiner la zone sélectionnée
        left_pixel = self.value_to_pixel(self.left_value)
        right_pixel = self.value_to_pixel(self.right_value)
        
        selected_rect = QRect(
            int(left_pixel),
            (self.height() - self.groove_height) // 2,
            int(right_pixel - left_pixel),
            self.groove_height
        )
        painter.setBrush(QBrush(QColor(52, 152, 219)))  # Bleu
        painter.drawRoundedRect(selected_rect, self.groove_height // 2, self.groove_height // 2)
        
        # Dessiner la poignée gauche (début)
        left_center = QPoint(int(left_pixel), self.height() // 2)
        painter.setBrush(QBrush(QColor(52, 152, 219)))  # Bleu
        painter.setPen(QPen(QColor(41, 128, 185), 2))
        painter.drawEllipse(left_center, self.handle_radius, self.handle_radius)
        
        # Icône sur la poignée gauche
        painter.setPen(QPen(Qt.GlobalColor.white, 2))
        painter.drawText(left_center.x() - 4, left_center.y() + 3, "▶")
        
        # Dessiner la poignée droite (fin)
        right_center = QPoint(int(right_pixel), self.height() // 2)
        painter.setBrush(QBrush(QColor(231, 76, 60)))  # Rouge
        painter.setPen(QPen(QColor(192, 57, 43), 2))
        painter.drawEllipse(right_center, self.handle_radius, self.handle_radius)
        
        # Icône sur la poignée droite
        painter.setPen(QPen(Qt.GlobalColor.white, 2))
        painter.drawText(right_center.x() - 4, right_center.y() + 3, "⏹")
        
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            left_pixel = self.value_to_pixel(self.left_value)
            right_pixel = self.value_to_pixel(self.right_value)
            
            # Vérifier quelle poignée est cliquée
            left_distance = abs(event.position().x() - left_pixel)
            right_distance = abs(event.position().x() - right_pixel)
            
            if left_distance <= self.handle_radius:
                self.dragging = 'left'
            elif right_distance <= self.handle_radius:
                self.dragging = 'right'
            else:
                # Clic ailleurs : déplacer la poignée la plus proche
                if left_distance < right_distance:
                    self.dragging = 'left'
                    self.left_value = self.pixel_to_value(event.position().x())
                else:
                    self.dragging = 'right'
                    self.right_value = self.pixel_to_value(event.position().x())
                
                # S'assurer que les valeurs restent dans l'ordre
                if self.left_value >= self.right_value:
                    if self.dragging == 'left':
                        self.left_value = self.right_value - 1
                    else:
                        self.right_value = self.left_value + 1
                
                self.update()
                self.rangeChanged.emit(self.left_value, self.right_value)
                
    def mouseMoveEvent(self, event):
        if self.dragging:
            new_value = self.pixel_to_value(event.position().x())
            
            if self.dragging == 'left':
                self.left_value = max(self.minimum, min(new_value, self.right_value - 1))
            elif self.dragging == 'right':
                self.right_value = max(self.left_value + 1, min(new_value, self.maximum))
                
            self.update()
            self.rangeChanged.emit(self.left_value, self.right_value)
            
    def mouseReleaseEvent(self, event):
        self.dragging = None


class AnalysisPage(QWidget):
    def __init__(self, go_back_callback):
        super().__init__()
        self.go_back_callback = go_back_callback
        self.df = None
        self.df_filtered = None  # DataFrame filtré selon la sélection
        self.comments_file = "comments.json"
        self.current_file = None
        self.current_index = 0
        self.graphs = []
        self.max_graphs = 6
        self.map_click_index = None
        self.advanced_mode = False
        
        # Variables pour la sélection de plage
        self.range_start = 0
        self.range_end = 0
        
        self.init_ui()

    def init_ui(self):
        self.header_height = 30

        # Panel d'informations (partie gauche)
        info_panel = QWidget()
        info_layout = QVBoxLayout()
        
        # Résumé du trajet
        stats_title = QLabel(" ")
        stats_title.setStyleSheet("font-weight: bold; font-size: 14px; color: white;")
        info_layout.addWidget(stats_title)
        
        self.general_stats = QLabel("Chargez un fichier")
        self.general_stats.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.general_stats.setMinimumWidth(200)
        info_layout.addWidget(self.general_stats)
        
        # NOUVEAU: Curseur de sélection de plage temporelle avec double poignée
        range_title = QLabel("🎯 Sélection de plage")
        range_title.setStyleSheet("font-weight: bold; font-size: 14px; color: white; margin-top: 10px;")
        info_layout.addWidget(range_title)
        
        # Container pour le curseur double
        range_container = QWidget()
        range_layout = QVBoxLayout()
        range_layout.setContentsMargins(10, 10, 10, 10)
        
        # Labels pour afficher les valeurs
        self.range_labels = QHBoxLayout()
        self.start_label = QLabel("Début: 0.0s")
        self.start_label.setStyleSheet("color: #3498db; font-weight: bold; font-size: 11px;")
        self.end_label = QLabel("Fin: 0.0s")
        self.end_label.setStyleSheet("color: #e74c3c; font-weight: bold; font-size: 11px;")
        self.range_labels.addWidget(self.start_label)
        self.range_labels.addStretch()
        self.range_labels.addWidget(self.end_label)
        
        range_layout.addLayout(self.range_labels)
        
        # Curseur à double poignée
        self.dual_slider = DualHandleSlider()
        self.dual_slider.rangeChanged.connect(self.on_range_change)
        range_layout.addWidget(self.dual_slider)
        
        
        
        range_container.setLayout(range_layout)
        range_container.setStyleSheet("""
            QWidget {
                background-color: #34495e;
                border-radius: 5px;
                margin: 2px;
            }
        """)
        info_layout.addWidget(range_container)
        
        # Séparateur
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        info_layout.addWidget(separator)
        
        # Informations instantanées
        detail_title = QLabel("📊 Informations instantanées")
        detail_title.setStyleSheet("font-weight: bold; font-size: 14px; color: white;")
        info_layout.addWidget(detail_title)
        
        self.instant_info = QLabel("Sélectionnez un point")
        self.instant_info.setStyleSheet("""
            background-color: #2c3e50; 
            color: white;
            padding: 10px; 
            border-radius: 5px; 
            font-family: monospace;
            font-size: 11px;
        """)
        self.instant_info.setWordWrap(True)
        self.instant_info.setMinimumHeight(120)
        info_layout.addWidget(self.instant_info)
        
        info_layout.addStretch()
        info_panel.setLayout(info_layout)

        # Zone graphiques optimisée
        graphs_panel = QWidget()
        graphs_layout = QVBoxLayout()
        graphs_layout.setContentsMargins(0, 0, 0, 0)
        graphs_layout.setSpacing(0)
        
        # Boutons de contrôle optimisés
        buttons_layout = QHBoxLayout()
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(3)
        
        # Boutons de même taille
        button_width = 120
        button_height = 30
        
        self.add_graph_button = QPushButton("➕ Ajouter")
        self.add_graph_button.setFixedSize(button_width, button_height)
        self.add_graph_button.setToolTip("Ajouter graphique")
        self.add_graph_button.clicked.connect(self.add_graph)
        
        self.remove_graph_button = QPushButton("➖ Supprimer")
        self.remove_graph_button.setFixedSize(button_width, button_height)
        self.remove_graph_button.setToolTip("Supprimer dernier")
        self.remove_graph_button.clicked.connect(self.remove_last_graph)
        
        self.mode_button = QPushButton("🔧 Mode Normal")
        self.mode_button.setFixedSize(button_width, button_height)
        self.mode_button.setCheckable(True)
        self.mode_button.setToolTip("Mode Normal/Avancé")
        self.mode_button.clicked.connect(self.toggle_advanced_mode)
        self.mode_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 5px;
                border-radius: 3px;
                font-size: 12px;
            }
            QPushButton:checked {
                background-color: #e3931b;
            }
        """)
        
        buttons_layout.addWidget(self.add_graph_button)
        buttons_layout.addWidget(self.remove_graph_button)
        buttons_layout.addWidget(self.mode_button)
        buttons_layout.addStretch()
        
        graphs_layout.addLayout(buttons_layout)
        
        # Container graphiques sans scroll et sans marges
        self.graphs_container = QWidget()
        self.graphs_container.setContentsMargins(0, 0, 0, 0)
        self.graphs_grid = QGridLayout()
        self.graphs_grid.setContentsMargins(0, 0, 0, 0)
        self.graphs_grid.setSpacing(0)
        self.graphs_container.setLayout(self.graphs_grid)
        
        graphs_layout.addWidget(self.graphs_container)
        graphs_panel.setLayout(graphs_layout)

        # Carte
        self.map_view = QWebEngineView()
        self.map_view.setMinimumSize(400, 250)

        # Commentaires
        comment_container = QWidget()
        comment_layout = QVBoxLayout()
        
        comment_title = QLabel("💬 Commentaires")
        comment_title.setStyleSheet("font-weight: bold; font-size: 14px; color: white;")
        comment_layout.addWidget(comment_title)
        
        self.comment_text = QTextEdit()
        self.comment_text.setPlaceholderText("Écrire vos commentaires ici...")
        self.save_comment_button = QPushButton("Enregistrer les commentaires")
        self.save_comment_button.clicked.connect(self.save_comments)
        
        comment_layout.addWidget(self.comment_text)
        comment_layout.addWidget(self.save_comment_button)
        comment_container.setLayout(comment_layout)

        # Header
        self.original_logo_pixmap = QPixmap("images/logo_styx_blanc.png")
        header_widget = QWidget()
        header_widget.setMinimumHeight(self.header_height)
        header_widget.setMaximumHeight(self.header_height)
        header_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.logo_label = QLabel(header_widget)
        self.logo_label.setScaledContents(True)
        self.logo_label.setGeometry(0, 0, self.width(), 100)

        self.title_label = QLabel("Trajet", header_widget)
        self.title_label.setStyleSheet("color: rgba(255, 255, 255, 180); font-weight: bold; font-size: 24pt;")
        self.title_label.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.title_label.setGeometry(self.width() // 5, 0, self.width() - self.width() // 5, 100)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        # Layouts principaux avec marges réduites
        left_splitter = QSplitter(Qt.Orientation.Vertical)
        left_splitter.addWidget(info_panel)
        left_splitter.addWidget(comment_container)
        left_splitter.setSizes([400, 50])
        
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.addWidget(graphs_panel)
        right_splitter.addWidget(self.map_view)
        right_splitter.setSizes([450, 250])
        
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.addWidget(left_splitter)
        main_splitter.addWidget(right_splitter)
        main_splitter.setSizes([300, 900])

        # Layout principal avec marges réduites
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(2, 2, 2, 2)
        main_layout.setSpacing(2)
        main_layout.addWidget(header_widget)
        main_layout.addWidget(main_splitter)

        # Layout des boutons en bas
        bottom_buttons_layout = QHBoxLayout()
        bottom_buttons_layout.setContentsMargins(0, 0, 0, 0)
        
        # Bouton génération de rapport (1/5 de la taille du bouton retour)
        self.generate_report_button = QPushButton("📊 Rapport")
        self.generate_report_button.setFixedHeight(30)  # Même hauteur que le bouton retour
        self.generate_report_button.clicked.connect(self.generate_report)
        self.generate_report_button.setStyleSheet("""
            QPushButton {
                background-color: #e67e22;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #d35400;
            }
        """)
        
        self.back_button = QPushButton("Retour")
        self.back_button.clicked.connect(self.go_back_callback)
        
        # Ajouter les boutons avec proportion 1/5 - 4/5
        bottom_buttons_layout.addWidget(self.generate_report_button, 1)
        bottom_buttons_layout.addWidget(self.back_button, 4)
        
        main_layout.addLayout(bottom_buttons_layout)

        self.setLayout(main_layout)
        
        # Créer le premier graphique
        self.add_graph()

    def on_range_change(self, start_value, end_value):
        """Callback pour le changement de plage du curseur double"""
        if self.df is None:
            return
            
        # Calculer les indices correspondants
        max_index = len(self.df) - 1
        self.range_start = int((start_value / 100.0) * max_index)
        self.range_end = int((end_value / 100.0) * max_index)
        
        # S'assurer que les indices sont valides
        self.range_start = max(0, min(self.range_start, max_index))
        self.range_end = max(self.range_start + 1, min(self.range_end, max_index))
        
        # Mettre à jour l'affichage des labels
        if "Temps" in self.df.columns:
            start_time = self.df.iloc[self.range_start]["Temps"]
            end_time = self.df.iloc[self.range_end]["Temps"]
            self.start_label.setText(f"Début: {start_time:.1f}s")
            self.end_label.setText(f"Fin: {end_time:.1f}s")
        
        # Appliquer le filtre
        self.apply_range_filter()

    def reset_range(self):
        """Remet le curseur aux positions initiales et met à jour tout"""
        if self.df is not None:
            # Réinitialiser les valeurs du slider
            self.dual_slider.set_values(0, 100)
            
            # Réinitialiser les indices
            self.range_start = 0
            self.range_end = len(self.df) - 1
            
            # Réappliquer le filtre complet
            self.df_filtered = self.df.copy()
            self.df_filtered = self.df_filtered.reset_index(drop=True)
            
            # Mettre à jour les labels
            if "Temps" in self.df.columns:
                start_time = self.df.iloc[0]["Temps"]
                end_time = self.df.iloc[-1]["Temps"]
                self.start_label.setText(f"Début: {start_time:.1f}s")
                self.end_label.setText(f"Fin: {end_time:.1f}s")
            
            # Forcer la mise à jour de tous les composants
            self.update_general_stats()
            self.update_graphs_data()
            self.display_map()
            
            # Réinitialiser le curseur au début
            if len(self.df_filtered) > 0:
                self.on_cursor_change(0)
                
            print("Plage réinitialisée : vue complète du trajet")

    def apply_range_filter(self):
        """Applique le filtre de plage sur les données"""
        if self.df is None:
            return
            
        # Filtrer le DataFrame
        self.df_filtered = self.df.iloc[self.range_start:self.range_end + 1].copy()
        
        # IMPORTANT: Réinitialiser l'index pour éviter les décalages
        self.df_filtered = self.df_filtered.reset_index(drop=True)
        
        # Mettre à jour tous les composants
        self.update_general_stats()
        self.update_graphs_data()
        self.display_map()
        
        # Réinitialiser la position du curseur au début de la sélection
        if len(self.df_filtered) > 0:
            self.on_cursor_change(0)

    def update_graphs_data(self):
        """Met à jour les données des graphiques avec auto-scale et forçage du redessin"""
        for graph in self.graphs:
            # Mettre à jour les données
            graph.set_data(self.df_filtered)
            
            # Forcer la mise à jour des options (inclut souvent le redessin)
            graph.update_options(self.advanced_mode)
            
            # Essayer différentes méthodes pour forcer le redessin
            try:
                # Méthode 1: méthode auto_scale personnalisée
                if hasattr(graph, 'auto_scale'):
                    graph.auto_scale()
                
                # Méthode 2: forcer le redessin avec repaint
                if hasattr(graph, 'repaint'):
                    graph.repaint()
                
                # Méthode 3: matplotlib avec canvas
                elif hasattr(graph, 'canvas'):
                    if hasattr(graph, 'ax'):
                        graph.ax.relim()
                        graph.ax.autoscale()
                    graph.canvas.draw()
                    graph.canvas.flush_events()
                
                # Méthode 4: matplotlib avec figure
                elif hasattr(graph, 'figure'):
                    for ax in graph.figure.get_axes():
                        ax.relim()
                        ax.autoscale()
                    graph.figure.canvas.draw()
                    graph.figure.canvas.flush_events()
                
                # Méthode 5: Widget Qt générique
                elif hasattr(graph, 'update'):
                    graph.update()
                
                # Méthode 6: Si c'est un QWidget, forcer la mise à jour
                if hasattr(graph, 'updateGeometry'):
                    graph.updateGeometry()
                    
            except Exception as e:
                print(f"Erreur lors de la mise à jour du graphique: {e}")
                continue

    def toggle_advanced_mode(self):
        """Bascule entre mode normal et avancé"""
        self.advanced_mode = not self.advanced_mode
        
        if self.advanced_mode:
            self.mode_button.setText("🔧 Mode Avancé")
        else:
            self.mode_button.setText("🔧 Mode Normal")
        
        for graph in self.graphs:
            graph.update_options(self.advanced_mode)
            if not self.advanced_mode:
                graph.reset_zoom()

    def get_advanced_mode(self):
        """Retourne l'état du mode avancé"""
        return self.advanced_mode

    def add_graph(self):
        if len(self.graphs) >= self.max_graphs:
            QMessageBox.information(self, "Maximum atteint", f"Maximum {self.max_graphs} graphiques.")
            return
            
        graph_widget = GraphWidget(
            len(self.graphs), 
            self.on_cursor_change, 
            self.on_zoom_change,
            self.get_advanced_mode
        )
        
        # Configuration optimisée de la grille (3 colonnes sur 2 lignes)
        row = len(self.graphs) // 3
        col = len(self.graphs) % 3
        
        self.graphs_grid.addWidget(graph_widget, row, col)
        self.graphs.append(graph_widget)
        
        if self.df_filtered is not None:
            graph_widget.set_data(self.df_filtered)
            graph_widget.update_options(self.advanced_mode)
            
                        
            
        # Mise à jour des boutons
        self.update_button_states()

    def remove_last_graph(self):
        if len(self.graphs) <= 1:
            return
            
        last_graph = self.graphs.pop()
        self.graphs_grid.removeWidget(last_graph)
        last_graph.deleteLater()
        
        # Mise à jour des boutons
        self.update_button_states()

    def update_button_states(self):
        """Met à jour l'état des boutons"""
        self.add_graph_button.setEnabled(len(self.graphs) < self.max_graphs)
        self.remove_graph_button.setEnabled(len(self.graphs) > 1)
        
        # S'assurer que le style reste cohérent
        button_style = """
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """
        self.add_graph_button.setStyleSheet(button_style)
        self.remove_graph_button.setStyleSheet(button_style)

    def on_cursor_change(self, index, lock=False):
        """Callback curseur - maintenant basé sur df_filtered"""
        if self.df_filtered is None:
            return
            
        self.current_index = index
        
        if not lock:
            for graph in self.graphs:
                graph.unlock()
        
        for graph in self.graphs:
            graph.update_cursor_position(index)
        
        if "Lat" in self.df_filtered.columns and "Lon" in self.df_filtered.columns and index < len(self.df_filtered):
            row = self.df_filtered.iloc[index]
            self.update_map_marker(row["Lat"], row["Lon"])
        
        self.update_instant_info(index)

    def on_zoom_change(self, source_graph_id, xlims):
        """Callback zoom synchronisé"""
        for i, graph in enumerate(self.graphs):
            if i != source_graph_id:
                graph.set_xlim(xlims)

    def update_instant_info(self, index):
        """Met à jour les informations instantanées - maintenant basé sur df_filtered"""
        if self.df_filtered is None or index >= len(self.df_filtered):
            self.instant_info.setText("Index invalide")
            return
            
        row = self.df_filtered.iloc[index]
        
        try:
            info_text = f"""<b>⏱️ Instant t = {row['Temps']:.1f}s</b><br><br>"""
            info_text += f"<b>Vitesse:</b> {row['Vitesse']:.1f} km/h<br>"
            
            if 'Alt' in self.df_filtered.columns:
                info_text += f"<b>Altitude:</b> {row['Alt']:.1f} m<br>"
            if 'Tension' in self.df_filtered.columns:
                info_text += f"<b>Tension:</b> {row['Tension']:.1f} V<br>"
            if 'CurrentIn' in self.df_filtered.columns:
                info_text += f"<b>Courant In:</b> {row['CurrentIn']:.2f} A<br>"
            if 'MotorCurrent' in self.df_filtered.columns:
                info_text += f"<b>Courant Motor:</b> {row['MotorCurrent']:.2f} A<br>"
            if 'Distance' in self.df_filtered.columns:
                info_text += f"<b>Distance (capteur):</b> {row['Distance']:.0f} m<br>"
            if 'Distance_GPS' in self.df_filtered.columns:
                info_text += f"<b>Distance (GPS):</b> {row['Distance_GPS']:.0f} m<br>"
            if 'WHCharged' in self.df_filtered.columns:
                info_text += f"<b>Wh Charged:</b> {row['WHCharged']:.2f}<br>"
            if 'WHDischarged' in self.df_filtered.columns:
                info_text += f"<b>Wh Discharged:</b> {row['WHDischarged']:.2f}<br>"
            
            self.instant_info.setText(info_text)
        except Exception as e:
            self.instant_info.setText(f"Erreur: {e}")

    def load_file(self, csv_path):
            try:
                df_raw = pd.read_csv(csv_path)
                df_clean = clean_data(df_raw)
                self.df = calculate_gps_distance(df_clean)

                try:
                    filename = os.path.basename(csv_path).split('.')[0]
                    timestamp_str = filename.replace("session_", "")
                    dt = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
                    formatted_date = dt.strftime("%d %B %Y, à %Hh%M")
                    self.title_label.setText(f"Trajet du {formatted_date}")
                except Exception:
                    self.title_label.setText("Trajet")

                self.current_file = csv_path
                self.load_comments()
                
                # Initialiser le curseur avec les nouvelles données
                max_index = len(self.df) - 1
                self.dual_slider.set_range(0, 100)
                self.dual_slider.set_values(0, 100)
                
                self.range_start = 0
                self.range_end = max_index
                self.df_filtered = self.df.copy()
                
                # Mettre à jour les labels des curseurs
                if "Temps" in self.df.columns:
                    start_time = self.df.iloc[0]["Temps"]
                    end_time = self.df.iloc[-1]["Temps"]
                    self.start_label.setText(f"Début: {start_time:.1f}s")
                    self.end_label.setText(f"Fin: {end_time:.1f}s")
                
                self.update_general_stats()
                
                for i, graph in enumerate(self.graphs):
                    graph.set_data(self.df_filtered)
                    graph.update_options(self.advanced_mode)
                    
                    # Sélection automatique pour le premier graphique
                    if i == 0:  # Premier graphique seulement
                        available_options = graph.get_available_options(self.advanced_mode)
                        if "Vitesse" in available_options:
                            graph.graph_selector.setCurrentText("Vitesse")
                            graph.update_graph("Vitesse")
                        elif len(available_options) > 1:
                            selected_option = available_options[1]
                            graph.graph_selector.setCurrentText(selected_option)
                            graph.update_graph(selected_option)
                    
                    # Forcer l'auto-scale
                    if hasattr(graph, 'auto_scale'):
                        graph.auto_scale()
                
                self.display_map()
                
                if len(self.df_filtered) > 0:
                    self.on_cursor_change(0)

            except Exception as e:
                QMessageBox.critical(self, "Erreur", f"Erreur de chargement : {e}")
    
    def update_general_stats(self):
        """Met à jour les statistiques générales - maintenant basé sur df_filtered"""
        if self.df_filtered is None or len(self.df_filtered) == 0:
            self.general_stats.setText("Aucune donnée")
            return
            
        try:
            if "Distance_GPS" in self.df_filtered.columns:
                # Pour la distance, prendre la différence entre fin et début
                dist_start = self.df_filtered["Distance_GPS"].iloc[0]
                dist_end = self.df_filtered["Distance_GPS"].iloc[-1]
                dist = dist_end - dist_start
                dist_type = "GPS"
            elif "Distance" in self.df_filtered.columns:
                dist_start = self.df_filtered["Distance"].iloc[0]
                dist_end = self.df_filtered["Distance"].iloc[-1]
                dist = dist_end - dist_start
                dist_type = "capteur"
            else:
                dist = 0
                dist_type = "N/A"
            
            temps_start = self.df_filtered["Temps"].iloc[0] if "Temps" in self.df_filtered.columns else 0
            temps_end = self.df_filtered["Temps"].iloc[-1] if "Temps" in self.df_filtered.columns else 0
            temps_sec = temps_end - temps_start
            temps_min = temps_sec / 60
            
            vitesse_moy = self.df_filtered["Vitesse"].mean() if "Vitesse" in self.df_filtered.columns else 0
            vitesse_max = self.df_filtered["Vitesse"].max() if "Vitesse" in self.df_filtered.columns else 0
            
            alt_max = self.df_filtered["Alt"].max() if "Alt" in self.df_filtered.columns else 0
            alt_min = self.df_filtered["Alt"].min() if "Alt" in self.df_filtered.columns else 0
            denivele = alt_max - alt_min
            
            wh_charged = self.df_filtered["WHCharged"].sum() if "WHCharged" in self.df_filtered.columns else 0
            wh_discharged = self.df_filtered["WHDischarged"].sum() if "WHDischarged" in self.df_filtered.columns else 0
            
            tension_moy = self.df_filtered["Tension"].mean() if "Tension" in self.df_filtered.columns else 0
            current_max = self.df_filtered["CurrentIn"].max() if "CurrentIn" in self.df_filtered.columns else 0

            info_text = f"""
            <h3>📊 Résumé du trajet (sélection)</h3>
            <b>Distance ({dist_type}) :</b> {dist:.0f} m<br>
            <b>Durée :</b> {temps_min:.1f} min<br>
            <b>Vitesse moy :</b> {vitesse_moy:.1f} km/h<br>
            <b>Vitesse max :</b> {vitesse_max:.1f} km/h
            <h3>⚡ Énergie</h3>
            <b>Chargée :</b> {wh_charged:.1f} Wh<br>
            <b>Déchargée :</b> {wh_discharged:.1f} Wh
            <h3>🏔️ Altitude</h3>
            <b>Min/Max :</b> {alt_min:.0f}/{alt_max:.0f} m<br>
            <b>Dénivelé :</b> {denivele:.0f} m<br>
            """
            
            self.general_stats.setText(info_text)
        except Exception as e:
            self.general_stats.setText(f"Erreur: {e}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        w = self.width()
        h = self.header_height
        self.logo_label.setGeometry(0, 0, w, h)
        if not self.original_logo_pixmap.isNull():
            scaled_pixmap = self.original_logo_pixmap.scaled(
                w, h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
            self.logo_label.setPixmap(scaled_pixmap)
        self.title_label.setGeometry(w // 5, 0, w - w // 5, h)

    def display_map(self):
        """Affiche la carte - maintenant basé sur df_filtered avec indicateurs départ/arrivée"""
        if self.df_filtered is None or "Lat" not in self.df_filtered.columns or "Lon" not in self.df_filtered.columns:
            self.map_view.setHtml("<h3>Colonnes GPS 'Lat' et 'Lon' non trouvées.</h3>")
            return

        latitudes = self.df_filtered["Lat"].tolist()
        longitudes = self.df_filtered["Lon"].tolist()

        valid_coords = [(lat, lon, i) for i, (lat, lon) in enumerate(zip(latitudes, longitudes)) 
                       if not (pd.isna(lat) or pd.isna(lon))]
        
        if not valid_coords:
            self.map_view.setHtml("<h3>Aucune donnée GPS valide.</h3>")
            return

        coordinates = [[lon, lat] for lat, lon, _ in valid_coords]
        geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coordinates},
                "properties": {}
            }]
        }
        
        geojson_str = json.dumps(geojson)
        center_lat = valid_coords[0][0]
        center_lon = valid_coords[0][1]
        
        # Coordonnées de départ et d'arrivée
        start_lat, start_lon = valid_coords[0][0], valid_coords[0][1]
        end_lat, end_lon = valid_coords[-1][0], valid_coords[-1][1]

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8" />
            <title>Carte GPS</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
            <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
            <style>
                #mapid {{ height: 100%; width: 100%; }}
                html, body {{ height: 100%; margin: 0; }}
                .custom-marker {{
                    background: transparent;
                    border: none;
                    font-size: 16px;
                }}
            </style>
        </head>
        <body>
            <div id="mapid"></div>
            <script>
                var map = L.map('mapid');
                L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                    maxZoom: 19,
                    attribution: '© OpenStreetMap contributors'
                }}).addTo(map);

                var geojson = {geojson_str};
                var routeLayer = L.geoJSON(geojson, {{
                    style: {{ color: '#3388ff', weight: 4, opacity: 0.8 }}
                }}).addTo(map);

                // Marqueur de départ (vert avec icône play) - taille uniforme
                var startIcon = L.divIcon({{
                    className: 'custom-marker',
                    html: '<div style="background-color: #27ae60; color: white; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.3); font-weight: bold; font-size: 12px;">▶</div>',
                    iconSize: [24, 24],
                    iconAnchor: [12, 12]
                }});
                
                var startMarker = L.marker([{start_lat}, {start_lon}], {{
                    icon: startIcon,
                    title: 'Départ'
                }}).addTo(map);

                map.fitBounds(routeLayer.getBounds());
                startMarker.bindPopup('<b>🏁 Départ</b><br>Début de la sélection');

                // Marqueur d'arrivée (rouge avec icône stop) - taille uniforme
                var endIcon = L.divIcon({{
                    className: 'custom-marker',
                    html: '<div style="background-color: #e74c3c; color: white; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.3); font-weight: bold; font-size: 12px;">⏹</div>',
                    iconSize: [24, 24],
                    iconAnchor: [12, 12]
                }});
                
                var endMarker = L.marker([{end_lat}, {end_lon}], {{
                    icon: endIcon,
                    title: 'Arrivée'
                }}).addTo(map);
                
                endMarker.bindPopup('<b>🏁 Arrivée</b><br>Fin de la sélection');

                // Curseur de position actuelle - taille uniforme
                var cursorMarker = null;
                window.clickedIndex = null;
                
                map.on('click', function(e) {{
                    var clickedLat = e.latlng.lat;
                    var clickedLon = e.latlng.lng;
                    var coordinates = geojson.features[0].geometry.coordinates;
                    var minDistance = Infinity;
                    var closestIndex = 0;
                    
                    coordinates.forEach(function(coord, index) {{
                        var lon = coord[0];
                        var lat = coord[1];
                        var distance = Math.sqrt(
                            Math.pow((lat - clickedLat) * 111000, 2) + 
                            Math.pow((lon - clickedLon) * 111000 * Math.cos(clickedLat * Math.PI / 180), 2)
                        );
                        if (distance < minDistance) {{
                            minDistance = distance;
                            closestIndex = index;
                        }}
                    }});
                    
                    window.clickedIndex = closestIndex;
                }});

                window.updateCursor = function(lat, lon) {{
                    if (cursorMarker) {{
                        map.removeLayer(cursorMarker);
                    }}
                    
                    if (lat != null && lon != null && !isNaN(lat) && !isNaN(lon)) {{
                        // Curseur de position actuelle (orange) - même taille que les autres
                        cursorMarker = L.circleMarker([lat, lon], {{
                            radius: 10,
                            fillColor: '#f39c12',
                            color: '#ffffff',
                            weight: 2,
                            opacity: 1,
                            fillOpacity: 0.8
                        }}).addTo(map);
                        
                        cursorMarker.bindPopup('<b>📍 Position actuelle</b>');
                    }}
                }};
                
                window.getClickedIndex = function() {{
                    var index = window.clickedIndex;
                    window.clickedIndex = null;
                    return index;
                }};
                
                // Pas de rezoom automatique - garder la vue actuelle
                // Commenté : map.fitBounds(group.getBounds().pad(0.1));
            </script>
        </body>
        </html>
        """
        
        self.map_view.setHtml(html)
        self.start_map_click_timer()

    def start_map_click_timer(self):
        """Timer pour détecter les clics carte"""
        self.map_timer = QTimer()
        self.map_timer.timeout.connect(self.check_map_clicks)
        self.map_timer.start(100)

    def check_map_clicks(self):
        """Vérifie les clics sur la carte - maintenant basé sur df_filtered"""
        js_code = "window.getClickedIndex ? window.getClickedIndex() : null;"
        
        def handle_result(result):
            if result is not None and isinstance(result, (int, float)):
                try:
                    index = int(result)
                    if 0 <= index < len(self.df_filtered):
                        self.on_cursor_change(index, lock=True)
                except (ValueError, TypeError):
                    pass
        
        self.map_view.page().runJavaScript(js_code, handle_result)

    def update_map_marker(self, lat, lon):
        """Met à jour le marqueur sur la carte"""
        if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
            return
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return
            
        js_code = f"""
        if (typeof window.updateCursor === 'function') {{
            window.updateCursor({lat}, {lon});
        }}
        """
        self.map_view.page().runJavaScript(js_code)

    def save_comments(self):
        comments = self.comment_text.toPlainText()
        data = {}
        if os.path.exists(self.comments_file):
            try:
                with open(self.comments_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except:
                data = {}
        if self.current_file:
            data[self.current_file] = comments
            with open(self.comments_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            self.save_comment_button.setText("Commentaires enregistrés ✓")

    
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


    def generate_report(self):
        """Génère un rapport complet du trajet sélectionné avec haute résolution"""
        if self.df_filtered is None or len(self.df_filtered) == 0:
            QMessageBox.warning(self, "Attention", "Aucune donnée à inclure dans le rapport.")
            return
        
        # Dialog personnalisé pour les options de rapport
        dialog = QDialog(self)
        dialog.setWindowTitle("Génération du Rapport")
        dialog.setFixedSize(450, 250)
        
        layout = QVBoxLayout()
        
        # Nom du rapport
        layout.addWidget(QLabel("Nom du rapport:"))
        name_input = QLineEdit(f"Rapport_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        layout.addWidget(name_input)
        
        # Option fichier de base
        layout.addWidget(QLabel(""))  # Espacement
        include_base_file = QCheckBox("Inclure le fichier CSV de base complet")
        include_base_file.setChecked(True)
        layout.addWidget(include_base_file)
        
        info_label = QLabel("⚠️ Le fichier de base contient TOUTES les données y compris\nles positions GPS de départ/arrivée complètes du trajet entier.\nSi vous ne voulez pas partager ces informations, décochez cette option.")
        info_label.setStyleSheet("color: #e67e22; font-size: 11px; background-color: #fef9e7; padding: 4px; border-radius: 4px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        
        
        # Boutons
        buttons_layout = QHBoxLayout()
        ok_button = QPushButton("Générer le Rapport")
        ok_button.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
        """)
        cancel_button = QPushButton("Annuler")
        
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)
        
        buttons_layout.addWidget(cancel_button)
        buttons_layout.addWidget(ok_button)
        layout.addLayout(buttons_layout)
        
        dialog.setLayout(layout)
        
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
            
        report_name = name_input.text().strip()
        include_base = include_base_file.isChecked()
        
        if not report_name:
            QMessageBox.warning(self, "Attention", "Veuillez saisir un nom de rapport.")
            return
        
        try:
            # Créer le dossier du rapport
            reports_dir = "rapports"
            if not os.path.exists(reports_dir):
                os.makedirs(reports_dir)
                
            report_dir = os.path.join(reports_dir, report_name)
            if not os.path.exists(report_dir):
                os.makedirs(report_dir)
            else:
                reply = QMessageBox.question(
                    self, 'Dossier existant', 
                    f'Le dossier "{report_name}" existe déjà. Écraser le contenu?',
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.No:
                    return
            
            # Dialogue de progression
            progress = QProgressDialog("Génération du rapport haute résolution...", "Annuler", 0, 6, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setAutoClose(True)
            progress.setAutoReset(True)
            progress.show()
            
            # Étape 1: Sauvegarder les données CSV
            progress.setLabelText("💾 Sauvegarde des données CSV...")
            progress.setValue(1)
            QApplication.processEvents()
            
            if include_base and self.current_file:
                # Copier le fichier original complet
                base_filename = os.path.join(report_dir, f"{report_name}_donnees_completes.csv")
                shutil.copy2(self.current_file, base_filename)
            
            # Sauvegarder les données filtrées
            filtered_filename = os.path.join(report_dir, f"{report_name}_donnees_selection.csv")
            self.df_filtered.to_csv(filtered_filename, index=False)
            
            # Étape 2: Générer les graphiques haute résolution
            progress.setLabelText("📊 Génération des graphiques 300 DPI...")
            progress.setValue(2)
            QApplication.processEvents()
            
            self.generate_high_res_graphs(report_dir)
            
            # Étape 3: Générer la carte GPS haute résolution
            progress.setLabelText("🗺️ Génération de la carte GPS 1200x800px...")
            progress.setValue(3)
            QApplication.processEvents()
            
            self.generate_high_res_map(report_dir)
            
            # Attendre la génération de la carte
            QTimer.singleShot(1000, lambda: self.finalize_report(progress, report_dir, report_name, include_base))
            
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Erreur lors de la génération du rapport:\n{e}")
            print(f"Erreur détaillée: {e}")
            import traceback
            traceback.print_exc()

    def finalize_report(self, progress, report_dir, report_name, include_base):
        """Finalise la génération du rapport"""
        try:
            # Étape 4: Générer le rapport HTML
            progress.setLabelText("📝 Génération du rapport HTML...")
            progress.setValue(4)
            QApplication.processEvents()
            
            self.generate_html_report_hd(report_dir, report_name, include_base)
            
            # Étape 5: Finalisation
            progress.setLabelText("✅ Finalisation...")
            progress.setValue(5)
            QApplication.processEvents()
            
            progress.setValue(6)
            
            # Compter les fichiers générés
            files_generated = []
            files_generated.append(f"• {report_name}.html (rapport principal HD)")
            files_generated.append(f"• {report_name}_donnees_selection.csv")
            if include_base:
                files_generated.append(f"• {report_name}_donnees_completes.csv")
            files_generated.append("• carte_gps.png (1200x800px)")
            files_generated.append(f"• {len(self.graphs)} graphiques (300 DPI)")
            
            QMessageBox.information(
                self, 
                "✅ Rapport Haute Résolution Généré", 
                f"Rapport sauvegardé dans:\n{os.path.abspath(report_dir)}\n\n" +
                f"Fichiers générés:\n" + "\n".join(files_generated)
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Erreur lors de la finalisation:\n{e}")

    def generate_high_res_graphs(self, report_dir):
        """Génère les graphiques en haute résolution 300 DPI"""
        graphs_dir = os.path.join(report_dir, "graphiques")
        if not os.path.exists(graphs_dir):
            os.makedirs(graphs_dir)
            
        for i, graph in enumerate(self.graphs):
            try:
                screenshot_path = os.path.join(graphs_dir, f"graphique_{i+1}.png")
                
                # Méthode 1: Matplotlib avec haute résolution
                if hasattr(graph, 'figure'):
                    graph.figure.savefig(
                        screenshot_path, 
                        dpi=300,  # Haute résolution
                        bbox_inches='tight',
                        facecolor='white',
                        edgecolor='none',
                        format='png',
                        pad_inches=0.1
                    )
                    print(f"Graphique {i+1} sauvegardé en 300 DPI (matplotlib)")
                
                # Méthode 2: Canvas matplotlib
                elif hasattr(graph, 'canvas') and hasattr(graph.canvas, 'figure'):
                    graph.canvas.figure.savefig(
                        screenshot_path, 
                        dpi=300,
                        bbox_inches='tight',
                        facecolor='white',
                        edgecolor='none',
                        format='png',
                        pad_inches=0.1
                    )
                    print(f"Graphique {i+1} sauvegardé en 300 DPI (canvas)")
                
                # Méthode 3: Widget Qt avec scaling
                elif hasattr(graph, 'grab'):
                    # Capturer à une résolution plus élevée
                    original_size = graph.size()
                    scale_factor = 3  # Tripler la résolution pour 300 DPI
                    
                    pixmap = graph.grab()
                    # Redimensionner pour une meilleure qualité
                    scaled_pixmap = pixmap.scaled(
                        original_size * scale_factor,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                    scaled_pixmap.save(screenshot_path, 'PNG', 98)  # Qualité maximale
                    print(f"Graphique {i+1} sauvegardé en haute résolution (Qt)")
                else:
                    print(f"Impossible de générer le graphique haute résolution {i+1}")
                    
            except Exception as e:
                print(f"Erreur lors de la génération haute résolution du graphique {i+1}: {e}")

    def generate_high_res_map(self, report_dir):
        """Génère la carte GPS en haute résolution 1200x800"""
        try:
            # Créer une carte temporaire plus grande pour la haute résolution
            self.create_high_res_map_html(report_dir)
            print("Carte HTML haute résolution créée")
            
        except Exception as e:
            print(f"Erreur lors de la génération de la carte haute résolution: {e}")

    def create_high_res_map_html(self, report_dir):
        import os, json, pandas as pd

        if self.df_filtered is None or "Lat" not in self.df_filtered.columns or "Lon" not in self.df_filtered.columns:
            return

        latitudes = self.df_filtered["Lat"].tolist()
        longitudes = self.df_filtered["Lon"].tolist()

        valid_coords = [(lat, lon) for lat, lon in zip(latitudes, longitudes) if not (pd.isna(lat) or pd.isna(lon))]

        if not valid_coords:
            return

        # Coordonnées pour Leaflet : [lon, lat]
        coordinates = [[lon, lat] for lat, lon in valid_coords]

        geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coordinates
                },
                "properties": {}
            }]
        }

        geojson_str = json.dumps(geojson)

        start_lat, start_lon = valid_coords[0]
        end_lat, end_lon = valid_coords[-1]

        # Centre de la carte au milieu de la trace
        center_lat = sum(lat for lat, _ in valid_coords) / len(valid_coords)
        center_lon = sum(lon for _, lon in valid_coords) / len(valid_coords)

        html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8" />
        <title>Carte GPS Haute Résolution</title>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
            #mapid {{ height: 800px; width: 1200px; }}
            html, body {{ margin: 0; padding: 0; background: white; }}
        </style>
    </head>
    <body>
        <div id="mapid"></div>
        <script>
            var map = L.map('mapid').setView([{center_lat}, {center_lon}], 13);

            L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                maxZoom: 19,
                attribution: '© OpenStreetMap contributors'
            }}).addTo(map);

            var geojson = {geojson_str};

            // Trace GPS (ligne bleue)
            L.geoJSON(geojson, {{
                style: function(feature) {{
                    return {{color: 'blue', weight: 5, opacity: 0.7}};
                }}
            }}).addTo(map);

            // Marqueur départ vert
            L.circleMarker([{start_lat}, {start_lon}], {{
                radius: 10,
                fillColor: 'green',
                color: 'white',
                weight: 2,
                opacity: 1,
                fillOpacity: 0.9
            }}).addTo(map).bindPopup("Départ");

            // Marqueur arrivée rouge
            L.circleMarker([{end_lat}, {end_lon}], {{
                radius: 10,
                fillColor: 'red',
                color: 'white',
                weight: 2,
                opacity: 1,
                fillOpacity: 0.9
            }}).addTo(map).bindPopup("Arrivée");
        </script>
    </body>
    </html>"""

        temp_map_path = os.path.join(report_dir, "temp_map.html")
        with open(temp_map_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        # Lancer la capture avec un délai pour laisser le temps au fichier d'être créé
        QTimer.singleShot(1000, lambda: self._capture_high_res_map(report_dir))

    def generate_high_res_map(self, report_dir):
        """Méthode simple: capture la carte existante en HD"""
        try:
            print("📸 Capture simple de la carte existante...")
            screenshot_path = os.path.join(report_dir, "carte_gps.png")
            
            # Capturer la carte actuelle
            pixmap = self.map_view.grab()
            
            if not pixmap.isNull():
                # Redimensionner en haute résolution
                hd_pixmap = pixmap.scaled(
                    1200, 800,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                
                success = hd_pixmap.save(screenshot_path, 'PNG', 98)
                if success:
                    file_size = os.path.getsize(screenshot_path)
                    print(f"✅ Carte simple sauvegardée: {file_size} bytes")
                    return True
                else:
                    print("❌ Échec sauvegarde")
            else:
                print("❌ Capture impossible")
            
            return False
        except Exception as e:
            print(f"❌ Erreur capture simple: {e}")
            return False

    def generate_html_report_hd(self, report_dir, report_name, include_base_file):
        """Génère le rapport HTML"""
        try:
            # Récupérer les statistiques
            stats = self.get_report_statistics()
            
            # Récupérer les commentaires
            comments = self.comment_text.toPlainText() or "Aucun commentaire"
            
            # Section fichiers de données
            files_section = f"""
        <div class="section">
            <h2>📁 Fichiers de Données</h2>
            <div class="files-info">
                <div class="file-item selection">
                    <h4>📊 Données de la sélection</h4>
                    <p><strong>Fichier:</strong> {report_name}_donnees_selection.csv</p>
                    <p><strong>Contenu:</strong> Données filtrées selon la plage sélectionnée ({stats['temps_debut']:.1f}s à {stats['temps_fin']:.1f}s)</p>
                    <p><strong>Points de données:</strong> {len(self.df_filtered):,}</p>
                    <p><strong>🔒 Confidentialité:</strong> Contient uniquement la portion sélectionnée</p>
                </div>"""
            
            if include_base_file:
                files_section += f"""
                <div class="file-item complete">
                    <h4>🗂️ Fichier de base complet</h4>
                    <p><strong>Fichier:</strong> {report_name}_donnees_completes.csv</p>
                    <p><strong>Contenu:</strong> Toutes les données du trajet original, y compris les positions GPS complètes</p>
                    <p><strong>⚠️ Attention:</strong> Ce fichier contient l'intégralité du parcours avec les positions de départ et d'arrivée exactes</p>
                </div>"""
            
            files_section += """
            </div>
        </div>"""
            
            # Générer le HTML complet
            html_content = f"""
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Rapport d'Analyse - {report_name}</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 40px;
                background-color: #f8fafc;
                color: #334155;
                line-height: 1.6;
            }}
            .header {{
                text-align: center;
                margin-bottom: 40px;
                padding: 30px;
                background: linear-gradient(135deg, #3b82f6, #1e40af);
                color: white;
                border-radius: 12px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            .header h1 {{
                margin: 0 0 10px 0;
                font-size: 2.5em;
            }}
            .header h2 {{
                margin: 0 0 15px 0;
                font-weight: 300;
                opacity: 0.9;
            }}
            .section {{
                background: white;
                margin: 30px 0;
                padding: 30px;
                border-radius: 12px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                border: 1px solid #e2e8f0;
            }}
            .section h2 {{
                margin-top: 0;
                color: #1e40af;
                border-bottom: 2px solid #e2e8f0;
                padding-bottom: 10px;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 20px;
                margin: 25px 0;
            }}
            .stat-item {{
                background: linear-gradient(135deg, #f1f5f9, #e2e8f0);
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                border: 1px solid #cbd5e1;
            }}
            .stat-value {{
                font-size: 28px;
                font-weight: bold;
                color: #1e40af;
                margin-bottom: 5px;
            }}
            .stat-label {{
                color: #64748b;
                font-size: 13px;
                font-weight: 500;
            }}
            .graphs-container {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
                gap: 25px;
                margin: 25px 0;
            }}
            .graph-item {{
                text-align: center;
                background: #f8fafc;
                padding: 15px;
                border-radius: 8px;
                border: 1px solid #e2e8f0;
            }}
            .graph-item h3 {{
                margin: 0 0 15px 0;
                color: #475569;
            }}
            .graph-item img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .map-container {{
                text-align: center;
                margin: 25px 0;
                background: #f8fafc;
                padding: 20px;
                border-radius: 8px;
                border: 1px solid #e2e8f0;
            }}
            .map-container img {{
                max-width: 100%;
                height: auto;
                border: 2px solid #cbd5e1;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            .files-info {{
                display: grid;
                gap: 20px;
            }}
            .file-item {{
                padding: 20px;
                border-radius: 8px;
                border: 1px solid #e2e8f0;
            }}
            .file-item.selection {{
                background: #f0f9ff;
                border-left: 4px solid #3b82f6;
            }}
            .file-item.complete {{
                background: #fef3c7;
                border-left: 4px solid #f59e0b;
            }}
            .file-item h4 {{
                margin: 0 0 12px 0;
                color: #1f2937;
            }}
            .file-item p {{
                margin: 6px 0;
                font-size: 14px;
            }}
            .comments {{
                background: #fffbeb;
                padding: 20px;
                border-left: 4px solid #f59e0b;
                border-radius: 8px;
                white-space: pre-wrap;
                font-family: 'Courier New', monospace;
            }}
            .info-box {{
                background: #f0fdf4;
                padding: 15px;
                border-left: 4px solid #22c55e;
                border-radius: 8px;
                margin: 15px 0;
                font-size: 13px;
                font-weight: 500;
            }}
            .tech-info {{
                background: #f8fafc;
                padding: 20px;
                border-radius: 8px;
                border: 1px solid #e2e8f0;
            }}
            .tech-info p {{
                margin: 8px 0;
            }}
            .footer {{
                text-align: center;
                margin-top: 50px;
                color: #64748b;
                font-size: 13px;
            }}
            @media print {{
                body {{ margin: 20px; }}
                .section {{ break-inside: avoid; }}
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📊 Rapport d'Analyse</h1>
            <h2>{report_name}</h2>
            <p>Généré le {datetime.now().strftime('%d %B %Y à %H:%M:%S')}</p>
        </div>

        <div class="section">
            <h2>📈 Statistiques du Trajet Analysé</h2>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value">{stats['distance']:.0f} m</div>
                    <div class="stat-label">Distance ({stats['distance_type']})</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{stats['duree']:.1f} min</div>
                    <div class="stat-label">Durée</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{stats['vitesse_moy']:.1f} km/h</div>
                    <div class="stat-label">Vitesse Moyenne</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{stats['vitesse_max']:.1f} km/h</div>
                    <div class="stat-label">Vitesse Maximum</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{stats['denivele']:.0f} m</div>
                    <div class="stat-label">Dénivelé</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{stats['wh_discharged']:.1f} Wh</div>
                    <div class="stat-label">Énergie Consommée</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>🗺️ Carte GPS</h2>
            <div class="map-container">
                <img src="carte_gps.png" alt="Carte GPS du trajet analysé">
            </div>
            <div class="info-box">
                📍 Carte générée avec marqueurs de départ (🟢) et d'arrivée (🔴)
            </div>
        </div>

        <div class="section">
            <h2>📊 Graphiques d'Analyse</h2>
            <div class="graphs-container">
    """
            
            # Ajouter les graphiques
            graphs_dir = os.path.join(report_dir, "graphiques")
            if os.path.exists(graphs_dir):
                for i in range(len(self.graphs)):
                    graph_file = f"graphique_{i+1}.png"
                    graph_path = os.path.join(graphs_dir, graph_file)
                    if os.path.exists(graph_path):
                        html_content += f'''
                <div class="graph-item">
                    <h3>Graphique {i+1}</h3>
                    <img src="graphiques/{graph_file}" alt="Graphique {i+1}">
                </div>'''
            
            html_content += f"""
            </div>
        </div>

        {files_section}

        <div class="section">
            <h2>💬 Commentaires</h2>
            <div class="comments">{comments}</div>
        </div>

        <div class="section">
            <h2>📋 Informations Techniques</h2>
            <div class="tech-info">
                <p><strong>Période analysée:</strong> {stats['temps_debut']:.1f}s à {stats['temps_fin']:.1f}s</p>
                <p><strong>Nombre de points analysés:</strong> {len(self.df_filtered):,}</p>
                <p><strong>Fichier source:</strong> {os.path.basename(self.current_file) if self.current_file else 'N/A'}</p>
            </div>
        </div>

        <div class="footer">
            <p>Rapport généré par STYX Analyse • {datetime.now().year}</p>
        </div>
    </body>
    </html>"""
            
            # Sauvegarder le fichier HTML
            html_path = os.path.join(report_dir, f"{report_name}.html")
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"✅ Rapport HTML généré: {html_path}")
                
        except Exception as e:
            print(f"❌ Erreur lors de la génération du HTML: {e}")



    def get_report_statistics(self):
        """Récupère les statistiques pour le rapport"""
        stats = {}
        
        if "Distance_GPS" in self.df_filtered.columns:
            dist_start = self.df_filtered["Distance_GPS"].iloc[0]
            dist_end = self.df_filtered["Distance_GPS"].iloc[-1]
            stats['distance'] = dist_end - dist_start
            stats['distance_type'] = "GPS"
        elif "Distance" in self.df_filtered.columns:
            dist_start = self.df_filtered["Distance"].iloc[0]
            dist_end = self.df_filtered["Distance"].iloc[-1]
            stats['distance'] = dist_end - dist_start
            stats['distance_type'] = "capteur"
        else:
            stats['distance'] = 0
            stats['distance_type'] = "N/A"
        
        temps_start = self.df_filtered["Temps"].iloc[0] if "Temps" in self.df_filtered.columns else 0
        temps_end = self.df_filtered["Temps"].iloc[-1] if "Temps" in self.df_filtered.columns else 0
        stats['duree'] = (temps_end - temps_start) / 60
        stats['temps_debut'] = temps_start
        stats['temps_fin'] = temps_end
        
        stats['vitesse_moy'] = self.df_filtered["Vitesse"].mean() if "Vitesse" in self.df_filtered.columns else 0
        stats['vitesse_max'] = self.df_filtered["Vitesse"].max() if "Vitesse" in self.df_filtered.columns else 0
        
        alt_max = self.df_filtered["Alt"].max() if "Alt" in self.df_filtered.columns else 0
        alt_min = self.df_filtered["Alt"].min() if "Alt" in self.df_filtered.columns else 0
        stats['denivele'] = alt_max - alt_min
        
        stats['wh_charged'] = self.df_filtered["WHCharged"].sum() if "WHCharged" in self.df_filtered.columns else 0
        stats['wh_discharged'] = self.df_filtered["WHDischarged"].sum() if "WHDischarged" in self.df_filtered.columns else 0
        
        return stats

    def load_comments(self):
        if not self.current_file or not os.path.exists(self.comments_file):
            self.comment_text.clear()
            return
        try:
            with open(self.comments_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            comments = data.get(self.current_file, "")
            self.comment_text.setPlainText(comments)
            self.save_comment_button.setText("Enregistrer les commentaires")
        except:
            self.comment_text.clear()   
    
# Note d'intégration:
# Pour utiliser ce code, assurez-vous d'avoir les imports suivants au début de votre fichier:
# 
# from PyQt6.QtWidgets import *
# from PyQt6.QtCore import *
# from PyQt6.QtGui import *
# from PyQt6.QtWebEngineWidgets import QWebEngineView
# import pandas as pd
# import json
# import os
# from datetime import datetime
#
# Et que les fonctions suivantes sont définies ailleurs dans votre code:
# - clean_data(df_raw)
# - calculate_gps_distance(df_clean)
# - GraphWidget (votre classe de widget graphique existante)

# --- Fenêtre principale ---
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Styx Analyse")
        self.setGeometry(100, 100, 1200, 800)
        self.init_ui()

    def init_ui(self):
        self.stack = QStackedWidget()
        self.analysis_page = AnalysisPage(self.go_home)
        self.home_page = HomePage(self.switch_to_analysis)
        self.stack.addWidget(self.home_page)
        self.stack.addWidget(self.analysis_page)
        layout = QVBoxLayout()
        layout.addWidget(self.stack)
        self.setLayout(layout)

    def switch_to_analysis(self, csv_path):
        self.analysis_page.load_file(csv_path)
        self.stack.setCurrentWidget(self.analysis_page)

    def go_home(self):
        self.stack.setCurrentWidget(self.home_page)


# --- Point d'entrée ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())

