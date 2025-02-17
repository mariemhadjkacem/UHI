from fastkml import kml
import os
import json
import csv

# 🔹 Chemin du fichier KML
kml_file = r"data\Building_Footprint.kml"  # Assurez-vous que le chemin est correct

# 📂 Vérifier que le fichier existe
if not os.path.exists(kml_file):
    print("❌ Fichier KML introuvable.")
    exit()

# 🏗 Lire le fichier en mode binaire
with open(kml_file, "rb") as f:
    kml_content = f.read()

# 🏗 Initialisation de FastKML
k = kml.KML()
k.from_string(kml_content)

# ✅ Extraction des données du KML
features = []

# Affichage des structures de documents pour débogage
for document in k.features:  # Accède aux documents
    print(f"📄 Document trouvé : {document.name}")

    for folder in document.features:  # Accède aux dossiers
        print(f"📂 Dossier trouvé : {folder.name}")

        for placemark in folder.features:  # Accède aux placemarks
            print(f"📌 Placemark trouvé : {placemark.name}")
            
            # Afficher les attributs du placemark pour vérifier la présence des données
            print(f"  - Attributs du Placemark : {placemark.attributes}")

            # Vérifier si le placemark contient une géométrie
            if hasattr(placemark, 'geometry') and placemark.geometry:
                geom = placemark.geometry
                print(f"  - Géométrie trouvée : {geom.geom_type}")

                # Vérification de la géométrie et des coordonnées
                if geom.geom_type == "MultiGeometry":
                    for geometry in geom.geometries():
                        if geometry.geom_type == "Polygon":
                            coordinates = list(geometry.exterior.coords)
                            print(f"    - Coordonnées Polygon : {coordinates}")
                            features.append({
                                "id": placemark.name or "N/A",
                                "type": "Polygon",
                                "coordinates": coordinates
                            })

                elif geom.geom_type == "Polygon":
                    coordinates = list(geom.exterior.coords)
                    print(f"    - Coordonnées Polygon : {coordinates}")
                    features.append({
                        "id": placemark.name or "N/A",
                        "type": "Polygon",
                        "coordinates": coordinates
                    })

# Vérifier si des géométries ont été extraites
if not features:
    print("❌ Aucune géométrie extraite. Vérifiez la structure du fichier KML.")
else:
    print(f"✅ {len(features)} géométries extraites.")

# 📌 Création du dossier de sortie
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# ✅ **Export en CSV**
csv_file = os.path.join(output_dir, "buildings.csv")
with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["ID", "Type", "Coordinates"])
    for feature in features:
        # Convertir les coordonnées en chaîne lisible pour CSV
        coordinates_str = " ".join([f"({x},{y})" for x, y in feature["coordinates"]])
        writer.writerow([feature["id"], feature["type"], coordinates_str])

print(f"✅ Fichier CSV enregistré : {csv_file}")

# ✅ **Export en GeoJSON**
geojson_file = os.path.join(output_dir, "buildings.geojson")
geojson_data = {
    "type": "FeatureCollection",
    "features": []
}

for feature in features:
    geojson_data["features"].append({
        "type": "Feature",
        "properties": {"id": feature["id"]},
        "geometry": {
            "type": feature["type"],
            "coordinates": feature["coordinates"]
        }
    })

with open(geojson_file, 'w', encoding='utf-8') as f:
    json.dump(geojson_data, f, indent=4)

print(f"✅ Fichier GeoJSON enregistré : {geojson_file}")
import xml.etree.ElementTree as ET

# Parse the KML file
tree = ET.parse('your_kml_file.kml')
root = tree.getroot()

# Define the KML namespace
namespace = {'kml': 'http://www.opengis.net/kml/2.2'}

# Extract polygons from the KML
polygons = []

for placemark in root.findall('.//kml:Placemark', namespace):
    # Look for <MultiGeometry> or <Polygon> elements
    polygon = placemark.find('.//kml:Polygon', namespace)
    if polygon is not None:
        # Extract coordinates from <coordinates> tag
        coordinates = polygon.find('.//kml:coordinates', namespace)
        if coordinates is not None:
            coord_text = coordinates.text.strip()
            coord_pairs = coord_text.split(' ')
            coords = [tuple(map(float, pair.split(','))) for pair in coord_pairs]
            polygons.append(coords)

# Print the extracted polygons
for polygon in polygons:
    print(polygon)
