import streamlit as st
import os
import pandas as pd
import geopandas as gpd
import joblib
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from io import BytesIO
import matplotlib.patheffects as path_effects
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker

# --- Config page ---
st.set_page_config(
    page_title="🌧️ Prévision des Inondations à Ouagadougou",
    layout="wide"
)

# --- Chemins relatifs ---
Prediction_inondation = os.path.dirname(os.path.abspath(__file__))
path_shapefile = os.path.join(Prediction_inondation, "data", "Secteurs_Ouaga.shp")
path_metadata = os.path.join(Prediction_inondation, "data", "donnee_statique.csv")
path_modele = os.path.join(Prediction_inondation, "model_inondation.pkl")

# --- Chargement du modèle bundle ---
@st.cache_data
def load_model_bundle():
    bundle = joblib.load(path_modele)
    return bundle['pipelines'], bundle['feature_names']

pipelines_final, feature_order = load_model_bundle()

# --- Chargement des données statiques ---
@st.cache_data
def load_shapefile():
    gdf = gpd.read_file(path_shapefile)
    gdf.columns = gdf.columns.str.strip()
    if "SECTEUR" in gdf.columns:
        gdf = gdf.rename(columns={"SECTEUR": "Secteur"})
    return gdf.to_crs(epsg=32630)

@st.cache_data
def load_metadata():
    df = pd.read_csv(path_metadata, sep=';')
    df.columns = df.columns.str.strip()
    return df

gdf_sectors = load_shapefile()
df_metadata = load_metadata()

# --- Dictionnaires de traduction ---

trads = {
    "title": {
        "Français": "🌧️ Prévision des Inondations à Ouagadougou",
        "English": "🌧️ Flood Forecasting in Ouagadougou"
    },
    "parameters": {
        "Français": "Paramètres d'entrée",
        "English": "Input Parameters"
    },
    "precipitation": {
        "Français": "Précipitation (mm)",
        "English": "Precipitation (mm)"
    },
    "year": {
        "Français": "Année",
        "English": "Year"
    },
    "month": {
        "Français": "Mois",
        "English": "Month"
    },
    "day": {
        "Français": "Jour",
        "English": "Day"
    },
    "select_sectors": {
        "Français": "Sélectionnez des secteurs :",
        "English": "Select sectors:"
    },
    "select_at_least": {
        "Français": "Veuillez sélectionner au moins un secteur.",
        "English": "Please select at least one sector."
    },
    "soil_moisture": {
        "Français": "Humidité du sol du secteur",
        "English": "Soil moisture of sector"
    },
    "button_calculate": {
        "Français": "Calculer la probabilité d'inondation",
        "English": "Calculate flood probability"
    },
    "info_click_button": {
        "Français": "Cliquez sur 'Calculer' pour générer les résultats et la carte.",
        "English": "Click 'Calculate' to generate results and map."
    },
    "missing_columns": {
        "Français": "Colonnes manquantes :",
        "English": "Missing columns:"
    },
    "map_title": {
        "Français": "Carte des probabilités d'inondation (%)",
        "English": "Map of flood probabilities (%)"
    },
    "global_confidence": {
        "Français": "Niveau de confiance global",
        "English": "Global confidence level"
    },
    "individual_confidence": {
        "Français": "Probabilité globale et niveau de confiance individuel",
        "English": "Global probability and individual confidence level"
    },
    "download_map": {
        "Français": "📷 Télécharger la carte",
        "English": "📷 Download map"
    },
    "download_results": {
        "Français": "🗅️ Télécharger les résultats",
        "English": "🗅️ Download results"
    }
}

col_names_translation = {
    "Français": {
        "Secteur": "Secteur",
        "Probabilité globale d'inondation": "Probabilité globale d'inondation",
        "Confiance_proxy": "Confiance_proxy",
        "Annee": "Année",
        "Mois": "Mois",
        "Jour": "Jour",
        "Precipitation": "Précipitation",
        "Humidite_sol": "Humidité_sol",
        "Superficie_depotoir": "Superficie_depotoir",
        "Longueur_caniveau": "Longueur_caniveau",
        "Plan_eau": "Plan_eau",
        "Type_sol": "Type_sol",
        "Relief": "Relief"
    },
    "English": {
        "Secteur": "Sector",
        "Probabilité globale d'inondation": "Global flood probability",
        "Confiance_proxy": "Confidence proxy",
        "Annee": "Year",
        "Mois": "Month",
        "Jour": "Day",
        "Precipitation": "Precipitation",
        "Humidite_sol": "Soil moisture",
        "Superficie_depotoir": "Dump area",
        "Longueur_caniveau": "Gutter length",
        "Plan_eau": "Water bodies",
        "Type_sol": "Soil type",
        "Relief": "Relief"
    }
}

# --- Choix langue ---
langue = st.selectbox("Langue / Language", ["Français", "English"])

# --- Titre principal ---
st.title(trads["title"][langue])

# --- Interface en deux colonnes ---
col_inputs, col_map = st.columns([1, 3])

with col_inputs:
    st.subheader(trads["parameters"][langue])

    precipitation = st.number_input(trads["precipitation"][langue], 0.0, 500.0, 0.0, step=0.1)
    annee = st.number_input(trads["year"][langue], 1980, 2050, 2024)
    mois = st.selectbox(trads["month"][langue], list(range(1, 13)), index=6)
    jour = st.selectbox(trads["day"][langue], list(range(1, 32)), index=14)

    secteurs_list = sorted(gdf_sectors['Secteur'].unique())
    secteur_options = (
        ["Ouagadougou_ville"] if langue == "Français" else ["Ouagadougou_city"]
    ) + [
        (f"Secteur {s}" if langue == "Français" else f"Sector {s}") for s in secteurs_list
    ]

    selection = st.multiselect(trads["select_sectors"][langue], options=secteur_options, default=[])

    if not selection:
        st.warning(trads["select_at_least"][langue])
        st.stop()

    if ("Ouagadougou_ville" if langue == "Français" else "Ouagadougou_city") in selection:
        selected_secteurs = secteurs_list
    else:
        # Extraire le numéro de secteur selon langue (Secteur X ou Sector X)
        prefix = "Secteur " if langue == "Français" else "Sector "
        selected_secteurs = [int(s.split(prefix)[1]) for s in selection if s.startswith(prefix)]

    humidites = {}
    for sec in selected_secteurs:
        humidites[sec] = st.slider(
            f"{trads['soil_moisture'][langue]} {sec}",
            min_value=0.0, max_value=1.0,
            value=0.5, key=f"hum_{sec}"
        )

with col_map:
    if st.button(trads["button_calculate"][langue]):
        df_sel = pd.DataFrame({"Secteur": selected_secteurs})
        df_full = df_sel.merge(df_metadata, on="Secteur", how="left")
        df_full["Annee"] = annee
        df_full["Mois"] = mois
        df_full["Jour"] = jour
        df_full["Precipitation"] = precipitation
        df_full["Humidite_sol"] = df_full["Secteur"].map(humidites)

        missing = [f for f in feature_order if f not in df_full.columns]
        if missing:
            st.error(f"{trads['missing_columns'][langue]} {missing}")
            st.stop()

        df_model = df_full[feature_order]
        probas = [pipe.predict_proba(df_model)[:, 1] for pipe in pipelines_final.values()]
        arr = np.column_stack(probas)
        df_full["Probabilité globale d'inondation"] = arr.mean(axis=1)
        df_full["Confiance_proxy"] = 1 - np.std(arr, axis=1)
        df_full["Probabilité globale d'inondation"] *= 100
        df_full["Confiance_proxy"] *= 100

        # Préparation GeoDataFrame pour la carte
        gdf_plot = gdf_sectors.merge(
            df_full[["Secteur", "Probabilité globale d'inondation"]],
            on="Secteur", how="left"
        ).fillna({"Probabilité globale d'inondation": 0})

        # Colormap
        cmap = LinearSegmentedColormap.from_list("risk", ["green", "yellow", "orange", "red"])
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.subplots_adjust(left=0.05, right=0.98, bottom=0.12, top=0.92)

        gdf_plot.plot(
            column="Probabilité globale d'inondation", cmap=cmap,
            linewidth=0.5, edgecolor="white", vmin=0, vmax=100, ax=ax, zorder=1
        )
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=0, vmax=100))



        for _, rr in gdf_plot.iterrows():
            if not rr.geometry.is_empty:
                x, y = rr.geometry.centroid.coords[0]
                ax.text(x, y, str(int(rr["Secteur"])),
                        ha="center", va="center", fontsize=7, fontweight="bold",
                        path_effects=[path_effects.withStroke(linewidth=1, foreground="white")])

        sb = ScaleBar(1, units="m", location='lower right', length_fraction=0.2, pad=-0.35,
                      box_color='white', box_alpha=0.7, font_properties={'size': 8})
        ax.add_artist(sb)

        bounds = gdf_plot.total_bounds
        x_arrow = bounds[2] - 500
        y_arrow = bounds[3] - 1000
        ax.annotate('N', xy=(x_arrow, y_arrow), xytext=(x_arrow, y_arrow - 0.00005),
                    arrowprops=dict(facecolor='black', width=4, headwidth=10),
                    ha='center', va='center', fontsize=14, fontweight='bold', zorder=3)

        cm = df_full["Confiance_proxy"].mean()
        ax.text(0.01, 0.99, f"{trads['global_confidence'][langue]}: {cm:.1f} %",
                transform=ax.transAxes, ha='left', va='top',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        


        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='4%', pad=0.02)
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=0, vmax=100))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax, orientation='vertical', label='%')
        cbar.ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0, symbol='%'))

        ax.text(0.01, -0.00, trads["map_title"][langue],
                transform=ax.transAxes, ha='left', va='bottom',
                fontsize=13, fontweight='bold')

        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

        st.pyplot(fig)

        # Traduction colonnes avant affichage et export
        df_export = df_full.rename(columns=col_names_translation[langue])
                
        with st.expander(trads["individual_confidence"][langue]):
            for _, r in df_export.iterrows():
                sec = r[col_names_translation[langue]["Secteur"]]
                prob = r[col_names_translation[langue]["Probabilité globale d'inondation"]]
                conf = r[col_names_translation[langue]["Confiance_proxy"]]

                if langue == "Français":
                    st.write(f"- Secteur {sec} : Probabilité = {prob:.1f} %, Confiance = {conf:.1f} %")
                else:
                    st.write(f"- Sector {sec}: Probability = {prob:.1f} %, Confidence = {conf:.1f} %")


        # Téléchargements
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        st.download_button(trads["download_map"][langue], buf, 
                           "carte_inondation.png" if langue=="Français" else "flood_map.png", 
                           "image/png")

        st.download_button(
            label=trads["download_results"][langue],
            data=df_export.to_csv(index=False).encode('utf-8-sig'),
            file_name="resultats_inondation.csv" if langue=="Français" else "flood_results.csv",
            mime="text/csv"
        )
    else:
        st.info(trads["info_click_button"][langue])
