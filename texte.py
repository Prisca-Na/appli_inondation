# ====== app.py ======
import streamlit as st
import os
import json
import pandas as pd
import geopandas as gpd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from io import BytesIO
import matplotlib.patheffects as path_effects
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ====== translations.json ======
# Place this file in data/translations.json
{
  "French": {
    "title": "🌧️ Prévision des Inondations à Ouagadougou",
    "precip": "Précipitation (mm)",
    "year": "Année",
    "month": "Mois",
    "day": "Jour",
    "select_sectors": "Sélectionnez des secteurs :",
    "all_sectors_label": "Ouagadougou_ville",
    "warning_no_sector": "Veuillez sélectionner au moins un secteur.",
    "humid_slider": "Humidité du sol du secteur {sec}",
    "calc_button": "Calculer la probabilité d'inondation",
    "info_click": "Cliquez sur 'Calculer' pour générer les résultats et la carte.",
    "expander_title": "Probabilité globale et niveau de confiance individuel",
    "download_map": "📷 Télécharger la carte",
    "download_results": "🗅️ Télécharger les résultats",
    "map_caption": "Carte des probabilités d'inondation",
    "sector_line": "- Secteur {sec}: Probabilité={prob:.3f}, Confiance={conf:.3f}",
    "global_confidence": "Niveau de confiance global : {cm:.3f}"
  },
  "English": {
    "title": "🌧️ Flood Forecasting in Ouagadougou",
    "precip": "Precipitation (mm)",
    "year": "Year",
    "month": "Month",
    "day": "Day",
    "select_sectors": "Select sectors:",
    "all_sectors_label": "Ouagadougou_city",
    "warning_no_sector": "Please select at least one sector.",
    "humid_slider": "Soil humidity for sector {sec}",
    "calc_button": "Compute flood probability",
    "info_click": "Click 'Compute' to generate the results and map.",
    "expander_title": "Global probability and individual confidence level",
    "download_map": "📷 Download map",
    "download_results": "🗅️ Download results",
    "map_caption": "Flood probability map",
    "sector_line": "- Sector {sec}: Probability={prob:.3f}, Confidence={conf:.3f}",
    "global_confidence": "Global confidence level: {cm:.3f}"
  }
}


# Load translations
with open("data/translations.json", "r", encoding="utf-8") as f:
    LANGUAGES = json.load(f)

# Streamlit config
st.set_page_config(
    page_title="🌧️ Prévision des Inondations à Ouagadougou",
    layout="wide"
)

# Language selector
lang = st.sidebar.selectbox(
    "Language / Langue",
    options=list(LANGUAGES.keys()),
    index=0
)
T = LANGUAGES[lang]

# Paths
Prediction_inondation = os.path.dirname(os.path.abspath(__file__))
path_shp = os.path.join(Prediction_inondation, "data", "Secteurs_Ouaga.shp")
path_meta = os.path.join(Prediction_inondation, "data", "donnee_statique.csv")
path_model = os.path.join(Prediction_inondation, "model_inondation.pkl")

# Load model bundle
@st.cache_data
def load_model_bundle():
    bundle = joblib.load(path_model)
    return bundle['pipelines'], bundle['feature_names']

pipelines_final, feature_order = load_model_bundle()

# Load spatial & metadata
@st.cache_data
def load_shapefile():
    gdf = gpd.read_file(path_shp)
    gdf.columns = gdf.columns.str.strip()
    if "SECTEUR" in gdf.columns:
        gdf = gdf.rename(columns={"SECTEUR": "Secteur"})
    return gdf.to_crs(epsg=32630)

@st.cache_data
def load_metadata():
    df = pd.read_csv(path_meta, sep=';')
    df.columns = df.columns.str.strip()
    return df


gdf_sectors = load_shapefile()
df_metadata = load_metadata()

# App UI
st.title(T['title'])
col_inputs, col_map = st.columns([1, 3])

# Input column
with col_inputs:
    precipitation = st.number_input(
        T['precip'], 0.0, 1000.0, 00.0, step=0.1
    )
    year = st.number_input(T['year'], 1980, 2050, 2024)
    month = st.selectbox(T['month'], list(range(1, 13)), index=4)
    day = st.selectbox(T['day'], list(range(1, 32)), index=14)

    sectors = sorted(gdf_sectors['Secteur'].unique())
    options = [T['all_sectors_label']] + [f"Secteur {s}" for s in sectors]
    selected = st.multiselect(
        T['select_sectors'], options=options, default=[]
    )

    if not selected:
        st.warning(T['warning_no_sector'])
        st.stop()

    if T['all_sectors_label'] in selected:
        sel_sectors = sectors
    else:
        sel_sectors = [int(s.split()[1]) for s in selected if s.startswith("Secteur")]

    humidities = {}
    for s in sel_sectors:
        humidities[s] = st.slider(
            T['humid_slider'].format(sec=s),
            min_value=0.0, max_value=1.0,
            value=0.5, key=f"hum_{s}"
        )

# Map & results column
with col_map:
    if st.button(T['calc_button']):
        # Prepare DataFrame
        df = pd.DataFrame({'Secteur': sel_sectors})
        df = df.merge(df_metadata, on='Secteur', how='left')
        df['Annee'], df['Mois'], df['Jour'] = year, month, day
        df['Precipitation'] = precipitation
        df['Humidite_sol'] = df['Secteur'].map(humidities)

        # Model prediction
        df_model = df[feature_order]
        probs = np.column_stack([
            pipe.predict_proba(df_model)[:, 1]
            for pipe in pipelines_final.values()
        ])
        df['Probabilité globale d'inondation'] = probs.mean(axis=1)
        df['Confiance_proxy'] = 1 - probs.std(axis=1)

        # CSV export with rename
        df_export = df.copy()
        if lang == 'English':
            rename_map = {
                'Secteur': 'Sector', 'Annee': 'Year',
                'Mois': 'Month', 'Jour': 'Day',
                'Precipitation': 'Precipitation',
                'Humidite_sol': 'Soil_humidity',
                'Probabilité globale d'inondation': 'Global_flood_probability',
                'Confiance_proxy': 'Confidence_proxy',
                'Superficie_depotoir': 'Dump_area',
                'Longueur_caniveau': 'Canal_length',
                'Plan_eau': 'Water_surface',
                'Type_sol': 'Soil_type',
                'Relief': 'Relief'
            }
            df_export.rename(columns=rename_map, inplace=True)

        # Plot map
        gdf_plot = gdf_sectors.merge(
            df[['Secteur', 'Probabilité globale d'inondation']],
            on='Secteur', how='left'
        ).fillna(0)
        cmap = LinearSegmentedColormap.from_list(
            'risk', ['green', 'yellow', 'orange', 'red']
        )
        fig, ax = plt.subplots(figsize=(8, 6))
        gdf_plot.plot(
            column='Probabilité globale d'inondation',
            cmap=cmap, linewidth=0.5, edgecolor='white',
            vmin=0, vmax=1, ax=ax, zorder=1
        )
        for _, r in gdf_plot.iterrows():
            if not r.geometry.is_empty:
                x, y = r.geometry.centroid.coords[0]
                ax.text(
                    x, y, str(int(r['Secteur'])),
                    ha='center', va='center', fontsize=7,
                    fontweight='bold',
                    path_effects=[
                        path_effects.withStroke(linewidth=1, foreground='white')
                    ]
                )
        ax.add_artist(ScaleBar(1, units='m', location='lower right'))
        ax.annotate(
            'N',
            xy=(ax.get_xlim()[1] - 500, ax.get_ylim()[1] - 1000),
            xytext=(0, -20), textcoords='offset points',
            arrowprops=dict(facecolor='black', width=4, headwidth=10),
            ha='center', va='center', fontsize=14, fontweight='bold', zorder=3
        )
        cm = df['Confiance_proxy'].mean()
        ax.text(
            0.01, 0.99,
            T['global_confidence'].format(cm=cm),
            transform=ax.transAxes, ha='left', va='top',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
        )
        cax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.02)
        plt.colorbar(
            plt.cm.ScalarMappable(
                cmap=cmap,
                norm=plt.Normalize(vmin=0, vmax=1)
            ),
            cax=cax,
            orientation='vertical'
        )
        ax.set_axis_off()
        st.pyplot(fig)

        # Expander details
        with st.expander(T['expander_title']):
            for _, row in df.iterrows():
                st.write(
                    T['sector_line'].format(
                        sec=int(row['Secteur']),
                        prob=row['Probabilité globale d'inondation'],
                        conf=row['Confiance_proxy']
                    )
                )

        # Downloads
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        st.download_button(T['download_map'], buf, 'map.png', 'image/png')
        st.download_button(
            T['download_results'],
            df_export.to_csv(index=False).encode('utf-8-sig'),
            'results.csv', 'text/csv'
        )
    else:
        st.info(T['info_click'])
