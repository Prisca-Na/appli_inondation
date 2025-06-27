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

st.title("🌧️ Prévision des Inondations à Ouagadougou")

# --- Interface en deux colonnes ---
col_inputs, col_map = st.columns([1, 3])

# --- Colonne de gauche: paramètres ---
with col_inputs:
    st.subheader("Paramètres d'entrée")
    precipitation = st.number_input("Précipitation (mm)", 0.0, 1000.0, 10.0, step=0.1)
    annee = st.number_input("Année", 1980, 2050, 2024)
    mois = st.selectbox("Mois", list(range(1, 13)), index=6)
    jour = st.selectbox("Jour", list(range(1, 32)), index=14)

    secteurs_list = sorted(gdf_sectors["Secteur"].unique())
    options = ["Tous les secteurs"] + secteurs_list
    selection = st.multiselect("Sélectionnez des secteurs :", options, default=[])

    if not selection:
        st.warning("Veuillez sélectionner au moins un secteur.")
        st.stop()

    selected_secteurs = secteurs_list if "Tous les secteurs" in selection else selection

    humidites = {}
    for sec in selected_secteurs:
        humidites[sec] = st.slider(
            f"Humidité du sol du secteur {sec}",
            min_value=0.0, max_value=1.0,
            value=0.5, key=f"hum_{sec}"
        )

# --- Colonne de droite : bouton, carte, téléchargements ---
with col_map:
    if st.button("Calculer la probabilité d'inondation"):
        df_sel = pd.DataFrame({"Secteur": selected_secteurs})
        df_full = df_sel.merge(df_metadata, on="Secteur", how="left")
        df_full["Annee"] = annee
        df_full["Mois"] = mois
        df_full["Jour"] = jour
        df_full["Precipitation"] = precipitation
        df_full["Humidite_sol"] = df_full["Secteur"].map(humidites)

        missing = [f for f in feature_order if f not in df_full.columns]
        if missing:
            st.error(f"Colonnes manquantes : {missing}")
            st.stop()

        df_model = df_full[feature_order]
        probas = [pipe.predict_proba(df_model)[:, 1] for pipe in pipelines_final.values()]
        arr = np.column_stack(probas)
        df_full["Probabilité globale d'inondation"] = arr.mean(axis=1)
        df_full["Confiance_proxy"] = 1 - np.std(arr, axis=1)

        gdf_plot = gdf_sectors.merge(
            df_full[["Secteur", "Probabilité globale d'inondation"]],
            on="Secteur", how="left"
        ).fillna({"Probabilité globale d'inondation": 0})

        cmap = LinearSegmentedColormap.from_list("risk", ["green", "yellow", "orange", "red"])
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.subplots_adjust(left=0.05, right=0.98, bottom=0.12, top=0.92)

        gdf_plot.plot(
            column="Probabilité globale d'inondation", cmap=cmap,
            linewidth=0.5, edgecolor="white", vmin=0, vmax=1, ax=ax, zorder=1
        )

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
        ax.text(0.01, 0.99, f"Niveau de confiance global: {cm:.3f}",
                transform=ax.transAxes, ha='left', va='top',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='4%', pad=0.02)
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        fig.colorbar(sm, cax=cax, orientation='vertical')

        ax.text(0.01, -0.00, "Carte des probabilités d'inondation",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=13, fontweight="bold")

        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

        st.pyplot(fig)

        with st.expander("Probabilité globale et niveau de confiance individuel"):
            for _, r in df_full.iterrows():
                sec = int(r["Secteur"])
                prob = r["Probabilité globale d'inondation"]
                conf = r["Confiance_proxy"]
                st.write(f"- Secteur {sec}: Probabilité={prob:.3f}, Confiance={conf:.3f}")

        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        st.download_button("📷 Télécharger la carte", buf, "carte_inondation.png", "image/png")
        st.download_button(
            "🗅️ Télécharger les résultats",
            df_full.to_csv(index=False).encode('utf-8-sig'),
            "resultats_inondation.csv", "text/csv"
        )

    else:
        st.info("Cliquez sur 'Calculer' pour générer les résultats et la carte.")
