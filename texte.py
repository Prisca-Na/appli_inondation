import streamlit as st
import os
import pandas as pd
import geopandas as gpd
import joblib
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.patheffects as path_effects
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from io import BytesIO


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

with col_inputs:
    st.subheader("Paramètres d'entrée")
    precipitation = st.number_input("Précipitation (mm)", 0.0, 1000.0, 10.0, step=0.1)
    annee = st.number_input("Année", 1980, 2050, 2024)
    mois = st.selectbox("Mois", list(range(1,13)), index=6)
    jour = st.selectbox("Jour", list(range(1,32)), index=14)

    secteurs_list = sorted(gdf_sectors["Secteur"].unique())
    options = ["Tous les secteurs"] + secteurs_list
    selection = st.multiselect("Sélectionnez des secteurs :", options, default=[])
    if "Tous les secteurs" in selection or not selection:
        selected_secteurs = secteurs_list
    else:
        selected_secteurs = selection

    # Sliders humidité pour chaque secteur sélectionné
    humidites = {}
    for sec in selected_secteurs:
        humidites[sec] = st.slider(
            f"Humidité du sol  {sec}", 0.0, 1.0, 0.5, key=f"h_{sec}"
        )

# ---- Colonne de droite: bouton, résultats et carte ----
with col_map:
    # Bouton Calculer (placeholder pour le faire disparaître)
    calc_ph = st.empty()
    if calc_ph.button("Calculer la probabilité d'inondation"):
        # faire disparaître le bouton
        calc_ph.empty()

        # Préparation des données
        df_sel = pd.DataFrame({"Secteur": selected_secteurs})
        df_full = df_sel.merge(df_metadata, on="Secteur", how="left")
        df_full["Annee"] = annee
        df_full["Mois"] = mois
        df_full["Jour"] = jour
        df_full["Precipitation"] = precipitation
        df_full["Humidite_sol"] = df_full["Secteur"].map(humidites)

        # Vérification des colonnes
        missing = [f for f in feature_order if f not in df_full.columns]
        if missing:
            st.error(f"Colonnes manquantes : {missing}")
            st.stop()

        df_model = df_full[feature_order]

        # Prédictions et fusion
        probas = []
        for name, pipe in pipelines_final.items():
            p = pipe.predict_proba(df_model)[:,1]
            df_full[f"Prob_{name}"] = p
            probas.append(p)
        arr = np.column_stack(probas)
        df_full["Probabilité globale d'inondation"] = p_fusion = arr.mean(axis=1)
        df_full["Confiance_proxy"] = 1 - np.std(arr, axis=1)

        # Expander: probabilité & confiance
        with st.expander("Probabilité globale et niveau de confiance individuel", expanded=True):
            for _, r in df_full.iterrows():
                sec = int(r["Secteur"])
                prob = r["Probabilité globale d'inondation"]
                conf = r["Confiance_proxy"]
                st.write(f"Secteur {sec}: Prob={prob:.3f}, Confiance={conf:.3f}")

        # Carte
        gdf_plot = gdf_sectors.merge(
            df_full[["Secteur","Probabilité globale d'inondation"]], on="Secteur", how="left"
        ).fillna({"Probabilité globale d'inondation": 0})
        cmap = LinearSegmentedColormap.from_list("risk", ["green","yellow","orange","red"])
        vmin, vmax = 0.0, 1.0

        fig, ax = plt.subplots(figsize=(8,6))
        ax.grid(True, linestyle="--", color="lightgray")
        gdf_plot.plot(
            column="Probabilité globale d'inondation", cmap=cmap,
            linewidth=0.5, edgecolor="white", vmin=vmin, vmax=vmax, ax=ax
        )
        for _, rr in gdf_plot.iterrows():
            if not rr.geometry.is_empty:
                x, y = rr.geometry.centroid.coords[0]
                ax.text(
                    x, y, str(int(rr["Secteur"])), ha='center', va='center', fontsize=7,
                    path_effects=[path_effects.withStroke(linewidth=1, foreground='white')]
                )
        # Nord & échelle
        sb = ScaleBar(1, units="m", location='lower right', length_fraction=0.2,
                      pad=-0.35, box_color='white', box_alpha=0.7, font_properties={'size': 8})
        ax.add_artist(sb)
        bts = gdf_plot.total_bounds
        ax.annotate('N', xy=(bts[2]-500,bts[3]-1000), xytext=(bts[2]-500,bts[3]-1000-0.00005),
                    arrowprops=dict(facecolor='black', width=4, headwidth=10), ha='center', va='center',
                    fontsize=14, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        # Confiance globale moyenne
        cg = df_full['Confiance_proxy'].mean()
        ax.text(0.01,0.99,f"Confiance moyenne: {cg:.3f}", transform=ax.transAxes,
                ha='left', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

        # Colorbar
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="4%", pad=0.02)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
        fig.colorbar(sm, cax=cax)

        # Affichage carte
        col_map.pyplot(fig)
        # Télécharger carte
        buf = BytesIO(); fig.savefig(buf, format='png', dpi=150); buf.seek(0)
        st.download_button("📷 Télécharger la carte", data=buf, file_name="carte.png", mime="image/png")
        # Télécharger CSV
        st.download_button(
            "📅 Télécharger les résultats",
            data=df_full.to_csv(index=False).encode('utf-8-sig'),
            file_name="resultats.csv", mime='text/csv'
        )
    # sinon afficher bouton
    elif calc_ph := st.empty():
        calc_ph.button("Calculer la probabilité d'inondation")
