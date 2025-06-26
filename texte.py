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
    # Météo et date
    precipitation = st.number_input("Précipitation (mm)", 0.0, 1000.0, 10.0, step=0.1)
    annee = st.number_input("Année", 1980, 2050, 2024)
    mois = st.selectbox("Mois", list(range(1, 13)), index=6)
    jour = st.selectbox("Jour", list(range(1, 32)), index=14)

    # Sélection des secteurs
    secteurs_list = sorted(gdf_sectors["Secteur"].unique())
    selected_secteurs = st.multiselect("Sélectionnez des secteurs :", secteurs_list)
    if not selected_secteurs:
        st.warning("Veuillez sélectionner au moins un secteur.")
        st.stop()

    # Sliders humidité
    humidites = {}
    for sec in selected_secteurs:
        humidites[sec] = st.slider(
            f"Humidité du sol Secteur {sec}", 0.0, 1.0, 0.5, key=f"h_{sec}"
        )

# --- Colonne de droite: bouton, carte, expander, téléchargements ---
with col_map:
    # Bouton Calculer
    calc_ph = st.empty()
    if calc_ph.button("Calculer la probabilité d'inondation"):
        calc_ph.empty()

        # Préparation des données
        df_sel = pd.DataFrame({"Secteur": selected_secteurs})
        df_full = df_sel.merge(df_metadata, on="Secteur", how="left")
        df_full["Annee"] = annee
        df_full["Mois"] = mois
        df_full["Jour"] = jour
        df_full["Precipitation"] = precipitation
        df_full["Humidite_sol"] = df_full["Secteur"].map(humidites)

        # Vérification colonnes
        missing = [f for f in feature_order if f not in df_full.columns]
        if missing:
            st.error(f"Colonnes manquantes : {missing}")
            st.stop()

        df_model = df_full[feature_order]

        # Prédictions et confiance
        probas = [pipe.predict_proba(df_model)[:,1] for pipe in pipelines_final.values()]
        arr = np.column_stack(probas)
        df_full["Probabilité globale d'inondation"] = arr.mean(axis=1)
        df_full["Confiance_proxy"] = 1 - np.std(arr, axis=1)

        # Carte
        gdf_plot = gdf_sectors.merge(
            df_full[["Secteur","Probabilité globale d'inondation"]],
            on="Secteur", how="left"
        ).fillna({"Probabilité globale d'inondation":0})
        cmap = LinearSegmentedColormap.from_list("risk", ["green","yellow","orange","red"])
        fig, ax = plt.subplots(figsize=(8,6))
        ax.grid(True, linestyle="--", color="lightgray")
        gdf_plot.plot(
            column="Probabilité globale d'inondation", cmap=cmap,
            linewidth=0.5, edgecolor="white", vmin=0, vmax=1, ax=ax
        )
        for _, rr in gdf_plot.iterrows():
            if not rr.geometry.is_empty:
                x,y = rr.geometry.centroid.coords[0]
                ax.text(x,y,str(int(rr["Secteur"])),ha='center',va='center',fontsize=7,
                        path_effects=[path_effects.withStroke(linewidth=1,foreground='white')])
        ScaleBar(1,units="m",location='lower right').add_to(ax)
        b = gdf_plot.total_bounds
        ax.annotate('N', xy=(b[2]-500,b[3]-1000), xytext=(b[2]-500,b[3]-1500),
                    arrowprops=dict(facecolor='black',width=4,headwidth=10),ha='center')
        ax.set_xticks([]); ax.set_yticks([])
        cm = df_full['Confiance_proxy'].mean()
        ax.text(0.01,0.99,f"Niveau de confiance global: {cm:.3f}",transform=ax.transAxes,
                ha='left',va='top',bbox=dict(facecolor='white',alpha=0.8,edgecolor='gray'))
        div = make_axes_locatable(ax); cax=div.append_axes('right',size='4%',pad=0.02)
        sm=mpl.cm.ScalarMappable(cmap=cmap,norm=mpl.colors.Normalize(0,1)); sm.set_array([])
        fig.colorbar(sm,cax=cax)
        col_map.pyplot(fig)

        # Expander sous la carte
        with st.expander("Probabilité globale et niveau de confiance individuel",expanded=True):
            for _, r in df_full.iterrows():
                st.write(f"Secteur {int(r['Secteur'])}: Prob={r['Probabilité globale d'inondation']:.3f}, Confiance={r['Confiance_proxy']:.3f}")

        # Téléchargements
        buf=BytesIO(); fig.savefig(buf,format='png',dpi=150); buf.seek(0)
        st.download_button("📷 Télécharger la carte",buf,"carte_inondation.png","image/png")
        st.download_button("📅 Télécharger les résultats",df_full.to_csv(index=False).encode('utf-8-sig'),"resultats_inondation.csv","text/csv")
    else:
        calc_ph.info("Cliquez sur 'Calculer' pour générer les résultats et la carte.")
