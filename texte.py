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

st.set_page_config(
    page_title="🌧️ Prévision des Inondations à Ouagadougou",
    layout="wide"
)


Prediction_inondation = os.path.dirname(os.path.abspath(__file__))
path_shapefile = os.path.join(Prediction_inondation, "data", "Secteurs_Ouaga.shp")
path_metadata = os.path.join(Prediction_inondation, "data", "donnee_statique.csv")
path_modele = os.path.join(Prediction_inondation, "model_inondation.pkl")
# --- 1. Charger le bundle (pipelines, seuils, confiance, ordre des variables) ---
@st.cache_data
def load_model_bundle():
    bundle = joblib.load(path_modele)
    return bundle['pipelines'], bundle['seuils'], bundle.get('niveau_confiance', None), bundle['feature_names']

pipelines_final, seuils_dict, niveau_confiance, feature_order = load_model_bundle()
seuil_fusion = seuils_dict.get('Fusion', None)

# --- 2. Charger la carte des secteurs et les métadonnées ---
@st.cache_data
def load_shapefile():
    gdf = gpd.read_file(path_shapefile)
    gdf.columns = gdf.columns.str.strip()
    if "SECTEUR" in gdf.columns:
        gdf = gdf.rename(columns={"SECTEUR": "Secteur"})
    return gdf.to_crs(epsg=32630)

#---- 3. Charger le fichier csv
@st.cache_data
def load_metadata():
    df = pd.read_csv(path_metadata, sep=';')
    df.columns = df.columns.str.strip()
    return df

gdf_sectors = load_shapefile()
df_metadata = load_metadata()

st.title("🌧️ Prévision des Inondations à Ouagadougou")

# --- 4. Mise en page en deux colonnes ---
col_inputs, col_map = st.columns([1, 3])

with col_inputs:
    st.subheader("Paramètres d'entrée")
    precipitation = st.number_input("Précipitation (mm)", min_value=0.0, step=0.1, value=10.0)
    annee = st.number_input("Année", min_value=1980, max_value=2050, value=2024)
    mois = st.selectbox("Mois", list(range(1, 13)), index=6)
    jour = st.selectbox("Jour", list(range(1, 32)), index=14)

    secteurs_list = sorted(gdf_sectors["Secteur"].unique().tolist())
    options_affichees = ["Tous les secteurs"] + secteurs_list

    selection = st.multiselect(
        "Sélectionnez un ou plusieurs secteurs :",
        options_affichees,
        default=[]
    )

    if "Tous les secteurs" in selection:
        selected_secteurs = secteurs_list
    else:
        selected_secteurs = selection

    humidites = {}
    for secteur in selected_secteurs:
        humidites[secteur] = st.slider(
            f"Humidité du sol Secteur {secteur}", 
            0.0, 1.0, 0.5, key=f"h_{secteur}"
        )

with col_map:
    placeholder_map = st.empty()
    st.markdown("---")
    calcul = st.button("Calculer la probabilité d'inondation")
    placeholder_probs = col_inputs.empty()

    if calcul:
        if not selected_secteurs:
            with col_inputs:
                st.warning("Veuillez sélectionner au moins un secteur.")
        else:
            df_sel = pd.DataFrame({"Secteur": selected_secteurs})
            df_full = df_sel.merge(df_metadata, on="Secteur", how="left")
            df_full["Annee"] = annee
            df_full["Mois"] = mois
            df_full["Jour"] = jour
            df_full["Precipitation"] = precipitation
            df_full["Humidite_sol"] = df_full["Secteur"].map(humidites)

            missing = [f for f in feature_order if f not in df_full.columns]
            if missing:
                st.error(f"Colonnes manquantes dans les données : {missing}")
                st.stop()

            df_model = df_full[feature_order]

            # Prédiction individuelle
            probas = []
            try:
                for name, pipe in pipelines_final.items():
                    p = pipe.predict_proba(df_model)[:, 1]
                    df_full[f"Prob_{name}"] = p
                    probas.append(p)
            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {e}")
                st.stop()

            # Fusion : moyenne des probabilités
            arr = np.column_stack(probas)
            df_full["Probabilité globale d'inondation"] = arr.mean(axis=1)
            df_full["Prediction"] = (df_full["Probabilité globale d'inondation"] >= seuil_fusion).astype(int)

            with col_inputs:
                placeholder_probs.markdown("Résultats par secteur")
                for _, r in df_full.iterrows():
                    sec = int(r["Secteur"])
                    pf = r["Probabilité globale d'inondation"]
                    pred = "🌊 Inondation" if r["Prediction"]==1 else "✅ Pas d’inondation"
                    placeholder_probs.write(f"- Secteur {sec}: prob={pf:.3f} → **{pred}**")

            # Carte des probabilités
            gdf_plot = gdf_sectors.merge(
                df_full[["Secteur", "Probabilité globale d'inondation"]],
                on="Secteur", how="left"
            ).fillna({"Probabilité globale d'inondation": 0})

            cmap = LinearSegmentedColormap.from_list("risk", ["green", "yellow", "orange", "red"])
            vmin, vmax = 0.0, 1.0

            fig, ax = plt.subplots(figsize=(8,6))
            fig.subplots_adjust(left=0.05, right=0.98, bottom=0.12, top=0.92)
            ax.set_axisbelow(True)
            ax.grid(True, linestyle="--", linewidth=0.5, color="lightgray")

            gdf_plot.plot(
                column="Probabilité globale d'inondation", cmap=cmap,
                linewidth=0.5, edgecolor="white",
                vmin=vmin, vmax=vmax, ax=ax, zorder=1
            )

            for _, rr in gdf_plot.iterrows():
                if not rr.geometry.is_empty:
                    x, y = rr.geometry.centroid.coords[0]
                    ax.text(x, y, str(int(rr["Secteur"])),
                            ha="center", va="center", fontsize=7, fontweight="bold",
                            path_effects=[path_effects.withStroke(linewidth=1, foreground="white")])

            sb = ScaleBar(1, units="m", location='lower right',
                          length_fraction=0.2, pad=-0.35,
                          box_color='white', box_alpha=0.7,
                          font_properties={'size': 8})
            ax.add_artist(sb)

            bounds = gdf_plot.total_bounds
            x_arrow = bounds[2] - 500
            y_arrow = bounds[3] - 1000
            ax.annotate(
                'N',
                xy=(x_arrow, y_arrow ),
                xytext=(x_arrow, y_arrow -0.00005),
                arrowprops=dict(facecolor='black', width=4, headwidth=10),
                ha='center', va='center', fontsize=14, fontweight='bold', zorder=3
            )

            if niveau_confiance is not None:
                ax.text(0.01, 0.99, f"Niveau de confiance: {niveau_confiance:.3f}",
                        transform=ax.transAxes, ha="left", va="top",
                        fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

            ax.set_xticks([])
            ax.set_yticks([])

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.02)
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            fig.colorbar(sm, cax=cax, orientation='vertical')

            ax.text(0.01, -0.00, "Carte des probabilités d'inondation",
                    transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=13, fontweight='bold')

            ax.text(0.01, -0.10, f"Seuil de décision optimal : {seuil_fusion:.3f}",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

            placeholder_map.pyplot(fig)

            with col_inputs:
                st.download_button(
                    label="📅 Télécharger les résultats",
                    data=df_full.to_csv(index=False).encode('utf-8-sig'),
                    file_name="resultats_inondation.csv",
                    mime='text/csv'
                )

    if not calcul:
        with col_map:
            st.info("Sélectionnez des secteurs et cliquez sur *Calculer* pour voir la carte.")
