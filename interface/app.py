import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# Charger les modèles une fois au démarrage
base_path = Path(__file__).resolve().parent.parent / "models"
rf_model_path = base_path / "random_forest_model.pkl"

# Charger le modèle
rf_loaded = joblib.load(rf_model_path)

# Interface utilisateur
st.title("Détection de Fraude par Carte Bancaire")
st.write("Chargez un fichier CSV contenant les caractéristiques des transactions pour prédire si elles sont frauduleuses.")

# Télécharger un exemple de CSV
example_data = """
Time,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27,V28,Amount
9,-0.33826175242575,1.11959337641566,1.04436655157316,-0.222187276738296,0.49936080649727,-0.24676110061991,0.651583206489972,0.0695385865186387,-0.736727316364109,-0.366845639206541,1.01761446783262,0.836389570307029,1.00684351373408,-0.443522816876142,0.150219101422635,0.739452777052119,-0.540979921943059,0.47667726004282,0.451772964394125,0.203711454727929,-0.246913936910008,-0.633752642406113,-0.12079408408185,-0.385049925313426,-0.0697330460416923,0.0941988339514961,0.246219304619926,0.0830756493473326,3.68
"""
st.download_button(
    label="Télécharger un exemple de fichier CSV",
    data=example_data,
    file_name="example_transaction.csv",
    mime="text/csv",
)

# Charger un fichier CSV
uploaded_file = st.file_uploader("Chargez un fichier CSV", type=["csv"])

if uploaded_file:
    # Lire le fichier CSV
    try:
        data = pd.read_csv(uploaded_file)
        st.write("Aperçu des données chargées :")
        st.write(data.head())

        # Vérifier que les colonnes attendues sont présentes
        expected_columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                            'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                            'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
        if not all(col in data.columns for col in expected_columns):
            st.error("Le fichier CSV doit contenir les colonnes suivantes :")
            st.write(expected_columns)
        else:
            # Faire les prédictions
            predictions = rf_loaded.predict(data)
            probabilities = rf_loaded.predict_proba(data)[:, 1]  # Probabilité pour la classe 1

            # Afficher les prédictions sous forme de phrases
            st.write("Résultats des prédictions :")
            for idx, (pred, prob) in enumerate(zip(predictions, probabilities)):
                if pred == 1:
                    st.error(
                        f"Transaction {idx + 1} suspectée de fraude avec une probabilité de {prob:.2%}."
                    )
                else:
                    st.success(
                        f"Transaction {idx + 1} considérée comme normale avec une probabilité de fraude de {prob:.2%}."
                    )

    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier : {e}")
