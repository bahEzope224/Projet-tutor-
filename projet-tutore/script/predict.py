import pandas as pd
import joblib
import os

def load_prediction_tools(model_path='model/best_model_xgb.joblib', encoders_path='model/label_encoders.joblib'):
    """Charge le modèle et les encodeurs sauvegardés."""
    if not os.path.exists(model_path) or not os.path.exists(encoders_path):
        raise FileNotFoundError("Modèle ou encodeurs introuvables. Lancez d'abord app.py.")
    
    model = joblib.load(model_path)
    le_dict = joblib.load(encoders_path)
    return model, le_dict

def predict_booking(new_data, model, le_dict):
    """
    Réalise une prédiction sur de nouvelles données.
    new_data: dict contenant les informations de réservation.
    """
    # 1. Conversion en DataFrame
    df_new = pd.DataFrame([new_data])
    
    # 2. Preprocessing identique à l'entraînement
    # Extraction du mois et jour (si fournis)
    if 'date of reservation' in df_new.columns:
        df_new['date of reservation'] = pd.to_datetime(df_new['date of reservation'], errors='coerce')
        df_new['booking_month'] = df_new['date of reservation'].dt.month.fillna(6).astype(int)
        df_new['arrival_num_day'] = df_new['date of reservation'].dt.dayofweek.fillna(0).astype(int)
        df_new = df_new.drop('date of reservation', axis=1)
    
    # Total nights si non fourni
    if 'total_nights' not in df_new.columns:
        df_new['total_nights'] = df_new['number of weekend nights'] + df_new['number of week nights']

    # Encodage catégoriel
    cat_cols = ['type of meal', 'room type', 'market segment type']
    for col in cat_cols:
        if col in df_new.columns:
            # On utilise l'encodeur correspondant
            try:
                df_new[col] = le_dict[col].transform(df_new[col])
            except ValueError:
                # Gérer les nouvelles catégories (fallback sur la première connue)
                print(f"Attention: Catégorie inconnue dans {col}, utilisation d'une valeur par défaut.")
                df_new[col] = 0
                
    # Assurer l'ordre des colonnes identique à l'entraînement
    # (On récupère l'ordre depuis le modèle ou les caractéristiques attendues)
    expected_cols = [
        'number of adults', 'number of children', 'number of weekend nights',
        'number of week nights', 'type of meal', 'car parking space', 
        'room type', 'lead time', 'market segment type', 'repeated', 
        'P-C', 'P-not-C', 'average price ', 'special requests', 
        'booking_month', 'arrival_num_day', 'total_nights'
    ]
    df_new = df_new[expected_cols]
    
    # 3. Prédiction
    prediction = model.predict(df_new)[0]
    probability = model.predict_proba(df_new)[0][1] # Probabilité de "Canceled"
    
    return "Canceled" if prediction == 1 else "Not_Canceled", probability

# --- EXEMPLE D'UTILISATION ---
if __name__ == "__main__":
    try:
        model, le_dict = load_prediction_tools()
        
        # Exemple de nouvelle réservation risquée (long délai, prix élevé, segment online)
        new_booking = {
            'number of adults': 2,
            'number of children': 0,
            'number of weekend nights': 1,
            'number of week nights': 2,
            'type of meal': 'Meal Plan 1',
            'car parking space': 0,
            'room type': 'Room_Type 1',
            'lead time': 150, # Délai long = Risque
            'market segment type': 'Online',
            'repeated': 0,
            'P-C': 0,
            'P-not-C': 0,
            'average price ': 120.0,
            'special requests': 0, # Pas de demande = Risque
            'date of reservation': '2026-06-15'
        }
        
        result, proba = predict_booking(new_booking, model, le_dict)
        print(f"\n--- Prédiction pour la nouvelle réservation ---")
        print(f"Statut prédit : {result}")
        print(f"Probabilité d'annulation : {proba:.2%}")
        
    except Exception as e:
        print(f"Erreur : {e}")
