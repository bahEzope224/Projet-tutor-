import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
import shap

# Configuration de l'esthétique des graphiques
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Création des dossiers
os.makedirs('model', exist_ok=True)
os.makedirs('output', exist_ok=True)

print("--- Chargement et Nettoyage Avancé ---")
df = pd.read_csv('data/HotelBookingCancellations.csv')
df = df.drop('Booking_ID', axis=1)
df['booking status'] = df['booking status'].map({'Canceled': 1, 'Not_Canceled': 0})

# Date Processing
df['date of reservation'] = pd.to_datetime(df['date of reservation'], format='mixed', dayfirst=False, errors='coerce')
df['booking_month'] = df['date of reservation'].dt.month
df['booking_month'] = df['booking_month'].fillna(df['booking_month'].mode()[0]).astype(int)
df['arrival_num_day'] = df['date of reservation'].dt.dayofweek
df['arrival_num_day'] = df['arrival_num_day'].fillna(df['arrival_num_day'].mode()[0]).astype(int)

# Feature Engineering
df['total_nights'] = df['number of weekend nights'] + df['number of week nights']
df['total_stay_cost'] = df['total_nights'] * df['average price ']

# --- 1. EDA Avancée ---
print("\n--- Génération de l'EDA Avancée ---")

# A. Heatmap de Saisonnalité (Mois vs Segment de Marché)
pivot_month_segment = df.pivot_table(index='booking_month', columns='market segment type', values='booking status', aggfunc='mean')
plt.figure(figsize=(12, 6))
sns.heatmap(pivot_month_segment, annot=True, cmap='YlOrRd', fmt=".2f")
plt.title('Taux d\'annulation par Mois et Segment de Marché')
plt.savefig('output/eda_seasonality_heatmap.png')

# B. Distribution de Lead Time et Prix
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
sns.kdeplot(data=df, x='lead time', hue='booking status', fill=True, ax=ax1)
ax1.set_title('Distribution du Lead Time (Délai de réservation)')
sns.kdeplot(data=df, x='average price ', hue='booking status', fill=True, ax=ax2)
ax2.set_title('Distribution de l\'Average Price')
plt.savefig('output/eda_distributions.png')

# C. Impact des Special Requests
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='special requests', y='booking status', estimator=np.mean)
plt.title('Taux d\'annulation vs Nombre de demandes spéciales')
plt.ylabel('Probabilité d\'annulation')
plt.savefig('output/eda_special_requests.png')

# --- 2. Encodage et Splitting ---
cat_cols = ['type of meal', 'room type', 'market segment type']
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Sauvegarde des encodeurs pour l'inférence
joblib.dump(le_dict, 'model/label_encoders.joblib')

X = df.drop(['booking status', 'date of reservation', 'total_stay_cost'], axis=1)
y = df['booking status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standarisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 3. Modélisation Avancée ---

# A. Logistic Regression (Baseline)
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)

# B. Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# C. XGBoost (Optimisé)
print("\n--- Entraînement : XGBoost ---")
xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)

# Évaluation
models = {'LogReg': log_reg, 'RF': rf_model, 'XGBoost': xgb_model}
results = {}

for name, model in models.items():
    X_ev = X_test_scaled if name == 'LogReg' else X_test
    y_pred = model.predict(X_ev)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results[name] = {'Accuracy': acc, 'F1-Score': f1}
    print(f"\n[{name}] Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")

# Sauvegarde du meilleur modèle (XGBoost)
joblib.dump(xgb_model, 'model/best_model_xgb.joblib')

# --- 4. Analyse SHAP Globale (1000 samples) ---
print("\n--- Analyse SHAP (Interprétabilité Globale) ---")
X_test_sample = X_test.sample(1000, random_state=42)
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_sample)

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_sample, show=False)
plt.title('SHAP Global Analysis - Top Factors leading to Cancellation')
plt.tight_layout()
plt.savefig('output/shap_global_summary.png')

# --- 5. Clustering : Identification de Paternes Similaires ---
print("\n--- Analyse des Paternes (Clustering K-Means) ---")

# Sélection des features pour le profilage
cluster_features = ['lead time', 'average price ', 'special requests', 'total_nights', 'number of adults']
X_cluster = df[cluster_features]

# Normalisation pour le clustering
scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

# Algorithme K-Means avec 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_transform(X_cluster_scaled).argmin(axis=1) # Utilisation de argmin car fit_transform renvoie distances

# Profilage des clusters
cluster_profile = df.groupby('cluster').agg({
    'booking status': 'mean',
    'lead time': 'mean',
    'average price ': 'mean',
    'special requests': 'mean',
    'total_nights': 'mean'
}).rename(columns={'booking status': 'Taux Annulation'})

print("\nProfils des Clusters identifiés :")
print(cluster_profile)

# Visualisation des profils
cluster_profile_norm = (cluster_profile - cluster_profile.min()) / (cluster_profile.max() - cluster_profile.min())
plt.figure(figsize=(12, 6))
sns.heatmap(cluster_profile_norm.drop('Taux Annulation', axis=1).T, annot=cluster_profile.drop('Taux Annulation', axis=1).T, cmap='Blues', fmt=".1f")
plt.title('Carte d\'identité des Segments (Valeurs moyennes)')
plt.savefig('output/cluster_profiles_heatmap.png')

# Taux d'annulation par cluster
plt.figure(figsize=(10, 6))
sns.barplot(x=cluster_profile.index, y=cluster_profile['Taux Annulation'])
plt.title('Risque d\'annulation par Segment de Clientèle')
plt.ylabel('Taux d\'annulation')
plt.xlabel('Segment ID')
plt.savefig('output/cluster_cancellation_risk.png')

# --- 6. Profondeur Financière ---
print("\n--- Analyse Financière Poussée ---")

# Perte par mois
monthly_loss = df[df['booking status'] == 1].groupby('booking_month')['total_stay_cost'].sum()
plt.figure(figsize=(12, 6))
monthly_loss.plot(kind='bar', color='salmon')
plt.title('Manque à Gagner Total par Mois (€)')
plt.xlabel('Mois')
plt.ylabel('Perte (€)')
plt.savefig('output/financial_monthly_loss.png')

# Perte par segment
segment_loss = df[df['booking status'] == 1].groupby('market segment type')['total_stay_cost'].sum()
# Re-mapping labels pour la clarté
# On réutilise les LabelEncoders si besoin, ou on fait un simple dictionnaire ici
segment_labels = {0: 'Aviation', 1: 'Complementary', 2: 'Corporate', 3: 'Offline', 4: 'Online'}
segment_loss.index = segment_loss.index.map(segment_labels)

plt.figure(figsize=(10, 6))
segment_loss.sort_values(ascending=False).plot(kind='pie', autopct='%1.1f%%')
plt.title('Répartition du Manque à Gagner par Segment de Marché')
plt.ylabel('')
plt.savefig('output/financial_segment_pie.png')

print(f"\nAnalyse terminée. Tous les graphiques sont dans 'output/'.")
print(f"Meilleur modèle (XGBoost) sauvegardé dans 'model/'.")
