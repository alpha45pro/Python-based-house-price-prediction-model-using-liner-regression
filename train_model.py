"""
train_model.py — Run this once to train and save the model.

Usage:
    python train_model.py

Optional — use real Kaggle data:
    pip install kagglehub
    Then set USE_KAGGLE = True below.
"""

USE_KAGGLE = False   # Set True if kagglehub is installed & you have Kaggle API key

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle, json, os, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression

os.makedirs('models', exist_ok=True)
os.makedirs('static', exist_ok=True)

# ── 1. Load data ───────────────────────────────────────────────────────────────
if USE_KAGGLE:
    import kagglehub
    path = kagglehub.dataset_download('juhibhojani/house-price')
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    df = pd.read_csv(os.path.join(path, csv_files[0]))
    print(f"Kaggle dataset loaded: {df.shape}")
    print(df.columns.tolist())
    # ── adapt column names to match expectations ──
    # Rename based on what juhibhojani/house-price actually contains:
    rename_map = {}  # e.g. {'Price': 'Price', 'Location': 'Location', ...}
    df = df.rename(columns=rename_map)
else:
    df = pd.read_csv('house_prices.csv')
    print(f"Synthetic dataset loaded: {df.shape}")

# ── 2. Preprocessing ───────────────────────────────────────────────────────────
print("\nMissing values:\n", df.isnull().sum())

# Outlier removal
for col in ['Price', 'Area']:
    if col in df.columns:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        df = df[(df[col] >= Q1 - 1.5*(Q3-Q1)) & (df[col] <= Q3 + 1.5*(Q3-Q1))]
print(f"After outlier removal: {df.shape}")

# ── 3. Feature engineering ─────────────────────────────────────────────────────
df['Room_ratio']  = df['Bathrooms'] / df['Bedrooms'].clip(1)
df['Total_rooms'] = df['Bedrooms'] + df['Bathrooms']

premium = ['Bandra','Juhu','Worli','Lower Parel','Dadar']
mid     = ['Andheri','Powai','Goregaon']
df['Location_tier'] = df['Location'].apply(
    lambda x: 'premium' if x in premium else ('mid' if x in mid else 'affordable'))

# ── 4. Feature / target split ──────────────────────────────────────────────────
features    = ['Area','Bedrooms','Bathrooms','Parking','Floor','Age_of_Property',
               'Room_ratio','Total_rooms','Location','Location_tier','Furnished_Status']
num_feats   = ['Area','Bedrooms','Bathrooms','Parking','Floor','Age_of_Property','Room_ratio','Total_rooms']
cat_feats   = ['Location','Location_tier','Furnished_Status']

X = df[features].copy()
y = np.log1p(df['Price'])

# ── 5. Preprocessing pipeline ──────────────────────────────────────────────────
num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')),
                     ('scaler',  StandardScaler())])
cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                     ('ohe',     OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
preprocessor = ColumnTransformer([('num', num_pipe, num_feats),
                                   ('cat', cat_pipe, cat_feats)])

model_pipe = Pipeline([
    ('prep',   preprocessor),
    ('poly',   PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
    ('select', SelectKBest(f_regression, k=60)),
    ('ridge',  Ridge(alpha=10.0))
])

# ── 6. Train / Evaluate ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipe.fit(X_train, y_train)

y_train_pred = model_pipe.predict(X_train)
y_test_pred  = model_pipe.predict(X_test)

y_train_actual     = np.expm1(y_train)
y_test_actual      = np.expm1(y_test)
y_train_pred_actual = np.expm1(y_train_pred)
y_test_pred_actual  = np.expm1(y_test_pred)

train_r2  = r2_score(y_train, y_train_pred)
test_r2   = r2_score(y_test,  y_test_pred)
train_mae = mean_absolute_error(y_train_actual, y_train_pred_actual)
test_mae  = mean_absolute_error(y_test_actual,  y_test_pred_actual)
cv_scores = cross_val_score(model_pipe, X_train, y_train, cv=5, scoring='r2')

print(f"\n{'='*50}")
print(f"  Train  R²  : {train_r2:.4f}")
print(f"  Test   R²  : {test_r2:.4f}")
print(f"  Train  MAE : ₹{train_mae:,.0f}")
print(f"  Test   MAE : ₹{test_mae:,.0f}")
print(f"  CV R² (5x) : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"{'='*50}")

# ── 7. Save ────────────────────────────────────────────────────────────────────
metadata = {
    'train_r2':  round(train_r2,  4),
    'test_r2':   round(test_r2,   4),
    'train_mae': round(train_mae, 2),
    'test_mae':  round(test_mae,  2),
    'cv_mean':   round(float(cv_scores.mean()), 4),
    'cv_std':    round(float(cv_scores.std()),  4),
    'locations': sorted(df['Location'].unique().tolist()),
    'furnished_options': sorted(df['Furnished_Status'].unique().tolist()),
}
with open('models/metadata.json', 'w') as f: json.dump(metadata, f, indent=2)
with open('models/model.pkl',     'wb') as f: pickle.dump(model_pipe, f)
print("\nModel & metadata saved to models/")

# ── 8. Plots ───────────────────────────────────────────────────────────────────
plt.style.use('dark_background')

# Actual vs Predicted
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='#0d0d1a')
fig.patch.set_facecolor('#0d0d1a')
for ax, (ya, yp, lbl, col) in zip(axes, [
    (y_train_actual/1e6, y_train_pred_actual/1e6, 'Training Set', '#00d4ff'),
    (y_test_actual/1e6,  y_test_pred_actual/1e6,  'Test Set',     '#ff6b6b')]):
    ax.set_facecolor('#111128')
    ax.scatter(ya, yp, alpha=0.4, s=16, color=col, edgecolors='none')
    mn, mx = min(ya.min(),yp.min()), max(ya.max(),yp.max())
    ax.plot([mn,mx],[mn,mx],'w--',lw=1.4,alpha=0.6,label='Perfect')
    ax.set_title(f'{lbl}  (R²={r2_score(ya,yp):.3f})', color='white', fontsize=12)
    ax.set_xlabel('Actual (₹M)', color='#aaa'); ax.set_ylabel('Predicted (₹M)', color='#aaa')
    ax.tick_params(colors='#666'); [sp.set_edgecolor('#333') for sp in ax.spines.values()]
    ax.legend(fontsize=9, facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')
plt.tight_layout()
plt.savefig('static/actual_vs_predicted.png', dpi=130, bbox_inches='tight', facecolor='#0d0d1a')
plt.close()

# Feature importance
coef = np.abs(model_pipe.named_steps['ridge'].coef_)
top  = np.argsort(coef)[-15:][::-1]
readable = {0:'Area',1:'Bedrooms',2:'Bathrooms',3:'Parking',4:'Floor',5:'Age',6:'Room Ratio',7:'Total Rooms'}
labels = [readable.get(i, f'Interaction {i}') for i in top]
fig2, ax2 = plt.subplots(figsize=(10,6), facecolor='#0d0d1a')
ax2.set_facecolor('#111128')
ax2.barh(range(15), coef[top][::-1], color=plt.cm.plasma(np.linspace(0.2,0.9,15))[::-1], height=0.65)
ax2.set_yticks(range(15)); ax2.set_yticklabels(labels[::-1], color='white', fontsize=10)
ax2.set_xlabel('|Coefficient|', color='#aaa')
ax2.set_title('Top 15 Feature Importances', color='white', fontsize=13, fontweight='bold')
ax2.tick_params(colors='#666'); [sp.set_edgecolor('#333') for sp in ax2.spines.values()]
plt.tight_layout()
plt.savefig('static/feature_importance.png', dpi=130, bbox_inches='tight', facecolor='#0d0d1a')
plt.close()

# CV scores
fig3, ax3 = plt.subplots(figsize=(8,4), facecolor='#0d0d1a')
ax3.set_facecolor('#111128')
bar_c = ['#00d4ff' if s>=cv_scores.mean() else '#ff6b6b' for s in cv_scores]
ax3.bar([f'Fold {i+1}' for i in range(5)], cv_scores, color=bar_c, width=0.5)
ax3.axhline(cv_scores.mean(), color='white', ls='--', lw=1.5, label=f'Mean={cv_scores.mean():.4f}')
ax3.set_ylim(max(0,cv_scores.min()-0.05), 1.0)
ax3.set_title('5-Fold CV R² Scores', color='white', fontsize=13, fontweight='bold')
ax3.set_ylabel('R²', color='#aaa'); ax3.tick_params(colors='#666')
[sp.set_edgecolor('#333') for sp in ax3.spines.values()]
ax3.legend(facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')
plt.tight_layout()
plt.savefig('static/cv_scores.png', dpi=130, bbox_inches='tight', facecolor='#0d0d1a')
plt.close()

print("Plots saved to static/")
