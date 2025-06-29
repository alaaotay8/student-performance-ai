import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb
from sklearn.feature_selection import SelectFromModel, RFE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
data = pd.read_csv('Student_performance_data _.csv')

# Examine data
print(f"Data shape: {data.shape}")
print(f"Columns: {data.columns.tolist()}")
print(f"Missing values: {data.isnull().sum().sum()}")

# Enhanced data preparation
print("\n=== Enhanced Data Preparation ===")

# Feature Engineering
def create_additional_features(df):
    """Create additional features that might be predictive"""
    df_enhanced = df.copy()
    
    # Interaction features (if applicable)
    if 'StudyTimeWeekly' in df.columns and 'Absences' in df.columns:
        df_enhanced['StudyTime_per_Absence'] = df_enhanced['StudyTimeWeekly'] / (df_enhanced['Absences'] + 1)
    
    # Binning continuous variables (example)
    if 'Age' in df.columns:
        df_enhanced['Age_Group'] = pd.cut(df_enhanced['Age'], bins=3, labels=['Young', 'Medium', 'Old'])
        df_enhanced['Age_Group'] = LabelEncoder().fit_transform(df_enhanced['Age_Group'])
    
    return df_enhanced

# Apply feature engineering
data_enhanced = create_additional_features(data)

# Prepare data
X = data_enhanced.drop(['StudentID', 'GPA', 'GradeClass'], axis=1, errors='ignore')
y = data_enhanced['GradeClass']

print(f"Input features: {X.columns.tolist()}")
print(f"Target class distribution:")
print(y.value_counts().sort_index())

# Encode categorical variables if any
categorical_columns = X.select_dtypes(include=['object']).columns
if len(categorical_columns) > 0:
    print(f"Encoding categorical columns: {categorical_columns.tolist()}")
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123, stratify=y
)

print(f"\nTraining set class distribution:")
print(y_train.value_counts().sort_index())

# Advanced preprocessing approaches
print("\n=== Advanced Preprocessing ===")

# 1. Robust Scaling (better for outliers)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Handle class imbalance with multiple strategies
def apply_sampling_strategy(X, y, strategy='smote'):
    """Apply different sampling strategies"""
    if strategy == 'smote':
        sampler = SMOTE(random_state=123)
    elif strategy == 'adasyn':
        sampler = ADASYN(random_state=123)
    elif strategy == 'smotetomek':
        sampler = SMOTETomek(random_state=123)
    elif strategy == 'undersample':
        sampler = RandomUnderSampler(random_state=123)
    else:
        return X, y
    
    return sampler.fit_resample(X, y)

# Test different sampling strategies
sampling_strategies = ['none', 'smote', 'adasyn', 'smotetomek']
best_sampling = 'none'
best_sampling_score = 0

print("Testing sampling strategies...")
for strategy in sampling_strategies:
    if strategy == 'none':
        X_sample, y_sample = X_train_scaled, y_train
    else:
        X_sample, y_sample = apply_sampling_strategy(X_train_scaled, y_train, strategy)
    
    # Quick test with Random Forest
    rf_test = RandomForestClassifier(n_estimators=50, random_state=123, class_weight='balanced')
    cv_scores = cross_val_score(rf_test, X_sample, y_sample, cv=3, scoring='f1_macro')
    avg_score = cv_scores.mean()
    
    print(f"{strategy}: F1-macro = {avg_score:.4f}")
    
    if avg_score > best_sampling_score:
        best_sampling_score = avg_score
        best_sampling = strategy

print(f"Best sampling strategy: {best_sampling}")

# Apply best sampling strategy
if best_sampling != 'none':
    X_train_final, y_train_final = apply_sampling_strategy(X_train_scaled, y_train, best_sampling)
else:
    X_train_final, y_train_final = X_train_scaled, y_train

print(f"Final training set shape: {X_train_final.shape}")
print(f"Final class distribution: {np.bincount(y_train_final)}")

# Enhanced model selection with hyperparameter tuning
print("\n=== Enhanced Model Selection ===")

# Compute class weights for imbalanced datasets
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# Define enhanced models with hyperparameter grids
models_enhanced = {
    'Random Forest': {
        'model': RandomForestClassifier(random_state=123, class_weight='balanced'),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'XGBoost': {
        'model': xgb.XGBClassifier(random_state=123, eval_metric='mlogloss'),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
    },
    'LightGBM': {
        'model': lgb.LGBMClassifier(random_state=123, verbose=-1),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'num_leaves': [31, 50, 100]
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(random_state=123),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    }
}

# Hyperparameter tuning with cross-validation
best_models = {}
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

print("Performing hyperparameter tuning...")
for name, model_info in models_enhanced.items():
    print(f"\nTuning {name}...")
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        model_info['model'],
        model_info['params'],
        cv=cv_strategy,
        scoring='f1_macro',  # Better for imbalanced datasets
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train_final, y_train_final)
    
    best_models[name] = {
        'model': grid_search.best_estimator_,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_
    }
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV F1-macro score: {grid_search.best_score_:.4f}")

# Select best individual model
best_individual_model = max(best_models.items(), key=lambda x: x[1]['best_score'])
print(f"\nBest individual model: {best_individual_model[0]} (F1-macro: {best_individual_model[1]['best_score']:.4f})")

# Create ensemble model
print("\n=== Creating Ensemble Model ===")
top_models = sorted(best_models.items(), key=lambda x: x[1]['best_score'], reverse=True)[:3]

ensemble_models = [(name, info['model']) for name, info in top_models]
voting_classifier = VotingClassifier(
    estimators=ensemble_models,
    voting='soft'  # Use probability-based voting
)

# Train ensemble model
voting_classifier.fit(X_train_final, y_train_final)

# Evaluate all models
print("\n=== Model Evaluation ===")
print("="*80)

models_to_evaluate = {
    **{name: info['model'] for name, info in best_models.items()},
    'Ensemble': voting_classifier
}

results = {}
for name, model in models_to_evaluate.items():
    # Prediction
    y_pred = model.predict(X_test_scaled)
    
    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    results[name] = {
        'Accuracy': accuracy,
        'F1-Macro': f1_macro,
        'F1-Weighted': f1_weighted
    }
    
    print(f"\n{name} Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1-Macro: {f1_macro:.4f}")
    print(f"  F1-Weighted: {f1_weighted:.4f}")
    print("-" * 50)

# Select final best model
final_best_model_name = max(results.items(), key=lambda x: x[1]['F1-Macro'])[0]
final_best_model = models_to_evaluate[final_best_model_name]

print(f"\nFinal best model: {final_best_model_name}")
print(f"Final best F1-Macro score: {results[final_best_model_name]['F1-Macro']:.4f}")

# Detailed evaluation of best model
print(f"\n=== Detailed Evaluation: {final_best_model_name} ===")
y_pred_final = final_best_model.predict(X_test_scaled)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_final))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_final)
print(cm)

# Feature importance (if available)
if hasattr(final_best_model, 'feature_importances_'):
    print("\nTop 10 Feature Importances:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': final_best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance.head(10))
elif hasattr(final_best_model, 'estimators_'):
    # For ensemble models, try to get feature importance from first estimator
    if hasattr(final_best_model.estimators_[0], 'feature_importances_'):
        print("\nTop 10 Feature Importances (from first estimator):")
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': final_best_model.estimators_[0].feature_importances_
        }).sort_values('importance', ascending=False)
        print(feature_importance.head(10))

# Save the final model
print(f"\n=== Saving Final Model ===")
model_data = {
    'model': final_best_model,
    'scaler': scaler,
    'feature_names': X.columns.tolist(),
    'model_name': final_best_model_name,
    'sampling_strategy': best_sampling,
    'class_distribution': dict(y_train.value_counts()),
    'performance_metrics': results[final_best_model_name],
    'requires_scaling': True
}

joblib.dump(model_data, 'enhanced_student_perf_model.pkl')
print("Enhanced model saved successfully to: enhanced_student_perf_model.pkl")

# Summary of improvements
print(f"\n=== Summary of Improvements ===")
print("1. ✓ Robust scaling instead of standard scaling")
print("2. ✓ Advanced sampling techniques for class imbalance")
print("3. ✓ Hyperparameter tuning with GridSearchCV")
print("4. ✓ Advanced ensemble methods")
print("5. ✓ Enhanced evaluation metrics (F1-macro, F1-weighted)")
print("6. ✓ Feature importance analysis")
print("7. ✓ Model comparison and selection")
print("\nModel is ready for enhanced predictions!")
