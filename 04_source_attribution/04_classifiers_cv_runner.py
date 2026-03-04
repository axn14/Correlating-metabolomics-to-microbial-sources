

# --------------------------------------------------------------------------"""
Classifiers and CV runner — get_classifiers(): RF(500 trees), GradientBoosting(300), LogisticRegression(multinomial), XGBoost(300). run_cv_clf(): 10×3 RepeatedStratifiedKFold; metrics: balanced_accuracy, f1_macro.
"""


# ============================================================
# SOURCE SPECIES CLASSIFICATION -- Cell 3: Models & CV runner
# ============================================================
def get_classifiers():
    return {
        "RandomForest":       RandomForestClassifier(
                                  n_estimators=500, max_depth=10,
                                  class_weight='balanced', n_jobs=-1, random_state=42),
        "GradientBoosting":   GradientBoostingClassifier(
                                  n_estimators=300, max_depth=5, random_state=42),
        "LogisticRegression": LogisticRegression(
                                  max_iter=1000, class_weight='balanced',
                                  multi_class='multinomial', random_state=42),
        "XGBoost":            XGBClassifier(
                                  n_estimators=300, max_depth=6,
                                  eval_metric='mlogloss', random_state=42),
    }

def run_cv_clf(X, y, clf, n_splits=10, n_repeats=3):
    """Repeated stratified k-fold CV; returns balanced_accuracy and f1_macro."""
    cv      = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    scoring = ['balanced_accuracy', 'f1_macro']
    res     = cross_validate(clf, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return {
        'balanced_accuracy_mean': res['test_balanced_accuracy'].mean(),
        'balanced_accuracy_std':  res['test_balanced_accuracy'].std(),
        'f1_macro_mean':          res['test_f1_macro'].mean(),
        'f1_macro_std':           res['test_f1_macro'].std(),
    }

print("get_classifiers and run_cv_clf defined.")