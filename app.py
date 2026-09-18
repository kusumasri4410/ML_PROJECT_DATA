import json
import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
METRICS_PATH = os.path.join(BASE_DIR, 'metrics.json')
TRAIN_PATH = os.path.join(BASE_DIR, 'train.csv')

model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

_train_df = None
_eda_cache = None


def get_train_df():
    global _train_df
    if _train_df is None and os.path.exists(TRAIN_PATH):
        _train_df = pd.read_csv(TRAIN_PATH)
    return _train_df


def get_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return {}


FEATURE_COLS = None


def get_feature_cols():
    global FEATURE_COLS
    if FEATURE_COLS is None:
        df = get_train_df()
        if df is not None:
            FEATURE_COLS = [c for c in df.columns if c != 'class']
    return FEATURE_COLS or []


def align_input(df_input):
    """Reindex user/train rows to the exact columns the model saw."""
    cols = get_feature_cols()
    df = df_input.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[cols]
    # coerce true numerics, leave the rest as strings for the pipeline
    for c in ['number_of_bruises', 'ID', 'mushroom_id']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def predict_df(df_input):
    data = align_input(df_input)
    preds = model.predict(data)
    proba = None
    if hasattr(model, 'predict_proba'):
        try:
            proba = model.predict_proba(data)
        except Exception:
            proba = None
    return preds, proba


def build_eda():
    global _eda_cache
    if _eda_cache is not None:
        return _eda_cache
    df = get_train_df()
    if df is None:
        return {}
    df = df.copy()

    class_counts = df['class'].value_counts().to_dict()

    # numeric summaries
    numeric_summary = {}
    for c in ['number_of_bruises', 'ID', 'mushroom_id']:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors='coerce')
            numeric_summary[c] = {
                'min': float(s.min()),
                'max': float(s.max()),
                'mean': round(float(s.mean()), 2),
                'median': float(s.median()),
            }

    # bruises histogram (bins) overall + split by class
    s = pd.to_numeric(df['number_of_bruises'], errors='coerce').fillna(0)
    bins = [0, 2, 4, 6, 8, 10, 12, 15, 25]
    labels = ['0-1', '2-3', '4-5', '6-7', '8-9', '10-11', '12-14', '15+']
    binned = pd.cut(s, bins=bins, labels=labels, include_lowest=True)
    hist_all = binned.value_counts().reindex(labels, fill_value=0).tolist()
    hist_by_class = {}
    for cls in sorted(df['class'].dropna().unique()):
        m = binned[df['class'] == cls].value_counts().reindex(labels, fill_value=0).tolist()
        hist_by_class[str(cls)] = m

    # categorical distributions (top categories) for key features
    key_cats = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
                'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape',
                'stalk-root', 'ring-type', 'spore-print-color', 'population', 'habitat']
    cat_dist = {}
    for c in key_cats:
        if c in df.columns:
            vc = df[c].fillna('missing').value_counts().head(10)
            cat_dist[c] = {'labels': vc.index.astype(str).tolist(),
                           'counts': vc.values.tolist()}

    # crosstabs vs class (stacked bars) for the most informative features
    cross_features = ['odor', 'gill-size', 'bruises', 'population', 'habitat',
                      'cap-color', 'spore-print-color', 'stalk-shape']
    crosstabs = {}
    classes = sorted(df['class'].dropna().unique().astype(str).tolist())
    for c in cross_features:
        if c in df.columns:
            ct = pd.crosstab(df[c].fillna('missing'), df['class'])
            for cls in classes:
                if cls not in ct.columns:
                    ct[cls] = 0
            ct = ct[classes]
            crosstabs[c] = {'labels': ct.index.astype(str).tolist(),
                            'classes': classes,
                            'counts': {cls: ct[cls].tolist() for cls in classes}}

    # dropdown options for the custom form (unique values)
    options = {}
    for c in get_feature_cols():
        if c in ('ID', 'mushroom_id', 'number_of_bruises'):
            continue
        if c in df.columns:
            vals = sorted([str(x) for x in df[c].dropna().unique()])
            options[c] = vals

    # feature importances from the trained RandomForest
    importances = []
    try:
        clf = model.named_steps['classifier']
        pre = model.named_steps['preprocessor']
        names = list(pre.get_feature_names_out())
        imps = clf.feature_importances_
        order = np.argsort(imps)[::-1][:15]
        importances = [{'feature': names[i], 'importance': round(float(imps[i]), 4)}
                       for i in order]
    except Exception:
        importances = []

    _eda_cache = {
        'n_rows': int(len(df)),
        'n_cols': int(len(df.columns)),
        'n_features': int(len(get_feature_cols())),
        'class_counts': {str(k): int(v) for k, v in class_counts.items()},
        'classes': classes,
        'numeric_summary': numeric_summary,
        'hist_labels': labels,
        'hist_all': hist_all,
        'hist_by_class': hist_by_class,
        'cat_dist': cat_dist,
        'crosstabs': crosstabs,
        'options': options,
        'importances': importances,
        'missing_total': int(df.isnull().sum().sum()),
    }
    return _eda_cache


@app.route('/')
def home():
    eda = build_eda()
    metrics = get_metrics()
    val = metrics.get('val_metrics', {})
    return render_template('home.html', eda=eda, val_metrics=val)


@app.route('/eda')
def eda():
    data = build_eda()
    return render_template('eda.html', eda=data)


@app.route('/metrics')
def metrics():
    m = get_metrics()
    return render_template('metrics.html', metrics=m)


@app.route('/playground', methods=['GET', 'POST'])
def playground():
    eda = build_eda()
    df = get_train_df()
    results = []
    batch_acc = None
    n = int(request.values.get('n', 5))
    n = max(1, min(n, 20))

    if model is None:
        return render_template('playground.html', error='Model not found.',
                               results=[], eda=eda, n=n)

    # Random sample from train set
    if request.values.get('action') in ('sample', None) and df is not None:
        # default GET also shows a sample so the page is alive
        sample = df.sample(n=n, random_state=None)
        X = sample.drop(columns=['class'], errors='ignore')
        preds, proba = predict_df(X)
        classes = list(getattr(model, 'classes_', ['e', 'p']))
        for i, (idx, row) in enumerate(sample.iterrows()):
            actual = str(row.get('class', ''))
            pred = str(preds[i])
            conf = None
            if proba is not None:
                try:
                    j = classes.index(pred)
                    conf = round(float(proba[i][j]), 3)
                except Exception:
                    conf = None
            results.append({
                'features': {c: str(row.get(c, '')) for c in
                             ['cap-shape', 'cap-surface', 'cap-color', 'bruises',
                              'odor', 'gill-size', 'gill-color', 'stalk-shape',
                              'population', 'habitat', 'spore-print-color']},
                'full_row': {c: str(row.get(c, '')) for c in get_feature_cols()},
                'actual': actual,
                'predicted': pred,
                'correct': (actual == pred),
                'confidence': conf,
            })
        if results:
            batch_acc = round(sum(1 for r in results if r['correct']) / len(results), 3)

    # Custom single prediction from the form
    custom = None
    if request.method == 'POST' and request.form.get('custom_submit'):
        data = {k: v for k, v in request.form.items() if k != 'custom_submit'}
        if 'ID' not in data:
            data['ID'] = 0
        if 'mushroom_id' not in data:
            data['mushroom_id'] = 0
        inp = pd.DataFrame([data])
        preds, proba = predict_df(inp)
        classes = list(getattr(model, 'classes_', ['e', 'p']))
        conf = None
        if proba is not None:
            try:
                conf = round(float(proba[0][list(classes).index(str(preds[0]))]), 3)
            except Exception:
                pass
        custom = {'input': data, 'predicted': str(preds[0]), 'confidence': conf}

    return render_template('playground.html', results=results, batch_acc=batch_acc,
                           n=n, eda=eda, custom=custom)


@app.route('/api/eda')
def api_eda():
    return jsonify(build_eda())


if __name__ == '__main__':
    app.run(debug=True)
