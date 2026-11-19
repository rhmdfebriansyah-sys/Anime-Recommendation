import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

kmeans = joblib.load(os.path.join(BASE_DIR, "kmeans.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

FEATURES = list(kmeans.feature_names_in_)  # urutan kolom persis seperti saat training
NUM_COLS = list(scaler.feature_names_in_)  # rating, members_log, episodes_log
GENRES = [f for f in FEATURES if f not in NUM_COLS and not f.startswith("type_")]
TYPES = [f[len("type_"):] for f in FEATURES if f.startswith("type_")]

# data anime + fitur yang sudah di-scale (baris sejajar satu sama lain)
anime = pd.read_csv(os.path.join(BASE_DIR, "anime_clustered.csv"))
X = pd.read_csv(os.path.join(BASE_DIR, "anime_features.csv"))[FEATURES].to_numpy(dtype=float)
assert len(anime) == len(X), "anime_clustered.csv dan anime_features.csv tidak sejajar"
NAMES_LOWER = anime["name"].fillna("").str.lower()
ID_TO_IDX = {int(a): i for i, a in enumerate(anime["anime_id"])}


@app.route("/")
def index():
    return render_template("index.html", genres=GENRES, types=TYPES)


@app.route("/search")
def search():
    q = request.args.get("q", "").strip().lower()
    if len(q) < 2:
        return jsonify([])
    mask = NAMES_LOWER.str.contains(q, regex=False)
    hits = anime[mask].sort_values("members", ascending=False).head(10)
    return jsonify([{"id": int(r.anime_id), "name": r["name"]} for _, r in hits.iterrows()])


@app.route("/recommend")
def recommend():
    try:
        anime_id = int(request.args.get("id"))
    except (TypeError, ValueError):
        return jsonify(error="ID anime tidak valid."), 400

    idx = ID_TO_IDX.get(anime_id)
    if idx is None:
        return jsonify(error="Anime tidak ditemukan."), 404

    # jarak Euclidean di ruang fitur yang sama dengan K-Means
    dist = np.linalg.norm(X - X[idx], axis=1)
    dist[idx] = np.inf  # jangan rekomendasikan dirinya sendiri
    nearest = np.argsort(dist)[:10]

    def to_item(i):
        r = anime.iloc[i]
        return {
            "id": int(r.anime_id),
            "name": r["name"],
            "genre": r["genre"] if isinstance(r["genre"], str) else "",
            "type": r["type"] if isinstance(r["type"], str) else "",
            "episodes": None if pd.isna(r["episodes"]) else int(r["episodes"]),
            "rating": None if pd.isna(r["rating"]) else float(r["rating"]),
            "cluster": int(r["cluster"]),
        }

    return jsonify(
        selected=to_item(idx),
        recommendations=[{**to_item(i), "distance": round(float(dist[i]), 4)} for i in nearest],
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    try:
        rating = float(data.get("rating"))
        members = float(data.get("members"))
        episodes = float(data.get("episodes"))
    except (TypeError, ValueError):
        return jsonify(error="Rating, members, dan episodes harus berupa angka."), 400

    if not 0 <= rating <= 10:
        return jsonify(error="Rating harus antara 0 dan 10."), 400
    if members < 0 or episodes < 0:
        return jsonify(error="Members dan episodes tidak boleh negatif."), 400

    genres = set(data.get("genres") or [])
    anime_type = data.get("type")
    if anime_type not in TYPES:
        return jsonify(error="Tipe anime tidak valid."), 400

    row = {f: 0.0 for f in FEATURES}
    for g in genres & set(GENRES):
        row[g] = 1.0
    row[f"type_{anime_type}"] = 1.0
    row["rating"] = rating
    row["members_log"] = float(np.log1p(members))
    row["episodes_log"] = float(np.log1p(episodes))

    df = pd.DataFrame([row], columns=FEATURES)
    df[NUM_COLS] = scaler.transform(df[NUM_COLS])

    cluster = int(kmeans.predict(df)[0])
    distances = kmeans.transform(df)[0]
    return jsonify(
        cluster=cluster,
        distances=[round(float(d), 4) for d in distances],
    )


if __name__ == "__main__":
    app.run(debug=True)
