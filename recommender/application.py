import os
import shutil
from datetime import timezone, datetime
from flask import Flask, jsonify, request
from prometheus_client import Counter, Gauge
from recommender import Recommender
from multiprocessing import Process, Value

application = Flask("recommender")

# Gunicorn runs multiple worker processes, each with its own metric state, so
# PROMETHEUS_MULTIPROC_DIR (set in production) switches to a registry that
# aggregates metrics across all workers from files on disk. Local dev
# (`flask run`, single-process gunicorn) doesn't set it and uses the simpler
# in-memory registry.
if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
    from prometheus_flask_exporter.multiprocess import GunicornInternalPrometheusMetrics as PrometheusMetrics
else:
    from prometheus_flask_exporter import PrometheusMetrics

prometheus = PrometheusMetrics(application, group_by="endpoint")

model_path = os.environ.get("MODEL_PATH", "data/recommender.pickle")
if not os.path.exists(model_path):
    shutil.copyfile("empty.pickle", model_path)

recommender = Recommender.load(model_path)
reload = Value("b", False)

TRAIN_TRIGGERED = Counter(
    "recommender_train_triggered_total", "Number of times a retrain was triggered via /train"
)

# These reflect the currently loaded model and are identical across gunicorn
# workers, so "max" aggregation (rather than the default sum) gives the
# correct value when scraped in multiprocess mode.
MODEL_USER_COUNT = Gauge(
    "recommender_model_user_count", "Number of users in the trained model", multiprocess_mode="max"
)
MODEL_POST_COUNT = Gauge(
    "recommender_model_post_count", "Number of posts in the trained model", multiprocess_mode="max"
)
MODEL_FACTORS = Gauge(
    "recommender_model_factors", "Number of ALS latent factors in the trained model", multiprocess_mode="max"
)
MODEL_SIZE_BYTES = Gauge(
    "recommender_model_size_bytes", "Approximate size of the trained model in bytes", multiprocess_mode="max"
)
MODEL_TRAINED_AT = Gauge(
    "recommender_model_trained_at_timestamp_seconds",
    "Unix timestamp of when the currently loaded model finished training",
    multiprocess_mode="max",
)
MODEL_TRAINING_DURATION = Gauge(
    "recommender_model_training_duration_seconds",
    "Duration of the currently loaded model's training run in seconds",
    multiprocess_mode="max",
)


def refresh_model_metrics(model):
    stats = model.metrics()
    MODEL_USER_COUNT.set(stats["user_count"])
    MODEL_POST_COUNT.set(stats["post_count"])
    MODEL_FACTORS.set(stats["factors"])
    MODEL_SIZE_BYTES.set(stats["model_size"])
    MODEL_TRAINED_AT.set(datetime.fromisoformat(stats["trained_at"]).replace(tzinfo=timezone.utc).timestamp())
    hours, minutes, seconds = (int(part) for part in stats["training_time"].split(":"))
    MODEL_TRAINING_DURATION.set(hours * 3600 + minutes * 60 + seconds)


refresh_model_metrics(recommender)


def get_model():
    global recommender
    if reload.value:
        recommender = Recommender.load(model_path)
        reload.value = False
        refresh_model_metrics(recommender)
    return recommender


@application.errorhandler(500)
def internal_server_error(error):
    return jsonify({"error": "Internal Server Error", "message": str(error)}), 500


@application.route("/recommend/<int:user_id>")
def recommend(user_id):
    limit = int(request.args.get('limit', 50))
    recommendations = get_model().recommend_for_user(user_id, limit)
    return jsonify(recommendations)


@application.route("/similar/<int:post_id>")
def similar(post_id):
    limit = int(request.args.get('limit', 50))
    recommendations = get_model().recommend_for_post(post_id, limit)
    return jsonify(recommendations)


@application.route("/info")
def info():
    return jsonify(get_model().metrics()), 200


@application.route("/train", methods=["PUT"])
def train():
    TRAIN_TRIGGERED.inc()
    process = Process(target=train_model, args=(reload,))
    process.start()
    return "", 201


def train_model(reload):
    Recommender.create()
    reload.value = True
