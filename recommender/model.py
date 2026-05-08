import os
import sys
import subprocess
import pickle
import time
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares


class Recommender:
    def __init__(self, **args):
        self.TRAINING_THREADS = int(args.get("training_threads", os.cpu_count()))
        self.ALS_FACTORS = args.get("als_factors", 128)
        self.ALS_REGULARIZATION = args.get("als_regularization", 1e-2)
        self.ALS_ITERATIONS = args.get("als_iterations", 15)
        self.MIN_POST_FAVS = args.get("min_post_favs", 5)
        self.MIN_USER_FAVS = args.get("min_user_favs", 50)
        self.MAX_FAVS = args.get("max_favs", 1e12)
        self.FAVS_PATH = args.get("favs_path", "data/favs.csv")
        self.MODEL_PATH = args.get("model_path", "data/recommender.pickle")
        self.DATABASE_URL = args.get("database_url", "postgresql://gayfurcity:@postgres/gayfurcity_development")

    @staticmethod
    def create(**args):
        env = {name.lower(): value for name, value in os.environ.items()}
        args = {**env, **args}

        recommender = Recommender(**args)
        recommender.dump_favorites()
        recommender.load_favorites()
        recommender.train()
        recommender.save(recommender.MODEL_PATH)

        return recommender

    @staticmethod
    def load(model_path):
        with open(model_path, "rb") as file:
            return pickle.load(file)

    def dump_favorites(self):
        query = f"""
      SELECT
        post_id,
        user_id
      FROM favorites
      WHERE
        post_id IN (SELECT id FROM posts WHERE fav_count > {self.MIN_POST_FAVS})
        AND user_id IN (SELECT id FROM users WHERE favorite_count > {self.MIN_USER_FAVS})
      ORDER BY post_id DESC
      LIMIT {self.MAX_FAVS}
    """

        self.shell(
            f"psql --no-psqlrc -c '\copy ({query}) TO STDOUT WITH (FORMAT CSV)' {self.DATABASE_URL} > {self.FAVS_PATH}")

    def load_favorites(self):
        favs_df = pd.read_csv(self.FAVS_PATH, dtype=np.int32, names=["post_id", "user_id"])
        favs_df = favs_df.astype("category")

        if favs_df.empty:
            self.favorites = csr_matrix([])
            self.users_to_id = {}
            self.posts_to_id = {}
            self.ids_to_post = {}
            self.empty = csr_matrix([])
            return

        self.favorites = csr_matrix((np.ones(favs_df.shape[0]), (favs_df["post_id"].cat.codes.copy(), favs_df["user_id"].cat.codes.copy())), dtype=np.int32)
        self.users_to_id = {k: v for v, k in enumerate(favs_df["user_id"].cat.categories)}
        self.posts_to_id = {k: v for v, k in enumerate(favs_df["post_id"].cat.categories)}
        self.ids_to_post = {k: v for v, k in self.posts_to_id.items()}
        self.empty = csr_matrix(self.favorites.shape)

    def train(self):
        self.model = AlternatingLeastSquares(
            calculate_training_loss=True,
            dtype=np.float32,
            num_threads=self.TRAINING_THREADS,
            factors=self.ALS_FACTORS,
            regularization=self.ALS_REGULARIZATION,
            iterations=self.ALS_ITERATIONS
        )

        start = time.monotonic()
        if not self.favorites.nnz == 0:
            self.model.fit(self.favorites)
        end = time.monotonic()
        dur = int(end - start)

        self.favorites = None
        self.trained_at = datetime.utcnow().isoformat()
        self.training_time = "{:02d}:{:02d}:{:02d}".format(dur // 3600, (dur % 3600 // 60), dur % 60)

    def recommend_for_user(self, user_id, limit=50):
        if not user_id in self.users_to_id:
            return []

        uid = self.users_to_id[user_id]
        recommendations = self.model.recommend(uid, self.empty, N=limit)
        recommendations = [(self.ids_to_post[id], float(score)) for id, score in recommendations]
        return recommendations

    def recommend_for_post(self, post_id, limit=50):
        if not post_id in self.posts_to_id:
            return []

        pid = self.posts_to_id[post_id]
        recommendations = self.model.similar_items(pid, N=limit)
        recommendations = [(self.ids_to_post[id], float(score)) for id, score in recommendations]
        return recommendations

    def metrics(self):
        return {
            "user_count": len(self.users_to_id),
            "post_count": len(self.posts_to_id),
            "factors": self.model.factors,
            "model_size": 4 * self.model.factors * (len(self.users_to_id) + len(self.posts_to_id)),
            "trained_at": self.trained_at,
            "training_time": self.training_time,
        }

    def save(self, model_path):
        with open(model_path, "wb") as file:
            pickle.dump(self, file)

    def shell(self, cmd):
        subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, shell=True, check=True)
