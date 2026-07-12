FROM python:3.7.12
WORKDIR /recommender

# https://github.com/python-poetry/poetry/discussions/1879#discussioncomment-216865
ENV \
  # https://stackoverflow.com/questions/59812009/what-is-the-use-of-pythonunbuffered-in-docker-file
  PYTHONUNBUFFERED=1 \
  # https://python-docs.readthedocs.io/en/latest/writing/gotchas.html#disabling-bytecode-pyc-files
  PYTHONDONTWRITEBYTECODE=1 \
  # https://stackoverflow.com/questions/45594707/what-is-pips-no-cache-dir-good-for
  PIP_NO_CACHE_DIR=1 \
  # https://stackoverflow.com/questions/46288847/how-to-suppress-pip-upgrade-warning
  PIP_DISABLE_PIP_VERSION_CHECK=1 \
  # Required for correct Prometheus metrics under gunicorn's multiple worker processes.
  PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus-multiproc

RUN apt-get update && \
  apt-get install -y --no-install-recommends tini postgresql-client && \
  pip install "poetry==1.1.11" && \
  mkdir -p "$PROMETHEUS_MULTIPROC_DIR"

COPY pyproject.toml poetry.lock ./
RUN poetry install --no-dev

COPY . .

EXPOSE 5000
ENTRYPOINT ["tini", "--"]
CMD ["python", "-m", "poetry", "run", "gunicorn", "wsgi", "--bind", "0.0.0.0:5000"]
