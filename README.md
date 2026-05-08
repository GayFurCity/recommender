# Recommender

Post recommendation service for [GayFurCity](https://github.com/GayFurCity/GayFurCity).

# Quickstart

    # Install pyenv (https://github.com/pyenv/pyenv)
    git clone https://github.com/pyenv/pyenv.git ~/.pyenv
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
    echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bash_profile

    # Install python dependencies
    sudo apt install build-essential libsqlite3-dev sqlite3 bzip2 libbz2-dev libffi-dev

    # Install python
    pyenv install 3.7.5

    # Install dependency manager (https://poetry.eustace.io)
    python -m pip install --user poetry

    # Install dependencies
    python -m poetry install

    # Edit config file
    cp env.sample .env
    vim .env

    # Train model
    python -m poetry run python bin/train

    # Run webserver (development)
    python -m poetry run flask run
    python -m poetry run gunicorn wsgi

    # Get recommendations for user #1
    curl http://localhost:5000/recommend/1

    # Get recommendations for post #1
    curl http://localhost:5000/similar/1

# System requirements

Training on the full dataset of ~80 million favorites takes ~17 minutes (on an
E5-1650v4) and requires ~4GB of RAM. The trained model requires ~2GB of RAM.
