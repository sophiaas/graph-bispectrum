Graphisomorphism
================

Use bispectrum on symmetric group to solve (*cough*) graph isomorphism.


## Setup

```bash
git clone git@github.com:rammie/graphisomorphism.git
cd graphisomorphism
virtualenv env
. env/bin/activate
pip install cython
pip install -r requirements.txt
python setup.py build_ext --inplace
nosetests -s tests
deactivate
```

## Docker

A `Dockerfile` and a `docker-compose.yml` are included in the root of the repo.

```bash
docker-compose build
docker-compose run graphisomorphism bash
```
 
 This should give you an interactive shell inside the container.
 The code is mounted in the /code directory.

 ```bash
 cd /code
 python setup.py build_ext --inplace --force
 nosetests -s tests
 ```
