Graphisomorphism
================

Use bispectrum on symmetric group to solve (*cough*) graph isomorphism.


## Setup

```bash
git clone git@github.com:rammie/graphisomorphism.git
cd graphisomorphism
virtualenv env
. env/bin/activate
pip install cython numpy==1.6.2
pip install -r requirements.txt
python setup.py build_ext --inplace
nosetests -s tests
deactivate
```
