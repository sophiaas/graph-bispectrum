Graphisomorphism
================

Use bispectrum on symmetric group to solve (*cough*) graph isomorphism.


Setup
=====

1. Clone the repo: `git clone git@github.com:rammie/graphisomorphism.git`
2. Ensure you have python 2.7 and virtualenv: `python --version`
3. Create a virtualenv, install the requirements, and run the tests.

```python
virtualenv env
. env/bin/activate
pip install cython
pip install -r requirements.txt
nosetests -s tests
deactivate
```
