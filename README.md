graphisomorphism
================

Use bispectrum on symmetric group to solve (*cough*) graph isomorphism.


```python
virtualenv env
. env/bin/activate
pip install cython
pip install -r requirements.txt
nosetests -s tests
deactivate
```
