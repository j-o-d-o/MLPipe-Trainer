rm -rf build
rm -rf dist
rm -rf mlpipe_trainer.egg-info
python setup.py sdist bdist_wheel
twine upload  dist/*

rm -rf build
rm -rf dist
rm -rf mlpipe_trainer.egg-info
