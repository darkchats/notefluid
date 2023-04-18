python setup.py install
rm -rf build
rm -rf dist
rm -rf notefluid.egg-info


git pull
git add -A
git commit -a -m "add"
git push