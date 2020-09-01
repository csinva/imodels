cd ../imodels
pdoc --html . --output-dir ../docs
cp -r ../docs/imodels/* ../docs/
rm -rf ../docs/imodels
