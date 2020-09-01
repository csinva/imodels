cd ../imodels
pdoc --html . --output-dir ../docs --external-links
cp -rf ../docs/imodels/* ../docs/
rm -rf ../docs/imodels
cd ../docs
python style_docs.py
