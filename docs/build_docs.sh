cd ../imodels
uv run pdoc --html . --output-dir ../docs --template-dir ../docs
cp -rf ../docs/imodels/* ../docs/
rm -rf ../docs/imodels
cd ../docs
rm -rf tests
uv run python style_docs.py
#bash paper/compile_paper.sh