cd ../imodels
uv run pdoc --html . --output-dir ../docs --template-dir ../docs
cp -rf ../docs/imodels/* ../docs/
rm -rf ../docs/imodels
cd ../docs
rm -rf tests
uv run python style_docs.py

# render the hand-written pages (figs, shrinkage, mdi_plus, gpgam) from the
# shared template, taking the head and sidebar from the freshly built index.html
uv run python build_pages.py
#bash paper/compile_paper.sh