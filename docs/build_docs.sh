cd ../imodels
uv run pdoc --html . --output-dir ../docs --template-dir ../docs
cd ../docs
# drop pages for modules pdoc no longer builds, so deleting a module from the
# package removes its page instead of leaving it to rot on an old template
uv run python prune_orphans.py
cp -rf imodels/* .
rm -rf imodels
rm -rf tests
uv run python style_docs.py

# render the hand-written pages (figs, shrinkage, mdi_plus, gpgam) from the
# shared template, taking the head and sidebar from the freshly built index.html
uv run python build_pages.py
#bash paper/compile_paper.sh