python -m pip uninstall pflacco --yes
python -m pip install -r requirements.txt
git clone https://github.com/Reiyan/pflacco.git
cp -rf MANIFEST.in pflacco/
cp -rf setup.py pflacco/
cp -r deep_ela pflacco/pflacco
cd pflacco
python -m pip install .