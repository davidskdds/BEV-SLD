# install venv
sudo apt install python3-venv

# create virtual environment
python3 -m venv bev_sld_env

# activate virtual environment
source bev_sld_env/bin/activate

# upgrade pip
pip install --upgrade pip

# install required packages
pip install -r requirements.txt