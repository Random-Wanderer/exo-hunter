# Exo-Hunter. Final project for Le Wagon batch #728
Other contributors:
https://github.com/Awle
https://github.com/lorcanob

# Data
https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=koi
https://www.kaggle.com/keplersmachines/kepler-labelled-time-series-data

## Goal
Our goal is to create a neural network that can either, take in raw data of light curves and help predict whether or not there are exo-planets present, or a Kepler object identifier number and return data on that system and a manim graphic. The end product will be in the form of a streamlit website.


![LoopOrbit_ManimCE_v0 12 0](https://user-images.githubusercontent.com/85910457/144986403-72fcfddc-fac9-4091-ba1e-963cbb6bac28.gif)



# Data analysis
- Document here the project: exo-hunter
- Description: Project Description
- Data Source:
- Type of analysis:

Please document the project the better you can.

# Startup the project

The initial setup.

Create virtualenv and install the project:
```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:
```bash
make clean install test
```

Check for exo-hunter in gitlab.com/{group}.
If your project is not set please add it:

- Create a new project on `gitlab.com/{group}/exo-hunter`
- Then populate it:

```bash
##   e.g. if group is "{group}" and project_name is "exo-hunter"
git remote add origin git@github.com:{group}/exo-hunter.git
git push -u origin master
git push -u origin --tags
```

Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
exo-hunter-run
```

# Install

Go to `https://github.com/{group}/exo-hunter` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:{group}/exo-hunter.git
cd exo-hunter
pip install -r requirements.txt
make clean install test                # install and test
```
Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
exo-hunter-run
```
