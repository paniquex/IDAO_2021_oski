#!/bin/bash

python3 -m venv idao_env
source ./idao_env/bin/activate

pip install -U pip

pip install -r reqirements.txt

git clone https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
pip install -e ./Ranger-Deep-Learning-Optimizer/
