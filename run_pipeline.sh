#!/bin/sh
cd scripts
python create_augmented_private.py
python training_with_pseudo.py
python inference.py
python ensemble.py
