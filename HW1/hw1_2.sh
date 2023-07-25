#!/bin/bash

# TODO - run your inference Python3 code

python3 p2_test.py $1 $2
python3 viz_mask.py --img_path ../hw1_data/p2_data/validation/0013_sat.jpg --seg_path output/0013_sat.png
python3 viz_mask.py --img_path ../hw1_data/p2_data/validation/0062_sat.jpg --seg_path output/0062_sat.png
python3 viz_mask.py --img_path ../hw1_data/p2_data/validation/0104_sat.jpg --seg_path output/0104_sat.png