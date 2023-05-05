#!/bin/sh
python3 train.py -m googlenet
python3 train.py -m mobilenet
python3 train.py -m efficient
python3 python_notification.py