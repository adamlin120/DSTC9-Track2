#!/usr/bin/env bash

mwoz_21_url="https://github.com/budzianowski/multiwoz/raw/master/data/MultiWOZ_2.1.zip"
mwoz_21_clean_url="https://github.com/ConvLab/ConvLab-2/raw/master/data/multiwoz/MultiWOZ2.1_Cleaned.zip"
mwoz_22_url="https://github.com/budzianowski/multiwoz/raw/master/data/MultiWOZ_2.2.zip"

wget ${mwoz_21_url} -O temp.zip
unzip temp.zip
wget ${mwoz_21_clean_url} -O temp.zip
unzip temp.zip
wget ${mwoz_22_url} -O temp.zip
unzip temp.zip
rm -r temp.zip __MACOSX