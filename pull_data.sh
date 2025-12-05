#!/bin/bash

if [[ ! -d "./data " ]]
then
  mkdir ./data
fi

curl https://zenodo.org/records/7711810/files/EuroSAT_MS.zip?download=1 -o ./data/EuroSAT_MS.zip
curl https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1 -o ./data/EuroSAT_RGB.zip

unzip ./data/EuroSAT_MS.zip -d ./data/
unzip ./data/EuroSAT_RGB.zip -d ./data/

if [[ $? -eq 0 ]]
then
  echo "Download done. :)"
fi
