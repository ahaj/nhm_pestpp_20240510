#!/bin/sh
pwd

tar xzf data.tar

cd data

export PATH=/mnt/condor_working/pws/bin:${PATH}

./pestpp-ies ies_hot.pst /h 172.16.202.229:9701
