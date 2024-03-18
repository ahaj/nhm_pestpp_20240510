#!/bin/sh
pwd

tar xzf data.tar

cd data

export PATH=/mnt/condor_working/pws/bin:${PATH}

./pestpp-ies prior_mc_reweight.pst /h 172.16.202.229:9701
