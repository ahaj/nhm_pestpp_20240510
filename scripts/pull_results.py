import pathlib as pl
import zipfile
import os

all_extractions = ['01473000','05431486','09112500']
#all_extractions = ['01473000']
#rootnm = 'prior_mc_reweight'
rootnm = 'ies_hot'


for cex in all_extractions:
    os.system(f"aws s3 cp s3://hytest-workspace/mnfienen/{rootnm}.{cex}.zip ../results/{rootnm}.{cex}.zip --endpoint-url https://usgs.osn.mghpcc.org/ --profile osn-hytest-workspace")
