import xarray as xr
import pathlib as pl
import numpy as np
import pandas as pd

#all_models = ['01473000', '05431486','09112500','14015000']
#rootdir = pl.Path('../NHM_extractions/20230110_pois_haj/')


#for vv, in all_models:
#    wkdir = pl.Path('../vv/')
#    cbh = xr.open_dataset(wkdir / 'baseline_AET_v11.nc')

cbh = xr.open_dataset('cbh.nc')
cbh = cbh.rename({'hruid': 'hru_id'})

cbh[['tmax']].to_netcdf('tmax.nc')
cbh[['tmin']].to_netcdf('tmin.nc')
cbh[['prcp']].to_netcdf('prcp.nc')

cbh.close()