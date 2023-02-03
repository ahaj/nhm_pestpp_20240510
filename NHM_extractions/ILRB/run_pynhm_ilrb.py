import pathlib as pl
import shutil
import time
import pynhm
from pynhm import PrmsParameters
from pynhm.utils.cbh_utils import cbh_files_to_netcdf

first_rodeo = True # if this is true, convert all the chb files to netCDF

work_dir = pl.Path("./projects/ILRB")
in_dir = work_dir / "input"
out_dir = work_dir / "output"
out_dir.mkdir(parents=True, exist_ok=True)

params = pynhm.PrmsParameters.load(work_dir / 'input' / "myparam.param")  # this line is necessary to avoid error in channel graph construction. apparently the parameters are edited. should probably copy them
control = pynhm.Control.load(work_dir / 'control' / "irlb.control", params=params)

if first_rodeo:
    cbh_files = {
        'prcp' : in_dir / 'prcp.cbh',
        'rhavg' : in_dir / 'rhavg.cbh',
        'tmax' : in_dir / 'tmax.cbh',
        'tmin' : in_dir / 'tmin.cbh'
    }

    for kk, vv in cbh_files.items():
        out_file = (in_dir / f"{kk}.nc")
        cbh_files_to_netcdf({kk: vv}, params, out_file)



multi_proc_model = pynhm.Model(
    pynhm.PRMSSolarGeometry,
    pynhm.PRMSAtmosphere,
    pynhm.PRMSCanopy,
    pynhm.PRMSSnow,
    pynhm.PRMSRunoff,
    pynhm.PRMSSoilzone,
    pynhm.PRMSGroundwater,
    pynhm.PRMSChannel,
    control=control,
    input_dir=work_dir / 'input',
    budget_type=None,
    calc_method = 'numba'
)

sttime = time.time()
multi_proc_model.run(netcdf_dir=out_dir, finalize=True)
print(f'That took {time.time()-sttime:.3f} looong seconds')
