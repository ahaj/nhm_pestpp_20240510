import pathlib as pl
import time
import pynhm


work_dir = pl.Path("./")
out_dir = work_dir / "output"
out_dir.mkdir(parents=True, exist_ok=True)

params = pynhm.PrmsParameters.load(work_dir / "myparam.param")  # this line is necessary to avoid error in channel graph construction. apparently the parameters are edited. should probably copy them
control = pynhm.Control.load(work_dir / "control.test", params=params)

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
    input_dir=work_dir,
    budget_type=None,
    calc_method = 'fortran'
)

sttime = time.time()
multi_proc_model.run(netcdf_dir=out_dir, finalize=True)
print(f'That took {time.time()-sttime:.3f} looong seconds')

