import pathlib as pl
import time
import pywatershed


work_dir = pl.Path("./")
out_dir = work_dir / "output"
out_dir.mkdir(parents=True, exist_ok=True)

params = pywatershed.parameters.PrmsParameters.load(work_dir / "parameters.json")  # this line is necessary to avoid error in channel graph construction. apparently the parameters are edited. should probably copy them
control = pywatershed.Control.load(work_dir / "control.test", params=params)

multi_proc_model = pywatershed.Model(
    pywatershed.PRMSSolarGeometry,
    pywatershed.PRMSAtmosphere,
    pywatershed.PRMSCanopy,
    pywatershed.PRMSSnow,
    pywatershed.PRMSRunoff,
    pywatershed.PRMSSoilzone,
    pywatershed.PRMSGroundwater,
    pywatershed.PRMSChannel,
    control=control,
    input_dir=work_dir,
    budget_type=None,
    verbose=False,
    calc_method='numba',
)
multi_proc_model.initialize_netcdf(
    output_dir=out_dir,
    separate_files=True,
)

sttime = time.time()
multi_proc_model.run(finalize=True)
print(f'That took {time.time()-sttime:.3f} looong seconds')

