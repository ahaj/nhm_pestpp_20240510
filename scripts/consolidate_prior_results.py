import pathlib as pl
import zipfile

all_extractions = ['01473000','05431486','09112500']
rootnm = 'prior_mc_reweight'
results_dir = pl.Path('../results')

for cex in all_extractions:
    cpath = pl.Path(f'../../CONDOR/{cex}/MASTER')
    files = []
    files.extend(cpath.glob(f'{rootnm}*pdc*'))
    files.extend(cpath.glob(f'{rootnm}*phi*'))
    files.extend(cpath.glob(f'{rootnm}.0*'))
    with zipfile.ZipFile(results_dir / f'{rootnm}.{cex}.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        [zipf.write(i, i.name) for i in files]
