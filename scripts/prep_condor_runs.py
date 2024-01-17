import pathlib as pl
import shutil

all_extractions = ['01473000']#,'05431486','09112500']
rootnm = 'prior_mc_reweight'
results_dir = pl.Path('../results')
hotstart = True
for cex in all_extractions:
    frompath = pl.Path(f'../NHM_extractions/20230110_pois_haj/{cex}/')
    topath = pl.Path(f'../../CONDOR/{cex}')
    tofolder = ['MASTER','data']

    files = [i for i in list(frompath.glob(f'{rootnm}*') ) if 'phi' not in i.name]
    files.extend([i for i in list(frompath.glob(f'*.tpl') ) ]) 
    files.extend([i for i in list(frompath.glob(f'*.py') ) ])
    files.extend([i for i in list(frompath.glob(f'*.ins') ) ])
    if hotstart:
        files.extend([i for i in list(frompath.glob(f'*hot*'))])
    files.append(frompath / 'loc.mat')

    for tf in tofolder:
        [shutil.copy2(f, topath / tf / f.name) for f in files]
