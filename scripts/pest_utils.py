import pandas as pd
import numpy as np

def pars_to_tpl_entries(pars, parname, hrus, segs, par_starting_vals,
                        hru_based=True, seg_based = True, month_based = True):
    # make parameter values into meaningful metadata-rich unique strings (i.e. names)
    tpl_pars = []
    dim_name = None
    if seg_based:
        dim_name = 'seg_'
        dim_vals = segs
    elif hru_based:
        dim_name = 'hru_'
        dim_vals = hrus
    if month_based:
        for cmonth in range(12):
            # oh hi there list comprehension - you are veeeery swanky
            tpl_pars.append(['~{0:^35}~'.format(f'{parname}:{dim_name}{cval}:mon_{cmonth+1}')
                                for cval in dim_vals])
    else:
        tpl_pars= ['~{0:^35}~'.format(f'{parname}:{dim_name}{cval}')
                                for cval in dim_vals]
    # cast into a numpy array cuz the json write will probably 💩 if given a list of lists
    tpl_pars = np.array(tpl_pars)
    # make sure we didn't mess up dimensions wrt months
    assert tpl_pars.shape == pars[parname].shape
    # grab the initial values that were in the original parameter file for starting values
    par_starting_vals = pd.concat((par_starting_vals,(pd.DataFrame(data=
                        {'parname':[v.replace('~','').strip() for v in tpl_pars.ravel()],
                                          'parval1':pars[parname].ravel()}))))
    # replace parameter values with names and delimiter for the TPL file
    pars[parname] = tpl_pars
    
    return par_starting_vals