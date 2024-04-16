import sys, os
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
sys.path.insert(0,'../scripts/')
from postprocessing_doublewide import setup_postproc, check_pdc, plot_phi, get_obs_and_noise, get_pars, plot_group, plot_pars_group

plot_obs = True
unzip_dirs = False
plot_streamflow = True
# cms = ['01473000']
cms = ['05431486']
# cms = ['09112500']
crrs = ['prior_mc_reweight','ies_hot']
# crrs = ['ies_hot']
phi_cutoffs = {cm:{crr:9e99 for crr in ['ies_hot','prior_mc_reweight']}
                for cm in ['01473000','05431486', '09112500']}
# catalog of cutoffs heuristically determined
# catalog of cutoffs heuristically determined
phi_cutoffs['01473000']['prior_mc_reweight'] = 4.0e7
phi_cutoffs['01473000']['ies_hot'] = 2.5e7
phi_cutoffs['05431486']['prior_mc_reweight'] = 5.7e8
phi_cutoffs['05431486']['ies_hot'] = 3.2e8
phi_cutoffs['09112500']['prior_mc_reweight'] = 1.2e9
phi_cutoffs['09112500']['ies_hot'] = 2.4e8


modens_list = []
for curr_model in cms:
    for curr_run_root in crrs:
        pstdir, results_file, tmp_res_path, fig_dir, obs, pst = setup_postproc(
            curr_model, curr_run_root, unzip_dirs
            )

        # ### look at PHI history
        phi = plot_phi(tmp_res_path, curr_run_root, curr_model, fig_dir)

        # ### Truncate PHI at a threshold
        best_iter = 2
        if 'prior' in curr_run_root:
            best_iter = 0
        best_iter
        
        # ## now rejection sampling for outlier PHI values
        orgphi = phi.loc[best_iter].iloc[5:].copy()
        ax = orgphi.hist(bins=50)
        lims = ax.get_xlim()

        phi_too_high = phi_cutoffs[curr_model][curr_run_root]

        phi = orgphi.loc[orgphi<=phi_too_high]
        fig,ax = plt.subplots(1,2)
        ### --> need to indicate which reals we will carry forward <-- ###
        orgphi.hist(bins=50, ax=ax[0])
        reals = phi.index 
        phi.hist(bins=50, ax=ax[1])
        ax[0].axvline(phi_too_high, color='orange')
        ax[1].set_xlim(lims)
        ax[0].set_title(f'Original PHI: {len(orgphi)} reals')
        ax[1].set_title(f'Truncated PHI: {len(phi)} reals')
        plt.savefig(fig_dir / 'phi_histogram.pdf')


        # # Now let's start looking at the fits
        modens, obens_noise = get_obs_and_noise(tmp_res_path, curr_run_root, reals, best_iter)
        modens_list.append(modens)
    # plot_group('sca_daily', obs, modens_list[0], obens_noise, fig_dir, curr_model, best_iter, curr_run_root, modens_list[1])

    # plot_group('actet_mean_mon', obs, modens_list[0], obens_noise, fig_dir, curr_model, best_iter, curr_run_root, modens_list[1])
    # plot_group('actet_mon', obs, modens_list[0], obens_noise, fig_dir, curr_model, best_iter, curr_run_root, modens_list[1])
    # plot_group('runoff_mon', obs, modens_list[0], obens_noise, fig_dir, curr_model, best_iter, curr_run_root, modens_list[1])
    # plot_group('soil_moist_mon', obs, modens_list[0], obens_noise, fig_dir, curr_model, best_iter, curr_run_root, modens_list[1])
    # plot_group('recharge_ann', obs, modens_list[0], obens_noise, fig_dir, curr_model, best_iter, curr_run_root, modens_list[1])

    # # streamflow_daily is a special case - all aggregated
    if plot_streamflow:
        plot_group('streamflow_daily', obs, modens_list[0], obens_noise, fig_dir, curr_model, best_iter, curr_run_root, modens_list[1])
        plot_group('streamflow_mean_mon', obs, modens_list[0], obens_noise, fig_dir, curr_model, best_iter, curr_run_root, modens_list[1])
        plot_group('streamflow_mon', obs, modens_list[0], obens_noise, fig_dir, curr_model, best_iter, curr_run_root, modens_list[1])
