import sys
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
sys.path.insert(0,'../scripts/')
from postprocessing import setup_postproc, check_pdc, plot_phi, get_obs_and_noise, get_pars, plot_group, plot_pars_group

eval_pdc = True
unzip_dirs = False
cms = ['01473000','05431486', '09112500']
crrs = ['ies','prior_mc_reweight']
phi_cutoffs = {cm:{crr:9e99 for crr in crrs}
                for cm in cms}
# catalog of cutoffs heuristically determined
phi_cutoffs['01473000']['prior_mc_reweight'] = .6e10
phi_cutoffs['01473000']['ies'] = 1.08e9
phi_cutoffs['05431486']['prior_mc_reweight'] = 2.2e9
phi_cutoffs['05431486']['ies'] = 7.5e8
phi_cutoffs['09112500']['prior_mc_reweight'] = 3.0e9
phi_cutoffs['09112500']['ies'] = 8.675e8

for curr_model in cms:
    for curr_run_root in crrs:
        pstdir, results_file, tmp_res_path, fig_dir, obs, pst = setup_postproc(
            curr_model, curr_run_root, unzip_dirs
            )
        if eval_pdc:
            pdc = check_pdc(tmp_res_path, curr_run_root, pst, obs)
            print(pdc)

        # ### look at PHI history
        phi = plot_phi(tmp_res_path, curr_run_root, curr_model, fig_dir)

        # ### Truncate PHI at a threshold
        best_iter = 3
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


        # # How about parameters?
        parens = get_pars(tmp_res_path, curr_run_root, reals, best_iter, pst)
        pargps = parens['pargroup'].unique()
        for cg in pargps:
            plot_pars_group(parens, cg, fig_dir, curr_model, best_iter, curr_run_root)

       # # Now let's start looking at the fits
        modens, obens_noise = get_obs_and_noise(tmp_res_path, curr_run_root, reals, best_iter)

        # plot_group('sca_daily', obs, modens, obens_noise, fig_dir, curr_model)

        plot_group('actet_mean_mon', obs, modens, obens_noise, fig_dir, curr_model, best_iter, curr_run_root)
        plot_group('actet_mon', obs, modens, obens_noise, fig_dir, curr_model, best_iter, curr_run_root)
        plot_group('runoff_mon', obs, modens, obens_noise, fig_dir, curr_model, best_iter, curr_run_root)
        plot_group('soil_moist_mon', obs, modens, obens_noise, fig_dir, curr_model, best_iter, curr_run_root)
        plot_group('recharge_ann', obs, modens, obens_noise, fig_dir, curr_model, best_iter, curr_run_root)

        # streamflow_daily is a special case - all aggregated
        plot_group('streamflow_daily', obs, modens, obens_noise, fig_dir, curr_model, best_iter, curr_run_root)
        plot_group('streamflow_mean_mon', obs, modens, obens_noise, fig_dir, curr_model, best_iter, curr_run_root)
        plot_group('streamflow_mon', obs, modens, obens_noise, fig_dir, curr_model, best_iter, curr_run_root)
