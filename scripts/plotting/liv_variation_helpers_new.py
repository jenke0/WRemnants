import hist
import numpy as np

from wums.boostHistHelpers import (
    addHists,
    broadcastSystHist,
    divideHists,
    multiplyHists,
    scaleHist,
)

mass_bin = 9


def mc_scaling(hist_in, hist_proj, lumi_scaling, weightsum, cross_sec):
    hist_in /= weightsum
    hist_in *= cross_sec
    hist_in *= 1000
    hist_in_2d = broadcastSystHist(hist_in, hist_proj)
    hist_in_2d = multiplyHists(hist_in_2d, lumi_scaling)

    return hist_in_2d


def correct_all_channels(
    iso_mc,
    dtdt_mc,
    dtst_mc,
    stst_mc,
    hist_proj_low,
    lumi_scaling,
    weightsum,
    cross_sec,
):
    iso = mc_scaling(iso_mc.copy(), hist_proj_low, lumi_scaling, weightsum, cross_sec)
    dtdt = mc_scaling(dtdt_mc.copy(), hist_proj_low, lumi_scaling, weightsum, cross_sec)
    dtst = mc_scaling(
        dtst_mc.copy(), hist_proj_low.copy(), lumi_scaling, weightsum, cross_sec
    )
    stst = mc_scaling(
        stst_mc.copy(), hist_proj_low.copy(), lumi_scaling, weightsum, cross_sec
    )
    return iso, dtdt, dtst, stst


def make_ones_hist(hist_ref):
    ones = np.ones_like(hist_ref.values())
    h_ones = hist_ref.copy()
    h_ones.values()[...] = ones
    return h_ones


def get_mc_lumis(
    input_data,
    time_proj_low,
    scaling,
    lumi_hists,
    weightsum,
    cross_sec,
):
    iso_h, dtdt_h, dtst_h, stst_h, iso_bg, dtdt_bg, dtst_bg, stst_bg = input_data

    lumi_h, lumi_bg = lumi_hists
    sum_lumis = addHists(lumi_bg, lumi_h)
    lumi_scaling_h = divideHists(lumi_h, sum_lumis)
    lumi_scaling_bg = divideHists(lumi_bg, sum_lumis)
    iso_h, dtdt_h, dtst_h, stst_h = correct_all_channels(
        iso_h,
        dtdt_h,
        dtst_h,
        stst_h,
        time_proj_low,
        multiplyHists(lumi_scaling_h, scaling),
        weightsum,
        cross_sec,
    )

    iso_bg, dtdt_bg, dtst_bg, stst_bg = correct_all_channels(
        iso_bg,
        dtdt_bg,
        dtst_bg,
        stst_bg,
        time_proj_low,
        multiplyHists(lumi_scaling_bg, scaling),
        weightsum,
        cross_sec,
    )

    iso = addHists(iso_bg, iso_h)
    dtdt = addHists(dtdt_bg, dtdt_h)
    dtst = addHists(dtst_bg, dtst_h)
    stst = addHists(stst_bg, stst_h)

    return iso, dtdt, dtst, stst


### i need to get good at coding so i dont need to pass in all these variables
def eta_phi_systematic(
    writer,
    input_data,
    time_hists,
    lumi_scaling,
    lumi_hists,
    weightsum,
    cross_sec,
    etaphi_num,
    mass_bin,
):
    (
        iso_H_stat,
        dtdt_H_stat,
        dtst_H_stat,
        stst_H_stat,
        iso_BG_stat,
        dtdt_BG_stat,
        dtst_BG_stat,
        stst_BG_stat,
    ) = input_data

    input_data = [
        iso_H_stat[{"etaPhiRegion": etaphi_num}],
        dtdt_H_stat[{"etaPhiRegion": etaphi_num}],
        dtst_H_stat[{"etaPhiRegion": etaphi_num}],
        stst_H_stat[{"etaPhiRegion": etaphi_num}],
        iso_BG_stat[{"etaPhiRegion": etaphi_num}],
        dtdt_BG_stat[{"etaPhiRegion": etaphi_num}],
        dtst_BG_stat[{"etaPhiRegion": etaphi_num}],
        stst_BG_stat[{"etaPhiRegion": etaphi_num}],
    ]

    iso_stat, dtdt_stat, dtst_stat, stst_stat = get_mc_lumis(
        input_data,
        time_hists,
        lumi_scaling,
        lumi_hists,
        weightsum,
        cross_sec,
    )
    iso_stat, dtdt_stat, dtst_stat, stst_stat = make_mutually_exclusive(
        iso_stat, dtdt_stat, dtst_stat, stst_stat
    )

    poi_prefire_stat_proj = iso_stat.project("time", "mll")
    writer.add_systematic(
        remove_bins(poi_prefire_stat_proj, "mll", mass_bin + 1),
        f"prefiring_stat_etaphi_{etaphi_num}",
        "Zmumu",
        "ch_iso_poi_high",
        constrained=True,
        groups=["prefiring_stat"],
    )
    writer.add_systematic(
        remove_bins(poi_prefire_stat_proj, "mll", mass_bin + 1, False),
        f"prefiring_stat_etaphi_{etaphi_num}",
        "Zmumu",
        "ch_iso_poi_low",
        constrained=True,
        groups=["prefiring_stat"],
    )

    writer.add_systematic(
        iso_stat[{"mll": mass_bin}].project("time", "pt_probe", "eta_probe"),
        f"prefiring_stat_etaphi_{etaphi_num}",
        "Zmumu",
        "ch_iso_eff",
        constrained=True,
        groups=["prefiring_stat"],
    )
    dtdt_stat = remove_bins(dtdt_stat)
    writer.add_systematic(
        dtdt_stat[{"mll": mass_bin}].project("time", "pt_probe", "eta_probe"),
        f"prefiring_stat_etaphi_{etaphi_num}",
        "Zmumu",
        "ch_dtdt_eff",
        constrained=True,
        groups=["prefiring_stat"],
    )
    writer.add_systematic(
        dtst_stat[{"mll": mass_bin}].project("time", "pt_probe", "eta_probe"),
        f"prefiring_stat_etaphi_{etaphi_num}",
        "Zmumu",
        "ch_dtst_eff",
        constrained=True,
        groups=["prefiring_stat"],
    )
    writer.add_systematic(
        stst_stat[{"mll": mass_bin}].project("time", "pt_probe", "eta_probe"),
        f"prefiring_stat_etaphi_{etaphi_num}",
        "Zmumu",
        "ch_stst_eff",
        constrained=True,
        groups=["prefiring_stat"],
    )


def get_era_vals(mc, trigger_cut, era, type_gen="pass", iso=False):
    if type_gen == "pass":
        if not iso:
            return (
                mc[f"{trigger_cut}_prpg_{era}"].get(),
                mc[f"{trigger_cut}_prpg_{era}_muonL1PrefireSyst"].get(),
                mc[f"{trigger_cut}_prpg_{era}_muonL1PrefireStat"].get(),
            )
        else:
            return (
                mc[f"{trigger_cut}_{era}"].get(),
                mc[f"{trigger_cut}_{era}_muonL1PrefireSyst"].get(),
                mc[f"{trigger_cut}_{era}_muonL1PrefireStat"].get(),
            )

    else:
        return (
            mc[f"{trigger_cut}_prfg_{era}"].get(),
            mc[f"{trigger_cut}_prfg_{era}_muonL1PrefireSyst"].get(),
            mc[f"{trigger_cut}_prfg_{era}_muonL1PrefireStat"].get(),
        )


def luminometer_syst(writer, luminometer, iso, dtdt, dtst, stst, syst):
    dtdt = remove_bins(dtdt)
    writer.add_systematic(
        iso[{"mll": mass_bin}].project("time", "pt_probe", "eta_probe"),
        f"{luminometer}_{syst}",
        "Zmumu",
        "ch_iso_eff",
        constrained=True,
        groups=[f"{syst}"],
    )

    writer.add_systematic(
        dtdt[{"mll": mass_bin}].project("time", "pt_probe", "eta_probe"),
        f"{luminometer}_{syst}",
        "Zmumu",
        "ch_dtdt_eff",
        constrained=True,
        groups=[f"{syst}"],
    )
    writer.add_systematic(
        dtst[{"mll": mass_bin}].project("time", "pt_probe", "eta_probe"),
        f"{luminometer}_{syst}",
        "Zmumu",
        "ch_dtst_eff",
        constrained=True,
        groups=[f"{syst}"],
    )
    writer.add_systematic(
        stst[{"mll": mass_bin}].project("time", "pt_probe", "eta_probe"),
        f"{luminometer}_{syst}",
        "Zmumu",
        "ch_stst_eff",
        constrained=True,
        groups=[f"{syst}"],
    )

    poi_iso_lumi_proj = iso.project("time", "mll")
    writer.add_systematic(
        remove_bins(poi_iso_lumi_proj, "mll", mass_bin + 1),
        f"{luminometer}_{syst}",
        "Zmumu",
        "ch_iso_poi_high",
        constrained=True,
        groups=[f"{syst}"],
    )
    writer.add_systematic(
        remove_bins(poi_iso_lumi_proj, "mll", mass_bin + 1, False),
        f"{luminometer}_{syst}",
        "Zmumu",
        "ch_iso_poi_low",
        constrained=True,
        groups=[f"{syst}"],
    )


def background_syst(
    writer,
    results,
    res_str,
    time_proj_low,
    lumi_scaling,
    lumi_hists,
    proc_name,
    bkg_name,
    fail_gen=False,
    mass_bin=mass_bin,
):

    MC = results[res_str]["output"]
    try:
        weightsum = results[res_str]["weight_sum"]
        cross_sec = results[res_str]["dataset"]["xsec"]
    except:
        weightsum = results["SingleMuon_2016PostVFP"]["weight_sum"]
        cross_sec = results["SingleMuon_2016PostVFP"]["dataset"]["xsec"]
    print(res_str)

    if res_str == "Ztautau_2016PostVFP":
        signal_proc = True
    else:
        signa_proc = False
    ### MAKE THIS IMPLEMENTATION NOT STUPID
    if fail_gen:
        dtdt_prpg_BG, _, _ = get_era_vals(MC, "dtdt", "BG", "fail")
        dtst_prpg_BG, _, _ = get_era_vals(MC, "dtst", "BG", "fail")
        stst_prpg_BG, _, _ = get_era_vals(MC, "stst", "BG", "fail")

        dtdt_prpg_H, _, _ = get_era_vals(MC, "dtdt", "H", "fail")
        dtst_prpg_H, _, _ = get_era_vals(MC, "dtst", "H", "fail")
        stst_prpg_H, _, _ = get_era_vals(MC, "stst", "H", "fail")
    else:
        iso_BG, _, _ = get_era_vals(MC, "pass_iso", "BG", iso=True)
        dtdt_prpg_BG, _, _ = get_era_vals(MC, "dtdt", "BG")
        dtst_prpg_BG, _, _ = get_era_vals(MC, "dtst", "BG")
        stst_prpg_BG, _, _ = get_era_vals(MC, "stst", "BG")

        iso_H, _, _ = get_era_vals(MC, "pass_iso", "H", iso=True)
        dtdt_prpg_H, _, _ = get_era_vals(MC, "dtdt", "H")
        dtst_prpg_H, _, _ = get_era_vals(MC, "dtst", "H")
        stst_prpg_H, _, _ = get_era_vals(MC, "stst", "H")

    prpg_all = [
        iso_H,
        dtdt_prpg_H,
        dtst_prpg_H,
        stst_prpg_H,
        iso_BG,
        dtdt_prpg_BG,
        dtst_prpg_BG,
        stst_prpg_BG,
    ]

    iso, dtdt, dtst, stst = get_mc_lumis(
        prpg_all,
        time_proj_low,
        lumi_scaling,
        lumi_hists,
        weightsum,
        cross_sec,
    )
    iso, dtdt, dtst, stst = make_mutually_exclusive(iso, dtdt, dtst, stst)
    dtdt = remove_bins(dtdt)

    iso_poi = iso.project("time", "mll")

    iso_poi_high = remove_bins(iso_poi, "mll", mass_bin + 1)
    writer.add_process(
        iso_poi_high,
        f"{proc_name}",
        "ch_iso_poi_high",
        signal=signal_proc,
    )
    iso_poi_low = remove_bins(
        iso_poi, "mll", mass_bin + 1, False
    )  ### is this the right one?
    writer.add_process(
        iso_poi_low,
        f"{proc_name}",
        "ch_iso_poi_low",
        signal=signal_proc,
    )

    # iso_low = expand_hist by duplicate axes
    # for i in range(nbins_mll):
    #     if i != mass_bin:
    #         if i < mass_bin:
    #             writer.add_systematic(
    #                 addHists(poi_mll_low[{"gen_mll": i}] * var_size, h3_poi_low),
    #                 f"n_mll{i}_{proc_name}",
    #                 f"{proc_name}",
    #                 "ch_iso_poi_low",
    #                 constrained=False,
    #                 groups=["nz"],
    #             )
    #         else:
    #             print("above mass bin")
    #             writer.add_systematic(
    #                 addHists(
    #                     poi_mll_high[{"gen_mll": i - (mass_bin + 1)}] * var_size,
    #                     h3_poi_high,
    #                 ),
    #                 f"n_mll{i}_{proc_name}",
    #                 f"{proc_name}",
    #                 "ch_iso_poi_high",
    #                 constrained=False,
    #                 groups=["nz"],
    #             )

    var = 1.01
    if proc_name == "W" or "Diboson":
        var = 1.001
    # writer.add_norm_systematic(
    #     f"{bkg_name}", f"{proc_name}", "ch_iso_poi_high", var, groups=["bkg"]
    # )

    # writer.add_norm_systematic(
    #     f"{bkg_name}", f"{proc_name}", "ch_iso_poi_low", var, groups=["bkg"]
    # )

    ##### in efficiency channels ####
    iso_proc = iso[{"mll": mass_bin}].project("time", "pt_probe", "eta_probe")
    dtdt_proc = dtdt[{"mll": mass_bin}].project("time", "pt_probe", "eta_probe")
    dtst_proc = dtst[{"mll": mass_bin}].project("time", "pt_probe", "eta_probe")
    stst_proc = stst[{"mll": mass_bin}].project("time", "pt_probe", "eta_probe")

    if proc_name == "W":
        dtdt_proc.variances()[...] = np.abs(dtdt_proc.variances())
        dtdt_proc.values()[...] = np.abs(
            dtdt_proc.variances()
        )  ## there are no variances less than -0.01 so setting this as positive per kenneth's recommendation

    writer.add_process(
        iso_proc, f"n_mll9_{proc_name}", "ch_iso_eff", signal=signal_proc
    )
    writer.add_process(
        dtdt_proc, f"n_mll9_{proc_name}", "ch_dtdt_eff", signal=signal_proc
    )
    writer.add_process(
        dtst_proc, f"n_mll9_{proc_name}", "ch_dtst_eff", signal=signal_proc
    )
    writer.add_process(
        stst_proc, f"n_mll9_{proc_name}", "ch_stst_eff", signal=signal_proc
    )

    var = 1.01
    if proc_name == "W" or "Diboson":
        var = 1.001
    writer.add_norm_systematic(
        f"{bkg_name}", f"{proc_name}", "ch_iso_eff", var, groups=["bkg"]
    )
    writer.add_norm_systematic(
        f"{bkg_name}", f"{proc_name}", "ch_dtdt_eff", var, groups=["bkg"]
    )
    writer.add_norm_systematic(
        f"{bkg_name}", f"{proc_name}", "ch_dtst_eff", var, groups=["bkg"]
    )
    writer.add_norm_systematic(
        f"{bkg_name}", f"{proc_name}", "ch_stst_eff", var, groups=["bkg"]
    )


def remove_bins(old_hist, ax_name="pt_probe", nbins=1, low=True):
    if ax_name == "pt_probe":
        nbins = old_hist.axes[ax_name].index(25)

    if low:
        new_edges = old_hist.axes[ax_name].edges[nbins:]
    elif not low:
        new_edges = old_hist.axes[ax_name].edges[:nbins]
    new_axis = hist.axis.Variable(new_edges, name=ax_name)
    ax_name_ind = old_hist.axes.name.index(ax_name)

    axes = list(old_hist.axes)
    axes[ax_name_ind] = new_axis

    slices = [slice(None)] * old_hist.ndim
    if low:
        slices[ax_name_ind] = slice(nbins, None)
    elif not low:
        slices[ax_name_ind] = slice(None, nbins - 1)

    if "probe" in ax_name:
        try:
            new_tag_axis = hist.axis.Variable(new_edges, name=f"{ax_name[:-5]}tag")
            tag_ind = old_hist.axes.name.index(f"{ax_name[:-5]}tag")
            axes[tag_ind] = new_tag_axis
            slices[tag_ind] = slice(nbins, None)
        except:  ## if there is no tag access for some reason
            pass

    new_hist = hist.Hist(*axes)
    new_hist.values()[...] = old_hist.values()[tuple(slices)]

    return new_hist


def make_mutually_exclusive(iso, dtdt, dtst, stst):
    iso_ex = iso
    dtdt_ex = addHists(dtdt, scaleHist(iso, -1))
    dtdt_iso_injected = dtdt.copy()
    dtdt_iso_injected.values()[:, :, 0, :, :, :] = iso[:, :, 0, :, :, :].values()
    dtst_ex = addHists(dtst, scaleHist(dtdt_iso_injected, -1))
    stst_ex = addHists(stst, scaleHist(dtst, -1))

    return iso_ex, dtdt_ex, dtst_ex, stst_ex


def create_variation(
    variation_hist,
    refererence_hist,
    i,
    j,
    k,
    nbins_total,
    muon="tag",
    h2=False,
    var_size=0.01,
):

    not_muon = "probe"
    var = variation_hist[{"gen_time": k, f"pt_{not_muon}": i, f"eta_{not_muon}": j}]

    var = var.project("time", "mll", f"pt_{muon}", f"eta_{muon}")
    var = broadcastSystHist(var, refererence_hist)
    var = multiplyHists(scaleHist(var, var_size / (nbins_total)), refererence_hist)
    var = var.project("time", "mll", "pt_probe", "eta_probe")
    return var


def variation_array(values, axes, selections, output_axes):
    """Select named bins and return a NumPy array in a requested axis order."""
    remaining_axes = list(axes.name)
    for axis_name, index in selections.items():
        axis_index = remaining_axes.index(axis_name)
        values = np.take(values, index, axis=axis_index)
        remaining_axes.pop(axis_index)
    return np.transpose(
        values, [remaining_axes.index(axis_name) for axis_name in output_axes]
    )


def variation_values(variation_hist):
    """Return the compact 4D NumPy variation array without duplicate axes."""
    return np.asarray(variation_hist.values())


def create_variation_array(
    variation_hist,
    reference_values,
    i,
    j,
    nbins_total,
    h2=False,
    var_size=0.01,
):
    """Create all generator-time variations as a NumPy array.

    The returned shape is ``(gen_time, time, mll, pt_probe, eta_probe)``.
    ``reference_values`` must already have the latter four axes in that order.
    """
    hist_i = i - 1 if h2 else i
    factors = variation_array(
        variation_hist.values(),
        variation_hist.axes,
        {"pt_probe": hist_i, "eta_probe": j},
        ["gen_time", "time", "mll", "pt_tag", "eta_tag"],
    )
    if factors.shape[1:] != reference_values.shape:
        raise ValueError(
            f"Variation shape {factors.shape[1:]} does not match reference shape "
            f"{reference_values.shape}"
        )
    return factors * reference_values[None, ...] * (var_size / nbins_total)


def create_probe_variation_array(variation_hist, reference_values, i, j, var_size=0.01):
    """Create all generator-time probe variations as a NumPy array."""
    factors = variation_array(
        variation_hist.values(),
        variation_hist.axes,
        {},
        ["gen_time", "time", "mll", "pt_probe", "eta_probe", "pt_tag", "eta_tag"],
    )[:, :, :, :, :, i, j]
    return factors * reference_values[None, ...] * var_size


def prefiring_syst(writer, iso_prefire, dtdt_prefire, dtst_prefire, stst_prefire):
    writer.add_systematic(
        iso_prefire[{"mll": mass_bin}].project("time", "pt_probe", "eta_probe"),
        f"prefiring_syst",
        "Zmumu",
        "ch_iso_eff",
        constrained=True,
        groups=["prefiring_syst"],
    )
    dtdt_prefire = remove_bins(dtdt_prefire)
    writer.add_systematic(
        dtdt_prefire[{"mll": mass_bin}].project("time", "pt_probe", "eta_probe"),
        f"prefiring_syst",
        "Zmumu",
        "ch_dtdt_eff",
        constrained=True,
        groups=["prefiring_syst"],
    )
    writer.add_systematic(
        dtst_prefire[{"mll": mass_bin}].project("time", "pt_probe", "eta_probe"),
        f"prefiring_syst",
        "Zmumu",
        "ch_dtst_eff",
        constrained=True,
        groups=["prefiring_syst"],
    )
    writer.add_systematic(
        stst_prefire[{"mll": mass_bin}].project(
            "time", "pt_probe", "eta_probe"
        ),  # used to be tag pt and eta
        f"prefiring_syst",
        "Zmumu",
        "ch_stst_eff",
        constrained=True,
        groups=["prefiring_syst"],
    )

    poi_prefire_proj = iso_prefire.project("time", "mll")
    writer.add_systematic(
        remove_bins(poi_prefire_proj, "mll", mass_bin + 1),
        f"prefiring_syst",
        "Zmumu",
        "ch_iso_poi_high",
        constrained=True,
        groups=["prefiring_syst"],
    )

    writer.add_systematic(
        remove_bins(poi_prefire_proj, "mll", mass_bin + 1, False),
        f"prefiring_syst",
        "Zmumu",
        "ch_iso_poi_low",
        constrained=True,
        groups=["prefiring_syst"],
    )


def get_corrected_mc(
    results,
    process,
    time_proj_low,
    lumi_hists,
    pcc_scaling,
    hfoc_scaling,
    ramses_scaling,
    hfoc_sbil,
    ramses_sbil,
    lumi_scaling,
    luminometers=False,
):
    print(process)
    MC = results[process]["output"]

    iso_BG, iso_BG_syst, iso_BG_stat = get_era_vals(MC, "pass_iso", "BG", iso=True)
    dtdt_BG, dtdt_BG_syst, dtdt_BG_stat = get_era_vals(MC, "dtdt", "BG")
    dtst_BG, dtst_BG_syst, dtst_BG_stat = get_era_vals(MC, "dtst", "BG")
    stst_BG, stst_BG_syst, stst_BG_stat = get_era_vals(MC, "stst", "BG")

    iso_H, iso_H_syst, iso_H_stat = get_era_vals(MC, "pass_iso", "H", iso=True)
    dtdt_H, dtdt_H_syst, dtdt_H_stat = get_era_vals(MC, "dtdt", "H")
    dtst_H, dtst_H_syst, dtst_H_stat = get_era_vals(MC, "dtst", "H")
    stst_H, stst_H_syst, stst_H_stat = get_era_vals(MC, "stst", "H")
    syst = [
        iso_H_syst[{"downUpVar": 0}],
        dtdt_H_syst[{"downUpVar": 0}],
        dtst_H_syst[{"downUpVar": 0}],
        stst_H_syst[{"downUpVar": 0}],
        iso_BG_syst[{"downUpVar": 0}],
        dtdt_BG_syst[{"downUpVar": 0}],
        dtst_BG_syst[{"downUpVar": 0}],
        stst_BG_syst[{"downUpVar": 0}],
    ]
    stat = [
        iso_H_stat[{"downUpVar": 0}],
        dtdt_H_stat[{"downUpVar": 0}],
        dtst_H_stat[{"downUpVar": 0}],
        stst_H_stat[{"downUpVar": 0}],
        iso_BG_stat[{"downUpVar": 0}],
        dtdt_BG_stat[{"downUpVar": 0}],
        dtst_BG_stat[{"downUpVar": 0}],
        stst_BG_stat[{"downUpVar": 0}],
    ]

    prpg_all = [
        iso_H,
        dtdt_H,
        dtst_H,
        stst_H,
        iso_BG,
        dtdt_BG,
        dtst_BG,
        stst_BG,
    ]

    weightsum = results[process]["weight_sum"]
    xsec = results[process]["dataset"]["xsec"]

    pass_gen = MC["pass_gen"].get()

    iso, dtdt, dtst, stst = get_mc_lumis(
        prpg_all,
        time_proj_low,
        lumi_scaling,
        lumi_hists,
        weightsum,
        xsec,
    )
    print("through nominal corrections")
    if luminometers:
        ### i think the way these should work is i do it in a single mass bin then project it across all the rest?

        hfoc_rescale = divideHists(hfoc_scaling, lumi_scaling)
        iso_hfoc = multiplyHists(hfoc_rescale, iso)
        dtdt_hfoc = multiplyHists(hfoc_rescale, dtdt)
        dtst_hfoc = multiplyHists(hfoc_rescale, dtst)
        stst_hfoc = multiplyHists(hfoc_rescale, stst)

        print("hfoc stability")
        pcc_rescale = divideHists(pcc_scaling, lumi_scaling)
        iso_pcc = multiplyHists(pcc_rescale, iso)
        dtdt_pcc = multiplyHists(pcc_rescale, dtdt)
        dtst_pcc = multiplyHists(pcc_rescale, dtst)
        stst_pcc = multiplyHists(pcc_rescale, stst)
        print("pcc stability")

        ramses_rescale = divideHists(ramses_scaling, lumi_scaling)
        iso_ramses = multiplyHists(ramses_rescale, iso)
        dtdt_ramses = multiplyHists(ramses_rescale, dtdt)
        dtst_ramses = multiplyHists(ramses_rescale, dtst)
        stst_ramses = multiplyHists(ramses_rescale, stst)
        print("ramses stability")

        hfoc_sbil_rescale = divideHists(hfoc_sbil, lumi_scaling)
        iso_sbil_hfoc = multiplyHists(hfoc_sbil_rescale, iso)
        dtdt_sbil_hfoc = multiplyHists(hfoc_sbil_rescale, dtdt)
        dtst_sbil_hfoc = multiplyHists(hfoc_sbil_rescale, dtst)
        stst_sbil_hfoc = multiplyHists(hfoc_sbil_rescale, stst)
        print("hfoc linearity")

        ramses_sbil_rescale = divideHists(ramses_sbil, lumi_scaling)
        iso_sbil_ramses = multiplyHists(ramses_sbil_rescale, iso)
        dtdt_sbil_ramses = multiplyHists(ramses_sbil_rescale, dtdt)
        dtst_sbil_ramses = multiplyHists(ramses_sbil_rescale, dtst)
        stst_sbil_ramses = multiplyHists(ramses_sbil_rescale, stst)
        print("ramses linearity")

        iso_pcc, dtdt_pcc, dtst_pcc, stst_pcc = make_mutually_exclusive(
            iso_pcc, dtdt_pcc, dtst_pcc, stst_pcc
        )
        iso_hfoc, dtdt_hfoc, dtst_hfoc, stst_hfoc = make_mutually_exclusive(
            iso_hfoc, dtdt_hfoc, dtst_hfoc, stst_hfoc
        )
        iso_sbil_hfoc, dtdt_sbil_hfoc, dtst_sbil_hfoc, stst_sbil_hfoc = (
            make_mutually_exclusive(
                iso_sbil_hfoc, dtdt_sbil_hfoc, dtst_sbil_hfoc, stst_sbil_hfoc
            )
        )
        iso_sbil_ramses, dtdt_sbil_ramses, dtst_sbil_ramses, stst_sbil_ramses = (
            make_mutually_exclusive(
                iso_sbil_ramses, dtdt_sbil_ramses, dtst_sbil_ramses, stst_sbil_ramses
            )
        )

    ### dtdt was negative from before this was all passed into a single function. need to investigate further
    iso_prefire, dtdt_prefire, dtst_prefire, stst_prefire = get_mc_lumis(
        syst,
        time_proj_low,
        lumi_scaling,
        lumi_hists,
        weightsum,
        xsec,
    )
    print("prefiring")

    iso_prefire, dtdt_prefire, dtst_prefire, stst_prefire = make_mutually_exclusive(
        iso_prefire, dtdt_prefire, dtst_prefire, stst_prefire
    )

    pass_gen = mc_scaling(
        pass_gen,
        time_proj_low,
        lumi_scaling,
        weightsum,
        xsec,
    )
    iso, dtdt, dtst, stst = make_mutually_exclusive(iso, dtdt, dtst, stst)
    corrected_mc = [iso, dtdt, dtst, stst]
    corrected_prefiring = [iso_prefire, dtdt_prefire, dtst_prefire, stst_prefire]

    if luminometers:
        pcc_stability = [iso_pcc, dtdt_pcc, dtst_pcc, stst_pcc]
        hfoc_stability = [iso_hfoc, dtdt_hfoc, dtst_hfoc, stst_hfoc]
        ramses_stability = [iso_ramses, dtdt_ramses, dtst_ramses, stst_ramses]
        hfoc_linearity = [iso_sbil_hfoc, dtdt_sbil_hfoc, dtst_sbil_hfoc, stst_sbil_hfoc]
        ramses_linearity = [
            iso_sbil_ramses,
            dtdt_sbil_ramses,
            dtst_sbil_ramses,
            stst_sbil_ramses,
        ]

        return (
            corrected_mc,
            pass_gen,
            corrected_prefiring,
            stat,
            weightsum,
            xsec,
            pcc_stability,
            hfoc_stability,
            ramses_stability,
            hfoc_linearity,
            ramses_linearity,
        )

    else:
        return corrected_mc, pass_gen, corrected_prefiring, stat
