"""
Core of the DeltaSNOW model. This module provides the public function 
:func:`deltasnow_snowpack_evolution` which implements the algorithm of the
deltaSNOW model with numba but does not validate input data. Users are
discouraged to use this function and should rather use the 
:func:`pydeltasnow.main.swe_deltasnow` function instead.

Significant parts of the code in this module are based on the work of Manuel 
Theurl: https://github.com/manueltheurl/snow_to_swe

"""
import numpy as np
from numba import njit

from pydeltasnow import __version__

__author__ = "Johannes Aschauer"
__copyright__ = "Johannes Aschauer"
__license__ = "GPL-2.0-or-later"

G = 9.81  # gravitational acceleration on earth
PRECISION = 10e-10  # floating point precision


@njit
def _assign_H(
        h_dd,
        swe_dd,
        age_dd,
        h,
        swe,
        age,
        H,
        SWE,
        t,
        day_tot,
):
    """
    Assign snowpack properties of the next timestep.
    """
    if t < day_tot:
        h[:, t + 1] = h_dd
        swe[:, t + 1] = swe_dd
        age[:, t + 1] = age_dd
        H[t + 1] = np.sum(h[:, t + 1])
        SWE[t + 1] = np.sum(swe[:, t + 1])

    return h, swe, age, H, SWE


@njit
def _compact_H(
        h,
        swe,
        swe_hat,
        age,
        ts,
        eta_null,
        k,
        rho_max,
):
    """
    Compaction of individual snow layers without additional load 
    Values of today are compacted to values of tomorrow.
    """
    age_d = 0 if h == 0 else age
    h_dd = h / (1 + (swe_hat * G * ts) / eta_null * np.exp(-k * swe / h))
    h_dd = swe / rho_max if swe / h_dd > rho_max else h_dd
    h_dd = 0 if h == 0 else h_dd
    swe_dd = swe
    age_dd = 0 if h == 0 else age_d + 1
    rho_dd = 0 if h == 0 else swe_dd / h_dd
    rho_dd = rho_max if rho_max - rho_dd < PRECISION else rho_dd
    return h_dd, swe_dd, age_dd, rho_dd


@njit
def _drench_H(
        t,
        ly,
        ly_tot,
        day_tot,
        Hobs,
        h,
        swe,
        age,
        H,
        SWE,
        ts,
        eta_null,
        k,
        rho_max
):
    """
    Drenching of the snowpack if the observed snowdepth decreased significantly.
    """
    Hobs_d = Hobs[t]
    h_d = h[:, t]
    swe_d = swe[:, t]
    age_d = age[:, t]

    # distribute mass top-down
    # reversed not working in numba
    for i in range(ly - 1, -1, -1):
        # for i in reversed(range(ly)):
        if np.sum(np.array([element for j, element in enumerate(h_d) if j != i])) + swe_d[
            i] / rho_max - Hobs_d >= PRECISION:
            # layers is densified to rho_max
            h_d[i] = swe_d[i] / rho_max
        else:
            # layer is densified as far as possible
            # but doesnt reach rho_max
            h_d[i] = swe_d[i] / rho_max + np.abs(
                np.sum(np.array([element for j, element in enumerate(h_d) if j != i])) + swe_d[i] / rho_max - Hobs_d)
            break

    true_false_list = np.ones(ly, dtype="bool")
    for i, (swe_d_val, h_d_val) in enumerate(zip(swe_d[:ly], h_d[:ly])):
        true_false_list[i] = rho_max - swe_d_val / h_d_val <= PRECISION

    if np.all(true_false_list):
        # no further compaction, runoff
        scale = Hobs_d / np.sum(h_d)
        h_d = h_d * scale  # all layers are compressed (and have rho_max) [m]
        swe_d = swe_d * scale

    h[:, t] = h_d
    swe[:, t] = swe_d
    age[:, t] = age_d
    H[t] = np.sum(h[:, t])
    SWE[t] = np.sum(swe[:, t])

    # no further compaction possible
    h_dd, swe_dd, age_dd, rho_dd = _dry_metamorphism(
        h[:, t],
        swe[:, t],
        age[:, t],
        ly_tot,
        ly,
        ts,
        eta_null,
        k,
        rho_max)

    # set values for next day
    h, swe, age, H, SWE = _assign_H(h_dd, swe_dd, age_dd, h, swe, age, H, SWE, t, day_tot)

    return h, swe, age, H, SWE, rho_dd


@njit
def _dry_metamorphism(
        h_d,
        swe_d,
        age_d,
        ly_tot,
        ly,
        ts,
        eta_null,
        k,
        rho_max
):
    """
    Compaction of dry snowpack. 
    """
    # overburden of current day (swe_hat_d)
    swe_hat_d = np.zeros(ly_tot)
    for i in range(ly_tot):
        swe_hat_d[i] = np.sum(swe_d[i:ly_tot])

    h_dd = h_d.copy()
    swe_dd = swe_d.copy()
    age_dd = age_d.copy()
    rho_dd = np.zeros(ly_tot)

    for i, (h, swe, swe_hat, age) in enumerate(zip(h_d[:ly], swe_d[:ly], swe_hat_d[:ly], age_d[:ly])):
        h_dd[i], swe_dd[i], age_dd[i], rho_dd[i] = _compact_H(
            h,
            swe,
            swe_hat,
            age,
            ts,
            eta_null,
            k,
            rho_max
        )

    return h_dd, swe_dd, age_dd, rho_dd


@njit
def _scale_H(
        t,
        ly,
        ly_tot,
        day_tot,
        deltaH,
        Hobs,
        h,
        swe,
        age,
        H,
        SWE,
        ts,
        eta_null,
        k,
        rho_max,
):
    """
    Scale snowpack if the settling was assumed a bit too strong or too weak.
    """
    Hobs_d = Hobs[t - 1]
    Hobs_dd = Hobs[t]
    h_d = h[:, t - 1]
    swe_d = swe[:, t - 1]
    age_d = age[:, t]

    # todays overburden
    swe_hat_d = np.zeros(ly_tot)
    for i in range(ly_tot):
        swe_hat_d[i] = np.sum(swe_d[i:ly_tot])

    # analytical solution for layerwise adapted viskosity eta
    # assumption: recompaction ~ linear height change of yesterdays layers (see paper)
    eta_cor = np.zeros(ly_tot)
    for i in range(ly_tot):
        if swe_d[i] == 0. or h_d[i] == 0:
            eta_cor[i] = 0.
        else:
            rho_d = swe_d[i] / h_d[i]
            x = ts * G * swe_hat_d[i] * np.exp(-k * rho_d)  # yesterday
            P = h_d[i] / Hobs_d  # yesterday
            eta_cor[i] = Hobs_dd * x * P / (h_d[i] - Hobs_dd * P) if (h_d[i] - Hobs_dd * P) != 0 else np.inf

    h_dd_cor = np.zeros(ly_tot)
    for i in range(ly_tot):
        if h_d[i] == 0 or eta_cor[i] == 0:
            h_dd_cor[i] = 0.
        else:
            h_dd_cor[i] = h_d[i] / (1 + (swe_hat_d[i] * G * ts) / eta_cor[i] * np.exp(-k * swe_d[i] / h_d[i]))
    h_dd_cor[np.isnan(h_dd_cor)] = 0
    H_dd_cor = np.sum(h_dd_cor)

    # and check, if Hd.cor is the same as Hobs.d
    if np.abs(H_dd_cor - Hobs_dd) > PRECISION:
        print("WARNING: error in exponential re-compaction: H.dd.cor-Hobs.dd")

    # which layers exceed rho.max?
    idx_max_arr = np.zeros(ly_tot, dtype='bool')
    for i, (swe_e_val, h_dd_cor_val) in enumerate(zip(swe_d, h_dd_cor)):
        try:
            idx_max_arr[i] = np.divide(swe_e_val, h_dd_cor_val) - rho_max > PRECISION
        except:
            idx_max_arr[i] = False
    idx_max_arr[np.isnan(idx_max_arr)] = False

    if np.any(idx_max_arr):
        if np.count_nonzero(idx_max_arr) < ly:
            # collect excess swe in those layers
            swe_excess = swe_d[idx_max_arr] - h_dd_cor[idx_max_arr] * rho_max

            # set affected layer(s) to rho.max
            swe_d[idx_max_arr] = swe_d[idx_max_arr] - swe_excess

            # distribute excess swe to other layers top-down
            lys = range(ly)
            lys_non_max = []
            for e in lys:
                if e not in np.nonzero(idx_max_arr)[0]:
                    lys_non_max.append(e)

            i = lys_non_max[-1]
            swe_excess_all = np.sum(swe_excess)

            while swe_excess_all > 0:
                # layer tolerates this swe amount to reach rho.max
                swe_res = h_dd_cor[i] * rho_max - swe_d[i]
                if swe_res > swe_excess_all:
                    swe_res = swe_excess_all

                swe_d[i] = swe_d[i] + swe_res
                swe_excess_all = swe_excess_all - swe_res
                i = i - 1
                if i < 0 < swe_excess_all:
                    # runoff
                    break
        else:
            # if all layers have density > rho.max
            # remove swe.excess from all layers (-> runoff)
            # (this sets density to rho.max)
            swe_excess = swe_d[idx_max_arr] - h_dd_cor[idx_max_arr] * rho_max
            swe_d[idx_max_arr] = swe_d[idx_max_arr] - swe_excess

    h[:, t] = h_dd_cor
    swe[:, t] = swe_d
    age[:, t] = age_d
    H[t] = np.sum(h[:, t])
    SWE[t] = np.sum(swe[:, t])

    # compact actual day
    h_dd, swe_dd, age_dd, rho_dd = _dry_metamorphism(
        h[:, t],
        swe[:, t],
        age[:, t],
        ly_tot,
        ly,
        ts,
        eta_null,
        k,
        rho_max,
    )

    # set values for next day
    h, swe, age, H, SWE = _assign_H(
        h_dd,
        swe_dd,
        age_dd,
        h,
        swe,
        age,
        H,
        SWE,
        t,
        day_tot,
    )
    return h, swe, age, H, SWE, rho_dd


@njit
def deltasnow_snowpack_evolution(
        Hobs,
        rho_max,
        rho_null,
        c_ov,
        k_ov,
        k,
        tau,
        eta_null,
        resolution,
):
    """
    This is the main loop in the delta snow model. Should be called on HS chunks
    of consecutive nonzeros.

    Parameters
    ----------
    Hobs : 1D :class:`numpy.ndarray` of floats 
        Measured snow height. Needs to be in [m].
        Must comply to the following constraints:
            - no nans
            - continuous entries (no missing dates)
    rho_max : float, optional
        Maximum density of an individual snow layer produced by the deltasnow 
        model in [kg/m3], rho_max needs to be positive. The default is 401.2588.
    rho_null : float, optional
        Fresh snow density for a newly created layer [kg/m3], rho_null needs to
        be positive. The default is 81.19417.
    c_ov : float, optional
        Overburden factor due to fresh snow [-], c_ov needs to be positive. The
        default is 0.0005104722.
    k_ov : float, optional
        Defines the impact of the individual layer density on the compaction due
        to overburden [-], k_ov need to be in the range [0,1].
        The default is 0.37856737.
    k : float, optional
        Exponent of the exponential-law compaction [m3/kg], k needs to be
        positive. The default is 0.02993175.
    tau : float, optional
        Uncertainty bound [m], tau needs to be positive.
        The default is 0.02362476.
    eta_null : float, optional
        Effective compactive viscosity of snow for "zero-density" [Pa s].
        The default is 8523356.
    resolution : float
        Timedelta in hours between snow observations.

    Raises
    ------
    RuntimeError
        If the snowpack evolution has somehow gone wrong.

    Returns
    -------
    SWE : 1D :class:`numpy.ndarray` of floats
        Calculated SWE in [mm]. Same shape as Hobs.

    """

    ly_tot = np.count_nonzero(Hobs)  # maximum number of layers [-]
    day_tot = len(Hobs)  # total days from first to last snowfall [-]

    # preallocate output arrays 
    H = np.zeros(day_tot)  # modeled total height of snow at any day [m]
    SWE = np.zeros(day_tot)  # modeled total SWE at any day [kg/m2]

    # preallocate matrix as days X layers
    h = np.zeros((ly_tot, day_tot))  # modeled height of snow in all layers [m]
    swe = np.zeros((ly_tot, day_tot))  # modeled swe in all layers [kg/m2]
    age = np.zeros((ly_tot, day_tot))  # age in all layers

    ly = 1  # layer number [-]
    ts = resolution * 3600

    for t in range(day_tot):
        # snowdepth = 0, no snow cover
        if Hobs[t] == 0:
            H[t] = 0
            SWE[t] = 0
            h[:, t] = 0
            swe[:, t] = 0

        # there is snow
        elif Hobs[t] > 0:  # redundant if, cause can snow height be negative?
            # first snow in/during season
            if Hobs[t - 1] == 0:
                ly = 1
                age[ly - 1, t] = 1
                h[ly - 1, t] = Hobs[t]
                H[t] = Hobs[t]
                swe[ly - 1, t] = rho_null * Hobs[t]
                SWE[t] = swe[ly - 1, t]

                # compact actual day
                h_dd, swe_dd, age_dd, rho_dd = _dry_metamorphism(
                    h[:, t],
                    swe[:, t],
                    age[:, t],
                    ly_tot,
                    ly,
                    ts,
                    eta_null,
                    k,
                    rho_max,
                )

                h, swe, age, H, SWE = _assign_H(
                    h_dd,
                    swe_dd,
                    age_dd,
                    h,
                    swe,
                    age,
                    H,
                    SWE,
                    t,
                    day_tot,
                )

            elif Hobs[t - 1] > 0:
                deltaH = Hobs[t] - H[t]
                if deltaH > tau:
                    sigma_null = deltaH * rho_null * G
                    epsilon = c_ov * sigma_null * np.exp(-k_ov * rho_dd / (rho_max - rho_dd))
                    h[:, t] = (1 - epsilon) * h[:, t]
                    swe[:, t] = swe[:, t - 1]
                    age[:ly, t] = age[:ly, t - 1] + 1
                    H[t] = np.sum(h[:, t])
                    SWE[t] = np.sum(swe[:, t])

                    # only for new layer
                    ly = ly + 1
                    h[ly - 1, t] = Hobs[t] - H[t]
                    swe[ly - 1, t] = rho_null * h[ly - 1, t]
                    age[ly - 1, t] = 1

                    # recompute
                    H[t] = np.sum(h[:, t])
                    SWE[t] = np.sum(swe[:, t])

                    # compact actual day
                    h_dd, swe_dd, age_dd, rho_dd = _dry_metamorphism(
                        h[:, t],
                        swe[:, t],
                        age[:, t],
                        ly_tot,
                        ly,
                        ts,
                        eta_null,
                        k,
                        rho_max,
                    )

                    # set values for next day
                    h, swe, age, H, SWE = _assign_H(
                        h_dd,
                        swe_dd,
                        age_dd,
                        h,
                        swe,
                        age,
                        H,
                        SWE,
                        t,
                        day_tot,
                    )

                # no mass gain or loss, but scaling
                elif -tau <= deltaH <= tau:
                    h, swe, age, H, SWE, rho_dd = _scale_H(
                        t,
                        ly,
                        ly_tot,
                        day_tot,
                        deltaH,
                        Hobs,
                        h,
                        swe,
                        age,
                        H,
                        SWE,
                        ts,
                        eta_null,
                        k,
                        rho_max,
                    )

                elif deltaH < -tau:
                    h, swe, age, H, SWE, rho_dd = _drench_H(
                        t,
                        ly,
                        ly_tot,
                        day_tot,
                        Hobs,
                        h,
                        swe,
                        age,
                        H,
                        SWE,
                        ts,
                        eta_null,
                        k,
                        rho_max,
                    )

                else:
                    raise RuntimeError("no valid calculated HS deviation.")
    return SWE
