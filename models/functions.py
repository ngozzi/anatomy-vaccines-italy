# libraries
import numpy as np
from datetime import datetime, timedelta
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt

# n. of compartments and age groups
ncomp = 63
nage = 10


def get_IFR(IFR_avg, Nk):
    """
    This function computes the IFR age stratified as reported in Verity et al
    :param IFR_avg: average IFR
    :param Nk: number of individuals in different compartments
    :return: new IFR
    """

    # Verity et al IFR
    IFR = [0.00161 / 100,  # 0-9
           0.00695 / 100,  # 10-19
           0.0309 / 100,   # 20-24
           0.0309 / 100,   # 25-29
           0.0844 / 100,   # 30-39
           0.161 / 100,    # 40-49
           0.595 / 100,    # 50-59
           1.93 / 100,     # 60-69
           4.28 / 100,     # 70-79
           7.80 / 100]     # 80+

    # get average IFR from Verity et al
    IFR_avg_verity = np.sum(np.array(IFR) * Nk) / np.sum(Nk)
    # compute the multiplying parameter and return new IFR
    gamma = IFR_avg / IFR_avg_verity
    return gamma * np.array(IFR)


def update_contacts(basin, date, baseline=False):
    """
    This functions compute the contacts matrix for a specific date
        :param basin: Basin object
        :param date: date
        :param baseline: if True baseline contacts matrix is returned
        :return: contacts matrix for the given date
    """

    global omega_s, omega_c, omega_w, alpha, C2_workplace, C6_stay_home, C1_school

    # weights of different contacts matrices
    weight_home = 4.11
    weight_school = 11.41
    weight_work = 8.07
    weight_comm = 2.79

    # baseline contacts matrix
    if baseline:
        return (weight_home * basin.contacts_home) + (weight_school * basin.contacts_school) + \
               (weight_work * basin.contacts_work) + (weight_comm * basin.contacts_community)

    # get year-week
    if date.isocalendar()[1] < 10:
        year_week = str(date.isocalendar()[0]) + "-0" + str(date.isocalendar()[1])
    else:
        year_week = str(date.isocalendar()[0]) + "-" + str(date.isocalendar()[1])

    # work and community setting
    omega_w = basin.work_reductions.loc[basin.work_reductions.year_week == year_week]["work_red"].values[0]
    omega_c = basin.comm_reductions.loc[basin.comm_reductions.year_week == year_week]["oth_red"].values[0]

    ### SCHOOL ###
    # before color scheme
    if date < datetime(2020, 11, 6):
        C1_school = basin.school_reductions.loc[basin.school_reductions.date == date]["C1_School closing"].values[0]
        C2_workplace = basin.school_reductions.loc[basin.school_reductions.date == date]["C2_Workplace closing"].values[0]
        C6_stay_home = basin.school_reductions.loc[basin.school_reductions.date == date]["C6_Stay at home requirements"].values[0]

        if C1_school == 3 and C2_workplace >= 2 and C6_stay_home >= 2:
            alpha = 1.0
            omega_s = 0.0

        elif C1_school == 3 and not (C2_workplace >= 2 and C6_stay_home >= 2):
            alpha = 0.3444
            omega_s = 1.0

        elif C1_school == 2:
            alpha = 0.3444
            omega_s = 1.0

        elif C1_school == 1:
            alpha = 1.0
            omega_s = 0.5

        elif C1_school == 0:
            alpha = 1.0
            omega_s = 1.0

    # color scheme
    elif date <= datetime(2021, 6, 8):
        color = basin.color.loc[basin.color.date == date]["colore_num_round"].values[0]
                
        if color == 0:
            alpha = 1.0
            omega_s = 0.5

        elif color == 1:
            alpha = 1.0
            omega_s = 0.5

        elif color == 2:
            alpha = 0.3444
            omega_s = 1.0

        elif color == 3:
            alpha = 1.0
            omega_s = 0.0
            
    # summer school closure
    else:
        alpha = 1.0
        omega_s = 0.0
        
    # contacts matrix with reductions
    C_hat = (weight_home * basin.contacts_home) + (omega_s * weight_school * basin.contacts_school) + \
            (omega_w * weight_work * basin.contacts_work) + (omega_c * weight_comm * basin.contacts_community)

    # multiply [0-9] (0), [10-19] (1), [20-24] (2) contacts by alpha
    for i in [0, 1, 2]:
        for j in [0, 1, 2]:
            C_hat[i, j] = C_hat[i, j] * alpha

    return C_hat


def compute_contacts(basin, start_date, end_date):
    """
    This function computes contacts matrices over a given time window
        :param basin: Basin object
        :param start_date: initial date
        :param end_date: last date
        :return: list of dates and dictionary of contacts matrices over time
    """

    # pre-compute contacts matrices over time
    Cs, date, dates = {}, start_date, [start_date]
    for i in range((end_date - start_date).days - 1):
        Cs[date] = update_contacts(basin, date)
        date += timedelta(days=1)
        dates.append(date)

    return Cs, dates


def VEM(VE, VES):
    """
    This function returns VEM given VE and VES (VE = 1 - (1 - VES) * (1 - VEM))
    :param VE: overall vaccine efficacy
    :param VES: vaccine efficacy against infection
    :return: VEM (vaccine efficacy against disease)
    """
    return 1 - (1 - VE) / (1 - VES)


def get_epi_params(increased_mortality=1.0, reduced_efficacy=False):
    """
    This function return the epidemiological parameters
    :param increased_mortality: increased mortality of VOC (float, defual=1.0)
    :param reduced_efficacy: if True vaccines are less effective against VOC (default=False)
    :return dictionary of params
    """

    params = {}

    # epidemiological parameters
    params["mu"] = 1 / 2.5
    params["eps"] = 1 / 4.0

    # Salje et al IFR
    params["IFR"] = np.array([0.001 / 100,  # 0-9
                              0.001 / 100,  # 10-19
                              0.005 / 100,  # 20-24
                              0.005 / 100,  # 25-29
                              0.020 / 100,  # 30-39
                              0.050 / 100,  # 40-49
                              0.200 / 100,  # 50-59
                              0.700 / 100,  # 60-69
                              1.900 / 100,  # 70-79
                              8.300 / 100])  # 80+
    params["IFR_VOC"] = increased_mortality * params["IFR"]

    # vaccine
    params["Delta_V"] = 14.0
    params["Delta_V2_pfizer"] = 21
    params["Delta_V2_moderna"] = 28
    params["Delta_V2_astrazeneca"] = 90

    # efficacy
    params["VE1_pfizer"] = 0.9
    params["VES1_pfizer"] = 0.8
    params["VEI1_pfizer"] = 0.4
    params["VEM1_pfizer"] = VEM(params["VE1_pfizer"], params["VES1_pfizer"])
    params["VE2_pfizer"] = 0.95
    params["VES2_pfizer"] = 0.9
    params["VEI2_pfizer"] = 0.4
    params["VEM2_pfizer"] = VEM(params["VE2_pfizer"], params["VES2_pfizer"])

    params["VE1_moderna"] = 0.9
    params["VES1_moderna"] = 0.8
    params["VEI1_moderna"] = 0.4
    params["VEM1_moderna"] = VEM(params["VE1_moderna"], params["VES1_moderna"])
    params["VE2_moderna"] = 0.95
    params["VEI2_moderna"] = 0.4
    params["VES2_moderna"] = 0.9
    params["VEM2_moderna"] = VEM(params["VE2_moderna"], params["VES2_moderna"])

    params["VE1_astrazeneca"] = 0.7
    params["VES1_astrazeneca"] = 0.6
    params["VEI1_astrazeneca"] = 0.4
    params["VEM1_astrazeneca"] = VEM(params["VE1_astrazeneca"], params["VES1_astrazeneca"])
    params["VE2_astrazeneca"] = 0.8
    params["VES2_astrazeneca"] = 0.7
    params["VEI2_astrazeneca"] = 0.4
    params["VEM2_astrazeneca"] = VEM(params["VE2_astrazeneca"], params["VES2_astrazeneca"])

    params["VE1_JJ"] = 0.7
    params["VES1_JJ"] = 0.6
    params["VEI1_JJ"] = 0.4
    params["VEM1_JJ"] = VEM(params["VE1_JJ"], params["VES1_JJ"])

    # efficacy VOC
    if reduced_efficacy == True:
        # efficacy
        params["VE1_pfizer_VOC"] = 0.5
        params["VES1_pfizer_VOC"] = 0.4
        params["VEI1_pfizer_VOC"] = 0.4
        params["VEM1_pfizer_VOC"] = VEM(params["VE1_pfizer_VOC"], params["VES1_pfizer_VOC"])
        params["VE2_pfizer_VOC"] = 0.9
        params["VES2_pfizer_VOC"] = 0.8
        params["VEI2_pfizer_VOC"] = 0.4
        params["VEM2_pfizer_VOC"] = VEM(params["VE2_pfizer_VOC"], params["VES2_pfizer_VOC"])

        params["VE1_moderna_VOC"] = 0.5
        params["VES1_moderna_VOC"] = 0.4
        params["VEI1_moderna_VOC"] = 0.4
        params["VEM1_moderna_VOC"] = VEM(params["VE1_moderna_VOC"], params["VES1_moderna_VOC"])
        params["VE2_moderna_VOC"] = 0.9
        params["VES2_moderna_VOC"] = 0.8
        params["VEI2_moderna_VOC"] = 0.4
        params["VEM2_moderna_VOC"] = VEM(params["VE2_moderna_VOC"], params["VES2_moderna_VOC"])

        params["VE1_astrazeneca_VOC"] = 0.5
        params["VES1_astrazeneca_VOC"] = 0.4
        params["VEI1_astrazeneca_VOC"] = 0.4
        params["VEM1_astrazeneca_VOC"] = VEM(params["VE1_astrazeneca_VOC"], params["VES1_astrazeneca_VOC"])
        params["VE2_astrazeneca_VOC"] = 0.75
        params["VES2_astrazeneca_VOC"] = 0.65
        params["VEI2_astrazeneca_VOC"] = 0.4
        params["VEM2_astrazeneca_VOC"] = VEM(params["VE2_astrazeneca_VOC"], params["VES2_astrazeneca_VOC"])

        params["VE1_JJ_VOC"] = 0.5
        params["VES1_JJ_VOC"] = 0.4
        params["VEI1_JJ_VOC"] = 0.4
        params["VEM1_JJ_VOC"] = VEM(params["VE1_JJ_VOC"], params["VES1_JJ_VOC"])

    else:
        for v in ["pfizer", "moderna", "astrazeneca", "JJ"]:
            params["VE1_" + v + "_VOC"] = params["VE1_" + v ]
            params["VES1_" + v + "_VOC"] = params["VES1_" + v]
            params["VEI1_" + v + "_VOC"] = params["VEI1_" + v]
            params["VEM1_" + v + "_VOC"] = params["VEM1_" + v]
            if v != "JJ":
                params["VE2_" + v + "_VOC"] = params["VE2_" + v]
                params["VES2_" + v + "_VOC"] = params["VES2_" + v]
                params["VEI2_" + v + "_VOC"] = params["VEI2_" + v]
                params["VEM2_" + v + "_VOC"] = params["VEM2_" + v]

    return params


def get_beta(R0, mu, C, Nk, date, seasonality_min, hemisphere):
    """
    This functions return beta for a SEIR model with age structure
        :param R0: basic reproductive number
        :param mu: recovery rate
        :param C: contacts matrix
        :param Nk: n. of individuals in different age groups
        :param date: current day
        :param seasonality_min: seasonality parameter
        :param hemisphere: hemisphere of basin (0: north, 1: tropical, 2; south)
        :return: returns the rate of infection beta for the given R0
    """
    # get seasonality adjustment
    seas_adj = apply_seasonality(date, seasonality_min, hemisphere)
    C_hat = np.zeros((C.shape[0], C.shape[1]))
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            C_hat[i, j] = (Nk[i] / Nk[j]) * C[i, j]
    return R0 * mu / (seas_adj * np.max([e.real for e in np.linalg.eig(C_hat)[0]]))


def apply_seasonality(day, seasonality_min, basin_hemispheres, seasonality_max=1):
    """
    This function computes the seasonality adjustment for transmissibility
        :param day: current day
        :param seasonality_min: seasonality parameter
        :param basin_hemispheres: hemisphere of basin (0: north, 1: tropical, 2; south)
        :param seasonality_max: seasonality parameter
        :return: returns seasonality adjustment
    """

    s_r = seasonality_min / seasonality_max
    day_max_north = datetime(day.year, 1, 15)
    day_max_south = datetime(day.year, 7, 15)

    seasonal_adjustment = np.empty(shape=(3,), dtype=np.float64)

    # north hemisphere
    seasonal_adjustment[0] = 0.5 * ((1 - s_r) * np.sin(2 * np.pi / 365 * (day - day_max_north).days + 0.5 * np.pi) + 1 + s_r)

    # tropical hemisphere
    seasonal_adjustment[1] = 1.0

    # south hemisphere
    seasonal_adjustment[2] = 0.5 * ((1 - s_r) * np.sin(2 * np.pi / 365 * (day - day_max_south).days + 0.5 * np.pi) + 1 + s_r)

    return seasonal_adjustment[basin_hemispheres]


def get_initial_conditions(basin, run_id):
    """
    This function returns the initial conditions (2020/09/01)
        :param basin: Basin object
        :param run_id: id of the run
        :return: initial conditions (2020/09/01)
    """

    # initial conditions
    initial_conditions = np.zeros((ncomp, nage))
    for age_id, age_str in zip(range(nage), ["0_9", "10_19", "20_24", "25_29", "30_39", "40_49", "50_59", "60_69", "70_79", "80_plus"]):
        # S: 0, L_WT: 1, L_VOC: 2, I_WT: 3, I_VOC: 4, R_WT: 5, R_VOC: 6,
        # V1r: 7, V1i: 8, L_WT_V1i: 9, L_VOC_V1i: 10, I_WT_V1i: 11, I_VOC_V1i: 12, R_WT_V1i: 13, R_VOC_V1i: 14,
        # V2r: 15, V2i: 16, L_WT_V2i: 17, L_VOC_V2i: 18, I_WT_V2i: 19, I_VOC_V2i: 20, R_WT_V2i: 21, R_VOC_V2i: 22
        initial_conditions[1, age_id] = int(np.random.poisson(basin.init_conditions.iloc[run_id]["Exposed_" + age_str]))
        initial_conditions[3, age_id] = int(np.random.poisson(basin.init_conditions.iloc[run_id]["Infectious_" + age_str]))
        initial_conditions[5, age_id] = int(np.random.poisson(basin.init_conditions.iloc[run_id]["Recovered_" + age_str]))
        initial_conditions[0, age_id] = int(basin.Nk[age_id] - initial_conditions[1, age_id] - initial_conditions[3, age_id] - initial_conditions[5, age_id])

    return initial_conditions


