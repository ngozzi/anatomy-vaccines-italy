# libraries
import numpy as np
from numba import jit
from functions import get_epi_params, apply_seasonality, get_beta

# n. of compartments and age groups
ncomp = 63
nage = 10

#@jit(fastmath=True, cache=True)
def compute_deaths(R_WT, R_VOC,
                   R_WT_V1i_pfizer, R_VOC_V1i_pfizer,
                   R_WT_V2i_pfizer, R_VOC_V2i_pfizer,
                   R_WT_V1i_moderna, R_VOC_V1i_moderna,
                   R_WT_V2i_moderna, R_VOC_V2i_moderna,
                   R_WT_V1i_astrazeneca, R_VOC_V1i_astrazeneca,
                   R_WT_V2i_astrazeneca, R_VOC_V2i_astrazeneca,
                   R_WT_V1i_JJ, R_VOC_V1i_JJ,
                   params):
    """
    This function computes the number of daily deaths
        :param R_WT: recovered for the wild type
        :param R_VOC: recovered for the variant of concern
        :param R_WT_V1i_pfizer: recovered vaccinated 1st dose for the wild type Pfizer
        :param R_VOC_V1i_pfizer: recovered vaccinated 1st dose for the variant of concern Pfizer
        :param R_WT_V2i_pfizer: recovered vaccinated 2nd dose for the wild type Pfizer
        :param R_VOC_V2i_pfizer: recovered vaccinated 2nd dose for the variant of concern Pfizer
        :param R_WT_V1i_moderna: recovered vaccinated 1st dose for the wild type Moderna
        :param R_VOC_V1i_moderna: recovered vaccinated 1st dose for the variant of concern Moderna
        :param R_WT_V2i_moderna: recovered vaccinated 2nd dose for the wild type Moderna
        :param R_VOC_V2i_moderna: recovered vaccinated 2nd dose for the variant of concern Moderna
        :param R_WT_V1i_astrazeneca: recovered vaccinated 1st dose for the wild type Astrazeneca
        :param R_VOC_V1i_astrazeneca: recovered vaccinated 1st dose for the variant of concern Astrazeneca
        :param R_WT_V2i_astrazeneca: recovered vaccinated 2nd dose for the wild type Astrazeneca
        :param R_VOC_V2i_astrazeneca: recovered vaccinated 2nd dose for the variant of concern Astrazeneca
        :param R_WT_V1i_JJ: recovered vaccinated 1st dose for the wild type J&J
        :param R_VOC_V1i_JJ: recovered vaccinated 1st dose for the variant of concern J&J
        :return: total daily deaths and for wild type and variant of concern and different vaccination statuses
    """

    # initialize deaths
    nage, T = R_WT.shape[0], R_WT.shape[1]
    deaths_WT, deaths_VOC = np.zeros((nage, T)), np.zeros((nage, T))
    deaths_vacc, deaths_vacc1dose, deaths_vacc2dose, deaths_novacc = np.zeros((nage, T)), np.zeros((nage, T)), \
                                                                     np.zeros((nage, T)), np.zeros((nage, T))
    ifr = np.zeros((nage, T))
    tot_rec, tot_deaths = np.zeros(T), np.zeros(T)

    for age in range(nage):
        # compute deaths
        novacc_WT = np.random.binomial(R_WT[age].astype(int), params["IFR"][age])
        novacc_VOC = np.random.binomial(R_VOC[age].astype(int), params["IFR_VOC"][age])

        pfizer_1dose_WT = np.random.binomial(R_WT_V1i_pfizer[age].astype(int), (1 - params["VEM1_pfizer"]) * params["IFR"][age])
        pfizer_2dose_WT = np.random.binomial(R_WT_V2i_pfizer[age].astype(int), (1 - params["VEM2_pfizer"]) * params["IFR"][age])
        pfizer_1dose_VOC = np.random.binomial(R_VOC_V1i_pfizer[age].astype(int), (1 - params["VEM1_pfizer_VOC"]) * params["IFR_VOC"][age])
        pfizer_2dose_VOC = np.random.binomial(R_VOC_V2i_pfizer[age].astype(int), (1 - params["VEM2_pfizer_VOC"]) * params["IFR_VOC"][age])

        moderna_1dose_WT = np.random.binomial(R_WT_V1i_moderna[age].astype(int), (1 - params["VEM1_moderna"]) * params["IFR"][age])
        moderna_2dose_WT = np.random.binomial(R_WT_V2i_moderna[age].astype(int), (1 - params["VEM2_moderna"]) * params["IFR"][age])
        moderna_1dose_VOC = np.random.binomial(R_VOC_V1i_moderna[age].astype(int), (1 - params["VEM1_moderna_VOC"]) * params["IFR_VOC"][age])
        moderna_2dose_VOC = np.random.binomial(R_VOC_V2i_moderna[age].astype(int), (1 - params["VEM2_moderna_VOC"]) * params["IFR_VOC"][age])

        astrazeneca_1dose_WT = np.random.binomial(R_WT_V1i_astrazeneca[age].astype(int), (1 - params["VEM1_astrazeneca"]) * params["IFR"][age])
        astrazeneca_2dose_WT = np.random.binomial(R_WT_V2i_astrazeneca[age].astype(int), (1 - params["VEM2_astrazeneca"]) * params["IFR"][age])
        astrazeneca_1dose_VOC = np.random.binomial(R_VOC_V1i_astrazeneca[age].astype(int), (1 - params["VEM1_astrazeneca_VOC"]) * params["IFR_VOC"][age])
        astrazeneca_2dose_VOC = np.random.binomial(R_VOC_V2i_astrazeneca[age].astype(int), (1 - params["VEM2_astrazeneca_VOC"]) * params["IFR_VOC"][age])

        JJ_1dose_WT = np.random.binomial(R_WT_V1i_JJ[age].astype(int), (1 - params["VEM1_JJ"]) * params["IFR"][age])
        JJ_1dose_VOC = np.random.binomial(R_VOC_V1i_JJ[age].astype(int), (1 - params["VEM1_JJ_VOC"]) * params["IFR_VOC"][age])


        deaths_WT[age] = novacc_WT + \
                         pfizer_1dose_WT + pfizer_2dose_WT + \
                         moderna_1dose_WT + moderna_2dose_WT + \
                         astrazeneca_1dose_WT + astrazeneca_2dose_WT + \
                         JJ_1dose_WT

        deaths_VOC[age] = novacc_VOC + \
                          pfizer_1dose_VOC + pfizer_2dose_VOC + \
                          moderna_1dose_VOC + moderna_2dose_VOC + \
                          astrazeneca_1dose_VOC + astrazeneca_2dose_VOC + \
                          JJ_1dose_VOC

        deaths_vacc[age] = pfizer_1dose_WT + pfizer_2dose_WT + \
                           moderna_1dose_WT + moderna_2dose_WT + \
                           astrazeneca_1dose_WT + astrazeneca_2dose_WT + \
                           JJ_1dose_WT + \
                           pfizer_1dose_VOC + pfizer_2dose_VOC + \
                           moderna_1dose_VOC + moderna_2dose_VOC + \
                           astrazeneca_1dose_VOC + astrazeneca_2dose_VOC + \
                           JJ_1dose_VOC

        deaths_novacc[age] = novacc_WT + novacc_VOC

        deaths_vacc1dose[age] = pfizer_1dose_WT + \
                                moderna_1dose_WT + \
                                astrazeneca_1dose_WT + \
                                JJ_1dose_WT + \
                                pfizer_1dose_VOC + \
                                moderna_1dose_VOC + \
                                astrazeneca_1dose_VOC + \
                                JJ_1dose_VOC

        deaths_vacc2dose[age] = pfizer_2dose_WT + \
                                moderna_2dose_WT + \
                                astrazeneca_2dose_WT + \
                                pfizer_2dose_VOC + \
                                moderna_2dose_VOC + \
                                astrazeneca_2dose_VOC

        tot_rec += (R_WT[age] + R_VOC[age] +
                    R_WT_V1i_pfizer[age] + R_WT_V2i_pfizer[age] + R_VOC_V1i_pfizer[age] + R_VOC_V2i_pfizer[age] +
                    R_WT_V1i_moderna[age] + R_WT_V2i_moderna[age] + R_VOC_V1i_moderna[age] + R_VOC_V2i_moderna[age] +
                    R_WT_V1i_astrazeneca[age] + R_WT_V2i_astrazeneca[age] + R_VOC_V1i_astrazeneca[age] + R_VOC_V2i_astrazeneca[age] +
                    R_WT_V1i_JJ[age] + R_VOC_V1i_JJ[age])

        tot_deaths += (deaths_WT[age] + deaths_VOC[age])

        # update ifr
        ifr[age] = (deaths_WT[age] + deaths_VOC[age]) / (R_WT[age] + R_VOC[age] +
                                                         R_WT_V1i_pfizer[age] + R_WT_V2i_pfizer[age] +
                                                         R_VOC_V1i_pfizer[age] + R_VOC_V2i_pfizer[age] +
                                                         R_WT_V1i_moderna[age] + R_WT_V2i_moderna[age] +
                                                         R_VOC_V1i_moderna[age] + R_VOC_V2i_moderna[age] +
                                                         R_WT_V1i_astrazeneca[age] + R_WT_V2i_astrazeneca[age] +
                                                         R_VOC_V1i_astrazeneca[age] + R_VOC_V2i_astrazeneca[age] +
                                                         R_WT_V1i_JJ[age] + R_VOC_V1i_JJ[age])

        # shift
        deaths_WT[age] = np.roll(deaths_WT[age], shift=int(params["Delta"]))
        deaths_WT[age][0:int(params["Delta"])] = 0

        deaths_VOC[age] = np.roll(deaths_VOC[age], shift=int(params["Delta"]))
        deaths_VOC[age][0:int(params["Delta"])] = 0

        deaths_vacc[age] = np.roll(deaths_vacc[age], shift=int(params["Delta"]))
        deaths_vacc[age][0:int(params["Delta"])] = 0

        deaths_novacc[age] = np.roll(deaths_novacc[age], shift=int(params["Delta"]))
        deaths_novacc[age][0:int(params["Delta"])] = 0

        deaths_vacc1dose[age] = np.roll(deaths_vacc1dose[age], shift=int(params["Delta"]))
        deaths_vacc1dose[age][0:int(params["Delta"])] = 0

        deaths_vacc2dose[age] = np.roll(deaths_vacc2dose[age], shift=int(params["Delta"]))
        deaths_vacc2dose[age][0:int(params["Delta"])] = 0

        ifr[age] = np.roll(ifr[age], shift=int(params["Delta"]))
        ifr[age][0:int(params["Delta"])] = 0

    ifr_tot = tot_deaths / tot_rec
    ifr_tot = np.roll(ifr_tot, shift=int(params["Delta"]))
    ifr_tot[0:int(params["Delta"])] = 0

    return {"deaths_TOT": deaths_WT + deaths_VOC,
            "deaths_WT": deaths_WT,
            "deaths_VOC": deaths_VOC,
            "deaths_vacc": deaths_vacc,
            "deaths_novacc": deaths_novacc,
            "deaths_vacc1dose": deaths_vacc1dose,
            "deaths_vacc2dose": deaths_vacc2dose,
            "ifr": ifr,
            "ifr_tot": ifr_tot}


def simulate(Cs, psi, Nk, initial_conditions, importations, vaccinations, R0, Delta, dates, seasonality_min,
             hemisphere, vaccine, increased_mortality=1.0, reduced_efficacy=True):
    """
    This function runs the 2-strain SEIRD model
        :param Cs: dictionary of contacts matrices over time
        :param psi: increased transmissibility of 2nd variant
        :param Nk: number of people in different age groups
        :param initial_conditions: initial conditions for different compartment/age groups
        :param importations: importations for different age groups
        :param vaccinations: dictionary of vaccinations
        :param R0: basic reproductive number
        :param Delta: delay (in days) in deaths
        :param dates: array of dates
        :param seasonality_min: seasonality parameter
        :param hemisphere: hemisphere of basin
        :param vaccine: vaccination strategy
        :param increased_mortality: increased mortality of VOC (float, defual=1.0)
        :param reduced_efficacy: if True vaccines are less effective against VOC (default=False)
        :return: returns simulation results
    """

    # get epi params
    params = get_epi_params(increased_mortality, reduced_efficacy)

    # get beta from Rt w.r.t to initial_date
    beta = get_beta(R0, params["mu"], Cs[dates[0]], Nk, dates[0], seasonality_min, hemisphere)

    # add parameters
    params["beta"] = beta
    params["Delta"] = Delta
    params["seasonality_min"] = seasonality_min
    params["hemisphere"] = hemisphere
    params["psi"] = psi

    # simulate
    results = stochastic_seird(Cs, initial_conditions, importations, vaccinations, Nk, dates, vaccine, params)

    # compute deaths
    results_deaths = compute_deaths(results["recovered_WT"],
                                    results["recovered_VOC"],
                                    results["recovered_WT_V1i_pfizer"],
                                    results['recovered_VOC_V1i_pfizer'],
                                    results['recovered_WT_V2i_pfizer'],
                                    results['recovered_VOC_V2i_pfizer'],
                                    results['recovered_WT_V1i_moderna'],
                                    results['recovered_VOC_V1i_moderna'],
                                    results['recovered_WT_V2i_moderna'],
                                    results['recovered_VOC_V2i_moderna'],
                                    results['recovered_WT_V1i_astrazeneca'],
                                    results['recovered_VOC_V1i_astrazeneca'],
                                    results['recovered_WT_V2i_astrazeneca'],
                                    results['recovered_VOC_V2i_astrazeneca'],
                                    results['recovered_WT_V1i_JJ'],
                                    results['recovered_VOC_V1i_JJ'],
                                    params)
    results.update(results_deaths)
    return results


#@jit(fastmath=True, cache=True)
def get_vaccinated(vaccine, vaccinations, compartments_step, date):
    """
    This function compute the number of daily vaccinated in different age groups for different allocation strategies
    :param vaccine: vaccine allocation strategy
    :param vaccinations: dictionary of vaccinations
    :param compartments_step: individuals in different compartments at this step
    :param date: current date
    :return: updated compartments and doses given by age group
    """

    # doses given by age group
    vaccines_given = np.zeros(nage)
    vaccines_baseline_given = np.zeros(nage)
    vaccines_nonbaseline_given = np.zeros(nage)

    S_indices = {"pfizer": 7, "moderna": 23, "astrazeneca": 39, "JJ": 55}
    L_WT_indices = {"pfizer": 9, "moderna": 25, "astrazeneca": 41, "JJ": 57}
    L_VOC_indices = {"pfizer": 10, "moderna": 26, "astrazeneca": 42, "JJ": 58}
    R_WT_indices = {"pfizer": 13, "moderna": 29, "astrazeneca": 45, "JJ": 61}
    R_VOC_indices = {"pfizer": 14, "moderna": 30, "astrazeneca": 46, "JJ": 62}

    # give according to real data
    if vaccine == "data-driven":

        for fornitore in ["pfizer", "moderna", "astrazeneca", "JJ"]:
            for age_vacc in range(nage):
                # vaccines available for this step
                vaccines_available = int(vaccinations["vaccinations_" + fornitore][date][age_vacc])

                # total number of people that can be vaccinated
                den = compartments_step[0, age_vacc] + compartments_step[1, age_vacc] + compartments_step[2, age_vacc] + \
                      compartments_step[5, age_vacc] + compartments_step[6, age_vacc]

                # distribute among S / L / R
                v_to_S = int(min(compartments_step[0, age_vacc], vaccines_available * compartments_step[0, age_vacc] / den))
                v_to_L_WT = int(min(compartments_step[1, age_vacc], vaccines_available * compartments_step[1, age_vacc] / den))
                v_to_L_VOC = int(min(compartments_step[2, age_vacc], vaccines_available * compartments_step[2, age_vacc] / den))
                v_to_R_WT = int(min(compartments_step[5, age_vacc], vaccines_available * compartments_step[5, age_vacc] / den))
                v_to_R_VOC = int(min(compartments_step[6, age_vacc], vaccines_available * compartments_step[6, age_vacc] / den))

                # S
                compartments_step[0, age_vacc] -= v_to_S
                compartments_step[S_indices[fornitore], age_vacc] += v_to_S

                # L_WT
                compartments_step[1, age_vacc] -= v_to_L_WT
                compartments_step[L_WT_indices[fornitore], age_vacc] += v_to_L_WT

                # L_VOC
                compartments_step[2, age_vacc] -= v_to_L_VOC
                compartments_step[L_VOC_indices[fornitore], age_vacc] += v_to_L_VOC

                # R_WT
                compartments_step[5, age_vacc] -= v_to_R_WT
                compartments_step[R_WT_indices[fornitore], age_vacc] += v_to_R_WT

                # R_VOC
                compartments_step[6, age_vacc] -= v_to_R_VOC
                compartments_step[R_VOC_indices[fornitore], age_vacc] += v_to_R_VOC

                vaccines_given[age_vacc] += (v_to_S + v_to_L_WT + v_to_L_VOC + v_to_R_WT + v_to_R_VOC)

    elif vaccine == "age-order":

        for fornitore in ["pfizer", "moderna", "astrazeneca", "JJ"]:

            vaccines_given_fornitore = np.zeros(nage)

            # give the baseline vaccine data-driven
            for age_vacc in range(nage):

                # total number of people that can be vaccinated in this age group
                den = compartments_step[0, age_vacc] + compartments_step[1, age_vacc] + compartments_step[2, age_vacc] + \
                      compartments_step[5, age_vacc] + compartments_step[6, age_vacc]

                # if there's still someone to vaccinate
                if den != 0:
                    # distribute among S / L / R
                    v_to_S = int(min(compartments_step[0, age_vacc], int(vaccinations["vaccinations_baseline_" + fornitore][date][age_vacc]) * compartments_step[0, age_vacc] / den))
                    v_to_L_WT = int(min(compartments_step[1, age_vacc], int(vaccinations["vaccinations_baseline_" + fornitore][date][age_vacc]) * compartments_step[1, age_vacc] / den))
                    v_to_L_VOC = int(min(compartments_step[2, age_vacc], int(vaccinations["vaccinations_baseline_" + fornitore][date][age_vacc]) * compartments_step[2, age_vacc] / den))
                    v_to_R_WT = int(min(compartments_step[5, age_vacc], int(vaccinations["vaccinations_baseline_" + fornitore][date][age_vacc]) * compartments_step[5, age_vacc] / den))
                    v_to_R_VOC = int(min(compartments_step[6, age_vacc], int(vaccinations["vaccinations_baseline_" + fornitore][date][age_vacc]) * compartments_step[6, age_vacc] / den))

                    # S
                    compartments_step[0, age_vacc] -= v_to_S
                    compartments_step[S_indices[fornitore], age_vacc] += v_to_S

                    # L_WT
                    compartments_step[1, age_vacc] -= v_to_L_WT
                    compartments_step[L_WT_indices[fornitore], age_vacc] += v_to_L_WT

                    # L_VOC
                    compartments_step[2, age_vacc] -= v_to_L_VOC
                    compartments_step[L_VOC_indices[fornitore], age_vacc] += v_to_L_VOC

                    # R_WT
                    compartments_step[5, age_vacc] -= v_to_R_WT
                    compartments_step[R_WT_indices[fornitore], age_vacc] += v_to_R_WT

                    # R_VOC
                    compartments_step[6, age_vacc] -= v_to_R_VOC
                    compartments_step[R_VOC_indices[fornitore], age_vacc] += v_to_R_VOC

                    # update doses given and left
                    vaccines_given[age_vacc] += (v_to_S + v_to_L_WT + v_to_L_VOC + v_to_R_WT + v_to_R_VOC)
                    vaccines_baseline_given[age_vacc] += (v_to_S + v_to_L_WT + v_to_L_VOC + v_to_R_WT + v_to_R_VOC)
                    vaccines_given_fornitore[age_vacc] += (v_to_S + v_to_L_WT + v_to_L_VOC + v_to_R_WT + v_to_R_VOC)

            # total number of vaccines available for this step
            tot_vaccine = int(vaccinations["vaccinations_" + fornitore][date].sum()) - np.sum(vaccines_given_fornitore)
            left_vaccine = tot_vaccine

            # distribute in decreasing order of age up to 50+
            # 9: 80+, 8: 70-79, 7: 60-69, 6: 50-59, 5: 40:49
            for age_vacc in np.arange(nage - 1, 5, -1):
                # total number of people that can be vaccinated
                den = compartments_step[0, age_vacc] + compartments_step[1, age_vacc] + compartments_step[2, age_vacc] + \
                      compartments_step[5, age_vacc] + compartments_step[6, age_vacc]

                # if there's still someone to vaccinate
                if den != 0 and left_vaccine > 0:

                    # distribute among S / L / R
                    v_to_S = int(min(compartments_step[0, age_vacc], left_vaccine * compartments_step[0, age_vacc] / den))
                    v_to_L_WT = int(min(compartments_step[1, age_vacc], left_vaccine * compartments_step[1, age_vacc] / den))
                    v_to_L_VOC = int(min(compartments_step[2, age_vacc], left_vaccine * compartments_step[2, age_vacc] / den))
                    v_to_R_WT = int(min(compartments_step[5, age_vacc], left_vaccine * compartments_step[5, age_vacc] / den))
                    v_to_R_VOC = int(min(compartments_step[6, age_vacc], left_vaccine * compartments_step[6, age_vacc] / den))

                    # S
                    compartments_step[0, age_vacc] -= v_to_S
                    compartments_step[S_indices[fornitore], age_vacc] += v_to_S

                    # L_WT
                    compartments_step[1, age_vacc] -= v_to_L_WT
                    compartments_step[L_WT_indices[fornitore], age_vacc] += v_to_L_WT

                    # L_VOC
                    compartments_step[2, age_vacc] -= v_to_L_VOC
                    compartments_step[L_VOC_indices[fornitore], age_vacc] += v_to_L_VOC

                    # R_WT
                    compartments_step[5, age_vacc] -= v_to_R_WT
                    compartments_step[R_WT_indices[fornitore], age_vacc] += v_to_R_WT

                    # R_VOC
                    compartments_step[6, age_vacc] -= v_to_R_VOC
                    compartments_step[R_VOC_indices[fornitore], age_vacc] += v_to_R_VOC

                    # update doses given and left
                    vaccines_given[age_vacc] += (v_to_S + v_to_L_WT + v_to_L_VOC + v_to_R_WT + v_to_R_VOC)
                    vaccines_nonbaseline_given[age_vacc] += (v_to_S + v_to_L_WT + v_to_L_VOC + v_to_R_WT + v_to_R_VOC)
                    left_vaccine -= (v_to_S + v_to_L_WT + v_to_L_VOC + v_to_R_WT + v_to_R_VOC)

            # give the remaining homogeneously in the under 50
            den = 0
            for age_vacc in np.arange(1, 6):
                den += compartments_step[0, age_vacc] + compartments_step[1, age_vacc] + compartments_step[2, age_vacc] + \
                       compartments_step[5, age_vacc] + compartments_step[6, age_vacc]

            # if there's still someone to vaccinate
            if den != 0:

                # iterate over the remaining age groups (under 50)
                for age_vacc in np.arange(1, 6):

                    # if there are still doses
                    if left_vaccine > 0:

                        # distribute among S / L / R
                        v_to_S = int(min(compartments_step[0, age_vacc], left_vaccine * compartments_step[0, age_vacc] / den))
                        v_to_L_WT = int(min(compartments_step[1, age_vacc], left_vaccine * compartments_step[1, age_vacc] / den))
                        v_to_L_VOC = int(min(compartments_step[2, age_vacc], left_vaccine * compartments_step[2, age_vacc] / den))
                        v_to_R_WT = int(min(compartments_step[5, age_vacc], left_vaccine * compartments_step[5, age_vacc] / den))
                        v_to_R_VOC = int(min(compartments_step[6, age_vacc], left_vaccine * compartments_step[6, age_vacc] / den))

                        # S
                        compartments_step[0, age_vacc] -= v_to_S
                        compartments_step[S_indices[fornitore], age_vacc] += v_to_S

                        # L_WT
                        compartments_step[1, age_vacc] -= v_to_L_WT
                        compartments_step[L_WT_indices[fornitore], age_vacc] += v_to_L_WT

                        # L_VOC
                        compartments_step[2, age_vacc] -= v_to_L_VOC
                        compartments_step[L_VOC_indices[fornitore], age_vacc] += v_to_L_VOC

                        # R_WT
                        compartments_step[5, age_vacc] -= v_to_R_WT
                        compartments_step[R_WT_indices[fornitore], age_vacc] += v_to_R_WT

                        # R_VOC
                        compartments_step[6, age_vacc] -= v_to_R_VOC
                        compartments_step[R_VOC_indices[fornitore], age_vacc] += v_to_R_VOC

                        # update doses given and left
                        vaccines_given[age_vacc] += (v_to_S + v_to_L_WT + v_to_L_VOC + v_to_R_WT + v_to_R_VOC)
                        vaccines_nonbaseline_given[age_vacc] += (v_to_S + v_to_L_WT + v_to_L_VOC + v_to_R_WT + v_to_R_VOC)
                        left_vaccine -= (v_to_S + v_to_L_WT + v_to_L_VOC + v_to_R_WT + v_to_R_VOC)

    elif vaccine == "reverse-age-order":

        for fornitore in ["pfizer", "moderna", "astrazeneca", "JJ"]:

            # give the baseline vaccine data-driven
            for age_vacc in range(nage):

                # total number of people that can be vaccinated in this age group
                den = compartments_step[0, age_vacc] + compartments_step[1, age_vacc] + compartments_step[2, age_vacc] + \
                      compartments_step[5, age_vacc] + compartments_step[6, age_vacc]

                # if there's still someone to vaccinate
                if den != 0:
                    # distribute among S / L / R
                    v_to_S = int(min(compartments_step[0, age_vacc], int(vaccinations["vaccinations_baseline_" + fornitore][date][age_vacc]) * compartments_step[0, age_vacc] / den))
                    v_to_L_WT = int(min(compartments_step[1, age_vacc], int(vaccinations["vaccinations_baseline_" + fornitore][date][age_vacc]) * compartments_step[1, age_vacc] / den))
                    v_to_L_VOC = int(min(compartments_step[2, age_vacc], int(vaccinations["vaccinations_baseline_" + fornitore][date][age_vacc]) * compartments_step[2, age_vacc] / den))
                    v_to_R_WT = int(min(compartments_step[5, age_vacc], int(vaccinations["vaccinations_baseline_" + fornitore][date][age_vacc]) * compartments_step[5, age_vacc] / den))
                    v_to_R_VOC = int(min(compartments_step[6, age_vacc], int(vaccinations["vaccinations_baseline_" + fornitore][date][age_vacc]) * compartments_step[6, age_vacc] / den))

                    # S
                    compartments_step[0, age_vacc] -= v_to_S
                    compartments_step[S_indices[fornitore], age_vacc] += v_to_S

                    # L_WT
                    compartments_step[1, age_vacc] -= v_to_L_WT
                    compartments_step[L_WT_indices[fornitore], age_vacc] += v_to_L_WT

                    # L_VOC
                    compartments_step[2, age_vacc] -= v_to_L_VOC
                    compartments_step[L_VOC_indices[fornitore], age_vacc] += v_to_L_VOC

                    # R_WT
                    compartments_step[5, age_vacc] -= v_to_R_WT
                    compartments_step[R_WT_indices[fornitore], age_vacc] += v_to_R_WT

                    # R_VOC
                    compartments_step[6, age_vacc] -= v_to_R_VOC
                    compartments_step[R_VOC_indices[fornitore], age_vacc] += v_to_R_VOC

                    # update doses given and left
                    vaccines_given[age_vacc] += (v_to_S + v_to_L_WT + v_to_L_VOC + v_to_R_WT + v_to_R_VOC)

                # total number of vaccines available for this step
            tot_vaccine = int(vaccinations["vaccinations_" + fornitore][date].sum()) - np.sum(vaccines_given)
            left_vaccine = tot_vaccine

            # distribute homogeneously to 20-49
            # 9: 80+, 8: 70-79, 7: 60-69, 6: 50-59, 5: 40:49, 4: 30-39, 3: 25-29, 2: 20-24, 1: 10-19, 0: 0-9
            den = 0
            for age_vacc in [5, 4, 3, 2]:
                # total number of people that can be vaccinated
                den += compartments_step[0, age_vacc] + compartments_step[1, age_vacc] + compartments_step[2, age_vacc] + \
                       compartments_step[5, age_vacc] + compartments_step[6, age_vacc]
            # if there's still someone to vaccinate
            if den != 0:

                # iterate over 20-49
                for age_vacc in [5, 4, 3, 2]:

                    # if there are still doses
                    if left_vaccine > 0:

                        # distribute among S / L / R
                        v_to_S = int(min(compartments_step[0, age_vacc], left_vaccine * compartments_step[0, age_vacc] / den))
                        v_to_L_WT = int(min(compartments_step[1, age_vacc], left_vaccine * compartments_step[1, age_vacc] / den))
                        v_to_L_VOC = int(min(compartments_step[2, age_vacc], left_vaccine * compartments_step[2, age_vacc] / den))
                        v_to_R_WT = int(min(compartments_step[5, age_vacc], left_vaccine * compartments_step[5, age_vacc] / den))
                        v_to_R_VOC = int(min(compartments_step[6, age_vacc], left_vaccine * compartments_step[6, age_vacc] / den))

                        # S
                        compartments_step[0, age_vacc] -= v_to_S
                        compartments_step[S_indices[fornitore], age_vacc] += v_to_S

                        # L_WT
                        compartments_step[1, age_vacc] -= v_to_L_WT
                        compartments_step[L_WT_indices[fornitore], age_vacc] += v_to_L_WT

                        # L_VOC
                        compartments_step[2, age_vacc] -= v_to_L_VOC
                        compartments_step[L_VOC_indices[fornitore], age_vacc] += v_to_L_VOC

                        # R_WT
                        compartments_step[5, age_vacc] -= v_to_R_WT
                        compartments_step[R_WT_indices[fornitore], age_vacc] += v_to_R_WT

                        # R_VOC
                        compartments_step[6, age_vacc] -= v_to_R_VOC
                        compartments_step[R_VOC_indices[fornitore], age_vacc] += v_to_R_VOC

                        # update doses given and left
                        vaccines_given[age_vacc] += (v_to_S + v_to_L_WT + v_to_L_VOC + v_to_R_WT + v_to_R_VOC)
                        left_vaccine -= (v_to_S + v_to_L_WT + v_to_L_VOC + v_to_R_WT + v_to_R_VOC)

            # give the remaining homogeneously to other age groups
            den = 0
            for age_vacc in [1, 6, 7, 8, 9]:
                den += compartments_step[0, age_vacc] + compartments_step[1, age_vacc] + compartments_step[2, age_vacc] + \
                       compartments_step[5, age_vacc] + compartments_step[6, age_vacc]

            # if there's still someone to vaccinate
            if den != 0:

                # iterate over remaining age groups
                for age_vacc in [1, 6, 7, 8, 9]:

                    # if there are still doses
                    if left_vaccine > 0:

                        # distribute among S / L / R
                        v_to_S = int(min(compartments_step[0, age_vacc], left_vaccine * compartments_step[0, age_vacc] / den))
                        v_to_L_WT = int(min(compartments_step[1, age_vacc], left_vaccine * compartments_step[1, age_vacc] / den))
                        v_to_L_VOC = int(min(compartments_step[2, age_vacc], left_vaccine * compartments_step[2, age_vacc] / den))
                        v_to_R_WT = int(min(compartments_step[5, age_vacc], left_vaccine * compartments_step[5, age_vacc] / den))
                        v_to_R_VOC = int(min(compartments_step[6, age_vacc], left_vaccine * compartments_step[6, age_vacc] / den))

                        # S
                        compartments_step[0, age_vacc] -= v_to_S
                        compartments_step[S_indices[fornitore], age_vacc] += v_to_S

                        # L_WT
                        compartments_step[1, age_vacc] -= v_to_L_WT
                        compartments_step[L_WT_indices[fornitore], age_vacc] += v_to_L_WT

                        # L_VOC
                        compartments_step[2, age_vacc] -= v_to_L_VOC
                        compartments_step[L_VOC_indices[fornitore], age_vacc] += v_to_L_VOC

                        # R_WT
                        compartments_step[5, age_vacc] -= v_to_R_WT
                        compartments_step[R_WT_indices[fornitore], age_vacc] += v_to_R_WT

                        # R_VOC
                        compartments_step[6, age_vacc] -= v_to_R_VOC
                        compartments_step[R_VOC_indices[fornitore], age_vacc] += v_to_R_VOC

                        # update doses given and left
                        vaccines_given[age_vacc] += (v_to_S + v_to_L_WT + v_to_L_VOC + v_to_R_WT + v_to_R_VOC)
                        left_vaccine -= (v_to_S + v_to_L_WT + v_to_L_VOC + v_to_R_WT + v_to_R_VOC)

    return compartments_step, vaccines_given, vaccines_nonbaseline_given, vaccines_baseline_given


#@jit(fastmath=True, cache=True)
def stochastic_seird(Cs, initial_conditions, importations, vaccinations, Nk, dates, vaccine, params):
    """
    This function simulates a stochastic SEIR model with two strains and vaccinations
        :param Cs: dictionary of contact matrices
        :param initial_conditions: initial conditions for different compartment/age groups
        :param importations: importations for different age groups
        :param vaccinations: dictionary of vaccinations
        :param Nk: number of people in different age groups
        :param dates: array of dates
        :param vaccine: vaccination strategy
        :param params: dictionary of parameters
        :return: returns evolution of n. of individuals in different compartments
    """

    # initial conditions
    T = len(dates)
    compartments = np.zeros((ncomp, nage, T))
    compartments[:, :, 0] = initial_conditions

    # recovered (to compute deaths)
    recovered_WT, recovered_VOC = np.zeros((nage, T)), np.zeros((nage, T))
    recovered_WT_V1i_pfizer, recovered_VOC_V1i_pfizer = np.zeros((nage, T)), np.zeros((nage, T))
    recovered_WT_V2i_pfizer, recovered_VOC_V2i_pfizer = np.zeros((nage, T)), np.zeros((nage, T))

    recovered_WT_V1i_moderna, recovered_VOC_V1i_moderna = np.zeros((nage, T)), np.zeros((nage, T))
    recovered_WT_V2i_moderna, recovered_VOC_V2i_moderna = np.zeros((nage, T)), np.zeros((nage, T))

    recovered_WT_V1i_astrazeneca, recovered_VOC_V1i_astrazeneca = np.zeros((nage, T)), np.zeros((nage, T))
    recovered_WT_V2i_astrazeneca, recovered_VOC_V2i_astrazeneca = np.zeros((nage, T)), np.zeros((nage, T))

    recovered_WT_V1i_JJ, recovered_VOC_V1i_JJ = np.zeros((nage, T)), np.zeros((nage, T))

    # incidence
    incidence_WT = np.zeros((nage, T))
    incidence_VOC = np.zeros((nage, T))

    # vaccinations per step
    vaccines_per_step = np.zeros((nage, T))
    vaccines_baseline_per_step = np.zeros((nage, T))
    vaccines_nonbaseline_per_step = np.zeros((nage, T))

    # simulate
    for i in range(T - 1):

        # importations
        if 12 <= i < 99 + 12:  # importations end on 2020/12/20
            for age_imp in range(nage):
                compartments[2, age_imp, i] += importations[age_imp, i - 12]  # L_VOC
                compartments[0, age_imp, i] -= importations[age_imp, i - 12]  # S

        # vaccinate
        compartments_step, vaccines_given, vaccines_nonbaseline_given, vaccines_baseline_given = get_vaccinated(vaccine, vaccinations, compartments[:, :, i], dates[i])
        compartments[:, :, i] = compartments_step
        vaccines_per_step[:, i] = vaccines_given
        vaccines_baseline_per_step[:, i] = vaccines_baseline_given
        vaccines_nonbaseline_per_step[:, i] = vaccines_nonbaseline_given

        # update contacts
        C = Cs[dates[i]]

        # seasonality adjustment
        seasonal_adjustment = apply_seasonality(dates[i], params["seasonality_min"], params["hemisphere"])

        # next step solution
        next_step = np.zeros((ncomp, nage))

        # iterate over ages
        for age1 in range(nage):

            # compute force of infections of the two strains
            force_inf_WT = np.sum(params["beta"] * seasonal_adjustment * C[age1, :] * (compartments[3, :, i] +
                                                                                       (1 - params["VEI1_pfizer"]) * compartments[11, :, i] +
                                                                                       (1 - params["VEI2_pfizer"]) * compartments[19, :, i] +
                                                                                       (1 - params["VEI1_moderna"]) * compartments[27, :, i] +
                                                                                       (1 - params["VEI2_moderna"]) * compartments[35, :, i] +
                                                                                       (1 - params["VEI1_astrazeneca"]) * compartments[43, :, i] +
                                                                                       (1 - params["VEI2_astrazeneca"]) * compartments[51, :, i] +
                                                                                       (1 - params["VEI1_JJ"]) * compartments[59, :, i]) / Nk)

            force_inf_VOC = np.sum(params["beta"] * seasonal_adjustment * (1 + params["psi"]) * C[age1, :] * (compartments[4, :, i] +
                                                                                                              (1 - params["VEI1_pfizer_VOC"]) * compartments[12, :, i] +
                                                                                                              (1 - params["VEI2_pfizer_VOC"]) * compartments[20, :, i] +
                                                                                                              (1 - params["VEI1_moderna_VOC"]) * compartments[28, :, i] +
                                                                                                              (1 - params["VEI2_moderna_VOC"]) * compartments[36, :, i] +
                                                                                                              (1 - params["VEI1_astrazeneca_VOC"]) * compartments[44, :, i] +
                                                                                                              (1 - params["VEI2_astrazeneca_VOC"]) * compartments[52, :, i] +
                                                                                                              (1 - params["VEI1_JJ_VOC"]) * compartments[60, :, i]) / Nk)

            # S -> L_WT and S -> L_VOC
            if force_inf_WT + force_inf_VOC == 0:
                new_latent_WT, new_latent_VOC = 0, 0
            else:
                new_latent_TOT = np.random.binomial(compartments[0, age1, i], force_inf_WT + force_inf_VOC)
                new_latent_WT = np.random.binomial(new_latent_TOT, force_inf_WT / (force_inf_WT + force_inf_VOC))
                new_latent_VOC = new_latent_TOT - new_latent_WT

            ################### PFIZER #####################
            # V1r -> L_WT and V1r -> L_VOC and V1r -> V1i
            leaving_fromV1r_pfizer = np.random.binomial(compartments[7, age1, i], force_inf_WT + force_inf_VOC + 1.0 / params["Delta_V"])
            new_latent_WT_fromV1r_pfizer = np.random.binomial(leaving_fromV1r_pfizer, force_inf_WT / (force_inf_WT + force_inf_VOC + 1.0 / params["Delta_V"]))
            new_latent_VOC_fromV1r_pfizer = np.random.binomial(leaving_fromV1r_pfizer - new_latent_WT_fromV1r_pfizer, force_inf_VOC / (force_inf_VOC + 1.0 / params["Delta_V"]))
            new_vaccinated_V1i_pfizer = leaving_fromV1r_pfizer - new_latent_WT_fromV1r_pfizer - new_latent_VOC_fromV1r_pfizer

            # V1i -> L_WT_V1i and V1i -> L_VOC_V1i and V1i -> V2r
            leaving_fromV1i_pfizer = np.random.binomial(compartments[8, age1, i], (1 - params["VES1_pfizer"]) * force_inf_WT + (1 - params["VES1_pfizer_VOC"]) * force_inf_VOC + 1.0 / params["Delta_V2_pfizer"])
            new_latent_WT_fromV1i_pfizer = np.random.binomial(leaving_fromV1i_pfizer, (1 - params["VES1_pfizer"]) * (force_inf_WT) / ((1 - params["VES1_pfizer"]) * force_inf_WT + (1 - params["VES1_pfizer_VOC"]) * force_inf_VOC + 1.0 / params["Delta_V2_pfizer"]))
            new_latent_VOC_fromV1i_pfizer = np.random.binomial(leaving_fromV1i_pfizer - new_latent_WT_fromV1i_pfizer, (1 - params["VES1_pfizer_VOC"]) * (force_inf_VOC) / ((1 - params["VES1_pfizer_VOC"]) * force_inf_VOC + 1.0 / params["Delta_V2_pfizer"]))
            new_vaccinated_V2r_pfizer = leaving_fromV1i_pfizer - new_latent_WT_fromV1i_pfizer - new_latent_VOC_fromV1i_pfizer

            # V2r -> L_WT_V1i and V2r -> L_VOC_V1i and V2r -> V2i
            leaving_fromV2r_pfizer = np.random.binomial(compartments[15, age1, i], (1 - params["VES1_pfizer"]) * force_inf_WT + (1 - params["VES1_pfizer_VOC"]) * force_inf_VOC + 1.0 / params["Delta_V"])
            new_latent_WT_fromV2r_pfizer = np.random.binomial(leaving_fromV2r_pfizer, (1 - params["VES1_pfizer"]) * (force_inf_WT) / ((1 - params["VES1_pfizer"]) * force_inf_WT + (1 - params["VES1_pfizer_VOC"]) * force_inf_VOC + 1.0 / params["Delta_V"]))
            new_latent_VOC_fromV2r_pfizer = np.random.binomial(leaving_fromV2r_pfizer - new_latent_WT_fromV2r_pfizer, (1 - params["VES1_pfizer_VOC"]) * (force_inf_VOC) / ((1 - params["VES1_pfizer_VOC"]) * force_inf_VOC + 1.0 / params["Delta_V"]))
            new_vaccinated_V2i_pfizer = leaving_fromV2r_pfizer - new_latent_WT_fromV2r_pfizer - new_latent_VOC_fromV2r_pfizer

            # V2i -> L_WT_V2i and V2i -> L_VOC_V2i
            if force_inf_WT + force_inf_VOC == 0:
                new_latent_WT_fromV2i_pfizer, new_latent_VOC_fromV2i_pfizer = 0, 0
            else:
                new_latent_TOT_fromV2i_pfizer = np.random.binomial(compartments[16, age1, i], (1 - params["VES2_pfizer"]) * force_inf_WT + (1 - params["VES2_pfizer_VOC"]) * force_inf_VOC)
                new_latent_WT_fromV2i_pfizer = np.random.binomial(new_latent_TOT_fromV2i_pfizer, (1 - params["VES2_pfizer"]) * force_inf_WT / ((1 - params["VES2_pfizer"]) * force_inf_WT + (1 - params["VES2_pfizer_VOC"]) * force_inf_VOC))
                new_latent_VOC_fromV2i_pfizer = new_latent_TOT_fromV2i_pfizer - new_latent_WT_fromV2i_pfizer
            ############################################################

            ################### MODERNA #####################
            # V1r -> L_WT and V1r -> L_VOC and V1r -> V1i
            leaving_fromV1r_moderna = np.random.binomial(compartments[23, age1, i], force_inf_WT + force_inf_VOC + 1.0 / params["Delta_V"])
            new_latent_WT_fromV1r_moderna = np.random.binomial(leaving_fromV1r_moderna, force_inf_WT / (force_inf_WT + force_inf_VOC + 1.0 / params["Delta_V"]))
            new_latent_VOC_fromV1r_moderna = np.random.binomial(leaving_fromV1r_moderna - new_latent_WT_fromV1r_moderna, force_inf_VOC / (force_inf_VOC + 1.0 / params["Delta_V"]))
            new_vaccinated_V1i_moderna = leaving_fromV1r_moderna - new_latent_WT_fromV1r_moderna - new_latent_VOC_fromV1r_moderna

            # V1i -> L_WT_V1i and V1i -> L_VOC_V1i and V1i -> V2r
            leaving_fromV1i_moderna = np.random.binomial(compartments[24, age1, i], (1 - params["VES1_moderna"]) * force_inf_WT + (1 - params["VES1_moderna_VOC"]) * force_inf_VOC + 1.0 / params["Delta_V2_moderna"])
            new_latent_WT_fromV1i_moderna = np.random.binomial(leaving_fromV1i_moderna, (1 - params["VES1_moderna"]) * (force_inf_WT) / ((1 - params["VES1_moderna"]) * force_inf_WT + (1 - params["VES1_moderna_VOC"]) * force_inf_VOC + 1.0 / params["Delta_V2_moderna"]))
            new_latent_VOC_fromV1i_moderna = np.random.binomial(leaving_fromV1i_moderna - new_latent_WT_fromV1i_moderna, (1 - params["VES1_moderna_VOC"]) * (force_inf_VOC) / ((1 - params["VES1_moderna_VOC"]) * force_inf_VOC + 1.0 / params["Delta_V2_moderna"]))
            new_vaccinated_V2r_moderna = leaving_fromV1i_moderna - new_latent_WT_fromV1i_moderna - new_latent_VOC_fromV1i_moderna

            # V2r -> L_WT_V1i and V2r -> L_VOC_V1i and V2r -> V2i
            leaving_fromV2r_moderna = np.random.binomial(compartments[31, age1, i], (1 - params["VES1_moderna"]) * force_inf_WT + (1 - params["VES1_moderna_VOC"]) * force_inf_VOC + 1.0 / params["Delta_V"])
            new_latent_WT_fromV2r_moderna = np.random.binomial(leaving_fromV2r_moderna, (1 - params["VES1_moderna"]) * (force_inf_WT) / ((1 - params["VES1_moderna"]) * force_inf_WT + params["VES1_moderna_VOC"] * force_inf_VOC + 1.0 / params["Delta_V"]))
            new_latent_VOC_fromV2r_moderna = np.random.binomial(leaving_fromV2r_moderna - new_latent_WT_fromV2r_moderna,(1 - params["VES1_moderna_VOC"]) * (force_inf_VOC) / ((1 - params["VES1_moderna_VOC"]) * force_inf_VOC + 1.0 / params["Delta_V"]))
            new_vaccinated_V2i_moderna = leaving_fromV2r_moderna - new_latent_WT_fromV2r_moderna - new_latent_VOC_fromV2r_moderna

            # V2i -> L_WT_V2i and V2i -> L_VOC_V2i
            if force_inf_WT + force_inf_VOC == 0:
                new_latent_WT_fromV2i_moderna, new_latent_VOC_fromV2i_moderna = 0, 0
            else:
                new_latent_TOT_fromV2i_moderna = np.random.binomial(compartments[32, age1, i], (1 - params["VES2_moderna"]) * force_inf_WT + (1 - params["VES2_moderna_VOC"]) * force_inf_VOC)
                new_latent_WT_fromV2i_moderna = np.random.binomial(new_latent_TOT_fromV2i_moderna, (1 - params["VES2_moderna"]) * force_inf_WT / ((1 - params["VES2_moderna"]) * force_inf_WT + (1 - params["VES2_moderna_VOC"]) * force_inf_VOC))
                new_latent_VOC_fromV2i_moderna = new_latent_TOT_fromV2i_moderna - new_latent_WT_fromV2i_moderna
            ############################################################

            ################### ASTRAZENECA #####################
            # V1r -> L_WT and V1r -> L_VOC and V1r -> V1i
            leaving_fromV1r_astrazeneca = np.random.binomial(compartments[39, age1, i], force_inf_WT + force_inf_VOC + 1.0 / params["Delta_V"])
            new_latent_WT_fromV1r_astrazeneca = np.random.binomial(leaving_fromV1r_astrazeneca, force_inf_WT / (force_inf_WT + force_inf_VOC + 1.0 / params["Delta_V"]))
            new_latent_VOC_fromV1r_astrazeneca = np.random.binomial(leaving_fromV1r_astrazeneca - new_latent_WT_fromV1r_astrazeneca, force_inf_VOC / (force_inf_VOC + 1.0 / params["Delta_V"]))
            new_vaccinated_V1i_astrazeneca = leaving_fromV1r_astrazeneca - new_latent_WT_fromV1r_astrazeneca - new_latent_VOC_fromV1r_astrazeneca

            # V1i -> L_WT_V1i and V1i -> L_VOC_V1i and V1i -> V2r
            leaving_fromV1i_astrazeneca = np.random.binomial(compartments[40, age1, i], (1 - params["VES1_astrazeneca"]) * force_inf_WT + (1 - params["VES1_astrazeneca_VOC"]) * force_inf_VOC + 1.0 / params["Delta_V2_astrazeneca"])
            new_latent_WT_fromV1i_astrazeneca = np.random.binomial(leaving_fromV1i_astrazeneca, (1 - params["VES1_astrazeneca"]) * (force_inf_WT) / ((1 - params["VES1_astrazeneca"]) * force_inf_WT + (1 - params["VES1_astrazeneca_VOC"]) * force_inf_VOC + 1.0 / params["Delta_V2_astrazeneca"]))
            new_latent_VOC_fromV1i_astrazeneca = np.random.binomial(leaving_fromV1i_astrazeneca - new_latent_WT_fromV1i_astrazeneca, (1 - params["VES1_astrazeneca_VOC"]) * (force_inf_VOC) / ((1 - params["VES1_astrazeneca_VOC"]) * force_inf_VOC + 1.0 / params["Delta_V2_astrazeneca"]))
            new_vaccinated_V2r_astrazeneca = leaving_fromV1i_astrazeneca - new_latent_WT_fromV1i_astrazeneca - new_latent_VOC_fromV1i_astrazeneca

            # V2r -> L_WT_V1i and V2r -> L_VOC_V1i and V2r -> V2i
            leaving_fromV2r_astrazeneca = np.random.binomial(compartments[47, age1, i], (1 - params["VES1_astrazeneca"]) * force_inf_WT + (1 - params["VES1_astrazeneca_VOC"]) * force_inf_VOC + 1.0 / params["Delta_V"])
            new_latent_WT_fromV2r_astrazeneca = np.random.binomial(leaving_fromV2r_astrazeneca, (1 - params["VES1_astrazeneca"]) * (force_inf_WT) / ((1 - params["VES1_astrazeneca"]) * force_inf_WT + (1 - params["VES1_astrazeneca_VOC"]) * force_inf_VOC + 1.0 / params["Delta_V"]))
            new_latent_VOC_fromV2r_astrazeneca = np.random.binomial(leaving_fromV2r_astrazeneca - new_latent_WT_fromV2r_astrazeneca, (1 - params["VES1_astrazeneca_VOC"]) * (force_inf_VOC) / ((1 - params["VES1_astrazeneca_VOC"]) * force_inf_VOC + 1.0 / params["Delta_V"]))
            new_vaccinated_V2i_astrazeneca = leaving_fromV2r_astrazeneca - new_latent_WT_fromV2r_astrazeneca - new_latent_VOC_fromV2r_astrazeneca

            # V2i -> L_WT_V2i and V2i -> L_VOC_V2i
            if force_inf_WT + force_inf_VOC == 0:
                new_latent_WT_fromV2i_astrazeneca, new_latent_VOC_fromV2i_astrazeneca = 0, 0
            else:
                new_latent_TOT_fromV2i_astrazeneca = np.random.binomial(compartments[48, age1, i], (1 - params["VES2_astrazeneca"]) * force_inf_WT + (1 - params["VES2_astrazeneca_VOC"]) * force_inf_VOC)
                new_latent_WT_fromV2i_astrazeneca = np.random.binomial(new_latent_TOT_fromV2i_astrazeneca, (1 - params["VES2_astrazeneca"]) * force_inf_WT / ((1 - params["VES2_astrazeneca"]) * force_inf_WT + (1 - params["VES2_astrazeneca_VOC"]) * force_inf_VOC))
                new_latent_VOC_fromV2i_astrazeneca = new_latent_TOT_fromV2i_astrazeneca - new_latent_WT_fromV2i_astrazeneca
            ############################################################

            ################### JJ #####################
            # V1r -> L_WT and V1r -> L_VOC and V1r -> V1i
            leaving_fromV1r_JJ = np.random.binomial(compartments[55, age1, i], force_inf_WT + force_inf_VOC + 1.0 / params["Delta_V"])
            new_latent_WT_fromV1r_JJ = np.random.binomial(leaving_fromV1r_JJ, force_inf_WT / (force_inf_WT + force_inf_VOC + 1.0 / params["Delta_V"]))
            new_latent_VOC_fromV1r_JJ = np.random.binomial(leaving_fromV1r_JJ - new_latent_WT_fromV1r_JJ, force_inf_VOC / (force_inf_VOC + 1.0 / params["Delta_V"]))
            new_vaccinated_V1i_JJ = leaving_fromV1r_JJ - new_latent_WT_fromV1r_JJ - new_latent_VOC_fromV1r_JJ

            # V1i -> L_WT_V1i and V1i -> L_VOC_V1i
            if force_inf_WT + force_inf_VOC == 0:
                new_latent_WT_fromV1i_JJ, new_latent_VOC_fromV1i_JJ = 0, 0
            else:
                leaving_fromV1i_JJ = np.random.binomial(compartments[56, age1, i], (1 - params["VES1_JJ"]) * force_inf_WT + (1 - params["VES1_JJ_VOC"]) * force_inf_VOC)
                new_latent_WT_fromV1i_JJ = np.random.binomial(leaving_fromV1i_JJ, (1 - params["VES1_JJ"]) * (force_inf_WT) / ((1 - params["VES1_JJ"]) * force_inf_WT + (1 - params["VES1_JJ_VOC"]) * force_inf_VOC))
                new_latent_VOC_fromV1i_JJ = leaving_fromV1i_JJ - new_latent_WT_fromV1i_JJ
            ############################################################

            # L_WT -> I_WT
            new_infected_WT = np.random.binomial(compartments[1, age1, i], params["eps"])

            # L_VOC -> I_VOC
            new_infected_VOC = np.random.binomial(compartments[2, age1, i], params["eps"])

            # I_WT -> R_WT
            new_recovered_WT = np.random.binomial(compartments[3, age1, i], params["mu"])

            # I_VOC -> R_VOC
            new_recovered_VOC = np.random.binomial(compartments[4, age1, i], params["mu"])


            ############### PFIZER ###############
            # L_WT_V1i -> I_WT_V1i
            new_infected_WT_fromV1i_pfizer = np.random.binomial(compartments[9, age1, i], params["eps"])

            # L_VOC_V1i -> I_VOC_V1i
            new_infected_VOC_fromV1i_pfizer = np.random.binomial(compartments[10, age1, i], params["eps"])

            # L_WT_V2i -> I_WT_V2i
            new_infected_WT_fromV2i_pfizer = np.random.binomial(compartments[17, age1, i], params["eps"])

            # L_VOC_V2i -> I_VOC_V2i
            new_infected_VOC_fromV2i_pfizer = np.random.binomial(compartments[18, age1, i], params["eps"])

            # I_WT_V1i -> R_WT_V1i
            new_recovered_WT_fromV1i_pfizer = np.random.binomial(compartments[11, age1, i], params["mu"])

            # I_VOC_V1i -> R_VOC_V1i
            new_recovered_VOC_fromV1i_pfizer = np.random.binomial(compartments[12, age1, i], params["mu"])

            # I_WT_V2i -> R_WT_V2i
            new_recovered_WT_fromV2i_pfizer = np.random.binomial(compartments[19, age1, i], params["mu"])

            # I_VOC_V2i -> R_VOC_V2i
            new_recovered_VOC_fromV2i_pfizer = np.random.binomial(compartments[20, age1, i], params["mu"])
            ##############################

            ############### moderna ###############
            # L_WT_V1i -> I_WT_V1i
            new_infected_WT_fromV1i_moderna = np.random.binomial(compartments[25, age1, i], params["eps"])

            # L_VOC_V1i -> I_VOC_V1i
            new_infected_VOC_fromV1i_moderna = np.random.binomial(compartments[26, age1, i], params["eps"])

            # L_WT_V2i -> I_WT_V2i
            new_infected_WT_fromV2i_moderna = np.random.binomial(compartments[33, age1, i], params["eps"])

            # L_VOC_V2i -> I_VOC_V2i
            new_infected_VOC_fromV2i_moderna = np.random.binomial(compartments[34, age1, i], params["eps"])

            # I_WT_V1i -> R_WT_V1i
            new_recovered_WT_fromV1i_moderna = np.random.binomial(compartments[27, age1, i], params["mu"])

            # I_VOC_V1i -> R_VOC_V1i
            new_recovered_VOC_fromV1i_moderna = np.random.binomial(compartments[28, age1, i], params["mu"])

            # I_WT_V2i -> R_WT_V2i
            new_recovered_WT_fromV2i_moderna = np.random.binomial(compartments[35, age1, i], params["mu"])

            # I_VOC_V2i -> R_VOC_V2i
            new_recovered_VOC_fromV2i_moderna = np.random.binomial(compartments[36, age1, i], params["mu"])
            ##############################

            ############### astrazeneca ###############
            # L_WT_V1i -> I_WT_V1i
            new_infected_WT_fromV1i_astrazeneca = np.random.binomial(compartments[41, age1, i], params["eps"])

            # L_VOC_V1i -> I_VOC_V1i
            new_infected_VOC_fromV1i_astrazeneca = np.random.binomial(compartments[42, age1, i], params["eps"])

            # L_WT_V2i -> I_WT_V2i
            new_infected_WT_fromV2i_astrazeneca = np.random.binomial(compartments[49, age1, i], params["eps"])

            # L_VOC_V2i -> I_VOC_V2i
            new_infected_VOC_fromV2i_astrazeneca = np.random.binomial(compartments[50, age1, i], params["eps"])

            # I_WT_V1i -> R_WT_V1i
            new_recovered_WT_fromV1i_astrazeneca = np.random.binomial(compartments[43, age1, i], params["mu"])

            # I_VOC_V1i -> R_VOC_V1i
            new_recovered_VOC_fromV1i_astrazeneca = np.random.binomial(compartments[44, age1, i], params["mu"])

            # I_WT_V2i -> R_WT_V2i
            new_recovered_WT_fromV2i_astrazeneca = np.random.binomial(compartments[51, age1, i], params["mu"])

            # I_VOC_V2i -> R_VOC_V2i
            new_recovered_VOC_fromV2i_astrazeneca = np.random.binomial(compartments[52, age1, i], params["mu"])
            ##############################

            ############### JJ ###############
            # L_WT_V1i -> I_WT_V1i
            new_infected_WT_fromV1i_JJ = np.random.binomial(compartments[57, age1, i], params["eps"])

            # L_VOC_V1i -> I_VOC_V1i
            new_infected_VOC_fromV1i_JJ = np.random.binomial(compartments[58, age1, i], params["eps"])

            # I_WT_V1i -> R_WT_V1i
            new_recovered_WT_fromV1i_JJ = np.random.binomial(compartments[59, age1, i], params["mu"])

            # I_VOC_V1i -> R_VOC_V1i
            new_recovered_VOC_fromV1i_JJ = np.random.binomial(compartments[60, age1, i], params["mu"])
            ##############################


            # update next step solution
            # S
            next_step[0, age1] = compartments[0, age1, i] - new_latent_WT - new_latent_VOC
            # L_WT
            next_step[1, age1] = compartments[1, age1, i] + new_latent_WT + new_latent_WT_fromV1r_pfizer + \
                                 new_latent_WT_fromV1r_moderna + new_latent_WT_fromV1r_astrazeneca + \
                                 new_latent_WT_fromV1r_JJ - new_infected_WT
            # L_VOC
            next_step[2, age1] = compartments[2, age1, i] + new_latent_VOC + new_latent_VOC_fromV1r_pfizer + \
                                 new_latent_VOC_fromV1r_moderna + new_latent_VOC_fromV1r_astrazeneca + \
                                 new_latent_VOC_fromV1r_JJ - new_infected_VOC
            # I_WT
            next_step[3, age1] = compartments[3, age1, i] + new_infected_WT - new_recovered_WT
            # I_VOC
            next_step[4, age1] = compartments[4, age1, i] + new_infected_VOC - new_recovered_VOC
            # R_WT
            next_step[5, age1] = compartments[5, age1, i] + new_recovered_WT
            # R_VOC
            next_step[6, age1] = compartments[6, age1, i] + new_recovered_VOC

            ############### PFIZER ###############
            # V1r
            next_step[7, age1] = compartments[7, age1, i] - new_latent_WT_fromV1r_pfizer - new_latent_VOC_fromV1r_pfizer - new_vaccinated_V1i_pfizer
            # V1i
            next_step[8, age1] = compartments[8, age1, i] - new_latent_WT_fromV1i_pfizer - new_latent_VOC_fromV1i_pfizer + new_vaccinated_V1i_pfizer - new_vaccinated_V2r_pfizer
            # L_WT_V1i
            next_step[9, age1] = compartments[9, age1, i] + new_latent_WT_fromV1i_pfizer + new_latent_WT_fromV2r_pfizer - new_infected_WT_fromV1i_pfizer
            # L_VOC_V1i
            next_step[10, age1] = compartments[10, age1, i] + new_latent_VOC_fromV1i_pfizer + new_latent_VOC_fromV2r_pfizer - new_infected_VOC_fromV1i_pfizer
            # I_WT_V1i
            next_step[11, age1] = compartments[11, age1, i] + new_infected_WT_fromV1i_pfizer - new_recovered_WT_fromV1i_pfizer
            # I_VOC_V1i
            next_step[12, age1] = compartments[12, age1, i] + new_infected_VOC_fromV1i_pfizer - new_recovered_VOC_fromV1i_pfizer
            # R_WT_V1i
            next_step[13, age1] = compartments[13, age1, i] + new_recovered_WT_fromV1i_pfizer
            # R_VOC_V1i
            next_step[14, age1] = compartments[14, age1, i] + new_recovered_VOC_fromV1i_pfizer
            # V2r
            next_step[15, age1] = compartments[15, age1, i] + new_vaccinated_V2r_pfizer - new_latent_WT_fromV2r_pfizer - new_latent_VOC_fromV2r_pfizer - new_vaccinated_V2i_pfizer
            # V2i
            next_step[16, age1] = compartments[16, age1, i] + new_vaccinated_V2i_pfizer - new_latent_WT_fromV2i_pfizer - new_latent_VOC_fromV2i_pfizer
            # L_WT_V2i
            next_step[17, age1] = compartments[17, age1, i] + new_latent_WT_fromV2i_pfizer - new_infected_WT_fromV2i_pfizer
            # L_VOC_V2i
            next_step[18, age1] = compartments[18, age1, i] + new_latent_VOC_fromV2i_pfizer - new_infected_VOC_fromV2i_pfizer
            # I_WT_V2i
            next_step[19, age1] = compartments[19, age1, i] + new_infected_WT_fromV2i_pfizer - new_recovered_WT_fromV2i_pfizer
            # I_VOC_V2i
            next_step[20, age1] = compartments[20, age1, i] + new_infected_VOC_fromV2i_pfizer - new_recovered_VOC_fromV2i_pfizer
            # R_WT_V2i
            next_step[21, age1] = compartments[21, age1, i] + new_recovered_WT_fromV2i_pfizer
            # R_VOC_V2i
            next_step[22, age1] = compartments[22, age1, i] + new_recovered_VOC_fromV2i_pfizer
            ##############################

            ############### moderna ###############
            # V1r
            next_step[23, age1] = compartments[23, age1, i] - new_latent_WT_fromV1r_moderna - new_latent_VOC_fromV1r_moderna - new_vaccinated_V1i_moderna
            # V1i
            next_step[24, age1] = compartments[24, age1, i] - new_latent_WT_fromV1i_moderna - new_latent_VOC_fromV1i_moderna + new_vaccinated_V1i_moderna - new_vaccinated_V2r_moderna
            # L_WT_V1i
            next_step[25, age1] = compartments[25, age1, i] + new_latent_WT_fromV1i_moderna + new_latent_WT_fromV2r_moderna - new_infected_WT_fromV1i_moderna
            # L_VOC_V1i
            next_step[26, age1] = compartments[26, age1, i] + new_latent_VOC_fromV1i_moderna + new_latent_VOC_fromV2r_moderna - new_infected_VOC_fromV1i_moderna
            # I_WT_V1i
            next_step[27, age1] = compartments[27, age1, i] + new_infected_WT_fromV1i_moderna - new_recovered_WT_fromV1i_moderna
            # I_VOC_V1i
            next_step[28, age1] = compartments[28, age1, i] + new_infected_VOC_fromV1i_moderna - new_recovered_VOC_fromV1i_moderna
            # R_WT_V1i
            next_step[29, age1] = compartments[29, age1, i] + new_recovered_WT_fromV1i_moderna
            # R_VOC_V1i
            next_step[30, age1] = compartments[30, age1, i] + new_recovered_VOC_fromV1i_moderna
            # V2r
            next_step[31, age1] = compartments[31, age1, i] + new_vaccinated_V2r_moderna - new_latent_WT_fromV2r_moderna - new_latent_VOC_fromV2r_moderna - new_vaccinated_V2i_moderna
            # V2i
            next_step[32, age1] = compartments[32, age1, i] + new_vaccinated_V2i_moderna - new_latent_WT_fromV2i_moderna - new_latent_VOC_fromV2i_moderna
            # L_WT_V2i
            next_step[33, age1] = compartments[33, age1, i] + new_latent_WT_fromV2i_moderna - new_infected_WT_fromV2i_moderna
            # L_VOC_V2i
            next_step[34, age1] = compartments[34, age1, i] + new_latent_VOC_fromV2i_moderna - new_infected_VOC_fromV2i_moderna
            # I_WT_V2i
            next_step[35, age1] = compartments[35, age1, i] + new_infected_WT_fromV2i_moderna - new_recovered_WT_fromV2i_moderna
            # I_VOC_V2i
            next_step[36, age1] = compartments[36, age1, i] + new_infected_VOC_fromV2i_moderna - new_recovered_VOC_fromV2i_moderna
            # R_WT_V2i
            next_step[37, age1] = compartments[37, age1, i] + new_recovered_WT_fromV2i_moderna
            # R_VOC_V2i
            next_step[38, age1] = compartments[38, age1, i] + new_recovered_VOC_fromV2i_moderna
            ##############################

            ############### astrazeneca ###############
            # V1r
            next_step[39, age1] = compartments[39, age1, i] - new_latent_WT_fromV1r_astrazeneca - new_latent_VOC_fromV1r_astrazeneca - new_vaccinated_V1i_astrazeneca
            # V1i
            next_step[40, age1] = compartments[40, age1, i] - new_latent_WT_fromV1i_astrazeneca - new_latent_VOC_fromV1i_astrazeneca + new_vaccinated_V1i_astrazeneca - new_vaccinated_V2r_astrazeneca
            # L_WT_V1i
            next_step[41, age1] = compartments[41, age1, i] + new_latent_WT_fromV1i_astrazeneca + new_latent_WT_fromV2r_astrazeneca - new_infected_WT_fromV1i_astrazeneca
            # L_VOC_V1i
            next_step[42, age1] = compartments[42, age1, i] + new_latent_VOC_fromV1i_astrazeneca + new_latent_VOC_fromV2r_astrazeneca - new_infected_VOC_fromV1i_astrazeneca
            # I_WT_V1i
            next_step[43, age1] = compartments[43, age1, i] + new_infected_WT_fromV1i_astrazeneca - new_recovered_WT_fromV1i_astrazeneca
            # I_VOC_V1i
            next_step[44, age1] = compartments[44, age1, i] + new_infected_VOC_fromV1i_astrazeneca - new_recovered_VOC_fromV1i_astrazeneca
            # R_WT_V1i
            next_step[45, age1] = compartments[45, age1, i] + new_recovered_WT_fromV1i_astrazeneca
            # R_VOC_V1i
            next_step[46, age1] = compartments[46, age1, i] + new_recovered_VOC_fromV1i_astrazeneca
            # V2r
            next_step[47, age1] = compartments[47, age1, i] + new_vaccinated_V2r_astrazeneca - new_latent_WT_fromV2r_astrazeneca - new_latent_VOC_fromV2r_astrazeneca - new_vaccinated_V2i_astrazeneca
            # V2i
            next_step[48, age1] = compartments[48, age1, i] + new_vaccinated_V2i_astrazeneca - new_latent_WT_fromV2i_astrazeneca - new_latent_VOC_fromV2i_astrazeneca
            # L_WT_V2i
            next_step[49, age1] = compartments[49, age1, i] + new_latent_WT_fromV2i_astrazeneca - new_infected_WT_fromV2i_astrazeneca
            # L_VOC_V2i
            next_step[50, age1] = compartments[50, age1, i] + new_latent_VOC_fromV2i_astrazeneca - new_infected_VOC_fromV2i_astrazeneca
            # I_WT_V2i
            next_step[51, age1] = compartments[51, age1, i] + new_infected_WT_fromV2i_astrazeneca - new_recovered_WT_fromV2i_astrazeneca
            # I_VOC_V2i
            next_step[52, age1] = compartments[52, age1, i] + new_infected_VOC_fromV2i_astrazeneca - new_recovered_VOC_fromV2i_astrazeneca
            # R_WT_V2i
            next_step[53, age1] = compartments[53, age1, i] + new_recovered_WT_fromV2i_astrazeneca
            # R_VOC_V2i
            next_step[54, age1] = compartments[54, age1, i] + new_recovered_VOC_fromV2i_astrazeneca
            ##############################

            ############### JJ ###############
            # V1r
            next_step[55, age1] = compartments[55, age1, i] - new_latent_WT_fromV1r_JJ - new_latent_VOC_fromV1r_JJ - new_vaccinated_V1i_JJ
            # V1i
            next_step[56, age1] = compartments[56, age1, i] - new_latent_WT_fromV1i_JJ - new_latent_VOC_fromV1i_JJ + new_vaccinated_V1i_JJ
            # L_WT_V1i
            next_step[57, age1] = compartments[57, age1, i] + new_latent_WT_fromV1i_JJ - new_infected_WT_fromV1i_JJ
            # L_VOC_V1i
            next_step[58, age1] = compartments[58, age1, i] + new_latent_VOC_fromV1i_JJ - new_infected_VOC_fromV1i_JJ
            # I_WT_V1i
            next_step[59, age1] = compartments[59, age1, i] + new_infected_WT_fromV1i_JJ - new_recovered_WT_fromV1i_JJ
            # I_VOC_V1i
            next_step[60, age1] = compartments[60, age1, i] + new_infected_VOC_fromV1i_JJ - new_recovered_VOC_fromV1i_JJ
            # R_WT_V1i
            next_step[61, age1] = compartments[61, age1, i] + new_recovered_WT_fromV1i_JJ
            # R_VOC_V1i
            next_step[62, age1] = compartments[62, age1, i] + new_recovered_VOC_fromV1i_JJ
            ##############################

            # update recovered
            recovered_WT[age1, i + 1] += new_recovered_WT
            recovered_VOC[age1, i + 1] += new_recovered_VOC

            ### pfizer
            recovered_WT_V1i_pfizer[age1, i + 1] += new_recovered_WT_fromV1i_pfizer
            recovered_VOC_V1i_pfizer[age1, i + 1] += new_recovered_VOC_fromV1i_pfizer
            recovered_WT_V2i_pfizer[age1, i + 1] += new_recovered_WT_fromV2i_pfizer
            recovered_VOC_V2i_pfizer[age1, i + 1] += new_recovered_VOC_fromV2i_pfizer

            ### moderna
            recovered_WT_V1i_moderna[age1, i + 1] += new_recovered_WT_fromV1i_moderna
            recovered_VOC_V1i_moderna[age1, i + 1] += new_recovered_VOC_fromV1i_moderna
            recovered_WT_V2i_moderna[age1, i + 1] += new_recovered_WT_fromV2i_moderna
            recovered_VOC_V2i_moderna[age1, i + 1] += new_recovered_VOC_fromV2i_moderna

            ### astrazeneca
            recovered_WT_V1i_astrazeneca[age1, i + 1] += new_recovered_WT_fromV1i_astrazeneca
            recovered_VOC_V1i_astrazeneca[age1, i + 1] += new_recovered_VOC_fromV1i_astrazeneca
            recovered_WT_V2i_astrazeneca[age1, i + 1] += new_recovered_WT_fromV2i_astrazeneca
            recovered_VOC_V2i_astrazeneca[age1, i + 1] += new_recovered_VOC_fromV2i_astrazeneca

            ### JJ
            recovered_WT_V1i_JJ[age1, i + 1] += new_recovered_WT_fromV1i_JJ
            recovered_VOC_V1i_JJ[age1, i + 1] += new_recovered_VOC_fromV1i_JJ

            # update incidence
            incidence_WT[age1, i + 1] += new_infected_WT + (new_infected_WT_fromV1i_pfizer +
                                                            new_infected_WT_fromV2i_pfizer +
                                                            new_infected_WT_fromV1i_moderna +
                                                            new_infected_WT_fromV2i_moderna +
                                                            new_infected_WT_fromV1i_astrazeneca +
                                                            new_infected_WT_fromV2i_astrazeneca +
                                                            new_infected_WT_fromV1i_JJ)
            incidence_VOC[age1, i + 1] += new_infected_VOC + (new_infected_VOC_fromV1i_pfizer +
                                                              new_infected_VOC_fromV2i_pfizer +
                                                              new_infected_VOC_fromV1i_moderna +
                                                              new_infected_VOC_fromV2i_moderna +
                                                              new_infected_VOC_fromV1i_astrazeneca +
                                                              new_infected_VOC_fromV2i_astrazeneca +
                                                              new_infected_VOC_fromV1i_JJ)

        # update solution at the next step
        compartments[:, :, i + 1] = next_step

    return {"compartments": compartments,
            "recovered_WT": recovered_WT,
            "recovered_VOC": recovered_VOC,
            "recovered_WT_V1i_pfizer": recovered_WT_V1i_pfizer,
            "recovered_VOC_V1i_pfizer": recovered_VOC_V1i_pfizer,
            "recovered_WT_V2i_pfizer": recovered_WT_V2i_pfizer,
            "recovered_VOC_V2i_pfizer": recovered_VOC_V2i_pfizer,
            "recovered_WT_V1i_moderna": recovered_WT_V1i_moderna,
            "recovered_VOC_V1i_moderna": recovered_VOC_V1i_moderna,
            "recovered_WT_V2i_moderna": recovered_WT_V2i_moderna,
            "recovered_VOC_V2i_moderna": recovered_VOC_V2i_moderna,
            "recovered_WT_V1i_astrazeneca": recovered_WT_V1i_astrazeneca,
            "recovered_VOC_V1i_astrazeneca": recovered_VOC_V1i_astrazeneca,
            "recovered_WT_V2i_astrazeneca": recovered_WT_V2i_astrazeneca,
            "recovered_VOC_V2i_astrazeneca": recovered_VOC_V2i_astrazeneca,
            "recovered_WT_V1i_JJ": recovered_WT_V1i_JJ,
            "recovered_VOC_V1i_JJ": recovered_VOC_V1i_JJ,
            "incidence_WT": incidence_WT,
            "incidence_VOC": incidence_VOC,
            "vaccines_per_step": vaccines_per_step,
            "vaccines_baseline_per_step": vaccines_baseline_per_step,
            "vaccines_nonbaseline_per_step": vaccines_nonbaseline_per_step}
