# libraries
from functions import compute_contacts, get_initial_conditions
from stochastic_SEIRD import simulate
from Basin import Basin
import pandas as pd
import numpy as np
from datetime import datetime
import uuid
import argparse
import warnings
warnings.filterwarnings("ignore")

# n. of compartments and age groups
ncomp = 63
nage = 10

    
# fit dates
start_date = datetime(2020, 9, 1)
end_date = datetime(2021, 7, 5)


def wmape(arr1, arr2):
    """
    Weighted Mean Absolute Percentage Error (WMAPE)
    """
    return np.sum(np.abs(arr1 - arr2)) / np.sum(np.abs(arr1))


def calibration(basin_name, nsim, threshold, step_save, psi, hemisphere, vaccine):
    """
    This function runs the calibration for a given basin
        :param basin_name: name of the basin
        :param nsim: number of total simulations
        :param threshold: ABC tolerance
        :param step_save: n. of runs of intermediate save
        :param psi: increased transmissibility of of 2nd variant
        :param hemisphere: hemisphere of basin (0: north, 1: tropical, 2; south)
        :param vaccine: vaccination strategy
        :return: 0
    """

    # create Basin object
    basin = Basin(basin_name, "../basins/", hemisphere)

    # pre-compute contacts matrices over time
    Cs, dates = compute_contacts(basin, start_date, end_date)

    # get real deaths (first month excluded for the delay Delta)
    real_deaths = basin.epi_data.loc[(basin.epi_data["Date"] >= datetime(2020, 9, 28)) &
                                     (basin.epi_data["Date"] < end_date)]["daily_Deaths"]

    # run calibration
    params = []
    for k in range(nsim):
        print(k)

        if k % 10000 == 0:
            print(k)

        # sample parameters (R0, Delta, initial conditions, seasonality, importations)
        R0 = np.random.uniform(0.8, 2.0)
        Delta = np.random.randint(12, 25)
        rnd_run = np.random.randint(0, basin.init_conditions.shape[0])
        seasonality_min = np.random.uniform(0.5, 1.0)
        run_id = np.random.randint(0, basin.importations.shape[0])
        imp_run = basin.importations[run_id]

        # initial conditions
        initial_conditions = get_initial_conditions(basin, rnd_run)

        # simulate and resample deaths weekly
        results = simulate(Cs, psi, basin.Nk, initial_conditions, imp_run,
                          {"vaccinations_pfizer": basin.vaccinations_pfizer,
                           "vaccinations_moderna": basin.vaccinations_moderna,
                           "vaccinations_astrazeneca": basin.vaccinations_astrazeneca,
                           "vaccinations_JJ": basin.vaccinations_JJ,
                           "vaccinations_baseline_pfizer": basin.vaccinations_baseline_pfizer,
                           "vaccinations_baseline_moderna": basin.vaccinations_baseline_moderna, 
                           "vaccinations_baseline_astrazeneca": basin.vaccinations_baseline_astrazeneca,
                           "vaccinations_baseline_JJ": basin.vaccinations_baseline_JJ},
                           R0, Delta, dates, seasonality_min, basin.hemisphere, vaccine)

        df_deaths = pd.DataFrame(data={"real_deaths": real_deaths.values, "sim_deaths": results["deaths_TOT"].sum(axis=0)[27:]}, index=dates[27:])
        df_deaths = df_deaths.resample("W").sum()

        # accept/reject
        err = wmape(df_deaths["real_deaths"].values, df_deaths["sim_deaths"].values)
        if err < threshold:
            params.append([R0, Delta, rnd_run, run_id, seasonality_min, err, df_deaths.sim_deaths.values])

        # save file every step_save steps 
        if k % step_save == 0 and k != 0:
            print(k)
            # create unique file name and save
            unique_filename = str(uuid.uuid4())
            np.savez_compressed("./posterior_samples/" + basin.name + \
                                "_thr" + str(threshold) + "_vaccine" + str(vaccine) + \
                                unique_filename + ".npz",
                                params)
            params = []

    return 0


if __name__ == "__main__":
    
    # parse basin name
    parser = argparse.ArgumentParser(description='Calibration ABC')
    parser.add_argument('--basin', type=str, help='name of the basin')
    parser.add_argument('--nsim', type=int, help='number of simulations')
    parser.add_argument('--th', type=float, help='threshold')
    parser.add_argument('--step', type=int, help='step save')
    parser.add_argument('--psi', type=float, help='increased transmission VOC')
    parser.add_argument('--vaccine', type=str, help='vaccine strategy')

    args = parser.parse_args()
    calibration(basin_name="Italy-" + args.basin,
                nsim=args.nsim,
                threshold=args.th,
                step_save=args.step,
                psi=args.psi,
                hemisphere=0,
                vaccine=args.vaccine)