# libraries
import numpy as np
import pandas as pd
import pickle as pkl
import os


class Basin:
    """
    This class create a basin object
    """

    def __init__(self, name, path_to_data, hemisphere):

        # name and hemisphere
        self.name = name
        self.hemisphere = hemisphere

        # contacts matrix
        self.contacts_work = np.load(os.path.join(path_to_data + name + "/contacts-matrices/work.npz"))["arr_0"]
        self.contacts_school = np.load(path_to_data + name + "/contacts-matrices/school.npz")["arr_0"]
        self.contacts_home = np.load(path_to_data + name + "/contacts-matrices/home.npz")["arr_0"]
        self.contacts_community = np.load(path_to_data + name + "/contacts-matrices/community.npz")["arr_0"]

        # demographic
        self.Nk = pd.read_csv(path_to_data + name + "/demographic/Nk.csv").value.values

        # epidemiological data
        self.epi_data = pd.read_csv(path_to_data + name + "/epidemiological-data/epi_data.csv")
        self.epi_data["Date"] = pd.to_datetime(self.epi_data ["Date"])

        # contacts reductions
        self.school_reductions = pd.read_csv(path_to_data + name + "/restrictions/school.csv")
        self.school_reductions["date"] = pd.to_datetime(self.school_reductions["date"])
        self.work_reductions = pd.read_csv(path_to_data + name + "/restrictions/work.csv")
        self.comm_reductions = pd.read_csv(path_to_data + name + "/restrictions/other_loc.csv")
        self.color = pd.read_csv(path_to_data + name + "/restrictions/color.csv")
        self.color["date"] = pd.to_datetime(self.color["date"])
        

        # initial conditions
        self.init_conditions = pd.read_csv(path_to_data + name + "/initial-conditions/run_ic_20200901.csv")

        # importations
        self.importations = np.load(path_to_data + name + "/importations/R0multi_starting_date2020-09-13.npz")["arr_0"]

        # vaccinations
        ### PFIZER
        with open(path_to_data + name + "/vaccinations/vaccinations_pfizer.pkl", "rb") as file:
            self.vaccinations_pfizer = pkl.load(file)

        with open(path_to_data + name + "/vaccinations/vaccinations_baseline_pfizer.pkl", "rb") as file:
            self.vaccinations_baseline_pfizer = pkl.load(file)

        ### MODERNA
        with open(path_to_data + name + "/vaccinations/vaccinations_moderna.pkl", "rb") as file:
            self.vaccinations_moderna = pkl.load(file)

        with open(path_to_data + name + "/vaccinations/vaccinations_baseline_moderna.pkl", "rb") as file:
            self.vaccinations_baseline_moderna = pkl.load(file)

        ### ASTRAZENECA
        with open(path_to_data + name + "/vaccinations/vaccinations_astrazeneca.pkl", "rb") as file:
            self.vaccinations_astrazeneca = pkl.load(file)

        with open(path_to_data + name + "/vaccinations/vaccinations_baseline_astrazeneca.pkl", "rb") as file:
            self.vaccinations_baseline_astrazeneca = pkl.load(file)

        ### JJ
        with open(path_to_data + name + "/vaccinations/vaccinations_JJ.pkl", "rb") as file:
            self.vaccinations_JJ = pkl.load(file)

        with open(path_to_data + name + "/vaccinations/vaccinations_baseline_JJ.pkl", "rb") as file:
            self.vaccinations_baseline_JJ = pkl.load(file)
