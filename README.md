# Anatomy of the first six months of COVID-19 vaccination campaign in Italy

Code for the paper "Anatomy of the first six months of COVID-19 vaccination campaign in Italy" (link to preprint: https://www.medrxiv.org/content/10.1101/2021.11.24.21266820.abstract)

Abstract: We analyze the effectiveness of the first six months of vaccination campaign against SARS-CoV-2 in Italy by using a computational epidemic model which takes into account demographic, mobility, vaccines data, as well as estimates of the introduction and spreading of the more transmissible Alpha variant. We consider six sub-national regions and study the effect of vaccines in terms of number of averted deaths, infections, and reduction in the Infection Fatality Rate (IFR) with respect to counterfactual scenarios with the actual non-pharmaceuticals interventions but no vaccine administration. Furthermore, we compare the effectiveness in counterfactual scenarios with different vaccines allocation strategies and vaccination rates. Our results show that, as of 2021/07/05, vaccines averted 29,350 (IQR: [16,454-42,826]) deaths and 4,256,332 (IQR: [1,675,564-6,980,070]) infections and a new pandemic wave in the country. During the same period, they achieved a -22.2% (IQR: [-31.4%; -13.9%]) IFR reduction. We show that a campaign that would have strictly prioritized age groups at higher risk of dying from COVID-19, besides frontline workers and the fragile population, would have implied additional benefits both in terms of avoided fatalities and reduction in the IFR. Strategies targeting the most active age groups would have prevented a higher number of infections but would have been associated with more deaths. Finally, we study the effects of different vaccination intake scenarios by rescaling the number of available doses in the time period under study to those administered in other countries of reference. The modeling framework can be applied to other countries to provide a mechanistic characterization of vaccination campaigns worldwide.


### Instructions
This repository contains data and code to run the model for the paper "Anatomy of the first six months of COVID-19 vaccination campaign in Italy":
- ```basins``` contains demographic, epidemiological, and NPIs data for different basins. This data is already pre-processed and is ready to be inputed in the model
- ```data-scripts``` contains the notebooks necessary to download and preprocess the vaccinations, demographic, and NPIs data
- ```models``` contains the implementation of the model
- ```simulation``` contains an example notebook to run the model 
- ```calibration``` contains a sample file that shows how the calibration is done
