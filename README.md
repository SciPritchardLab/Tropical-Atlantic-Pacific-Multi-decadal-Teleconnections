# Diagnosing Biases in Tropical Atlantic-Pacific Multi-Decadal Teleconnections Across CMIP6 and E3SM Models

This repository contains code and data for analyzing biases in tropical Atlantic-Pacific multi-decadal teleconnections across CMIP6 and E3SM climate models.

## Overview

This study investigates how current climate models represent interactions between the tropical Atlantic and Pacific Oceans on decadal timescales. These basin interactions play a crucial role in global climate variability, yet most climate models show systematic biases in simulating them. As shown in Figure 1, observations (ERSST and HadISST) reveal that warming in the North Tropical Atlantic typically leads to cooling in the Central-Eastern Pacific. However, the CMIP6 multi-model mean shows the opposite response - Pacific warming during Atlantic warming periods.

![Correlation Coefficient of Low Pass Filtered SSTA on NTA Index (1950-2015)](results/figures/Fig1_Correlation_Coefficient_of_Low_Pass_Filtered_SSTA_on_NTA_Index_(1950-2015).png)

Through comprehensive analysis of 27 CMIP6 models and two configurations of the DOE Energy Exascale Earth System Model (E3SM), we:

1. Demonstrate that most CMIP6 models show Pacific-driven teleconnections that contradict observed patterns
2. Identify four high-skill CMIP6 models and E3SMv2 that better capture the observed interactions while showing local biases
3. Evaluate how different treatments of cloud-scale processes influence these basin interactions

This repository provides the analysis code and workflows used to diagnose these systematic biases.

## Key Features

- Analysis of tropical Atlantic-Pacific teleconnections in CMIP6 models
- Evaluation of E3SM model performance
- Diagnosis of systematic biases and their potential sources
- Comparison with observational datasets

## Data

The analysis uses:
- CMIP6 model output
- E3SM simulation data  
- Observational/Reanalysis datasets

## Code Structure

- `data/`: Input data and processed results
- `notebooks/`: Jupyter notebooks for analysis
- `results/`: Results from the analysis

## Requirements

- Python 3.9+
- Required packages listed in `requirements.txt`

## Usage

1. Clone the repository
2. Install dependencies
3. Run notebooks in `notebooks/`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or feedback, please open an issue in this repository.
