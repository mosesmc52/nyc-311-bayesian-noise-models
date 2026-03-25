
![Screenshot of project](images/ODW-2026-banner.png)

## NYC 311 Noise Complaint Modeling (Bayesian)
Bayesian hierarchical models for forecasting NYC 311 noise complaints and quantifying urban stress.

[Presentation](https://docs.google.com/presentation/d/1vLMbCHvBLH5sb2UQLPi-YPjA2fshaUcXq7H0l-lGHhA/edit?usp=sharing)


### Kepler Maps

This zip file contains the Kepler maps used in the presentation:

- **Slide 24 – Where Are Complaints Concentrated Across NYC?**  
  - `03_01.html`

- **Slide 26 – Where Do Prediction Errors Occur?**  
  - `03_02.html`

[Download Kepler Maps](https://drive.google.com/file/d/1KNrjP8SRHJfdn7d5JvioaDCx-55OHC1w/view?usp=sharing)


### Project Overview

This project models and predicts NYC 311 noise complaints using Bayesian hierarchical models built with PyMC.


The goal is to:

- Understand spatial and temporal patterns in noise complaints

- Quantify uncertainty in predictions

- Demonstrate the impact of partial pooling (hierarchical modeling)

Key modeling approaches:

- Baseline Poisson model (unpooled)

- Hierarchical Poisson (PUMA ↔ NTA pooling)

### Why This Project Matters

311 data provides a real-time signal of urban stress and quality of life.

This project demonstrates how Bayesian methods can:

- Turn noisy civic data into structured insight

- Quantify uncertainty (not just point estimates)

- Support better decision-making at the neighborhood level

### Project Scope

This project focuses on noise complaints from the summer including June - August.

- Training Data 2021-2024
- Test Data: 2025

### Key Findings

- Hierarchical models outperform unpooled models in predictive accuracy (LOO / WAIC)
- The model tends to overestimate high complaint counts
- Credible intervals are too narrow → the model is **overconfident**
- Suggests the Poisson assumption is too restrictive (Negative Binomial likely better)


---
### Repo Organization
```bash
nyc-311-bayesian-noise-model/
│
├── data/                  # Raw and processed datasets
│   ├── raw/
│   ├── processed/
│
├── images/
│   ├── ODW-2026-banner.png
│
├── notebooks/             # Jupyter notebooks (EDA + modeling)
│   ├── 0_get_that_data.ipynb
│   ├── 01_explore_noise_patterns.ipynb
│   ├── 02_fully_pooled_citywide_baseline.ipynb
│   ├── 02_unpooled_nta_neighborhood_differences.ipynb
│   ├── 03_partially_pooled_nta_puma_grouped.ipynb
│   ├── 04_testing_predictions_on_2025.ipynb
│
├── queries/
│   ├── nyc_311_noise.soql
│
├── scripts/
│   ├── aggregate/
│   │   ├── build_noise_counts_with_lookup.py
│   ├── ingest/
│   │   ├── download_nta_geojson.py
│   │   ├── download_nyc_311_noise.py
│   │   ├── download_pumas_geojson.py
│   ├── lookups/   
│   │   ├──  build_puma_nta_lookup.py
├── .env.example
├── pyproject.toml
├── README.md
```

---
### System dependencies

#### Python Environment
```bash
poetry install
```

#### Environment Variables

This project uses environment variables for accessing the NYC Open Data API.

1. Copy the example file:

```bash
cp .env.example .env
```

2. Add your credentials to .env:

```env
NYC_API_KEY=your_api_key_here
NYC_API_SECRET=your_api_secret_here
```

[Instructions for retrieving API Key](
https://support.socrata.com/hc/en-us/articles/210138558-Generating-App-Tokens-and-API-Keys)

---


### How to Run

```bash
# install dependencies
poetry install
```


```bash
# launch notebooks
poetry run jupyter lab
```

Geographic data is visualized using [Kepler.gl](https://kepler.gl/) for interactive spatial exploration.

---

### Datasets
Raw data is ingested via scripts in `scripts/ingest/`

- [NYC PUMA Boundaries (Census 2020)](https://www2.census.gov/geo/tiger/TIGER2020PL/STATE/36_NEW_YORK/)

  → Used for aggregating complaints at the PUMA level

- [NYC NTA Boundaries (Neighborhood Tabulation Areas)](https://www.nyc.gov/content/planning/pages/resources/datasets/neighborhood-tabulation)

  → Used for hierarchical modeling and partial pooling

- [NYC 311 Service Requests (Noise Complaints)](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2020-to-Present/erm2-nwe9/explore/query/SELECT%0A%20%20%60unique_key%60%2C%0A%20%20%60created_date%60%2C%0A%20%20%60closed_date%60%2C%0A%20%20%60agency%60%2C%0A%20%20%60agency_name%60%2C%0A%20%20%60complaint_type%60%2C%0A%20%20%60descriptor%60%2C%0A%20%20%60descriptor_2%60%2C%0A%20%20%60location_type%60%2C%0A%20%20%60incident_zip%60%2C%0A%20%20%60incident_address%60%2C%0A%20%20%60street_name%60%2C%0A%20%20%60cross_street_1%60%2C%0A%20%20%60cross_street_2%60%2C%0A%20%20%60intersection_street_1%60%2C%0A%20%20%60intersection_street_2%60%2C%0A%20%20%60address_type%60%2C%0A%20%20%60city%60%2C%0A%20%20%60landmark%60%2C%0A%20%20%60facility_type%60%2C%0A%20%20%60status%60%2C%0A%20%20%60due_date%60%2C%0A%20%20%60resolution_description%60%2C%0A%20%20%60resolution_action_updated_date%60%2C%0A%20%20%60community_board%60%2C%0A%20%20%60council_district%60%2C%0A%20%20%60police_precinct%60%2C%0A%20%20%60bbl%60%2C%0A%20%20%60borough%60%2C%0A%20%20%60x_coordinate_state_plane%60%2C%0A%20%20%60y_coordinate_state_plane%60%2C%0A%20%20%60open_data_channel_type%60%2C%0A%20%20%60park_facility_name%60%2C%0A%20%20%60park_borough%60%2C%0A%20%20%60vehicle_type%60%2C%0A%20%20%60taxi_company_borough%60%2C%0A%20%20%60taxi_pick_up_location%60%2C%0A%20%20%60bridge_highway_name%60%2C%0A%20%20%60bridge_highway_direction%60%2C%0A%20%20%60road_ramp%60%2C%0A%20%20%60bridge_highway_segment%60%2C%0A%20%20%60latitude%60%2C%0A%20%20%60longitude%60%2C%0A%20%20%60location%60%0AWHERE%20caseless_contains%28%60complaint_type%60%2C%20%22noise%22%29%0AORDER%20BY%20%60created_date%60%20DESC%20NULL%20FIRST%0ASEARCH%20%22noise%22/page/filter)

  → Primary dataset of complaint events filtered for noise-related incidents

### Notebooks

**0. Download the Data (`0_get_that_data.ipynb`)**

- Download the raw datasets from their source systems
- Clean and preprocess the data for analysis
- Build the geographic lookup tables used in later notebooks

**1. Exploratory Data Analysis (`01_explore_noise_patterns.ipynb`)**

- Review summary statistics for NYC noise complaints
- Explore complaint categories and temporal patterns, including weekday effects
- Rank the quietest and noisiest neighborhoods

**2. Model 1: Fully Pooled (`02_model1_fully_pooled_citywide_baseline.ipynb`)**

- Fit a citywide fully pooled baseline over complaint category and weekday
- Establish the simplest benchmark before introducing geography-specific structure
- Visualize pooled weekday patterns across complaint groups

**3. Model 2: Unpooled NTA (`02_model2_unpooled_nta_neighborhood_differences.ipynb`)**

- Compare raw city-relative intensity across neighborhoods
- Build an unpooled Poisson model as a baseline
- Model complaint counts at the NTA level

**4. Model 3: Partially Pooled NTA | PUMA Grouped (`03_model3_partially_pooled_nta_puma_grouped.ipynb`)**

- Build a hierarchical model with partial pooling
- Use an NTA-level baseline with PUMA-level deviations
- Compare the unpooled NTA model against the partially pooled NTA | PUMA grouped model

**5. Model Evaluation (`04_testing_predictions_on_2025.ipynb`)**

- Evaluate the partially pooled NTA | PUMA grouped model on 2025 test data
- Use calibration plots to assess predictive performance and identify areas for improvement
- Perform coverage analysis using credible intervals


### Results Summary

- Hierarchical models outperform unpooled models in predictive accuracy (LOO / WAIC)
- The model tends to overestimate high complaint counts in high-activity areas
- Credible intervals are too narrow → the model is **overconfident**
- This suggests the Poisson likelihood is too restrictive for this problem

### Opportunities for Improvement

- Replace Poisson with a **Negative Binomial likelihood** to handle overdispersion
- Incorporate covariates:
  - Weather data
  - Holidays and events
