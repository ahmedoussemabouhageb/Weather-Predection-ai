# Mediterranean Sea Weather Forecasting with ConvLSTM

A deep learning project for short-term spatiotemporal weather forecasting over the Mediterranean Sea basin. The model predicts key atmospheric and oceanic variables using a ConvLSTM encoder architecture trained on ERA5 reanalysis data spanning 2019 to 2024.

---

## Overview

This project tackles the challenge of predicting weather conditions across the Mediterranean Sea by learning from historical gridded reanalysis data. Instead of treating each location independently, the model captures spatial patterns across the entire domain simultaneously, making it well-suited for modeling ocean-atmosphere interactions.

The pipeline covers everything from raw GRIB data ingestion to model training, evaluation, and forecast generation.

---

## Predicted Variables

The model forecasts the following variables at a 5-hour horizon:

- **SWH** — Significant Wave Height (metres)
- **SST** — Sea Surface Temperature (Kelvin)
- **T2M** — 2-metre Air Temperature (Kelvin)
- **MSL** — Mean Sea Level Pressure (hPa)
- **U10** — 10-metre U-component of Wind (m/s)
- **V10** — 10-metre V-component of Wind (m/s)
- **Wind Speed** — Derived magnitude from U10 and V10 (m/s)

---

## Data

- **Source:** ERA5 reanalysis GRIB files (ECMWF)
- **Years covered:** 2019, 2022, 2023, 2024
- **Spatial domain:** Latitude 30°N to 46°N, Longitude 6°W to 37°E
- **Temporal resolution:** Hourly
- **Spatial resolution:** 0.25° (atmospheric variables) and 0.5° (ocean wave variables, interpolated to 0.25°)

---

## Architecture

The model follows an encoder-decoder design built around ConvLSTM cells, which extend standard LSTM by replacing matrix multiplications with convolutions. This allows the model to capture both temporal dynamics and spatial structure at each time step.

**Encoder**

Two stacked ConvLSTM cells process a 48-hour input sequence frame by frame, producing a latent spatial state that summarises recent atmospheric and oceanic conditions across the domain.

**Reconstruction branch**

A lightweight convolutional head reconstructs the current timestep from the latent state. This acts as a regulariser and encourages the encoder to capture meaningful spatial features, not just temporal transitions.

**Forecasting branch**

A deeper convolutional head predicts the residual change (delta) between the last observed timestep and the target horizon. The final forecast is obtained by adding this delta to the last observed frame, which stabilises training by keeping the model focused on changes rather than absolute values.

Each variable is trained as an independent model. This keeps memory usage manageable and allows each variable to specialise without interference from others.

---

## Feature Engineering

Beyond the raw variables, the following derived features are computed and used as inputs:

- Wind speed magnitude from U10 and V10 components
- Pressure tendency (first-order time difference of MSL)
- Day-of-year encoded as sine and cosine to capture seasonal cycles
- Hour-of-day encoded as sine and cosine to capture diurnal cycles

---

## Land Masking

Ocean-only variables like SST and SWH are undefined over land. To prevent the model from learning spurious land signals, a land mask is applied during both training and evaluation. Loss is computed exclusively over ocean grid points. Two separate masks are used: one derived from SST NaN patterns (0.25° resolution) and one for SWH which uses a union of SST and wave-grid NaN patterns to account for interpolation artefacts.

---

## Training Setup

- **Input sequence length:** 48 hourly timesteps
- **Forecast horizon:** 5 hours ahead
- **Train / validation / test split:** 70% / 15% / 15% (chronological, no shuffling across splits)
- **Sampling stride:** Every 6 hours to reduce temporal redundancy
- **Loss function:** Masked MSE over ocean points only, combining reconstruction loss (weight 0.1) and forecast loss (weight 0.9)
- **Optimiser:** Adam with weight decay
- **Learning rate schedule:** ReduceLROnPlateau with patience of 2 epochs
- **Early stopping:** Patience of 4 epochs
- **Gradient clipping:** Max norm of 1.0

---

## Normalisation

Each variable is normalised to zero mean and unit variance using statistics computed from the training set only, preventing data leakage. Normalisation statistics are saved to disk and used during evaluation to convert predictions back to physical units.

---

## Evaluation Metrics

Models are evaluated on the held-out test set in original physical units after denormalisation. The following metrics are reported for each variable:

- **RMSE** — Root Mean Squared Error
- **MAE** — Mean Absolute Error
- **Pearson Correlation** — Linear correlation between predictions and targets
- **Strict accuracy** — Percentage of predictions within 0.5 × RMSE of the target
- **Fair accuracy** — Percentage within 1.0 × RMSE
- **Loose accuracy** — Percentage within 2.0 × RMSE

---

## Project Structure

```
.
├── train.py            Main training and evaluation script
├── normstats.pkl       Saved normalisation statistics
├── best-swh.pt         Best model checkpoint for SWH
├── best-sst.pt         Best model checkpoint for SST
├── best-t2m.pt         Best model checkpoint for T2M
├── best-msl.pt         Best model checkpoint for MSL
├── best-u10.pt         Best model checkpoint for U10
├── best-v10.pt         Best model checkpoint for V10
```

---

## Requirements

- Python 3.9 or later
- PyTorch
- xarray
- cfgrib
- numpy
- scipy
- pandas

---

## Author

Oussema Bouhajeb
