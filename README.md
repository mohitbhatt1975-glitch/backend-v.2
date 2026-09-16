# Ladakh Shelter Thermal Simulator — v2

Physics-informed ML surrogate model for predicting shelter indoor temperature,
solar gain, and heat loss in high-altitude cold regions (Ladakh), built for SIH.

## What changed from v1, and why

| PS requirement (verbatim) | v1 status | v2 fix |
|---|---|---|
| "Prediction of shelter inside temperature" | Present, but trained on data with no day/night correlation | `generate_data.py` now generates a realistic diurnal outside-temperature and solar curve per synthetic shelter |
| "Prediction of thermal energy generated from solar radiation" | Present (`Solar_Gain`) | Now also scaled by `Orientation_Factor`, so orientation actually affects the number |
| "Heat flow details... for a defined time period" | Single lumped `Heat_Loss` | Split into `Conduction_Loss` and `Infiltration_Loss` so you can show *where* heat is escaping |
| "Effect of openings" | Not modelled — windows only ever helped (solar), never cost heat | Windows now have their own `Window_U_Value` and lose heat by conduction; `ACH` (air changes/hour) models infiltration separately |
| "Composite multi-material" walls | Single arbitrary `r_value`, thermal mass was a fixed constant | `materials_library.py` gives you real material properties (mud brick, stone, EPS, rockwool, canvas, straw bale, steel, PCM board) and `compute_composite_wall()` turns any layer stack into the `R_Value` / `Thermal_Mass_Factor` pair the model actually uses |
| "Orientation" | Not modelled | `Orientation_Factor` input (0.3 poor .. 1.0 optimal), scales solar gain |
| "Comparative analysis with different materials... under same ambient condition" | Not implemented | New `POST /compare` endpoint: give it 2–8 configs, it runs them all against the *same* weather and ranks them |
| Credibility of synthetic dataset | No independent check | `train_model.py` reports held-out RMSE/R² per target — put this table in your PPT |
| "Software based model... user friendly, real time data, material properties" | Yes (NASA API) but fragile | NASA fetch is now async, retries backward through recent dates, rejects `-999` fill values, and input bounds are enforced by Pydantic |

## Files

- `generate_data.py` — synthetic dataset generator (run this first)
- `materials_library.py` — real material properties + composite-wall calculator
- `train_model.py` — trains 4 separate XGBoost models with monotonic constraints, saves `thermal_model_v4.pkl`
- `main.py` — FastAPI backend (`/simulate`, `/compare`, `/materials`, `/materials/composite`)
- `requirements.txt`

## Run it locally in VS Code

1. Open this folder in VS Code (`File > Open Folder`).
2. Open a terminal (`` Ctrl+` ``) and create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS/Linux
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Generate the dataset (takes ~10–20 seconds):
   ```
   python generate_data.py
   ```
   Check the printed sanity check at the end — hour 14 should show high solar/warm temp, hour 2 should show zero solar/cold temp. If that looks wrong, stop and check before training.
5. Train the model (~30–60 seconds) and note the RMSE/R² table it prints:
   ```
   python train_model.py
   ```
6. Run the API:
   ```
   uvicorn main:app --reload
   ```
7. Open `http://127.0.0.1:8000/docs` — FastAPI's interactive Swagger UI lets you test `/simulate`, `/compare`, and `/materials/composite` directly in the browser before wiring up the React frontend.

## Known simplifications (be ready for these questions)

- Orientation is a single 0.3–1.0 factor, not a true directional solar-angle calculation — good enough for comparative screening, not for architectural-grade orientation optimization.
- PCM (phase-change material) is approximated with an elevated effective specific heat, not true latent-heat phase-transition modelling.
- The baseline used for kerosene-savings comparison (canvas tent, R=0.1, ACH=4.0) is a reasonable "no passive design" reference, not a specific real product — worth stating explicitly if asked.
- This is a validated-by-holdout-split surrogate, not yet cross-validated against ANSYS or physical sensor data. That's the honest next step (see conversation notes on Phase 2 / digital twin roadmap).
