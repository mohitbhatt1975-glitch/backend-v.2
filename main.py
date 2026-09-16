"""
main.py  (v2)

Changes vs the original:

1. Pydantic Field bounds on every input -- matches the ranges the model was
   actually trained on (see generate_data.py BOUNDS), so out-of-range or
   malformed requests get a clean 422 instead of reaching the model.
2. NASA fetch is async (httpx, non-blocking), wrapped in proper error
   handling, walks backward from today to find the most recent date with
   complete data (NASA POWER hourly typically lags a few days), and rejects
   -999 fill-value hours instead of silently feeding them to the model.
3. Sustainability math (kerosene saved / CO2 prevented) is now a genuine
   comparison: actual required heater power vs a baseline "no passive
   design" shelter (canvas tent, minimal insulation, high infiltration) --
   simulated with the SAME model -- instead of double counting solar gain
   that just leaks back out.
4. New /compare endpoint: batch-simulates N configs against the same
   weather and ranks them -- this is what the PS's "comparative analysis
   with different materials... to predict the most efficient combination"
   actually asks for.
5. New /materials endpoint exposes the real-material composite-wall
   calculator from materials_library.py.
"""

import math
from datetime import date, timedelta
from typing import List, Optional

import httpx
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from materials_library import MATERIALS, compute_composite_wall

app = FastAPI(title="Ladakh Shelter Thermal Simulator", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Load the model bundle once at startup ----
MODEL_PATH = "thermal_model_v4.pkl"
try:
    bundle = joblib.load(MODEL_PATH)
    MODELS = bundle["models"]
    FEATURES = bundle["features"]
except FileNotFoundError:
    MODELS = None
    FEATURES = None
    print(f"WARNING: {MODEL_PATH} not found. Run generate_data.py then train_model.py first.")

# Baseline "no passive design" reference shelter, used to compute genuine
# fuel savings (canvas skin, poor insulation, high infiltration).
BASELINE_OVERRIDES = {
    "r_value": 0.1,
    "thermal_mass_factor": 2000.0,
    "window_u_value": 5.8,
    "ach": 4.0,
    "orientation_factor": 0.3,
}

KEROSENE_ENERGY_DENSITY_J_PER_L = 35_000_000
KEROSENE_CO2_KG_PER_L = 2.5


class ShelterInput(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    shape_code: int = Field(..., ge=1, le=5, description="1=box 2=dome 3=A-frame 4=vertical cylinder 5=half-cylinder")
    length: float = Field(..., gt=0, le=15)
    width: float = Field(..., gt=0, le=15)
    height: float = Field(..., gt=0, le=6)
    r_value: float = Field(..., gt=0, le=4.0, description="m^2*K/W; use /materials/composite to derive from real layers")
    thermal_mass_factor: float = Field(..., gt=0, le=700000.0, description="J/(m^2*K) of wall area")
    window_area: float = Field(..., ge=0, le=15)
    window_u_value: float = Field(2.8, ge=0.5, le=7.0)
    orientation_factor: float = Field(0.7, ge=0.3, le=1.0)
    ach: float = Field(1.5, ge=0.2, le=6.0, description="air changes per hour")
    occupants: int = Field(0, ge=0, le=30)
    initial_temp: float = Field(10.0, ge=-35, le=50)
    date: Optional[str] = Field(None, description="YYYY-MM-DD; defaults to most recent day with complete NASA data")


class CompareRequest(BaseModel):
    configs: List[ShelterInput] = Field(..., min_length=1, max_length=8)
    labels: Optional[List[str]] = None


NASA_PARAMS = "T2M,ALLSKY_SFC_SW_DWN,WS10M"
NASA_LAG_DAYS = 3        # POWER meteorology trails real time by ~2-3 days
NASA_WINDOW_DAYS = 17    # how far back a single scan reaches


def _group_hours_by_day(params: dict) -> dict:
    """Turn NASA's flat {YYYYMMDDHH: value} maps into {YYYYMMDD: {param: [24 values]}}."""
    days: dict = {}
    for key in ("T2M", "ALLSKY_SFC_SW_DWN", "WS10M"):
        for stamp, value in params[key].items():
            day = stamp[:8]
            days.setdefault(day, {"T2M": [], "ALLSKY_SFC_SW_DWN": [], "WS10M": []})
            days[day][key].append((stamp, value))
    return {
        day: {k: [v for _, v in sorted(pairs)] for k, pairs in series.items()}
        for day, series in days.items()
    }


def _pick_most_recent_valid_day(days: dict) -> Optional[dict]:
    """Newest day in the window with 24 complete, non-fill hours for all three params."""
    for day in sorted(days.keys(), reverse=True):
        series = days[day]
        temps = series["T2M"]
        solar = series["ALLSKY_SFC_SW_DWN"]
        wind = series["WS10M"]
        if len(temps) < 24 or len(solar) < 24 or len(wind) < 24:
            continue
        if any(v <= -900 for v in temps[:24] + solar[:24] + wind[:24]):
            continue
        return {"temps": temps[:24], "solar": solar[:24], "wind": wind[:24], "date_used": day}
    return None


async def fetch_nasa_weather(lat: float, lon: float, requested_date: Optional[str]) -> dict:
    """
    Fetch 24 hours of T2M / ALLSKY_SFC_SW_DWN / WS10M from NASA POWER.

    NASA POWER publishes its two products on very different schedules: the
    meteorology (T2M, WS10M -- MERRA-2) is current to within a few days, but
    the hourly solar product (ALLSKY_SFC_SW_DWN -- SYN1DEG) lags by months.
    Recent days therefore come back with valid temperature/wind and -999 fill
    values for every solar hour.

    So: scan a recent window in ONE range request; if no day in it has
    complete data, fall back to the same calendar window in previous years,
    which is fully populated and seasonally equivalent. The day actually used
    is reported back to the caller, and seasonal fallbacks are flagged rather
    than passed off as current weather.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        if requested_date:
            ds = requested_date.replace("-", "")
            windows = [(ds, ds, False)]
        else:
            today = date.today()
            recent_end = today - timedelta(days=NASA_LAG_DAYS)
            recent_start = today - timedelta(days=NASA_WINDOW_DAYS)
            windows = [(recent_start.strftime("%Y%m%d"), recent_end.strftime("%Y%m%d"), False)]
            for years_back in (1, 2, 3):
                shift = timedelta(days=365 * years_back)
                windows.append((
                    (recent_start - shift).strftime("%Y%m%d"),
                    (recent_end - shift).strftime("%Y%m%d"),
                    True,
                ))

        last_error = None
        for start_ds, end_ds, is_fallback in windows:
            url = (
                "https://power.larc.nasa.gov/api/temporal/hourly/point?"
                f"parameters={NASA_PARAMS}&community=RE&"
                f"longitude={lon}&latitude={lat}&start={start_ds}&end={end_ds}&format=JSON"
            )
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                params = resp.json()["properties"]["parameter"]
                found = _pick_most_recent_valid_day(_group_hours_by_day(params))
                if found is not None:
                    found["is_seasonal_fallback"] = is_fallback
                    return found
                last_error = f"No day with complete, non-fill data in {start_ds}-{end_ds}"
            except (httpx.HTTPError, KeyError, ValueError) as e:
                last_error = str(e)
                continue

        raise HTTPException(
            status_code=502,
            detail=f"Could not fetch complete NASA weather data near ({lat},{lon}). Last error: {last_error}",
        )


def predict_hour(features_row: pd.DataFrame) -> dict:
    return {target: float(model.predict(features_row)[0]) for target, model in MODELS.items()}


def simulate_config(cfg: ShelterInput, weather: dict, overrides: Optional[dict] = None) -> dict:
    """Run the 24-hour recursive simulation for one shelter config against one weather profile."""
    params = cfg.model_dump()
    if overrides:
        params.update(overrides)

    current_temp = params["initial_temp"]
    length, width, height = params["length"], params["width"], params["height"]
    shape_code = params["shape_code"]

    # Recompute shape geometry exactly as generate_data.py does, so Shape_Factor is consistent
    if shape_code == 1:
        envelope_area = (2 * length * height) + (2 * width * height) + (length * width)
    elif shape_code == 2:
        r = length / 2.0
        envelope_area = 2 * math.pi * (r ** 2)
    elif shape_code == 3:
        slant = math.sqrt((width / 2) ** 2 + height ** 2)
        envelope_area = (2 * slant * length) + (width * height)
    elif shape_code == 4:
        r = length / 2.0
        envelope_area = (2 * math.pi * r * height) + (math.pi * (r ** 2))
    else:
        r = width / 2.0
        envelope_area = (math.pi * r * length) + (math.pi * (r ** 2))

    if shape_code == 1:
        volume = length * width * height
    elif shape_code == 2:
        r = length / 2.0
        volume = (2 / 3) * math.pi * (r ** 3)
    elif shape_code == 3:
        volume = 0.5 * width * length * height
    elif shape_code == 4:
        r = length / 2.0
        volume = math.pi * (r ** 2) * height
    else:
        r = width / 2.0
        volume = 0.5 * math.pi * (r ** 2) * length

    shape_factor = envelope_area / volume

    hourly = {"inside_temp": [], "solar_gain": [], "conduction_loss": [], "infiltration_loss": []}
    required_heater_joules = 0.0

    for hour in range(24):
        row = pd.DataFrame([[
            shape_code, length, width, height, shape_factor,
            params["r_value"], params["thermal_mass_factor"],
            params["window_area"], params["window_u_value"],
            params["orientation_factor"], params["ach"], params["occupants"],
            weather["temps"][hour], weather["solar"][hour], weather["wind"][hour],
            current_temp,
        ]], columns=FEATURES)

        preds = predict_hour(row)
        next_temp = preds["Inside_Temp"]
        solar_gain = preds["Solar_Gain"]
        conduction_loss = preds["Conduction_Loss"]
        infiltration_loss = preds["Infiltration_Loss"]

        hourly["inside_temp"].append(round(next_temp, 2))
        hourly["solar_gain"].append(round(solar_gain, 1))
        hourly["conduction_loss"].append(round(conduction_loss, 1))
        hourly["infiltration_loss"].append(round(infiltration_loss, 1))

        # Heater power actually needed this hour to hold comfort (only counts when losses exceed free gains)
        internal_heat = params["occupants"] * 100.0
        net_free_heat = solar_gain + internal_heat - conduction_loss - infiltration_loss
        heater_needed_watts = max(0.0, -net_free_heat)
        required_heater_joules += heater_needed_watts * 3600

        current_temp = next_temp

    return {
        "hourly": hourly,
        "required_heater_joules": required_heater_joules,
        "final_temp": current_temp,
    }


def compute_sustainability(actual_joules: float, baseline_joules: float) -> dict:
    saved_joules = max(0.0, baseline_joules - actual_joules)
    liters_saved = saved_joules / KEROSENE_ENERGY_DENSITY_J_PER_L
    co2_prevented = liters_saved * KEROSENE_CO2_KG_PER_L
    return {
        "kerosene_saved_liters": round(liters_saved, 2),
        "co2_emissions_prevented_kg": round(co2_prevented, 2),
    }


def compute_livability(inside_temps: List[float]) -> int:
    score = 100.0
    for t in inside_temps:
        if t < 15.0:
            score -= 1.5
        elif t > 25.0:
            score -= 1.0
    return max(0, round(score))


@app.get("/materials")
def list_materials():
    return MATERIALS


class CompositeWallRequest(BaseModel):
    layers: List[List] = Field(..., description='e.g. [["mud_brick_adobe", 0.25], ["eps_insulation", 0.05]]')
    envelope_area: float = Field(..., gt=0)


@app.post("/materials/composite")
def materials_composite(req: CompositeWallRequest):
    try:
        layer_tuples = [(name, float(thickness)) for name, thickness in req.layers]
        return compute_composite_wall(layer_tuples, req.envelope_area)
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/simulate")
async def run_simulation(data: ShelterInput):
    if MODELS is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded. Run train_model.py to produce {MODEL_PATH}.")

    weather = await fetch_nasa_weather(data.lat, data.lon, data.date)
    actual = simulate_config(data, weather)
    baseline = simulate_config(data, weather, overrides=BASELINE_OVERRIDES)

    sustainability = compute_sustainability(actual["required_heater_joules"], baseline["required_heater_joules"])
    livability = compute_livability(actual["hourly"]["inside_temp"])

    return {
        "weather_date_used": weather["date_used"],
        "weather_is_seasonal_fallback": weather.get("is_seasonal_fallback", False),
        "hourly_inside_temp": actual["hourly"]["inside_temp"],
        "hourly_solar_gain": actual["hourly"]["solar_gain"],
        "hourly_conduction_loss": actual["hourly"]["conduction_loss"],
        "hourly_infiltration_loss": actual["hourly"]["infiltration_loss"],
        "sustainability": sustainability,
        "livability_score": livability,
    }


@app.post("/compare")
async def compare_configs(req: CompareRequest):
    if MODELS is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded. Run train_model.py to produce {MODEL_PATH}.")

    first = req.configs[0]
    weather = await fetch_nasa_weather(first.lat, first.lon, first.date)

    labels = req.labels or [f"Config {i+1}" for i in range(len(req.configs))]
    results = []
    for label, cfg in zip(labels, req.configs):
        actual = simulate_config(cfg, weather)
        baseline = simulate_config(cfg, weather, overrides=BASELINE_OVERRIDES)
        sustainability = compute_sustainability(actual["required_heater_joules"], baseline["required_heater_joules"])
        livability = compute_livability(actual["hourly"]["inside_temp"])
        results.append({
            "label": label,
            "livability_score": livability,
            "kerosene_saved_liters": sustainability["kerosene_saved_liters"],
            "co2_emissions_prevented_kg": sustainability["co2_emissions_prevented_kg"],
            "min_inside_temp": round(min(actual["hourly"]["inside_temp"]), 2),
            "max_inside_temp": round(max(actual["hourly"]["inside_temp"]), 2),
            "hourly_inside_temp": actual["hourly"]["inside_temp"],
        })

    ranked = sorted(results, key=lambda r: r["livability_score"], reverse=True)
    return {
        "weather_date_used": weather["date_used"],
        "weather_is_seasonal_fallback": weather.get("is_seasonal_fallback", False),
        "ranked_results": ranked,
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODELS is not None}
