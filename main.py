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
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional

import httpx
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import physics
import solar
from materials_library import MATERIALS, compute_composite_wall

app = FastAPI(title="Ladakh Shelter Thermal Simulator", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# ---- Load the model bundle once at startup ----
# Resolved against this file so the server works from any working directory.
MODEL_PATH = BASE_DIR / "thermal_model_v4.pkl"
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

# Comfort band. The v2 score deducted a flat penalty per hour outside it, which
# made every Ladakh winter shelter score identically (nothing clears 15 C in
# January, so everything lost the same 36 points). Degree-hours weight each hour
# by HOW far outside the band it sits, so a shelter holding -9 C separates from
# one at -22 C.
COMFORT_MIN_C = 15.0
COMFORT_MAX_C = 25.0
COMFORT_SCALE_DEGREE_HOURS = 300.0  # decay constant: 300 degC*h -> ~37/100

# NASA POWER is a public service and its data for a past date never changes,
# so cache it. Also keeps a demo responsive when judges re-run one location.
WEATHER_CACHE_TTL_S = 6 * 3600
_WEATHER_CACHE = {}
_GROUND_CACHE = {}


async def fetch_ground_temperature(client, lat: float, lon: float):
    """
    Undisturbed ground temperature, approximated by the annual mean air
    temperature at the site.

    A few metres down, soil sits close to the yearly average and barely moves
    with the weather -- which in a Ladakh January is far warmer than the night
    air. v4 treated the floor as another cold wall losing heat to -18 C, which
    overstated losses. NASA POWER publishes the climatological annual mean, so
    use it; on any failure return None and the engine falls back to air
    temperature, i.e. the old behaviour.
    """
    key = (round(lat, 2), round(lon, 2))
    if key in _GROUND_CACHE:
        return _GROUND_CACHE[key]
    url = ("https://power.larc.nasa.gov/api/temporal/climatology/point?"
           f"parameters=T2M&community=RE&longitude={lon}&latitude={lat}&format=JSON")
    try:
        resp = await client.get(url)
        resp.raise_for_status()
        monthly = resp.json()["properties"]["parameter"]["T2M"]
        annual = monthly.get("ANN")
        if annual is None or annual <= -900:
            values = [v for k, v in monthly.items() if k != "ANN" and v > -900]
            annual = sum(values) / len(values) if values else None
        _GROUND_CACHE[key] = annual
        return annual
    except (httpx.HTTPError, KeyError, ValueError, ZeroDivisionError):
        _GROUND_CACHE[key] = None
        return None


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
    surrogate_check: bool = Field(True, description="also score the ML surrogate against the physics engine")


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
    cache_key = (round(lat, 3), round(lon, 3), requested_date or "recent")
    hit = _WEATHER_CACHE.get(cache_key)
    if hit and (time.time() - hit[0]) < WEATHER_CACHE_TTL_S:
        return hit[1]

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
                payload = resp.json()
                params = payload["properties"]["parameter"]
                found = _pick_most_recent_valid_day(_group_hours_by_day(params))
                if found is not None:
                    found["is_seasonal_fallback"] = is_fallback
                    # NASA returns [lon, lat, elevation] for the grid point.
                    coords = payload.get("geometry", {}).get("coordinates", [])
                    found["elevation_m"] = float(coords[2]) if len(coords) > 2 else 0.0
                    # ALLSKY_SFC_SW_DWN is global HORIZONTAL irradiance; windows
                    # are vertical, so project it onto the glazing plane.
                    day = datetime.strptime(found["date_used"], "%Y%m%d").date()
                    doy = day.timetuple().tm_yday
                    found["solar_horizontal"] = list(found["solar"])
                    found["solar"] = solar.convert_day(found["solar_horizontal"], lat, doy)
                    # Opaque surfaces see different irradiance again: the roof is
                    # horizontal, the walls average over four orientations.
                    found["solar_roof"] = list(found["solar_horizontal"])
                    found["solar_wall"] = solar.convert_day_walls(
                        found["solar_horizontal"], lat, doy)
                    found["sun_track"] = solar.sun_track(lat, doy)
                    found["ground_temp"] = await fetch_ground_temperature(client, lat, lon)
                    _WEATHER_CACHE[cache_key] = (time.time(), found)
                    return found
                last_error = f"No day with complete, non-fill data in {start_ds}-{end_ds}"
            except (httpx.HTTPError, KeyError, ValueError) as e:
                last_error = str(e)
                continue

        raise HTTPException(
            status_code=502,
            detail=f"Could not fetch complete NASA weather data near ({lat},{lon}). Last error: {last_error}",
        )


def _geometry(params: dict):
    """Envelope, roof, floor, volume and shape factor for a config."""
    envelope_area, roof_area, floor_area, volume = physics.shape_geometry(
        params["shape_code"], params["length"], params["width"], params["height"]
    )
    return envelope_area, roof_area, floor_area, volume, envelope_area / volume


def simulate_config(cfg: ShelterInput, weather: dict, overrides: Optional[dict] = None) -> dict:
    """
    Run the 24-hour simulation for one shelter against one weather profile.

    This evaluates the energy balance in physics.py directly. v2 served the
    XGBoost surrogate's approximation of it instead, which is where the
    impossible outputs came from: solar gain going negative at night, indoor air
    falling below ambient in an occupied shelter, conduction flipping sign at
    dawn. Those were approximation error, not physics, and they are gone.
    """
    params = cfg.model_dump()
    if overrides:
        params.update(overrides)

    envelope_area, roof_area, floor_area, volume, _ = _geometry(params)
    density_ratio = physics.air_density_ratio(weather.get("elevation_m", 0.0))
    ground_temp = weather.get("ground_temp")
    capacitance = physics.thermal_capacitance(
        params["thermal_mass_factor"], envelope_area, volume, density_ratio)
    window_area = min(params["window_area"], physics.max_window_area(envelope_area))

    current_temp = params["initial_temp"]
    hourly = {"inside_temp": [], "solar_gain": [], "conduction_loss": [],
              "infiltration_loss": [], "ground_loss": [],
              "wall_conduction": [], "roof_conduction": [], "window_conduction": []}
    required_heater_joules = 0.0
    previous_temps = []

    for hour in range(24):
        previous_temps.append(current_temp)
        r = physics.step(
            current_temp, weather["temps"][hour], weather["solar"][hour], weather["wind"][hour],
            envelope_area=envelope_area, volume=volume, r_value=params["r_value"],
            window_area=window_area, window_u_value=params["window_u_value"],
            orientation_factor=params["orientation_factor"], ach=params["ach"],
            occupants=params["occupants"], capacitance=capacitance,
            density_ratio=density_ratio,
            roof_area=roof_area, floor_area=floor_area, ground_temp=ground_temp,
            roof_irradiance=weather.get("solar_roof", [0] * 24)[hour],
            wall_irradiance=weather.get("solar_wall", [0] * 24)[hour],
        )
        hourly["inside_temp"].append(round(r["next_temp"], 2))
        hourly["solar_gain"].append(round(r["solar_gain"], 1))
        hourly["conduction_loss"].append(round(r["conduction_loss"], 1))
        hourly["infiltration_loss"].append(round(r["infiltration_loss"], 1))
        hourly["ground_loss"].append(round(r["ground_loss"], 1))
        hourly["wall_conduction"].append(round(r["wall_conduction"], 1))
        hourly["roof_conduction"].append(round(r["roof_conduction"], 1))
        hourly["window_conduction"].append(round(r["window_conduction"], 1))

        # Fuel is measured against a held setpoint rather than the floating
        # temperature -- see physics.heating_load for why the old form read zero.
        required_heater_joules += physics.heating_load(
            COMFORT_MIN_C, weather["temps"][hour], weather["solar"][hour], weather["wind"][hour],
            envelope_area=envelope_area, volume=volume, r_value=params["r_value"],
            window_area=window_area, window_u_value=params["window_u_value"],
            orientation_factor=params["orientation_factor"], ach=params["ach"],
            occupants=params["occupants"], density_ratio=density_ratio,
            floor_area=floor_area, ground_temp=ground_temp,
        ) * 3600

        current_temp = r["next_temp"]

    return {
        "hourly": hourly,
        "required_heater_joules": required_heater_joules,
        "final_temp": current_temp,
        "previous_temps": previous_temps,
        "envelope_area": envelope_area,
        "effective_window_area": window_area,
    }


def surrogate_agreement(cfg: ShelterInput, weather: dict, truth: dict) -> Optional[dict]:
    """
    Score the trained XGBoost surrogate against the physics engine it learned from.

    One-step (teacher-forced) comparison: each hour the surrogate is given the
    physics engine's own indoor temperature and asked for the next one, so the
    number measures surrogate fidelity rather than compounding drift. This is
    what the ML half of the project actually demonstrates.
    """
    if MODELS is None:
        return None

    params = cfg.model_dump()
    envelope_area, _roof, _floor, volume, shape_factor = _geometry(params)
    window_area = min(params["window_area"], physics.max_window_area(envelope_area))

    rows = []
    for hour in range(24):
        rows.append([
            params["shape_code"], params["length"], params["width"], params["height"], shape_factor,
            params["r_value"], params["thermal_mass_factor"], window_area, params["window_u_value"],
            params["orientation_factor"], params["ach"], params["occupants"],
            weather.get("elevation_m", 0.0),
            weather.get("ground_temp") if weather.get("ground_temp") is not None else weather["temps"][hour],
            weather["temps"][hour], weather["solar"][hour],
            weather.get("solar_roof", [0] * 24)[hour],
            weather.get("solar_wall", [0] * 24)[hour],
            weather["wind"][hour],
            truth["previous_temps"][hour],
        ])
    frame = pd.DataFrame(rows, columns=FEATURES)

    predicted = MODELS["Inside_Temp"].predict(frame)
    actual = truth["hourly"]["inside_temp"]
    errors = [float(predicted[i]) - actual[i] for i in range(24)]
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))

    return {
        "inside_temp_rmse_c": round(rmse, 3),
        "max_abs_error_c": round(max(abs(e) for e in errors), 3),
        "note": "XGBoost surrogate vs the physics engine, one-step teacher-forced over 24 h.",
    }


def compute_sustainability(actual_joules: float, baseline_joules: float) -> dict:
    """
    Fuel saved against an undesigned reference shelter, both holding COMFORT_MIN_C.

    The absolute loads are reported alongside the saving so the figure can be
    checked rather than taken on trust. Note what the comparison means: it is the
    cost of holding 15 C in an uninsulated, leaky shelter, which in a Ladakh
    January is deliberately punishing. Occupants of such a shelter in practice
    wear layers and accept a far colder interior -- the baseline is a like-for-
    like thermal comparison, not a claim about what anyone currently burns.
    """
    saved_joules = max(0.0, baseline_joules - actual_joules)
    liters_saved = saved_joules / KEROSENE_ENERGY_DENSITY_J_PER_L
    co2_prevented = liters_saved * KEROSENE_CO2_KG_PER_L
    return {
        "kerosene_saved_liters": round(liters_saved, 2),
        "co2_emissions_prevented_kg": round(co2_prevented, 2),
        "heating_load_kwh_per_day": round(actual_joules / 3_600_000, 1),
        "baseline_load_kwh_per_day": round(baseline_joules / 3_600_000, 1),
        "setpoint_c": COMFORT_MIN_C,
        "basis": "energy to hold the setpoint for 24 h, versus an uninsulated leaky shelter of the same size",
    }


def compute_comfort(inside_temps: List[float]) -> dict:
    """
    Comfort as degree-hours outside the band, plus a 0-100 score derived from it.

    Degree-hours is the honest primary number: it is auditable, has units, and
    keeps discriminating when every option is cold. The score is a monotonic
    transform of it for presentation, and never saturates.
    """
    deficit = sum(max(0.0, COMFORT_MIN_C - t) for t in inside_temps)
    excess = sum(max(0.0, t - COMFORT_MAX_C) for t in inside_temps)
    degree_hours = deficit + excess
    score = 100.0 * math.exp(-degree_hours / COMFORT_SCALE_DEGREE_HOURS)
    return {
        "livability_score": int(round(score)),
        "comfort_degree_hours": round(degree_hours, 1),
        "cold_degree_hours": round(deficit, 1),
        "hot_degree_hours": round(excess, 1),
        "hours_in_band": sum(1 for t in inside_temps if COMFORT_MIN_C <= t <= COMFORT_MAX_C),
    }


def validate_geometry(cfg: ShelterInput) -> None:
    """
    Reject configurations that are individually in range but jointly impossible.

    Pydantic checks each field alone, so v2 accepted a 3 m dome (14.1 m^2 of
    surface) carrying 15 m^2 of glazing -- more glass than the shelter has
    envelope -- and returned confident numbers for it. Glazing displaces opaque
    wall, so it cannot exceed a sane fraction of the envelope.
    """
    envelope_area, _roof, _floor, _volume = physics.shape_geometry(
        cfg.shape_code, cfg.length, cfg.width, cfg.height
    )
    limit = physics.max_window_area(envelope_area)
    if cfg.window_area > limit:
        raise HTTPException(
            status_code=422,
            detail=(
                f"window_area {cfg.window_area:.1f} m^2 exceeds what this shelter can carry. "
                f"A {cfg.length:.1f}x{cfg.width:.1f}x{cfg.height:.1f} m shape-{cfg.shape_code} "
                f"envelope is {envelope_area:.1f} m^2, so glazing is capped at {limit:.1f} m^2 "
                f"(60% of the envelope)."
            ),
        )


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
    validate_geometry(data)

    weather = await fetch_nasa_weather(data.lat, data.lon, data.date)
    actual = simulate_config(data, weather)
    baseline = simulate_config(data, weather, overrides=BASELINE_OVERRIDES)

    sustainability = compute_sustainability(actual["required_heater_joules"], baseline["required_heater_joules"])
    comfort = compute_comfort(actual["hourly"]["inside_temp"])

    return {
        "weather_date_used": weather["date_used"],
        "weather_is_seasonal_fallback": weather.get("is_seasonal_fallback", False),
        "hourly_outside_temp": [round(t, 2) for t in weather["temps"]],
        "hourly_solar_power": [round(v, 1) for v in weather["solar"]],
        "hourly_solar_horizontal": [round(v, 1) for v in weather.get("solar_horizontal", [])],
        "site_elevation_m": round(weather.get("elevation_m", 0.0)),
        "hourly_inside_temp": actual["hourly"]["inside_temp"],
        "hourly_solar_gain": actual["hourly"]["solar_gain"],
        "hourly_conduction_loss": actual["hourly"]["conduction_loss"],
        "hourly_infiltration_loss": actual["hourly"]["infiltration_loss"],
        "hourly_ground_loss": actual["hourly"]["ground_loss"],
        "loss_breakdown": {
            "walls": actual["hourly"]["wall_conduction"],
            "roof": actual["hourly"]["roof_conduction"],
            "windows": actual["hourly"]["window_conduction"],
            "draughts": actual["hourly"]["infiltration_loss"],
            "floor": actual["hourly"]["ground_loss"],
        },
        "ground_temp_c": (round(weather["ground_temp"], 1)
                          if weather.get("ground_temp") is not None else None),
        "sun_track": [{"altitude": round(a, 1), "azimuth": round(z, 1)}
                      for a, z in weather.get("sun_track", [])],
        "geometry": {
            "shape_code": data.shape_code,
            "length": data.length, "width": data.width, "height": data.height,
            "window_area": actual["effective_window_area"],
            "envelope_area": round(actual["envelope_area"], 1),
        },
        "sustainability": sustainability,
        "livability_score": comfort["livability_score"],
        "comfort": comfort,
        "engine": "physics",
        "surrogate_agreement": surrogate_agreement(data, weather, actual) if data.surrogate_check else None,
    }


@app.post("/compare")
async def compare_configs(req: CompareRequest):
    for cfg in req.configs:
        validate_geometry(cfg)

    first = req.configs[0]
    weather = await fetch_nasa_weather(first.lat, first.lon, first.date)

    labels = req.labels or [f"Config {i+1}" for i in range(len(req.configs))]
    results = []
    for label, cfg in zip(labels, req.configs):
        actual = simulate_config(cfg, weather)
        baseline = simulate_config(cfg, weather, overrides=BASELINE_OVERRIDES)
        sustainability = compute_sustainability(actual["required_heater_joules"], baseline["required_heater_joules"])
        comfort = compute_comfort(actual["hourly"]["inside_temp"])
        results.append({
            "label": label,
            "livability_score": comfort["livability_score"],
            "comfort_degree_hours": comfort["comfort_degree_hours"],
            "kerosene_saved_liters": sustainability["kerosene_saved_liters"],
            "co2_emissions_prevented_kg": sustainability["co2_emissions_prevented_kg"],
            "min_inside_temp": round(min(actual["hourly"]["inside_temp"]), 2),
            "max_inside_temp": round(max(actual["hourly"]["inside_temp"]), 2),
            "hourly_inside_temp": actual["hourly"]["inside_temp"],
        })

    ranked = sorted(results, key=lambda r: r["comfort_degree_hours"])
    return {
        "weather_date_used": weather["date_used"],
        "weather_is_seasonal_fallback": weather.get("is_seasonal_fallback", False),
        "ranked_results": ranked,
    }


@app.get("/health")
def health():
    return {"status": "ok", "engine": "physics", "surrogate_loaded": MODELS is not None}


@app.get("/", include_in_schema=False)
def dashboard():
    """Serve the operator dashboard."""
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
