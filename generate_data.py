"""
generate_data.py  (v2)

Fixes vs the original version:

1. CRITICAL: Outside_Temp and Solar_Power used to be drawn independently at
   random for EVERY hour, with zero correlation to time-of-day. That taught
   the model there's no such thing as day/night -- so at inference time,
   when it's fed real NASA data (which *is* smoothly diurnal), predictions
   drift/misbehave. This version samples one realistic daily temperature
   curve and one realistic daily solar curve per synthetic shelter, so the
   model actually learns the day/night thermal-lag behaviour the PS is
   about.

2. Windows now have their own U-value and lose heat by conduction, instead
   of only ever contributing solar gain. Previously more window area could
   only ever help, which is physically backwards.

3. Infiltration/openings (air changes per hour, ACH) is now a first-class,
   separate heat-loss term -- directly answers the PS's "effect of
   openings" requirement.

4. Thermal mass is no longer an arbitrary constant. It's computed from
   Thermal_Mass_Factor (J/(m^2*K) of wall area), which materials_library.py
   derives from real material properties -- this is what lets you study
   "composite multi-material" walls.

5. Orientation_Factor represents how favourably the window-bearing wall
   faces the sun (0.3 = poor orientation, 1.0 = optimal).

6. Latin Hypercube Sampling instead of pure uniform random, for more even
   coverage of the input space (matters most at the edges, which is where
   the old "flatline" extrapolation problem bit hardest).
"""

import math
import random

import numpy as np
import pandas as pd
from scipy.stats import qmc

import physics
from physics import shape_geometry, thermal_capacitance

random.seed(42)
np.random.seed(42)

N_SHELTERS = 2000
HOURS = 24

# ---- Sampling bounds for continuous static + daily-weather parameters ----
# Order matters -- must match the LHS dimension order below.
BOUNDS = {
    "length":            (3.0, 10.0),      # m
    "width":             (3.0, 10.0),      # m
    "height":            (2.5, 4.0),       # m
    "r_value":           (0.05, 3.0),      # m^2*K/W  (canvas tent .. thick insulated wall)
    "thermal_mass_factor": (500.0, 600000.0),  # J/(m^2*K) of wall area (light fabric .. thick stone/adobe)
    "window_area":       (0.5, 8.0),       # m^2
    "window_u_value":    (1.0, 6.0),       # W/(m^2*K) (triple-glazed .. single pane)
    "orientation_factor": (0.3, 1.0),      # 0.3 = poorly oriented, 1.0 = optimal solar-facing
    "ach":               (0.3, 5.0),       # air changes per hour (well-sealed .. leaky)
    "daily_mean_temp":   (-25.0, 5.0),     # deg C, Ladakh-representative winter ambient
    "daily_amplitude":   (4.0, 15.0),      # deg C day/night swing
    "daily_peak_solar":  (500.0, 1100.0),  # W/m^2, matches Ladakh's high irradiance
}
DIM_ORDER = list(BOUNDS.keys())

def sample_inputs(n):
    sampler = qmc.LatinHypercube(d=len(DIM_ORDER), seed=42)
    unit_samples = sampler.random(n)
    lowers = np.array([BOUNDS[k][0] for k in DIM_ORDER])
    uppers = np.array([BOUNDS[k][1] for k in DIM_ORDER])
    scaled = qmc.scale(unit_samples, lowers, uppers)
    return pd.DataFrame(scaled, columns=DIM_ORDER)


def diurnal_outside_temp(hour, daily_mean, daily_amplitude, rng):
    # Warmest around 14:00, coldest around 02:00 -- matches typical Ladakh
    # winter behaviour where daytime solar warms the air, then it drops
    # sharply after sunset.
    base = daily_mean + daily_amplitude * math.cos(2 * math.pi * (hour - 14) / 24)
    noise = rng.uniform(-1.5, 1.5)
    return base + noise


def diurnal_solar_power(hour, daily_peak_solar, rng):
    if 6 <= hour <= 18:
        shape = math.sin(math.pi * (hour - 6) / 12)
        cloud_factor = rng.uniform(0.75, 1.0)  # mild variability, Ladakh is mostly clear-sky
        return max(0.0, daily_peak_solar * shape * cloud_factor)
    return 0.0


def build_dataset():
    print("Sampling input space with Latin Hypercube Sampling...")
    static_df = sample_inputs(N_SHELTERS)

    rows = []
    rng = random.Random(42)

    print(f"Simulating {N_SHELTERS} shelters x {HOURS} hours (diurnal physics)...")
    for i in range(N_SHELTERS):
        row = static_df.iloc[i]
        shape_code = rng.choice([1, 2, 3, 4, 5])
        occupants = rng.randint(0, 15)

        length, width, height = row["length"], row["width"], row["height"]
        r_value = row["r_value"]
        thermal_mass_factor = row["thermal_mass_factor"]
        window_area = min(row["window_area"], 0.6 * length * width)  # can't exceed a sane fraction of floor area
        window_u_value = row["window_u_value"]
        orientation_factor = row["orientation_factor"]
        ach = row["ach"]
        daily_mean_temp = row["daily_mean_temp"]
        daily_amplitude = row["daily_amplitude"]
        daily_peak_solar = row["daily_peak_solar"]

        envelope_area, volume = shape_geometry(shape_code, length, width, height)
        shape_factor = envelope_area / volume

        c_total = thermal_capacitance(thermal_mass_factor, envelope_area, volume)

        current_inside_temp = rng.uniform(daily_mean_temp - 2, daily_mean_temp + 8)

        for hour in range(HOURS):
            outside_temp = diurnal_outside_temp(hour, daily_mean_temp, daily_amplitude, rng)
            solar_power = diurnal_solar_power(hour, daily_peak_solar, rng)
            wind_speed = rng.uniform(0, 20)

            r = physics.step(
                current_inside_temp, outside_temp, solar_power, wind_speed,
                envelope_area=envelope_area, volume=volume, r_value=r_value,
                window_area=window_area, window_u_value=window_u_value,
                orientation_factor=orientation_factor, ach=ach,
                occupants=occupants, capacitance=c_total,
            )
            solar_gain = r["solar_gain"]
            conduction_loss = r["conduction_loss"]
            infiltration_loss = r["infiltration_loss"]
            next_inside_temp = r["next_temp"]

            rows.append([
                shape_code, length, width, height, shape_factor,
                r_value, thermal_mass_factor, window_area, window_u_value,
                orientation_factor, ach, occupants,
                outside_temp, solar_power, wind_speed, current_inside_temp,
                next_inside_temp, solar_gain, conduction_loss, infiltration_loss,
            ])

            current_inside_temp = next_inside_temp

        if (i + 1) % 400 == 0:
            print(f"  ...{i + 1}/{N_SHELTERS} shelters simulated")

    columns = [
        "Shape_Code", "Length", "Width", "Height", "Shape_Factor",
        "R_Value", "Thermal_Mass_Factor", "Window_Area", "Window_U_Value",
        "Orientation_Factor", "ACH", "Occupants",
        "Outside_Temp", "Solar_Power", "Wind_Speed", "Previous_Temp",
        "Inside_Temp", "Solar_Gain", "Conduction_Loss", "Infiltration_Loss",
    ]
    return pd.DataFrame(rows, columns=columns)


if __name__ == "__main__":
    df = build_dataset()
    df.to_csv("training_data.csv", index=False)
    print(f"Dataset created successfully: {len(df)} rows -> training_data.csv")

    # Quick sanity check that the diurnal fix actually worked
    check = df.groupby(df.index % 24)[["Outside_Temp", "Solar_Power"]].mean()
    print("\nSanity check -- mean value by hour-of-day (should show a clear diurnal shape):")
    print(f"  Hour 2  (should be coldest, ~0 solar):  Outside_Temp={check.loc[2, 'Outside_Temp']:.1f}  Solar={check.loc[2, 'Solar_Power']:.1f}")
    print(f"  Hour 14 (should be warmest, peak solar): Outside_Temp={check.loc[14, 'Outside_Temp']:.1f}  Solar={check.loc[14, 'Solar_Power']:.1f}")
