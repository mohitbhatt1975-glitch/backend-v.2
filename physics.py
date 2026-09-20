"""
physics.py

The shelter energy balance, in one place.

Why this file exists
--------------------
v2 had the same physics written twice: once in `generate_data.py` to build the
training set, and once (as geometry only) in `main.py`. Two copies of an
equation drift. Worse, `main.py` served the *ML surrogate's approximation* of
this equation rather than the equation, which is where every physically
impossible output came from -- negative solar gain, indoor air colder than
ambient in an occupied shelter, conduction changing sign at dawn.

So: this module is the single source of truth. `generate_data.py` uses it to
produce training data, and `main.py` uses it to answer `/simulate` exactly.
The XGBoost surrogate is still trained and still reported -- but as a validated
*approximation of this engine*, not as the thing standing in for it.

What each version added
-----------------------
v3  exact (closed-form) integration instead of clamped forward Euler
v4  altitude air-density correction; irradiance projected onto the window plane
v5  the floor exchanges heat with the ground, not outdoor air; opaque surfaces
    absorb sunlight and radiate to the night sky (both via sol-air temperature)

Units
-----
    temperature        deg C
    area               m^2
    volume             m^3
    R_value            m^2*K/W   (thermal resistance; U = 1/R)
    U_value            W/(m^2*K)
    thermal_mass       J/(m^2*K) of envelope area
    power / heat flow  W
"""

import math

SOLAR_TRANSMITTANCE = 0.6          # fraction of incident solar passing the glazing as usable heat
AIR_VOLUMETRIC_HEAT_CAPACITY = 1200.0   # J/(m^3*K) at sea level
INFILTRATION_COEFFICIENT = 0.34    # W/(m^3*K) per air-change-per-hour, sea level
OCCUPANT_HEAT_W = 100.0            # sensible heat output per person
WIND_SENSITIVITY = 0.05            # conduction multiplier per m/s of wind

SEA_LEVEL_PRESSURE_PA = 101325.0

# --- opaque-surface radiation (v5) ---
SOLAR_ABSORPTIVITY = 0.6           # mid-tone render/mud; 0.3 whitewashed, 0.9 dark stone
FILM_BASE = 16.0                    # outside surface film coefficient, W/(m^2*K)
FILM_WIND = 2.7                    # ...and its wind sensitivity, per m/s.
                                   # Anchored on ASHRAE (34.0 at 6.7 m/s) and ISO 6946,
                                   # whose external surface resistance implies h_o ~ 25
                                   # even in calm air, because the coefficient includes
                                   # radiative exchange and not just wind. An earlier
                                   # 5.7 + 3.8v fell to 5.7 in still air and produced a
                                   # 105 K sol-air lift in full sun -- also inconsistent
                                   # with the 3.9 K sky term, which assumes h_o ~ 22.7.
MAX_IRRADIANCE = 1400.0            # W/m^2; nothing at the surface exceeds this, so
                                   # clamp defensively rather than trust the caller
SKY_DEPRESSION_K = 3.9             # ASHRAE sol-air correction for a surface facing
                                   # the open sky. This is the night-radiation term:
                                   # with no sun, a roof behaves as if the air were
                                   # ~4 C colder than it is. Vertical walls see as
                                   # much ground as sky, so they take no correction.

# --- ground coupling (v5) ---
GROUND_RESISTANCE = 1.0            # m^2*K/W of soil and contact resistance under the floor

SECONDS_PER_HOUR = 3600.0
TEMP_FLOOR = -35.0
TEMP_CEILING = 50.0


def air_density_ratio(elevation_m):
    """
    Air density at this altitude relative to sea level, from the barometric
    formula for the standard atmosphere.

    Everything that moves *air* rather than heat through a solid scales with
    density: the heat capacity of the air inside, and the energy carried out by
    infiltration. v3 used sea-level constants, which for Leh at 4533 m overstated
    infiltration loss by a factor of 1.8 -- an awkward error in a project whose
    entire subject is high altitude, especially since NASA returns the site
    elevation in the same response the weather arrives in.

    This is a pressure ratio only, so sea level is exactly 1.0. The temperature
    dependence of density is left inside AIR_VOLUMETRIC_HEAT_CAPACITY and
    INFILTRATION_COEFFICIENT, which are standard-condition values.
    """
    if not elevation_m or elevation_m <= 0:
        return 1.0
    pressure = SEA_LEVEL_PRESSURE_PA * (1.0 - 2.25577e-5 * elevation_m) ** 5.25588
    return pressure / SEA_LEVEL_PRESSURE_PA


def shape_geometry(shape_code, length, width, height):
    """
    Surfaces and volume for each shelter form.

    Returns (above_ground_area, roof_area, floor_area, volume).

    `above_ground_area` is what it always was -- walls plus roof, the surfaces
    exchanging heat with outdoor air. The floor is now reported separately
    because it exchanges heat with the *ground*, which in a Ladakh winter is far
    warmer than the air; treating it as another cold wall, as v4 did, overstated
    losses. Roof area is split out so sky-facing surfaces can be handled
    differently from vertical walls.
    """
    if shape_code == 1:  # rectangular box
        roof_area = length * width
        above_ground_area = (2 * length * height) + (2 * width * height) + roof_area
        floor_area = length * width
        volume = length * width * height
    elif shape_code == 2:  # dome / hemisphere
        r = length / 2.0
        above_ground_area = 2 * math.pi * (r ** 2)
        roof_area = above_ground_area * 0.5   # upper half faces the sky
        floor_area = math.pi * (r ** 2)
        volume = (2 / 3) * math.pi * (r ** 3)
    elif shape_code == 3:  # A-frame / triangular tent
        slant = math.sqrt((width / 2) ** 2 + height ** 2)
        roof_area = 2 * slant * length
        above_ground_area = roof_area + (width * height)   # slopes plus gable ends
        floor_area = width * length
        volume = 0.5 * width * length * height
    elif shape_code == 4:  # vertical cylinder
        r = length / 2.0
        roof_area = math.pi * (r ** 2)
        above_ground_area = (2 * math.pi * r * height) + roof_area
        floor_area = math.pi * (r ** 2)
        volume = math.pi * (r ** 2) * height
    else:  # 5: half-cylinder / Quonset hut
        r = width / 2.0
        roof_area = math.pi * r * length
        above_ground_area = roof_area + (math.pi * (r ** 2))
        floor_area = width * length
        volume = 0.5 * math.pi * (r ** 2) * length
    return above_ground_area, roof_area, floor_area, volume


def max_window_area(envelope_area):
    """
    Largest glazing area that is physically meaningful for this envelope.

    Glazing displaces opaque wall, so it can never exceed the envelope, and a
    shelter that is entirely glass has no wall left to insulate.
    """
    return 0.6 * envelope_area


def thermal_capacitance(thermal_mass_factor, envelope_area, volume, density_ratio=1.0):
    """Total heat capacity of structure + enclosed air, in J/K."""
    return (thermal_mass_factor * envelope_area
            + volume * AIR_VOLUMETRIC_HEAT_CAPACITY * density_ratio)


def film_coefficient(wind_speed):
    """Outside surface film coefficient, W/(m^2*K)."""
    return FILM_BASE + FILM_WIND * max(0.0, wind_speed)


def sol_air_temperature(outside_temp, irradiance, wind_speed, faces_sky,
                        absorptivity=SOLAR_ABSORPTIVITY):
    """
    The temperature an opaque surface *behaves* as if the outside air were.

    Standard building-physics device that folds two effects into one number:

        T_sol-air = T_out + (alpha * I) / h_o  -  (sky correction)

    Sunlight absorbed by the surface pushes it above air temperature; longwave
    radiation to a cold sky pulls it below. v4 modelled neither -- walls and
    roofs gained nothing from the sun, and nothing radiated to the night sky. At
    4500 m under clear skies both are significant, and they act in opposite
    directions, which is why they belong in the same term.

    `faces_sky` is True for roofs, which see the open sky hemisphere. Vertical
    walls see roughly as much ground as sky, so they take no sky correction.
    """
    h_o = film_coefficient(wind_speed)
    capped = min(MAX_IRRADIANCE, max(0.0, irradiance))
    solar_lift = (absorptivity * capped) / h_o
    sky_drop = SKY_DEPRESSION_K if faces_sky else 0.0
    return outside_temp + solar_lift - sky_drop


def conductance(envelope_area, window_area, r_value, window_u_value, ach, volume,
                wind_speed, density_ratio=1.0, floor_area=0.0):
    """
    Total heat-loss conductance UA of the shelter, in W/K.

    Includes the floor path through the ground, which puts the wall's own
    resistance and the soil beneath it in series.
    """
    wind_multiplier = 1 + (wind_speed * WIND_SENSITIVITY)
    opaque_area = max(0.0, envelope_area - window_area)
    ua_floor = floor_area / (r_value + GROUND_RESISTANCE) if floor_area else 0.0
    return ((1 / r_value) * opaque_area * wind_multiplier
            + window_u_value * window_area * wind_multiplier
            + ach * volume * INFILTRATION_COEFFICIENT * density_ratio
            + ua_floor)


def heating_load(setpoint_c, outside_temp, solar_power, wind_speed, *,
                 envelope_area, volume, r_value, window_area, window_u_value,
                 orientation_factor, ach, occupants, density_ratio=1.0,
                 floor_area=0.0, ground_temp=None):
    """
    Heater power needed to hold `setpoint_c` for this hour, in W.

    This is the honest way to compare designs on fuel. v2 instead summed the
    energy imbalance at whatever temperature the shelter happened to drift to --
    but a shelter left to float settles at steady state, where that imbalance is
    zero by construction, so every design "needed" no heat.

    Holding a fixed setpoint is the standard comparison: free heat (sun plus
    bodies) offsets the loss, and whatever is left is fuel.
    """
    wind_multiplier = 1 + (wind_speed * WIND_SENSITIVITY)
    opaque_area = max(0.0, envelope_area - window_area)
    ua_air = ((1 / r_value) * opaque_area * wind_multiplier
              + window_u_value * window_area * wind_multiplier
              + ach * volume * INFILTRATION_COEFFICIENT * density_ratio)

    loss = ua_air * (setpoint_c - outside_temp)
    if floor_area:
        ground = outside_temp if ground_temp is None else ground_temp
        loss += (floor_area / (r_value + GROUND_RESISTANCE)) * (setpoint_c - ground)

    free_heat = (solar_power * window_area * SOLAR_TRANSMITTANCE * orientation_factor
                 + occupants * OCCUPANT_HEAT_W)
    return max(0.0, loss - free_heat)


def step(inside_temp, outside_temp, solar_power, wind_speed, *,
         envelope_area, volume, r_value, window_area, window_u_value,
         orientation_factor, ach, occupants, capacitance, density_ratio=1.0,
         roof_area=0.0, floor_area=0.0, ground_temp=None,
         roof_irradiance=0.0, wall_irradiance=0.0,
         absorptivity=SOLAR_ABSORPTIVITY):
    """
    Advance one hour of the energy balance, integrated exactly.

    Over one hour the weather is held constant, which makes the balance a linear
    first-order ODE:

        C dT/dt = Q_in - UA (T - T_equivalent)

    with the closed-form solution

        T_ss  = T_equivalent + Q_in / UA               (steady state)
        T(t)  = T_ss + (T_0 - T_ss) e^(-t/tau),        tau = C / UA

    v2 stepped this with forward Euler over a full hour and clamped the result
    to +/-6 deg C/hr to stop it exploding. That clamp hid instability rather
    than preventing it: a canvas tent has tau of roughly four minutes, so an
    hour-long Euler step overshoots the steady state badly -- which is how v2
    produced indoor air colder than ambient inside an occupied shelter. The
    exponential form cannot overshoot: T moves monotonically toward T_ss.

    `T_equivalent` is the conductance-weighted mix of what each surface actually
    faces -- sol-air temperature for sunlit roof and walls, plain outdoor air for
    windows and leaking air, and ground temperature under the floor. Heat flows
    are reported as hour averages from the analytic mean temperature, so the
    reported net heat exactly accounts for the temperature change over the hour.
    """
    wind_multiplier = 1 + (wind_speed * WIND_SENSITIVITY)
    opaque_area = max(0.0, envelope_area - window_area)

    # Split the opaque envelope into sky-facing roof and vertical wall.
    roof = min(roof_area, opaque_area)
    wall = max(0.0, opaque_area - roof)

    ua_roof = (1 / r_value) * roof * wind_multiplier
    ua_wall = (1 / r_value) * wall * wind_multiplier
    ua_window = window_u_value * window_area * wind_multiplier
    ua_infiltration = ach * volume * INFILTRATION_COEFFICIENT * density_ratio
    ua_floor = floor_area / (r_value + GROUND_RESISTANCE) if floor_area else 0.0
    ua_total = ua_roof + ua_wall + ua_window + ua_infiltration + ua_floor

    # What each surface actually exchanges heat with
    t_roof = sol_air_temperature(outside_temp, roof_irradiance, wind_speed, True, absorptivity)
    t_wall = sol_air_temperature(outside_temp, wall_irradiance, wind_speed, False, absorptivity)
    t_ground = outside_temp if ground_temp is None else ground_temp

    # Heat in through the glazing, plus bodies
    solar_gain = solar_power * window_area * SOLAR_TRANSMITTANCE * orientation_factor
    internal_heat = occupants * OCCUPANT_HEAT_W
    heat_in = solar_gain + internal_heat

    if ua_total <= 0.0:
        mean_temp = inside_temp + (heat_in * SECONDS_PER_HOUR) / (2 * capacitance)
        next_temp = inside_temp + (heat_in * SECONDS_PER_HOUR) / capacitance
    else:
        t_equivalent = ((ua_roof * t_roof + ua_wall * t_wall
                         + (ua_window + ua_infiltration) * outside_temp
                         + ua_floor * t_ground) / ua_total)
        steady_state = t_equivalent + heat_in / ua_total
        tau = capacitance / ua_total
        decay = math.exp(-SECONDS_PER_HOUR / tau)
        next_temp = steady_state + (inside_temp - steady_state) * decay
        mean_temp = steady_state + (inside_temp - steady_state) * (tau / SECONDS_PER_HOUR) * (1 - decay)

    roof_conduction = ua_roof * (mean_temp - t_roof)
    wall_conduction = ua_wall * (mean_temp - t_wall)
    window_conduction = ua_window * (mean_temp - outside_temp)
    conduction_loss = roof_conduction + wall_conduction + window_conduction
    infiltration_loss = ua_infiltration * (mean_temp - outside_temp)
    ground_loss = ua_floor * (mean_temp - t_ground)
    net_heat = heat_in - conduction_loss - infiltration_loss - ground_loss

    next_temp = max(TEMP_FLOOR, min(TEMP_CEILING, next_temp))

    return {
        "solar_gain": solar_gain,
        "internal_heat": internal_heat,
        "roof_conduction": roof_conduction,
        "wall_conduction": wall_conduction,
        "window_conduction": window_conduction,
        "conduction_loss": conduction_loss,
        "infiltration_loss": infiltration_loss,
        "ground_loss": ground_loss,
        "net_heat": net_heat,
        "next_temp": next_temp,
        "mean_temp": mean_temp,
        "sol_air_roof": t_roof,
        "sol_air_wall": t_wall,
        "time_constant_s": (capacitance / ua_total) if ua_total > 0 else float("inf"),
    }
