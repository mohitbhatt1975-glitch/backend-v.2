"""
physics.py

The shelter energy balance, in one place.

Why this file exists
--------------------
v2 had the same physics written twice: once in `generate_data.py` to build the
training set, and once (as geometry only) in `main.py` to rebuild Shape_Factor
at inference time. Two copies of an equation drift. Worse, `main.py` served the
*ML surrogate's approximation* of this equation rather than the equation, which
is where every physically-impossible output came from -- negative solar gain,
indoor air colder than ambient in an occupied shelter, conduction changing sign
at dawn.

So: this module is the single source of truth. `generate_data.py` uses it to
produce training data, and `main.py` uses it to answer `/simulate` exactly.
The XGBoost surrogate is still trained and still reported -- but as a validated
*approximation of this engine*, not as the thing standing in for it.

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
AIR_VOLUMETRIC_HEAT_CAPACITY = 1200.0   # J/(m^3*K)
INFILTRATION_COEFFICIENT = 0.34    # W/(m^3*K) per air-change-per-hour (standard HVAC constant)
OCCUPANT_HEAT_W = 100.0            # sensible heat output per person
WIND_SENSITIVITY = 0.05            # conduction multiplier per m/s of wind

SECONDS_PER_HOUR = 3600.0
TEMP_FLOOR = -35.0
TEMP_CEILING = 50.0


def shape_geometry(shape_code, length, width, height):
    """Envelope (heat-losing) area and enclosed volume for each shelter form."""
    if shape_code == 1:  # rectangular box
        envelope_area = (2 * length * height) + (2 * width * height) + (length * width)
        volume = length * width * height
    elif shape_code == 2:  # dome / hemisphere
        r = length / 2.0
        envelope_area = 2 * math.pi * (r ** 2)
        volume = (2 / 3) * math.pi * (r ** 3)
    elif shape_code == 3:  # A-frame / triangular tent
        slant = math.sqrt((width / 2) ** 2 + height ** 2)
        envelope_area = (2 * slant * length) + (width * height)
        volume = 0.5 * width * length * height
    elif shape_code == 4:  # vertical cylinder
        r = length / 2.0
        envelope_area = (2 * math.pi * r * height) + (math.pi * (r ** 2))
        volume = math.pi * (r ** 2) * height
    else:  # 5: half-cylinder / Quonset hut
        r = width / 2.0
        envelope_area = (math.pi * r * length) + (math.pi * (r ** 2))
        volume = 0.5 * math.pi * (r ** 2) * length
    return envelope_area, volume


def max_window_area(envelope_area):
    """
    Largest glazing area that is physically meaningful for this envelope.

    Glazing displaces opaque wall, so it can never exceed the envelope, and a
    shelter that is *entirely* glass has no wall left to insulate. The training
    set capped windows at 0.6 * floor area; 60% of the envelope is the same
    ceiling expressed against the surface the window actually occupies.
    """
    return 0.6 * envelope_area


def thermal_capacitance(thermal_mass_factor, envelope_area, volume):
    """Total heat capacity of structure + enclosed air, in J/K."""
    return thermal_mass_factor * envelope_area + volume * AIR_VOLUMETRIC_HEAT_CAPACITY


def conductance(envelope_area, window_area, r_value, window_u_value, ach, volume, wind_speed):
    """Total heat-loss conductance UA of the shelter, in W/K."""
    wind_multiplier = 1 + (wind_speed * WIND_SENSITIVITY)
    opaque_area = max(0.0, envelope_area - window_area)
    return ((1 / r_value) * opaque_area * wind_multiplier
            + window_u_value * window_area * wind_multiplier
            + ach * volume * INFILTRATION_COEFFICIENT)


def heating_load(setpoint_c, outside_temp, solar_power, wind_speed, *,
                 envelope_area, volume, r_value, window_area, window_u_value,
                 orientation_factor, ach, occupants):
    """
    Heater power needed to hold `setpoint_c` for this hour, in W.

    This is the honest way to compare designs on fuel. v2 instead summed the
    energy imbalance at whatever temperature the shelter happened to drift to --
    but a shelter left to float settles at steady state, where that imbalance is
    zero by construction. Measured that way every design "needs" no heat, which
    says nothing about how expensive it is to actually live in.

    Holding a fixed setpoint is the standard building-physics comparison: free
    heat (sun plus bodies) offsets the loss, and whatever is left is fuel.
    """
    ua_total = conductance(envelope_area, window_area, r_value, window_u_value,
                           ach, volume, wind_speed)
    free_heat = (solar_power * window_area * SOLAR_TRANSMITTANCE * orientation_factor
                 + occupants * OCCUPANT_HEAT_W)
    return max(0.0, ua_total * (setpoint_c - outside_temp) - free_heat)


def step(inside_temp, outside_temp, solar_power, wind_speed, *,
         envelope_area, volume, r_value, window_area, window_u_value,
         orientation_factor, ach, occupants, capacitance):
    """
    Advance one hour of the energy balance, integrated exactly.

    Over one hour the weather is held constant, which makes the balance a linear
    first-order ODE:

        C dT/dt = Q_in - UA (T - T_out)

    with the closed-form solution

        T_ss  = T_out + Q_in / UA                      (steady state)
        T(t)  = T_ss + (T_0 - T_ss) e^(-t/tau),        tau = C / UA

    v2 stepped this with forward Euler over a full hour and clamped the result
    to +/-6 deg C/hr to stop it exploding. That clamp was hiding instability
    rather than preventing it: a canvas tent has tau of roughly four minutes, so
    an hour-long Euler step overshoots the steady state badly -- which is how v2
    produced indoor air colder than ambient inside an occupied shelter.

    The exponential form is unconditionally stable and cannot overshoot: T moves
    monotonically toward T_ss and never crosses it. Since Q_in >= 0 implies
    T_ss >= T_out, indoor air can no longer fall below outdoor air while the
    shelter is being heated. No clamp needed.

    Heat flows are reported as hour *averages* (from the analytic mean
    temperature), so the reported net heat exactly accounts for the temperature
    change over the hour rather than only describing its first instant.
    """
    wind_multiplier = 1 + (wind_speed * WIND_SENSITIVITY)
    opaque_area = max(0.0, envelope_area - window_area)

    # Conductances, W/K
    ua_wall = (1 / r_value) * opaque_area * wind_multiplier
    ua_window = window_u_value * window_area * wind_multiplier
    ua_infiltration = ach * volume * INFILTRATION_COEFFICIENT
    ua_total = ua_wall + ua_window + ua_infiltration

    # Heat in, W
    solar_gain = solar_power * window_area * SOLAR_TRANSMITTANCE * orientation_factor
    internal_heat = occupants * OCCUPANT_HEAT_W
    heat_in = solar_gain + internal_heat

    if ua_total <= 0.0:
        # A perfectly sealed, perfectly insulated shelter: nothing escapes.
        mean_temp = inside_temp + (heat_in * SECONDS_PER_HOUR) / (2 * capacitance)
        next_temp = inside_temp + (heat_in * SECONDS_PER_HOUR) / capacitance
    else:
        steady_state = outside_temp + heat_in / ua_total
        tau = capacitance / ua_total
        decay = math.exp(-SECONDS_PER_HOUR / tau)
        next_temp = steady_state + (inside_temp - steady_state) * decay
        # Analytic mean of T over the hour
        mean_temp = steady_state + (inside_temp - steady_state) * (tau / SECONDS_PER_HOUR) * (1 - decay)

    mean_delta = mean_temp - outside_temp
    wall_conduction = ua_wall * mean_delta
    window_conduction = ua_window * mean_delta
    conduction_loss = wall_conduction + window_conduction
    infiltration_loss = ua_infiltration * mean_delta
    net_heat = heat_in - conduction_loss - infiltration_loss

    next_temp = max(TEMP_FLOOR, min(TEMP_CEILING, next_temp))

    return {
        "solar_gain": solar_gain,
        "internal_heat": internal_heat,
        "wall_conduction": wall_conduction,
        "window_conduction": window_conduction,
        "conduction_loss": conduction_loss,
        "infiltration_loss": infiltration_loss,
        "net_heat": net_heat,
        "next_temp": next_temp,
        "mean_temp": mean_temp,
        "time_constant_s": (capacitance / ua_total) if ua_total > 0 else float("inf"),
    }
