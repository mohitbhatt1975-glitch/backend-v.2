"""
solar.py

Converts NASA's horizontal irradiance onto the plane of a vertical window.

Why this exists
---------------
NASA POWER's `ALLSKY_SFC_SW_DWN` is *global horizontal* irradiance -- what a
sensor lying flat on the ground sees. Shelter windows are vertical. v3 fed the
horizontal number straight into a vertical window, which understates winter
solar gain badly: at 34 deg N in mid-January the sun peaks only ~33 deg above
the horizon, so a south-facing wall catches roughly 1.5x the direct beam a
horizontal surface does. That matters here because free solar heat is the whole
argument for passive design in Ladakh.

Method (standard, Duffie & Beckman):
  1. Sun position from latitude, day-of-year and hour angle.
  2. Split the measured horizontal irradiance into beam and diffuse using the
     Erbs correlation on the clearness index.
  3. Project the beam onto the vertical plane, add the sky-diffuse the wall can
     see (half the sky) and ground reflection (half the ground).

Assumptions, stated plainly:
  * NASA POWER hourly data is requested with `time_standard = LST`, treated
    here as local solar time, so the hour index is the solar hour.
  * The glazing is modelled as facing the equator (due south in the northern
    hemisphere). How far a real shelter deviates from that is what the
    `orientation_factor` input derates -- so orientation is still the user's
    lever, applied on top of the ideal-plane irradiance computed here.
"""

import math

SOLAR_CONSTANT = 1367.0   # W/m^2
DEFAULT_ALBEDO = 0.3      # bare high-altitude ground; snow cover runs far higher


def _declination_deg(day_of_year):
    """Cooper's equation."""
    return 23.45 * math.sin(math.radians(360.0 * (284 + day_of_year) / 365.0))


def sun_altitude_sin(lat_deg, day_of_year, solar_hour):
    """sin(solar altitude); negative means the sun is below the horizon."""
    phi = math.radians(lat_deg)
    delta = math.radians(_declination_deg(day_of_year))
    omega = math.radians(15.0 * (solar_hour - 12.0))
    return (math.sin(delta) * math.sin(phi)
            + math.cos(delta) * math.cos(phi) * math.cos(omega))


def cos_incidence_vertical(lat_deg, day_of_year, solar_hour):
    """
    cos of the angle between the sun and the normal of an equator-facing
    vertical surface. Duffie & Beckman 1.6.2 with tilt = 90 deg.
    """
    phi = math.radians(lat_deg)
    delta = math.radians(_declination_deg(day_of_year))
    omega = math.radians(15.0 * (solar_hour - 12.0))
    facing_equator = 1.0 if lat_deg >= 0 else -1.0
    return facing_equator * (-math.sin(delta) * math.cos(phi)
                             + math.cos(delta) * math.sin(phi) * math.cos(omega))


def _diffuse_fraction(clearness_index):
    """Erbs correlation: how much of the horizontal irradiance is diffuse."""
    kt = clearness_index
    if kt <= 0.22:
        return 1.0 - 0.09 * kt
    if kt <= 0.80:
        return (0.9511 - 0.1604 * kt + 4.388 * kt ** 2
                - 16.638 * kt ** 3 + 12.336 * kt ** 4)
    return 0.165


def vertical_irradiance(ghi, lat_deg, day_of_year, solar_hour, albedo=DEFAULT_ALBEDO):
    """
    Irradiance on an equator-facing vertical surface, W/m^2, from measured
    global horizontal irradiance.
    """
    if ghi <= 0:
        return 0.0

    sin_alt = sun_altitude_sin(lat_deg, day_of_year, solar_hour)
    if sin_alt <= 0.01:          # sun at or below the horizon: no usable beam
        return ghi * albedo * 0.5

    # Clearness index against the extraterrestrial horizontal irradiance
    eccentricity = 1.0 + 0.033 * math.cos(math.radians(360.0 * day_of_year / 365.0))
    extraterrestrial = SOLAR_CONSTANT * eccentricity * sin_alt
    kt = min(1.0, max(0.0, ghi / extraterrestrial)) if extraterrestrial > 0 else 0.0

    diffuse = ghi * _diffuse_fraction(kt)
    beam_horizontal = max(0.0, ghi - diffuse)
    beam_normal = beam_horizontal / sin_alt

    cos_theta = max(0.0, cos_incidence_vertical(lat_deg, day_of_year, solar_hour))

    beam_on_wall = beam_normal * cos_theta
    sky_on_wall = diffuse * 0.5          # a vertical wall sees half the sky dome
    ground_on_wall = ghi * albedo * 0.5  # and half the ground

    return beam_on_wall + sky_on_wall + ground_on_wall


def convert_day(hourly_ghi, lat_deg, day_of_year, albedo=DEFAULT_ALBEDO):
    """Convert 24 hourly horizontal values to the vertical window plane."""
    return [
        vertical_irradiance(ghi, lat_deg, day_of_year, hour + 0.5, albedo)
        for hour, ghi in enumerate(hourly_ghi)
    ]
