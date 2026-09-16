"""
materials_library.py

Real building-material properties, used to convert a "composite wall"
(a stack of physical material layers) into the two lumped parameters our
ML model actually trains on: R_Value (thermal resistance) and
Thermal_Mass_Factor (thermal capacitance per unit wall area).

This is what lets the project satisfy the PS requirement of studying
"application of suitable materials and application of thermal mass storage
material, composite multi-material etc." without blowing up the ML feature
space -- the model only ever sees (R_Value, Thermal_Mass_Factor), but those
two numbers can now be traced back to real, citable material science instead
of an arbitrary constant.

Values below are representative figures commonly used in building-physics
references (ASHRAE Handbook of Fundamentals; standard building-material
property tables). Treat them as reasonable engineering defaults for a
demo/prototype -- for a production tool you'd want to let users override
them or cite region-specific test data.

Units:
    conductivity_k   -> W/(m*K)      thermal conductivity
    density          -> kg/m^3
    specific_heat    -> J/(kg*K)
"""

MATERIALS = {
    "mud_brick_adobe":     {"conductivity_k": 0.50, "density": 1700, "specific_heat": 900,
                             "note": "Traditional Ladakh construction material; high thermal mass, moderate insulation."},
    "stone_masonry":       {"conductivity_k": 1.70, "density": 2500, "specific_heat": 800,
                             "note": "Very high thermal mass, poor insulation on its own."},
    "timber_plywood":      {"conductivity_k": 0.13, "density": 550,  "specific_heat": 1600,
                             "note": "Low mass, moderate insulation, common in temporary shelter frames."},
    "eps_insulation":      {"conductivity_k": 0.035, "density": 20,   "specific_heat": 1450,
                             "note": "Expanded polystyrene. Excellent insulation, negligible thermal mass."},
    "rockwool_insulation": {"conductivity_k": 0.040, "density": 100,  "specific_heat": 840,
                             "note": "Mineral wool insulation. Similar role to EPS, slightly denser."},
    "canvas_tent_fabric":  {"conductivity_k": 0.05,  "density": 300,  "specific_heat": 1300,
                             "note": "Baseline lightweight shelter skin; poor insulation, near-zero thermal mass."},
    "straw_bale":          {"conductivity_k": 0.08,  "density": 110,  "specific_heat": 1500,
                             "note": "Natural, low-cost insulation with modest thermal mass."},
    "steel_sheet":         {"conductivity_k": 50.0,  "density": 7850, "specific_heat": 490,
                             "note": "Common relief-shelter cladding. Very poor insulation on its own."},
    "pcm_wallboard":       {"conductivity_k": 0.20,  "density": 860,  "specific_heat": 5000,
                             "note": "Phase-change-material board. Effective specific heat approximated as "
                                     "elevated to represent latent heat absorbed near its phase-transition "
                                     "temperature -- a simplification of true PCM behaviour, good enough for "
                                     "comparative screening, not for precise PCM sizing."},
}


def compute_composite_wall(layers, envelope_area):
    """
    Convert a stack of material layers into (R_Value, Thermal_Mass_Factor).

    layers: list of (material_name: str, thickness_m: float), given in order
            from outside to inside (order doesn't affect the lumped result,
            but keep it consistent for your own bookkeeping).
    envelope_area: total exterior surface area of the shelter (m^2), used
            only to sanity-check inputs -- the returned R_Value and
            Thermal_Mass_Factor are both already per-unit-area quantities,
            so they plug directly into the ML model regardless of shelter size.

    Returns: dict with r_value (m^2*K/W) and thermal_mass_factor (J/(m^2*K))
    """
    if not layers:
        raise ValueError("Provide at least one material layer.")

    r_total = 0.0          # series thermal resistance, m^2*K/W
    heat_capacity_area = 0.0  # J/(m^2*K), i.e. capacitance per unit wall area

    for name, thickness_m in layers:
        if name not in MATERIALS:
            raise ValueError(f"Unknown material '{name}'. Available: {list(MATERIALS)}")
        if thickness_m <= 0:
            raise ValueError(f"Thickness for '{name}' must be positive.")

        props = MATERIALS[name]
        r_total += thickness_m / props["conductivity_k"]
        heat_capacity_area += thickness_m * props["density"] * props["specific_heat"]

    return {
        "r_value": round(r_total, 4),
        "thermal_mass_factor": round(heat_capacity_area, 2),
        "layers_used": [l[0] for l in layers],
        "envelope_area_m2": envelope_area,
    }


if __name__ == "__main__":
    # Quick sanity check: compare a lightweight tent vs an insulated adobe wall
    tent = compute_composite_wall([("canvas_tent_fabric", 0.005)], envelope_area=60)
    insulated_adobe = compute_composite_wall(
        [("mud_brick_adobe", 0.25), ("eps_insulation", 0.05)], envelope_area=60
    )
    print("Canvas tent:      ", tent)
    print("Insulated adobe:  ", insulated_adobe)
