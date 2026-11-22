def ldg_dist_ws(p: dict[str, float]) -> float:
    s_rwy = p["distance_runway_m"]
    s_flare = p["distance_flare_m"]
    rho = p["competition_density_kg_m3"]
    mu_ldg = p["mu_LDG_unitless"]
    g = p["gravity_sea_level_m_per_s2"]
    cl_max_LDG = p["cl_max_LDG_unitless"]
    
    s_ground_roll_m = s_rwy - s_flare

    return (mu_ldg * g * s_ground_roll_m * rho * cl_max_LDG) / (1.15 ** 2)