import pandas as pd
import numpy as np
import os
from math import radians, sin, cos, sqrt, atan2

from model import load_model, predict_score, score_to_label, FEATURES, compute_raw_score, RISK_BINS, RISK_LABELS
from data_generator import save_all, ANTIDOTE_MAP

DATA_DIR = "data"

# Chemical → firefighting requirement
CHEM_FOAM_NEEDED = {
    "Chlorine":           2000,
    "Ammonia":            1500,
    "Sulfuric Acid":      2500,
    "Nitric Acid":        2500,
    "LPG":                3000,
    "Benzene":            3000,
    "Acetone":            2000,
    "Hydrogen Peroxide":  1800,
    "Sodium Hydroxide":   1200,
    "Hydrochloric Acid":  2200,
}

CHEM_HAZMAT_NEEDED = {
    "Chlorine": True, "Ammonia": True, "Benzene": True,
    "LPG": True, "Nitric Acid": True,
}

GAP_RADIUS_KM = 50   # flag gap if nearest resource > this distance


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def nearest_resource(factory_row, resource_df, extra_cols=None):
    distances = resource_df.apply(
        lambda r: haversine(factory_row.latitude, factory_row.longitude, r.latitude, r.longitude),
        axis=1
    )
    idx = distances.idxmin()
    dist = distances[idx]
    row  = resource_df.loc[idx]
    result = {"distance_km": round(dist, 2), "name": row["name"], "id": idx}
    if extra_cols:
        for c in extra_cols:
            result[c] = row[c]
    return result


def check_supply_gaps(factory_row, nearest_hosp, nearest_fire, nearest_amb):
    gaps = []
    chem = factory_row["chemical_type"]

    # Hospital checks
    required_antidote = ANTIDOTE_MAP.get(chem, "")
    antidotes_available = str(nearest_hosp.get("antidote_stock", "")).split("|")
    if nearest_hosp["distance_km"] > GAP_RADIUS_KM:
        gaps.append(f"No hospital within {GAP_RADIUS_KM}km (nearest: {nearest_hosp['distance_km']}km)")
    elif required_antidote and required_antidote not in antidotes_available:
        gaps.append(f"No {required_antidote} within {GAP_RADIUS_KM}km for {chem} storage")

    # Fire station checks
    foam_needed = CHEM_FOAM_NEEDED.get(chem, 1500)
    hazmat_needed = CHEM_HAZMAT_NEEDED.get(chem, False)
    if nearest_fire["distance_km"] > GAP_RADIUS_KM:
        gaps.append(f"No fire station within {GAP_RADIUS_KM}km (nearest: {nearest_fire['distance_km']}km)")
    else:
        if nearest_fire.get("chemical_foam_kl", 0) < foam_needed:
            gaps.append(f"Fire station foam capacity insufficient ({nearest_fire.get('chemical_foam_kl',0)}kL available, {foam_needed}kL needed for {chem})")
        if hazmat_needed and not nearest_fire.get("hazmat_team", False):
            gaps.append(f"No HAZMAT team at nearest fire station — required for {chem} incident")

    # Ambulance checks
    if nearest_amb["distance_km"] > GAP_RADIUS_KM:
        gaps.append(f"No ambulance depot within {GAP_RADIUS_KM}km (nearest: {nearest_amb['distance_km']}km)")
    elif nearest_amb.get("fleet_size", 0) < 5:
        gaps.append(f"Ambulance fleet too small ({nearest_amb.get('fleet_size',0)} units) — minimum 5 needed for Critical factory")

    return gaps


def generate_recommendations(factory_row, nearest_hosp, nearest_fire, nearest_amb, gaps):
    recs = []
    chem     = factory_row["chemical_type"]
    fname    = factory_row["name"]
    district = factory_row["district"]
    antidote = ANTIDOTE_MAP.get(chem, "General emergency kit")
    foam     = CHEM_FOAM_NEEDED.get(chem, 1500)

    recs.append(f"🏥 {nearest_hosp['name']} should pre-stock {antidote} (minimum 20 units) for potential {chem} exposure incident at {fname}.")
    recs.append(f"🚒 {nearest_fire['name']} must deploy at least {foam}kL of chemical foam and position one truck on standby within 10km of {fname}.")
    recs.append(f"🚑 {nearest_amb['name']} should place {min(nearest_amb.get('fleet_size', 3), 3)} ALS ambulances on immediate standby for {fname} in {district}.")

    if CHEM_HAZMAT_NEEDED.get(chem, False) and not nearest_fire.get("hazmat_team", False):
        recs.append(f"⚠️ HAZMAT team must be dispatched from the next available station — {nearest_fire['name']} does not have HAZMAT capability for {chem} incident.")

    overflow = factory_row.get("storage_overflow_pct", 0)
    if overflow > 20:
        recs.append(f"📋 District Collector {district} should issue immediate notice to {fname} to reduce {chem} storage — currently {overflow:.1f}% over licensed limit.")

    if factory_row.get("months_since_inspection", 0) > 24:
        recs.append(f"🔍 Schedule emergency safety inspection at {fname} — last inspected {factory_row['months_since_inspection']:.0f} months ago.")

    return recs


def load_all_data():
    paths = {
        "factories":        f"{DATA_DIR}/factories.csv",
        "hospitals":        f"{DATA_DIR}/hospitals.csv",
        "fire_stations":    f"{DATA_DIR}/fire_stations.csv",
        "ambulance_depots": f"{DATA_DIR}/ambulance_depots.csv",
    }
    missing = [k for k, v in paths.items() if not os.path.exists(v)]
    if missing:
        save_all(DATA_DIR)

    factories  = pd.read_csv(paths["factories"])
    hospitals  = pd.read_csv(paths["hospitals"])
    fire_st    = pd.read_csv(paths["fire_stations"])
    amb_depots = pd.read_csv(paths["ambulance_depots"])
    return factories, hospitals, fire_st, amb_depots


def run_risk_engine():
    factories, hospitals, fire_st, amb_depots = load_all_data()
    clf, scaler = load_model()

    np.random.seed(42)

    # Compute risk scores
    if "raw_score" not in factories.columns:
        factories["raw_score"] = factories.apply(compute_raw_score, axis=1)

    factories["risk_score"] = factories.apply(
        lambda row: predict_score(clf, scaler, row.to_dict()), axis=1
    )
    factories["risk_label"] = factories["risk_score"].apply(score_to_label)

    # Build gap & recommendation records for Critical factories
    critical = factories[factories["risk_label"] == "Critical"].copy()

    gap_records  = []
    rec_records  = []

    for _, frow in critical.iterrows():
        nh = nearest_resource(frow, hospitals,  extra_cols=["antidote_stock", "bed_count", "icu_beds"])
        nf = nearest_resource(frow, fire_st,    extra_cols=["chemical_foam_kl", "hazmat_team", "truck_count"])
        na = nearest_resource(frow, amb_depots, extra_cols=["fleet_size", "als_units"])

        gaps = check_supply_gaps(frow, nh, nf, na)
        recs = generate_recommendations(frow, nh, nf, na, gaps)

        gap_records.append({
            "factory_id":         frow["factory_id"],
            "factory_name":       frow["name"],
            "district":           frow["district"],
            "chemical_type":      frow["chemical_type"],
            "risk_score":         frow["risk_score"],
            "nearest_hospital":   nh["name"],
            "hospital_dist_km":   nh["distance_km"],
            "hospital_beds":      nh.get("bed_count", "N/A"),
            "nearest_fire_stn":   nf["name"],
            "fire_dist_km":       nf["distance_km"],
            "fire_foam_kl":       nf.get("chemical_foam_kl", "N/A"),
            "hazmat_available":   nf.get("hazmat_team", False),
            "nearest_ambulance":  na["name"],
            "ambulance_dist_km":  na["distance_km"],
            "ambulance_fleet":    na.get("fleet_size", "N/A"),
            "gap_count":          len(gaps),
            "gaps":               " | ".join(gaps) if gaps else "No critical gaps",
        })

        rec_records.append({
            "factory_id":    frow["factory_id"],
            "factory_name":  frow["name"],
            "district":      frow["district"],
            "chemical_type": frow["chemical_type"],
            "risk_score":    frow["risk_score"],
            "latitude":      frow["latitude"],
            "longitude":     frow["longitude"],
            "recommendations": "\n".join(recs),
        })

    gaps_df = pd.DataFrame(gap_records)
    recs_df = pd.DataFrame(rec_records)

    return factories, hospitals, fire_st, amb_depots, gaps_df, recs_df
