import pandas as pd
import numpy as np
import os

np.random.seed(42)

DISTRICTS = {
    "Jaipur":       (26.9124, 75.7873),
    "Jodhpur":      (26.2389, 73.0243),
    "Kota":         (25.2138, 75.8648),
    "Ajmer":        (26.4499, 74.6399),
    "Bhiwadi":      (28.2000, 76.8500),
    "Alwar":        (27.5665, 76.6200),
    "Udaipur":      (24.5854, 73.7125),
    "Bikaner":      (28.0229, 73.3119),
    "Sikar":        (27.6094, 75.1399),
    "Chittorgarh":  (24.8887, 74.6269),
}

CHEMICAL_TYPES = [
    "Chlorine", "Ammonia", "Sulfuric Acid", "Nitric Acid",
    "LPG", "Benzene", "Acetone", "Hydrogen Peroxide",
    "Sodium Hydroxide", "Hydrochloric Acid"
]

FACTORY_PREFIXES = [
    "Rajasthan", "Marwar", "Mewar", "Shekhawati", "Thar",
    "Aravalli", "Chambal", "Luni", "Banas", "Mahi"
]
FACTORY_TYPES = [
    "Chemical Works", "Petrochemicals Ltd", "Industries Pvt Ltd",
    "Manufacturing Co", "Processing Plant", "Refinery Ltd",
    "Chemicals & Polymers", "Agro Chemicals", "Pharma Chemicals", "Dye Works"
]

ANTIDOTE_MAP = {
    "Chlorine":           "Chlorine antidote kit",
    "Ammonia":            "Ammonia neutralizer",
    "Sulfuric Acid":      "Acid burn kit",
    "Nitric Acid":        "Acid burn kit",
    "LPG":                "Burn treatment kit",
    "Benzene":            "Activated charcoal",
    "Acetone":            "General solvent kit",
    "Hydrogen Peroxide":  "Oxidant antidote kit",
    "Sodium Hydroxide":   "Alkali burn kit",
    "Hydrochloric Acid":  "Acid burn kit",
}

def jitter(lat, lon, spread=0.3):
    return lat + np.random.uniform(-spread, spread), lon + np.random.uniform(-spread, spread)

def generate_factories(n=500):
    rows = []
    districts = list(DISTRICTS.keys())
    per_district = n // len(districts)
    factory_id = 1

    for district in districts:
        base_lat, base_lon = DISTRICTS[district]
        count = per_district if district != districts[-1] else n - len(rows)
        for _ in range(count):
            lat, lon = jitter(base_lat, base_lon, 0.25)
            prefix = np.random.choice(FACTORY_PREFIXES)
            ftype  = np.random.choice(FACTORY_TYPES)
            name   = f"{prefix} {ftype} {factory_id}"
            chemical = np.random.choice(CHEMICAL_TYPES)
            licensed_capacity = np.random.randint(500, 5000)
            storage_used      = licensed_capacity * np.random.uniform(0.5, 1.4)
            violations        = int(np.random.exponential(2.5))
            months_since_insp = int(np.random.uniform(1, 48))
            complaints        = int(np.random.exponential(3))
            accident_history  = int(np.random.poisson(1.2))

            rows.append({
                "factory_id":          factory_id,
                "name":                name,
                "district":            district,
                "latitude":            round(lat, 5),
                "longitude":           round(lon, 5),
                "chemical_type":       chemical,
                "licensed_capacity_kl": licensed_capacity,
                "storage_used_kl":     round(storage_used, 1),
                "storage_overflow_pct": round((storage_used / licensed_capacity - 1) * 100, 2),
                "violation_count":     violations,
                "months_since_inspection": months_since_insp,
                "worker_complaints":   complaints,
                "accident_history":    accident_history,
            })
            factory_id += 1

    return pd.DataFrame(rows)


def generate_hospitals(n=50):
    rows = []
    districts = list(DISTRICTS.keys())
    for i in range(n):
        district = districts[i % len(districts)]
        base_lat, base_lon = DISTRICTS[district]
        lat, lon = jitter(base_lat, base_lon, 0.4)
        antidote_count = np.random.randint(2, 7)
        antidotes = list(np.random.choice(list(set(ANTIDOTE_MAP.values())), antidote_count, replace=False))
        rows.append({
            "hospital_id":  i + 1,
            "name":         f"{district} District Hospital {i+1}",
            "district":     district,
            "latitude":     round(lat, 5),
            "longitude":    round(lon, 5),
            "bed_count":    np.random.randint(50, 500),
            "antidote_stock": "|".join(antidotes),
            "icu_beds":     np.random.randint(5, 50),
        })
    return pd.DataFrame(rows)


def generate_fire_stations(n=30):
    rows = []
    districts = list(DISTRICTS.keys())
    for i in range(n):
        district = districts[i % len(districts)]
        base_lat, base_lon = DISTRICTS[district]
        lat, lon = jitter(base_lat, base_lon, 0.35)
        rows.append({
            "station_id":       i + 1,
            "name":             f"{district} Fire Station {i+1}",
            "district":         district,
            "latitude":         round(lat, 5),
            "longitude":        round(lon, 5),
            "truck_count":      np.random.randint(2, 12),
            "chemical_foam_kl": np.random.randint(500, 5000),
            "hazmat_team":      np.random.choice([True, False]),
            "capacity_score":   np.random.randint(40, 100),
        })
    return pd.DataFrame(rows)


def generate_ambulance_depots(n=20):
    rows = []
    districts = list(DISTRICTS.keys())
    for i in range(n):
        district = districts[i % len(districts)]
        base_lat, base_lon = DISTRICTS[district]
        lat, lon = jitter(base_lat, base_lon, 0.45)
        rows.append({
            "depot_id":    i + 1,
            "name":        f"{district} Ambulance Depot {i+1}",
            "district":    district,
            "latitude":    round(lat, 5),
            "longitude":   round(lon, 5),
            "fleet_size":  np.random.randint(3, 20),
            "als_units":   np.random.randint(1, 8),
            "on_standby":  False,
        })
    return pd.DataFrame(rows)


def save_all(out_dir="data"):
    os.makedirs(out_dir, exist_ok=True)
    factories  = generate_factories(500)
    hospitals  = generate_hospitals(50)
    fire_st    = generate_fire_stations(30)
    amb_depots = generate_ambulance_depots(20)

    factories.to_csv(f"{out_dir}/factories.csv",        index=False)
    hospitals.to_csv(f"{out_dir}/hospitals.csv",         index=False)
    fire_st.to_csv(f"{out_dir}/fire_stations.csv",       index=False)
    amb_depots.to_csv(f"{out_dir}/ambulance_depots.csv", index=False)
    print(f"[data_generator] Saved {len(factories)} factories, {len(hospitals)} hospitals, "
          f"{len(fire_st)} fire stations, {len(amb_depots)} ambulance depots → {out_dir}/")
    return factories, hospitals, fire_st, amb_depots


if __name__ == "__main__":
    save_all()
