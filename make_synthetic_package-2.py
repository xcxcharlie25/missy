"""
make_synthetic_package.py   Synthetic data package for D-Wave, same structure as the real pipeline.

Everything here is invented: areas, pharmacies, chains (CHAIN A to F), members, drugs, prices. No real
location, company, member or claim. State and metro codes are synthetic (XA, A001).

Like the real files, every pharmacy is a retail network pharmacy. PHR_DISPNSR_TYPE_DESC is the sub
category inside retail (Community or Retail, Clinic, Compounding, VA, DME, Home Infusion, ...) and
PHR_DISPNSR_CLASS_DESC the class (Chain, Independent, Franchise, Government, Alternate Dispensing Site).
Type and class together are the KIND of pharmacy; the normal kind is Community or Retail with Chain,
Independent or Franchise. Members mostly use one normal pharmacy, some also a special kind.

Produces, in OUT_DIR, the two files D-Wave gets:
    synthetic_claims_2025.csv       one row per fill, all clients, exact real column names
    universal_pharmacies.csv        the search space: one row per pharmacy x drug (GPI + generic/brand) with the pharmacy's
                                    columns from the claims (kind, chain, state, metro, location, volume, capacity) and the
                                    average cost of that drug there: unit ingredient cost and dispensing fee per fill, from the
                                    client's own claims where it filled the drug there, else from all clients' claims

With IN_HOUSE_TABLES = True it also writes, in OUT_DIR/in_house, the tables the in house model reads
(book_tables and member_tables as parquet, built with the same rules as book_pharmacy_extract.py and
fedex_member_extract.py) plus a README. Those stay in house: run  python pnm_run.py OUT_DIR/in_house

Run:  python make_synthetic_package.py            (edit CONFIG below)
Needs: pandas, numpy, pyarrow
"""
import os, math, json
import numpy as np, pandas as pd

# ================================================================== CONFIG =====
CONFIG = dict(
    OUT_DIR = "synthetic_package",
    SEED = 2025,
    YEAR = 2025,
    # synthetic areas: a centre, a synthetic state and metro code, pharmacies, members of the client and of other clients
    AREAS = [
        dict(name="AREA 1", lat=41.20, lon=-96.00,  state="XA", msa="A001", pharmacies=260, client_members=4500, other_members=18000),
        dict(name="AREA 2", lat=36.40, lon=-108.30, state="XB", msa="A002", pharmacies=200, client_members=3500, other_members=14000),
        dict(name="AREA 3", lat=44.70, lon=-84.60,  state="XC", msa="A003", pharmacies=140, client_members=2000, other_members=8000),
    ],
    AREA_SPREAD_MILES = 14,              # how far pharmacies spread around an area centre
    RURAL = dict(pharmacies=40, client_members=300, other_members=1200, state="XD", lat=39.0, lon=-99.0, spread_miles=120),
    CLIENT = "CLIENT A",                 # the client whose network is designed
    OTHER_CLIENTS = ["CLIENT B", "CLIENT C", "CLIENT D"],
    ACCOUNTS = ["001", "003", "021"], GROUPS = ["GRP01", "GRP02", "GRP03", "GRP04"],
    CLIENT_PHARMACY_SHARE = 0.45,        # share of retail pharmacies the client's members use at all
    CHAINS = ["CHAIN A", "CHAIN B", "CHAIN C", "CHAIN D", "CHAIN E", "CHAIN F"],
    CHAIN_RATE = {"CHAIN A": 0.82, "CHAIN B": 0.86, "CHAIN C": 0.90, "CHAIN D": 0.95, "CHAIN E": 1.00, "CHAIN F": 1.05,
                  "FRANCHISE X": 1.02, "FRANCHISE Y": 1.06, "Independent": 1.12, "GOVERNMENT": 0.95, "ALTERNATE": 1.08},
    CHAIN_FEE = {"CHAIN A": 1.00, "CHAIN B": 1.25, "CHAIN C": 1.50, "CHAIN D": 1.50, "CHAIN E": 1.75, "CHAIN F": 2.00,
                 "FRANCHISE X": 1.75, "FRANCHISE Y": 2.00, "Independent": 2.25, "GOVERNMENT": 1.00, "ALTERNATE": 2.50},
    # dispenser type (sub category inside retail) mix of the universe. These are the values seen in the real data;
    # paste the full real list here if it has more
    TYPE_MIX = {"Community or Retail Pharmacy": 0.88, "Clinic Pharmacy": 0.03, "Compounding Pharmacy": 0.02,
                "Department of Veterans Affairs VA Pharmacy": 0.01, "DME": 0.01, "Home Infusion Therapy Provider": 0.025,
                "Indian Health Services": 0.005, "Institutional Pharmacy": 0.02},
    GOVERNMENT_TYPES = ["Department of Veterans Affairs VA Pharmacy", "Indian Health Services"],   # these types get class Government Pharmacy
    # dispensing class mix of Community or Retail pharmacies (other types are Chain or Independent, or Government per the list above)
    CLASS_MIX = {"Chain Pharmacy": 0.62, "Independent Pharmacy": 0.22, "Franchise Pharmacy": 0.08, "Government Pharmacy": 0.04, "Alternate Dispensing Site": 0.04},
    NORMAL_TYPE = "Community or Retail Pharmacy", NORMAL_CLASSES = ["Chain Pharmacy", "Independent Pharmacy", "Franchise Pharmacy"],
    SPECIAL_KIND_SHARE = 0.12,           # share of members who also use one special kind of pharmacy (government, clinic, ...)
    SPECIAL_ONLY_SHARE = 0.02,           # share of members whose home pharmacy is a special kind (they fill nowhere normal)
    N_DRUGS = 300, GENERIC_SHARE = 0.80, MAINT_SHARE = 0.40, SPECIALTY_SHARE = 0.05,
    FILLS_PER_MEMBER = (4, 16),          # acute plus maintenance refills land in this range for most members
    REJECT_RATE = 0.03,                  # rows with m_nf_app_due_amt = 0 (rejected or reversed), dropped by the extracts
    CAPACITY_SHARE = 0.90,               # 2025 volume = this share of a pharmacy's capacity
    IN_HOUSE_TABLES = False,             # True: also write the model's own tables to OUT_DIR/in_house (never shared)
)
C = CONFIG
rng = np.random.default_rng(C["SEED"])
OUT = C["OUT_DIR"]; os.makedirs(OUT, exist_ok=True)
HOUSE = os.path.join(OUT, "in_house") if C["IN_HOUSE_TABLES"] else None
if HOUSE:
    os.makedirs(os.path.join(HOUSE, "book_tables"), exist_ok=True); os.makedirs(os.path.join(HOUSE, "member_tables"), exist_ok=True)
R_EARTH = 3958.7613
def haversine(lat1, lon1, lat2, lon2):
    la1, lo1, la2, lo2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((la2-la1)/2)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
    return 2*R_EARTH*np.arcsin(np.sqrt(np.clip(a, 0, 1)))
def scatter(lat, lon, miles, n):
    """points around a centre, roughly within `miles`"""
    r = miles*np.sqrt(rng.random(n)); t = rng.uniform(0, 2*np.pi, n)
    return lat + r*np.sin(t)/69.0, lon + r*np.cos(t)/(69.0*np.cos(np.radians(lat)))

# ================================================================== 1. PHARMACIES ==
rows = []; pid = 1000000
def add_pharmacies(n, lat0, lon0, spread, state, msa, city):
    global pid
    la, lo = scatter(lat0, lon0, spread, n)
    types = rng.choice(list(C["TYPE_MIX"]), n, p=np.array(list(C["TYPE_MIX"].values()))/sum(C["TYPE_MIX"].values()))
    for i in range(n):
        t = types[i]
        if t == C["NORMAL_TYPE"]:
            cls = rng.choice(list(C["CLASS_MIX"]), p=np.array(list(C["CLASS_MIX"].values()))/sum(C["CLASS_MIX"].values()))
        elif t in C["GOVERNMENT_TYPES"]: cls = "Government Pharmacy"
        else: cls = "Chain Pharmacy" if rng.random() < 0.5 else "Independent Pharmacy"
        if cls == "Chain Pharmacy": chain = rng.choice(C["CHAINS"])
        elif cls == "Franchise Pharmacy": chain = rng.choice(["FRANCHISE X", "FRANCHISE Y"])
        elif cls == "Government Pharmacy": chain = "GOVERNMENT"
        elif cls == "Alternate Dispensing Site": chain = "ALTERNATE"
        else: chain = "Independent"
        name = f"{chain} STORE {pid}" if chain not in ("Independent",) else f"PHARMACY {pid}"
        rows.append(dict(PHR_NCPDP_ID=f"{pid:07d}", PHR_NPI_NO=f"19{pid:08d}", PHR_NM=name, PHR_AFFL_NM=(chain if cls == "Chain Pharmacy" else None),
                         a_nf_pharmacy=chain, PHR_DISPNSR_TYPE_DESC=t, PHR_DISPNSR_CLASS_DESC=cls, PHR_ST_CD=state, LOC_MSA_CD=msa,
                         PHR_ZIP_CD=f"{90000 + rng.integers(0, 900):05d}", PHR_CITY_NM=city, PHR_ADDR1_TXT=f"{rng.integers(1, 9999)} {rng.choice(['MAIN', 'OAK', 'PARK', 'LAKE', 'HILL'])} ST",
                         phr_lat=round(float(la[i]), 6), phr_long=round(float(lo[i]), 6),
                         store_factor=float(rng.uniform(0.94, 1.06)), area=city))
        pid += 1
for a in C["AREAS"]: add_pharmacies(a["pharmacies"], a["lat"], a["lon"], C["AREA_SPREAD_MILES"], a["state"], a["msa"], a["name"])
ru = C["RURAL"]; add_pharmacies(ru["pharmacies"], ru["lat"], ru["lon"], ru["spread_miles"], ru["state"], None, "RURAL")
ph = pd.DataFrame(rows)
ph["rate"] = ph.a_nf_pharmacy.map(C["CHAIN_RATE"]).fillna(1.0) * ph.store_factor
ph["fee"] = ph.a_nf_pharmacy.map(C["CHAIN_FEE"]).fillna(1.5)
ph["kind"] = ph.PHR_DISPNSR_TYPE_DESC + " | " + ph.PHR_DISPNSR_CLASS_DESC
ph["normal"] = (ph.PHR_DISPNSR_TYPE_DESC == C["NORMAL_TYPE"]) & ph.PHR_DISPNSR_CLASS_DESC.isin(C["NORMAL_CLASSES"])
ret = ph.reset_index(drop=True)                                          # every pharmacy is a retail network pharmacy
client_ok = rng.random(len(ret)) < C["CLIENT_PHARMACY_SHARE"]           # pharmacies the client's members may use
client_ret = ret[client_ok].reset_index(drop=True)
ret_normal = ret[ret.normal].reset_index(drop=True); client_normal = client_ret[client_ret.normal].reset_index(drop=True)
print(f"pharmacies {len(ph):,} | normal kind {int(ph.normal.sum()):,} | the client can use {int(client_ok.sum()):,}")
print("kinds: " + ", ".join(f"{k} {n}" for k, n in ph.kind.value_counts().items()))

# ================================================================== 2. DRUGS ==
n_d = C["N_DRUGS"]
gpi = [f"{rng.integers(10, 99):02d}{rng.integers(10, 99):02d}{rng.integers(10, 99):02d}{rng.integers(10, 99):02d}{rng.integers(100000, 999999):06d}" for _ in range(n_d)]
is_spec = rng.random(n_d) < C["SPECIALTY_SHARE"]
is_gen = (rng.random(n_d) < C["GENERIC_SHARE"]) & ~is_spec
is_maint = rng.random(n_d) < C["MAINT_SHARE"]
base = np.where(is_spec, rng.uniform(80, 400, n_d), np.where(is_gen, rng.uniform(0.04, 0.9, n_d), rng.uniform(3, 25, n_d)))
drugs = pd.DataFrame(dict(a_nt_gpi=gpi, a_irpg_drugtype=np.where(is_gen, "G", "B"), a_nf_specialty_flag=np.where(is_spec, "Y", "N"),
                          a_nt_maintenance_flag=np.where(is_maint, "Y", "N"), base_unit=base, qpd=rng.choice([1, 1, 1, 2], n_d),
                          weight=np.where(is_maint, 3.0, 1.0) * np.where(is_gen, 2.0, 1.0)))
drugs["weight"] /= drugs.weight.sum()
print(f"drugs {n_d}: generic {int(is_gen.sum())}, brand {int((~is_gen & ~is_spec).sum())}, specialty {int(is_spec.sum())}, maintenance {int(is_maint.sum())}")

# ================================================================== 3. MEMBERS ==
# each member lives near a home pharmacy of the normal kind, mostly fills there, sometimes at a second pharmacy of the
# same kind; some also use one special kind (government, clinic, VA, ...). Other clients' members use any pharmacy.
def make_members(n, area_ret, client, start_id, for_client, home_miles=1.5):
    if for_client: pool = area_ret[client_ok[area_ret.index]] if len(area_ret[client_ok[area_ret.index]]) else area_ret
    else: pool = area_ret
    normal = pool[pool.normal]
    if len(normal) == 0: normal = pool
    special = pool[~pool.normal]
    hi = rng.integers(0, len(normal), n); home = normal.iloc[hi].reset_index(drop=True)
    if len(special):                                                        # a few members live on a special kind of pharmacy only
        so = np.where(rng.random(n) < C["SPECIAL_ONLY_SHARE"])[0]
        home.iloc[so] = special.iloc[rng.integers(0, len(special), len(so))].values
    la, lo = scatter(0, 0, home_miles, n); la = home.phr_lat.values + la; lo = home.phr_long.values + lo
    m = pd.DataFrame(dict(MBR_ID=np.arange(start_id, start_id + n), mbr_lat=np.round(la, 6), mbr_long=np.round(lo, 6),
                          CARRIER_ID=client, ACCOUNT_ID=rng.choice(C["ACCOUNTS"], n), EMPLOYER_GROUP_ID=rng.choice(C["GROUPS"], n),
                          home=home.PHR_NCPDP_ID.values, home_kind=home.kind.values))
    # second pharmacy of the same kind within reach; sometimes one special kind of pharmacy (the nearest of a random special kind)
    second, extra = [], []
    kinds_special = sorted(special.kind.unique())
    for i in range(n):
        same = pool[(pool.kind == m.home_kind.iat[i]) & (pool.PHR_NCPDP_ID != m.home.iat[i])]
        if len(same):
            d = haversine(m.mbr_lat.iat[i], m.mbr_long.iat[i], same.phr_lat.values, same.phr_long.values)
            second.append(same.PHR_NCPDP_ID.values[np.argsort(d)[min(len(d)-1, rng.integers(0, 3))]])
        else: second.append(m.home.iat[i])
        if kinds_special and rng.random() < C["SPECIAL_KIND_SHARE"]:
            cand = special[special.kind == kinds_special[rng.integers(0, len(kinds_special))]]
            d = haversine(m.mbr_lat.iat[i], m.mbr_long.iat[i], cand.phr_lat.values, cand.phr_long.values); extra.append(cand.PHR_NCPDP_ID.values[np.argmin(d)])
        else: extra.append(None)
    m["second"] = second; m["extra"] = extra
    return m
members = []; next_id = 1
for a in C["AREAS"] + [dict(name="RURAL", client_members=ru["client_members"], other_members=ru["other_members"])]:
    area_ret = ret[ret.area == a["name"]]; hm = 1.5 if a["name"] != "RURAL" else 12.0      # rural members live farther from their pharmacy
    members.append(make_members(a["client_members"], area_ret, C["CLIENT"], next_id, True, hm)); next_id += a["client_members"]
    per = a["other_members"] // len(C["OTHER_CLIENTS"])
    for oc in C["OTHER_CLIENTS"]:
        members.append(make_members(per, area_ret, oc, next_id, False, hm)); next_id += per
members = pd.concat(members, ignore_index=True)
print(f"members {len(members):,} | {C['CLIENT']} {int((members.CARRIER_ID == C['CLIENT']).sum()):,}")

# ================================================================== 4. CLAIMS ==
PH_RATE = ph.set_index("PHR_NCPDP_ID").rate.to_dict(); PH_FEE = ph.set_index("PHR_NCPDP_ID").fee.to_dict()
client_factor = {C["CLIENT"]: 1.00}; client_factor.update({oc: float(rng.uniform(0.97, 1.03)) for oc in C["OTHER_CLIENTS"]})
year0 = pd.Timestamp(f"{C['YEAR']}-01-01"); ndays = 365
claims = []
lo_f, hi_f = C["FILLS_PER_MEMBER"]
for mi, m in enumerate(members.itertuples()):
    n_drugs = int(rng.integers(1, 7))
    dk = rng.choice(n_d, n_drugs, replace=False, p=drugs.weight.values)
    for k in dk:
        d = drugs.iloc[k]
        if d.a_nt_maintenance_flag == "Y":
            supply = int(rng.choice([30, 90], p=[0.7, 0.3])); start = int(rng.integers(0, 60)); n_fill = max(1, (ndays - start) // supply)
            n_fill = int(round(n_fill * rng.uniform(0.6, 1.0)))                # adherence below 100%
            dates = [start + j*supply + int(rng.integers(0, 4)) for j in range(n_fill)]
        else:
            supply = int(rng.choice([5, 7, 10, 14, 30])); dates = [int(rng.integers(0, ndays)) for _ in range(int(rng.integers(1, 3)))]
        for day in dates:
            if day >= ndays: continue
            u = rng.random()
            if u < 0.80: p_id = m.home
            elif u < 0.95: p_id = m.second
            elif u < 0.995: p_id = m.extra if isinstance(m.extra, str) else m.home       # the member's special kind of pharmacy, if they have one
            else:                                                                       # a one off fill at some normal pharmacy
                pool_r = client_normal if m.CARRIER_ID == C["CLIENT"] else ret_normal
                p_id = pool_r.PHR_NCPDP_ID.values[rng.integers(0, len(pool_r))]
            qty = float(supply * d.qpd)
            ingred = round(float(d.base_unit * qty * PH_RATE[p_id] * client_factor[m.CARRIER_ID] * rng.uniform(0.97, 1.03)), 2)
            fee = float(PH_FEE[p_id])
            claims.append((m.MBR_ID, m.CARRIER_ID, m.ACCOUNT_ID, m.EMPLOYER_GROUP_ID, m.mbr_lat, m.mbr_long, p_id, d.a_nt_gpi, d.a_irpg_drugtype,
                           d.a_nf_specialty_flag, d.a_nt_maintenance_flag, qty, supply, day, ingred, fee))
cl = pd.DataFrame(claims, columns=["MBR_ID", "CARRIER_ID", "ACCOUNT_ID", "EMPLOYER_GROUP_ID", "mbr_lat", "mbr_long", "PHR_NCPDP_ID", "a_nt_gpi", "a_irpg_drugtype",
                                   "a_nf_specialty_flag", "a_nt_maintenance_flag", "a_nf_drg_qty", "DAYS_SUPPLY", "day", "m_nf_clt_ingred_cost_paid", "m_nf_clt_dispensing_fee"])
cl = cl.sample(frac=1.0, random_state=C["SEED"]).reset_index(drop=True)
cl["FILLED_DT"] = (year0 + pd.to_timedelta(cl.day, unit="D")).dt.strftime("%Y-%m-%d"); cl = cl.drop(columns="day")
cl["CLAIM_ID"] = [f"C{C['YEAR']}{i:09d}" for i in range(1, len(cl) + 1)]
cl["m_nf_app_due_amt"] = (cl.m_nf_clt_ingred_cost_paid + cl.m_nf_clt_dispensing_fee).round(2)
cl["a_nf_clt_prc_typ"] = np.where(cl.a_irpg_drugtype == "G", "MAC", rng.choice(["AWP", "U&C"], len(cl), p=[0.9, 0.1]))
cl["a_nf_phr_prc_typ"] = cl.a_nf_clt_prc_typ
rej = rng.random(len(cl)) < C["REJECT_RATE"]
cl.loc[rej, "m_nf_app_due_amt"] = 0.0; cl.loc[rej & (rng.random(len(cl)) < 0.5), "a_nf_clt_prc_typ"] = "ZERO"
cl = cl.merge(ph.drop(columns=["store_factor", "rate", "fee", "area", "kind", "normal"]), on="PHR_NCPDP_ID", how="left")
cols = ["CLAIM_ID", "CARRIER_ID", "ACCOUNT_ID", "EMPLOYER_GROUP_ID", "MBR_ID", "mbr_lat", "mbr_long",
        "PHR_NCPDP_ID", "PHR_NPI_NO", "PHR_NM", "PHR_AFFL_NM", "a_nf_pharmacy", "PHR_DISPNSR_TYPE_DESC", "PHR_DISPNSR_CLASS_DESC",
        "PHR_ST_CD", "LOC_MSA_CD", "PHR_ZIP_CD", "PHR_CITY_NM", "PHR_ADDR1_TXT", "phr_lat", "phr_long",
        "a_nt_gpi", "a_irpg_drugtype", "a_nf_specialty_flag", "a_nt_maintenance_flag", "a_nf_drg_qty", "DAYS_SUPPLY", "FILLED_DT",
        "m_nf_clt_ingred_cost_paid", "m_nf_clt_dispensing_fee", "m_nf_app_due_amt", "a_nf_clt_prc_typ", "a_nf_phr_prc_typ"]
cl = cl[cols]
cl.to_csv(os.path.join(OUT, f"synthetic_claims_{C['YEAR']}.csv"), index=False)                      # D-Wave file 1
if HOUSE: cl.to_parquet(os.path.join(HOUSE, f"synthetic_claims_{C['YEAR']}.parquet"), index=False)
print(f"claims {len(cl):,} | {C['CLIENT']} {int((cl.CARRIER_ID == C['CLIENT']).sum()):,} | rejected rows {int(rej.sum()):,}")

# ================================================================== 5. THE TABLES, same rules as the extracts ==
# D-Wave file 2 = book_pharmacies joined to book_price_table (one row per pharmacy x drug). The in house model's own
# tables are written only with IN_HOUSE_TABLES
paid = cl[(cl.m_nf_app_due_amt > 0) & cl.a_nf_clt_prc_typ.fillna("").str.strip().ne("") & (cl.a_nf_clt_prc_typ != "ZERO")].copy()
paid["is_client"] = (paid.CARRIER_ID == C["CLIENT"]).astype(int)
def save(df, folder, name):
    if HOUSE: df.to_parquet(os.path.join(HOUSE, folder, name + ".parquet"), index=False)

# pharmacy universe
attrs = ["PHR_NPI_NO", "PHR_NM", "PHR_AFFL_NM", "a_nf_pharmacy", "PHR_DISPNSR_TYPE_DESC", "PHR_DISPNSR_CLASS_DESC", "PHR_ST_CD", "LOC_MSA_CD", "PHR_ZIP_CD", "PHR_CITY_NM", "PHR_ADDR1_TXT"]
g = paid.groupby("PHR_NCPDP_ID")
bp = g[attrs].first()
bp["PHR_LAT"] = g.phr_lat.median(); bp["PHR_LON"] = g.phr_long.median()
bp["fills_all"] = g.size(); bp["fills_client"] = g.is_client.sum()
bp["ingred_all"] = g.m_nf_clt_ingred_cost_paid.sum(); bp["fee_all"] = g.m_nf_clt_dispensing_fee.sum()
bp["ingred_client"] = paid[paid.is_client == 1].groupby("PHR_NCPDP_ID").m_nf_clt_ingred_cost_paid.sum().reindex(bp.index).fillna(0.0)
bp["fee_client"] = paid[paid.is_client == 1].groupby("PHR_NCPDP_ID").m_nf_clt_dispensing_fee.sum().reindex(bp.index).fillna(0.0)
bp["first_fill"] = g.FILLED_DT.min(); bp["last_fill"] = g.FILLED_DT.max()
bp["used_by_client"] = bp.fills_client > 0
bp["capacity_fills"] = np.ceil(bp.fills_all / C["CAPACITY_SHARE"]).astype(int)
bp["phase1_retail"] = bp.PHR_DISPNSR_TYPE_DESC == "Community or Retail Pharmacy"
bp = bp.reset_index()
save(bp, "book_tables", "book_pharmacies")

# price list
k = ["PHR_NCPDP_ID", "a_nt_gpi", "a_irpg_drugtype"]
pc = paid[paid.is_client == 1].groupby(k).agg(ingred_client=("m_nf_clt_ingred_cost_paid", "sum"), fee_client=("m_nf_clt_dispensing_fee", "sum"), qty_client=("a_nf_drg_qty", "sum"), claims_client=("CLAIM_ID", "size"))
pa = paid.groupby(k).agg(ingred_all=("m_nf_clt_ingred_cost_paid", "sum"), fee_all=("m_nf_clt_dispensing_fee", "sum"), qty_all=("a_nf_drg_qty", "sum"), claims_all=("CLAIM_ID", "size"))
pt = pa.join(pc, how="left").fillna({"ingred_client": 0.0, "fee_client": 0.0, "qty_client": 0.0, "claims_client": 0}).reset_index()
pt["claims_client"] = pt.claims_client.astype(int)
own = pt.claims_client > 0
pt["unit_ingred"] = np.where(own, pt.ingred_client / pt.qty_client.replace(0, np.nan), pt.ingred_all / pt.qty_all)
pt["fee_per_fill"] = np.where(own, pt.fee_client / pt.claims_client.replace(0, np.nan), pt.fee_all / pt.claims_all)
pt["price_source"] = np.where(own, "client", "all clients")
pt = pt[k + ["ingred_client", "fee_client", "qty_client", "claims_client", "ingred_all", "fee_all", "qty_all", "claims_all", "unit_ingred", "fee_per_fill", "price_source"]]
save(pt, "book_tables", "book_price_table")

# D-Wave file 2: the universal pharmacy data, one row per pharmacy x drug. Pharmacy columns repeat on every row of the pharmacy
ph_cols = ["PHR_NCPDP_ID"] + attrs + ["PHR_LAT", "PHR_LON", "fills_all", "fills_client", "used_by_client", "capacity_fills"]
up = bp[ph_cols].merge(pt, on="PHR_NCPDP_ID", how="inner").sort_values(["PHR_NCPDP_ID", "a_nt_gpi", "a_irpg_drugtype"])
up = up[ph_cols + ["a_nt_gpi", "a_irpg_drugtype", "claims_client", "qty_client", "ingred_client", "fee_client",
                   "claims_all", "qty_all", "ingred_all", "fee_all", "unit_ingred", "fee_per_fill", "price_source"]]
up.to_csv(os.path.join(OUT, "universal_pharmacies.csv"), index=False)

# chain fallback
pch = paid.groupby(["a_nf_pharmacy", "a_nt_gpi", "a_irpg_drugtype"]).agg(ingred_all=("m_nf_clt_ingred_cost_paid", "sum"), fee_all=("m_nf_clt_dispensing_fee", "sum"),
                                                                        qty_all=("a_nf_drg_qty", "sum"), claims_all=("CLAIM_ID", "size")).reset_index().rename(columns={"a_nf_pharmacy": "chain"})
pch["unit_ingred"] = pch.ingred_all / pch.qty_all; pch["fee_per_fill"] = pch.fee_all / pch.claims_all
save(pch, "book_tables", "book_chain_price_table")

# the client's tables (same rules as fedex_member_extract.py: nothing filtered by type or class, kind = type | class)
cp = paid[paid.is_client == 1].copy()
cp["kind"] = cp.PHR_DISPNSR_TYPE_DESC + " | " + cp.PHR_DISPNSR_CLASS_DESC
cp["is_normal"] = ((cp.PHR_DISPNSR_TYPE_DESC == C["NORMAL_TYPE"]) & cp.PHR_DISPNSR_CLASS_DESC.isin(C["NORMAL_CLASSES"])).astype(int)
cn = cp[cp.is_normal == 1]
g = cp.groupby("MBR_ID")
mb = pd.DataFrame({"ACCOUNT_ID": g.ACCOUNT_ID.first(), "EMPLOYER_GROUP_ID": g.EMPLOYER_GROUP_ID.first(), "MBR_LAT": g.mbr_lat.median(), "MBR_LON": g.mbr_long.median(),
                   "fills_all": g.size(), "fills_normal": g.is_normal.sum(), "ingred_all": g.m_nf_clt_ingred_cost_paid.sum(), "fee_all": g.m_nf_clt_dispensing_fee.sum()})
gn = cn.groupby("MBR_ID")
mb["ingred_normal"] = gn.m_nf_clt_ingred_cost_paid.sum().reindex(mb.index).fillna(0.0); mb["fee_normal"] = gn.m_nf_clt_dispensing_fee.sum().reindex(mb.index).fillna(0.0)
mb["first_fill"] = g.FILLED_DT.min(); mb["last_fill"] = g.FILLED_DT.max()
mb["pharmacies_used"] = g.PHR_NCPDP_ID.nunique()
def most_used(df):
    return df.groupby(["MBR_ID", "PHR_NCPDP_ID"]).size().reset_index(name="n").sort_values(["MBR_ID", "n"], ascending=[True, False]).drop_duplicates("MBR_ID").set_index("MBR_ID").PHR_NCPDP_ID
mb["main_pharmacy"] = most_used(cp).reindex(mb.index)
mb["main_normal_pharmacy"] = most_used(cn).reindex(mb.index)
mb["kinds_used"] = g.kind.nunique()
mb["kinds"] = g.kind.apply(lambda s: " ; ".join(sorted(set(s))))
dm = cp.groupby(["MBR_ID", "PHR_DISPNSR_TYPE_DESC", "PHR_DISPNSR_CLASS_DESC", "a_nt_gpi", "a_irpg_drugtype"]).agg(
        fills=("CLAIM_ID", "size"), qty=("a_nf_drg_qty", "sum"), days=("DAYS_SUPPLY", "sum"),
        ingred=("m_nf_clt_ingred_cost_paid", "sum"), fee=("m_nf_clt_dispensing_fee", "sum"),
        maint=("a_nt_maintenance_flag", lambda s: int((s == "Y").any()))).reset_index()
mb["drugs"] = dm.groupby("MBR_ID").size().reindex(mb.index).fillna(0).astype(int); mb["maint_drugs"] = dm.groupby("MBR_ID").maint.sum().reindex(mb.index).fillna(0).astype(int)
mb["has_normal"] = mb.fills_normal > 0
mb = mb.reset_index(); save(mb, "member_tables", "client_members")
save(dm, "member_tables", "client_demand")
mp = cp.groupby(["MBR_ID", "PHR_NCPDP_ID"]).agg(PHR_DISPNSR_TYPE_DESC=("PHR_DISPNSR_TYPE_DESC", "first"), PHR_DISPNSR_CLASS_DESC=("PHR_DISPNSR_CLASS_DESC", "first"),
                                                 fills=("CLAIM_ID", "size"), visits=("FILLED_DT", "nunique"), ingred=("m_nf_clt_ingred_cost_paid", "sum"), fee=("m_nf_clt_dispensing_fee", "sum")).reset_index()
save(mp, "member_tables", "client_member_pharmacy")
mf = cp[cp.a_nt_maintenance_flag == "Y"][["MBR_ID", "PHR_NCPDP_ID", "a_nt_gpi", "a_irpg_drugtype", "FILLED_DT", "DAYS_SUPPLY"]].sort_values(["MBR_ID", "a_nt_gpi", "FILLED_DT"])
save(mf, "member_tables", "client_maint_fills")

# ================================================================== 6. README (in house only) ==
readme = f"""# Synthetic data package, pharmacy network design (phase 1, retail). In house notes, not shared

Everything in this package is invented. Areas, states (XA to XD), metro codes (A001 to A003), pharmacies, chains (CHAIN A to F),
members, drugs and prices are synthetic. Column names and table structures are the ones used by the real pipeline.

Every pharmacy is a retail network pharmacy. PHR_DISPNSR_TYPE_DESC is the sub category inside retail and PHR_DISPNSR_CLASS_DESC
the class; together they are the kind of pharmacy. The normal kind is {C['NORMAL_TYPE']} with {', '.join(C['NORMAL_CLASSES'])}.
A member is only moved within the kind they use today; every member ends up with one normal pharmacy.

## Files
D-Wave gets the first two only (csv, in the package folder). The rest are the in house model's tables (parquet, in in_house/).

| file | rows | one row per | what it is |
|---|---|---|---|
| synthetic_claims_{C['YEAR']}.csv | {len(cl):,} | fill | all clients' claims for one year. {C['CLIENT']} is the client whose network is designed ({int((cl.CARRIER_ID == C['CLIENT']).sum()):,} rows); {', '.join(C['OTHER_CLIENTS'])} are the rest of the book |
| universal_pharmacies.csv | {len(up):,} | pharmacy x GPI x generic/brand | the search space: every pharmacy with a paid claim, with its columns and its average cost per drug (= book_pharmacies joined to book_price_table) |
| in_house/book_tables/book_pharmacies | {len(bp):,} | pharmacy | the universe: every pharmacy with a paid claim, any client |
| book_tables/book_price_table | {len(pt):,} | pharmacy x GPI x generic/brand | the price list |
| book_tables/book_chain_price_table | {len(pch):,} | chain x GPI x generic/brand | price fallback for a drug nobody filled at a store |
| member_tables/client_members | {len(mb):,} | member of {C['CLIENT']} | location, totals, today's pharmacy |
| member_tables/client_demand | {len(dm):,} | member x type x class x GPI x G/B | what each member fills, per kind of pharmacy |
| member_tables/client_member_pharmacy | {len(mp):,} | member x pharmacy | today's pharmacies and visits |
| member_tables/client_maint_fills | {len(mf):,} | maintenance fill | input to the adherence (quality) score |

Rules used to build the tables from the claims: paid claims only (m_nf_app_due_amt > 0 and a_nf_clt_prc_typ not blank or ZERO);
sums per pharmacy and per pharmacy x drug; prices are volume weighted (sum of money / sum of units); a drug at a pharmacy is priced
from the client's own claims when the client filled it there (price_source = client), otherwise from all clients (all clients);
a drug nobody filled at a store takes the chain's price. Generic and brand are never mixed: the drug key is a_nt_gpi + a_irpg_drugtype.
Capacity: 2025 volume is {int(C['CAPACITY_SHARE']*100)}% of what a pharmacy can take (capacity_fills = fills_all / {C['CAPACITY_SHARE']}).

## Claims columns
CLAIM_ID claim number | CARRIER_ID client | ACCOUNT_ID, EMPLOYER_GROUP_ID levels under the client | MBR_ID member | mbr_lat, mbr_long member location |
PHR_NCPDP_ID pharmacy key | PHR_NPI_NO, PHR_NM, PHR_AFFL_NM labels | a_nf_pharmacy chain | PHR_DISPNSR_TYPE_DESC dispenser type, the sub category inside retail ({', '.join(C['TYPE_MIX'])}) |
PHR_DISPNSR_CLASS_DESC dispensing class (Chain, Independent, Franchise, Government, Alternate Dispensing Site) | PHR_ST_CD, LOC_MSA_CD, PHR_ZIP_CD, PHR_CITY_NM, PHR_ADDR1_TXT location labels |
phr_lat, phr_long pharmacy location | a_nt_gpi 14 digit drug code | a_irpg_drugtype G generic / B brand | a_nf_specialty_flag | a_nt_maintenance_flag |
a_nf_drg_qty units dispensed | DAYS_SUPPLY | FILLED_DT fill date | m_nf_clt_ingred_cost_paid client ingredient cost | m_nf_clt_dispensing_fee client fee |
m_nf_app_due_amt amount due (0 = rejected or reversed) | a_nf_clt_prc_typ client price basis (MAC, AWP, U&C, ZERO) | a_nf_phr_prc_typ pharmacy price basis

## book_pharmacies columns
PHR_NCPDP_ID key | attributes as above | PHR_LAT, PHR_LON | fills_all, fills_client paid claims (all clients, the client) | ingred_all, fee_all, ingred_client, fee_client money |
first_fill, last_fill | used_by_client | capacity_fills | phase1_retail (label: type = {C['NORMAL_TYPE']}; not a filter)

## book_price_table columns
PHR_NCPDP_ID, a_nt_gpi, a_irpg_drugtype key | ingred_client, fee_client, qty_client, claims_client the client's sums | ingred_all, fee_all, qty_all, claims_all all clients' sums |
unit_ingred price per unit | fee_per_fill | price_source client or all clients

## book_chain_price_table columns
chain, a_nt_gpi, a_irpg_drugtype key | ingred_all, fee_all, qty_all, claims_all | unit_ingred | fee_per_fill

## client_members columns
MBR_ID | ACCOUNT_ID, EMPLOYER_GROUP_ID | MBR_LAT, MBR_LON | fills_all, fills_normal (at the normal kind) | ingred_all, fee_all, ingred_normal, fee_normal (what the client paid in the year) |
first_fill, last_fill | pharmacies_used | main_pharmacy (most fills, any kind), main_normal_pharmacy | kinds_used, kinds | drugs, maint_drugs | has_normal

## client_demand columns
MBR_ID, PHR_DISPNSR_TYPE_DESC, PHR_DISPNSR_CLASS_DESC, a_nt_gpi, a_irpg_drugtype key | fills, qty, days | ingred, fee (paid today) | maint

## client_member_pharmacy columns
MBR_ID, PHR_NCPDP_ID key | PHR_DISPNSR_TYPE_DESC, PHR_DISPNSR_CLASS_DESC | fills | visits (distinct fill dates) | ingred, fee

## client_maint_fills columns
MBR_ID, PHR_NCPDP_ID, a_nt_gpi, a_irpg_drugtype, FILLED_DT, DAYS_SUPPLY

## Model (for reference)
minimize  drug bill + network cost + A x visits moved + lambda x visits x extra miles
drug bill of a member at a pharmacy = sum over the member's drugs of unit_ingred x qty + fee_per_fill x fills, priced from book_price_table by the rule above.
Decisions: x[p] pharmacy in the network, a[m,p] member (per kind of pharmacy) served by p, z[chain, state].
Rules: every member and kind served once, only by open pharmacies of the same kind, inside the access radius (2 / 5 / 15 miles urban / suburban / rural, today's pharmacy may be kept),
capacity for new members = capacity_fills minus 2025 volume, chains in or out by state, mandated pharmacies stay, quality floor on adherence, a normal pharmacy within reach for everyone.
"""
if HOUSE:
    with open(os.path.join(HOUSE, "README.md"), "w", encoding="utf-8") as f: f.write(readme)
    with open(os.path.join(HOUSE, "config_used.json"), "w") as f: json.dump(C, f, indent=2)
print(f"tables: pharmacies {len(bp):,} (used by client {int(bp.used_by_client.sum()):,}) | price rows {len(pt):,} "
      f"(client priced {int(own.sum()):,}) | chain rows {len(pch):,} | members {len(mb):,} (with a normal pharmacy {int(mb.has_normal.sum()):,}, "
      f"also using a special kind {int((mb.fills_all > mb.fills_normal).sum()):,}, more than one kind {int((mb.kinds_used > 1).sum()):,}) | demand {len(dm):,} | member x pharmacy {len(mp):,} | maint fills {len(mf):,}")
print(f"for D-Wave: {os.path.abspath(OUT)}: synthetic_claims_{C['YEAR']}.csv ({len(cl):,} rows), universal_pharmacies.csv ({len(up):,} pharmacy x drug rows)"
      + (f"   |   in house tables: {os.path.abspath(HOUSE)}" if HOUSE else ""))
