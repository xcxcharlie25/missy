"""
pnm_run.py   Pharmacy Network Design, phase 1 retail. Real data, stays in house.

Reads the tables written by book_pharmacy_extract.py (whole book) and fedex_member_extract.py
(one client), builds the network model metro by metro, solves it with HiGHS, and writes the
results workbook plus the two PETL files.

    python pnm_run.py             full run, edit CONFIG first
    python pnm_run.py --smoke     the 2 largest metros and 3 designs, to check everything works

Needs: pip install duckdb highspy scipy pandas numpy openpyxl matplotlib

The model, solved metro by metro
    minimize   drug bill + network cost + A * visits moved + lambda * visits * extra miles
    decisions  x[p] pharmacy p in the network, a[m,p] member m (per dispensing class) served by p,
               z[chain, state] chain in or out as a whole in a state
    rules      C1 every member and class served exactly once        C2 only by an open pharmacy
               C3 inside the access radius (today's pharmacy may be kept)
               C4 capacity from 2025 volume, busy pharmacies keep their own members
               C5 chains in or out by state                        C6 mandated pharmacies stay
               C7 quality floor, maintenance adherence (PDC)       C8 a normal pharmacy within reach for everyone
               C10 network size cap when network cost is unknown   C11 all decisions 0/1
Memory: raw claims are never loaded. The price list is cut once to the client's drugs and to the
pharmacies near its members, then read per metro. The model is sparse matrices straight into HiGHS.
"""
import os, sys, time, math, json
import numpy as np, pandas as pd
import duckdb, highspy
from scipy.spatial import cKDTree
from scipy import sparse

# ================================================================== CONFIG =====
CONFIG = dict(
    BOOK_DIR   = r"C:\Users\psaxen41\Downloads\Real_Data\book_tables",     # from book_pharmacy_extract.py
    MEMBER_DIR = r"C:\Users\psaxen41\Downloads\Real_Data\member_tables",   # from fedex_member_extract.py
    OUT_DIR    = r"C:\Users\psaxen41\Downloads\Real_Data\model_results",
    CLIENT     = "FEDEX",
    RETAIL_TYPE = "Community or Retail Pharmacy",

    # who gets what kind of pharmacy
    ASSIGN_BY_CLASS = True,          # True: one pharmacy per dispensing class the member uses (what we agreed)
                                     # False: one retail pharmacy per member, classes pooled (requirements document wording)
    NORMAL_CLASSES  = ["Chain Pharmacy", "Independent Pharmacy", "Franchise Pharmacy"],
    NORMAL_GUARANTEE = True,         # members using only Government / Alternate sites, or with no retail fills, get a normal
                                     # pharmacy within reach kept open and allotted (zero fills, never counted as moved)

    # access (C3). Member setting is derived from pharmacy density around the member, editable
    ACCESS_MILES = dict(URBAN=2.0, SUBURBAN=5.0, RURAL=15.0),
    URBAN_MIN_WITHIN_2MI = 5,        # at least this many retail pharmacies within 2 miles: URBAN
    SUBURBAN_MIN_WITHIN_5MI = 1,     # else at least this many within 5 miles: SUBURBAN, else RURAL
    KEEP_TODAY_BEYOND_RADIUS = True, # a member may always keep the pharmacy they use today
    SEARCH_NEAREST = 40,             # candidates inside the radius that get priced
    KEEP_NEAREST = 8, KEEP_CHEAPEST = 6,   # of those, the ones the optimizer sees (plus today's)

    # capacity (C4). 2025 volume across all clients is this share of what the pharmacy can take
    CAPACITY_SHARE = 0.90,
    HEADROOM_RESERVED = 0.0,         # share of the free room held back for members in areas not covered yet (0 to 1)

    # network cost and size (objective term 2, C10)
    NETWORK_COST_PER_PHARMACY = 0.0, # dollars a year to keep a pharmacy in the network. PLACEHOLDER, no source yet
    NETWORK_SIZE_SHARE = 1.0,        # with no network cost: open pharmacies per metro <= this share of what the client uses today
                                     # None = no cap

    # chains and mandates (C5, C6)
    ALL_OR_NOTHING_CHAINS = [],      # names as in a_nf_pharmacy, e.g. ["CVS", "WALGREENS"]. Empty = rule off
    MANDATED_STATES = [],            # any willing pharmacy states: every retail pharmacy there stays in, e.g. ["WI"]
    MANDATED_NCPDP = [],             # specific pharmacies that must stay in

    # quality (C7): adherence on maintenance drugs, proportion of days covered
    QUALITY_FLOOR = True,
    QUALITY_MIN_PAIRS = 20,          # member x drug pairs needed for a pharmacy's own score, else chain, else overall
    QUALITY_MAX_DAYS = 365,

    # disruption and distance weights, the trade off curve
    A_HOLD = 5.0, LAMBDA_HOLD = 2.0,
    A_SWEEP = [0, 1, 2, 5, 10, 25, 50],
    LAMBDA_SWEEP = [0, 0.5, 1, 2, 5, 10],
    BANDS = [(-1e9, 0.0, "as close or closer"), (0.0, 2.0, "up to 2"), (2.0, 5.0, "2 to 5"), (5.0, 8.0, "5 to 8"),
             (8.0, 15.0, "8 to 15"), (15.0, 1e9, "over 15")],

    # named network scenarios, e.g. {"Standard Select": "standard_select_ncpdp.csv"}: CSV with a PHR_NCPDP_ID column
    FIXED_NETWORKS = {},
    # optional footprint for territory coverage (areas with no members): CSV with ZIP, LAT, LON, SETTING
    FOOTPRINT_FILE = None, ADEQUACY_COUNT = dict(URBAN=2, SUBURBAN=1, RURAL=1),
    TREND_GPI2 = {},                 # utilization trend by first 2 digits of GPI, e.g. {"27": 1.03}. Applied to demand

    SOLVER_TIME_LIMIT = 180, SOLVER_GAP = 0.0002, THREADS = 4, TIGHT_LINKING_MAX_VARS = 200_000,
    MAX_PAIRS_PER_BLOCK = 25_000,    # a metro with more member x class pairs is cut into geographic blocks solved one after another
    DUCKDB_MEMORY = "2GB",
    SMOKE_METROS = 2,
)
if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):          # python pnm_run.py <folder with book_tables and member_tables>
    CONFIG.update(BOOK_DIR=os.path.join(sys.argv[1], "book_tables"), MEMBER_DIR=os.path.join(sys.argv[1], "member_tables"),
                  OUT_DIR=os.path.join(sys.argv[1], "model_results"))
SMOKE = "--smoke" in sys.argv
T0 = time.time()
def stamp(msg): print(f"[{time.time()-T0:6.0f}s] {msg}", flush=True)

R_EARTH = 3958.7613
def haversine(lat1, lon1, lat2, lon2):
    la1, lo1, la2, lo2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((la2-la1)/2)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
    return 2*R_EARTH*np.arcsin(np.sqrt(np.clip(a, 0, 1)))
def xyz(lat, lon):
    la, lo = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    return np.c_[np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la)]
def chord(miles): return 2*np.sin(np.asarray(miles, float)/(2*R_EARTH))   # unit sphere chord of a great circle distance

C = CONFIG
OUT = C["OUT_DIR"]; os.makedirs(OUT, exist_ok=True)
BOOK, MEM = C["BOOK_DIR"], C["MEMBER_DIR"]
P = lambda d, f: os.path.join(d, f).replace("\\", "/")
inf = highspy.kHighsInf

# ================================================================== 1. DATA IN ==
stamp("loading tables")
ph = pd.read_parquet(P(BOOK, "book_pharmacies.parquet"))
ph["PHR_NCPDP_ID"] = ph.PHR_NCPDP_ID.astype(str)
ph = ph[ph.phase1_retail.astype(bool)].reset_index(drop=True)
ph["chain"] = ph.a_nf_pharmacy.fillna("Independent").astype(str)
ph["cls"] = ph.PHR_DISPNSR_CLASS_DESC.fillna("Independent Pharmacy").astype(str)
ph["state"] = ph.PHR_ST_CD.astype("string").fillna("NA")
ph["metro"] = ph.LOC_MSA_CD.astype("string").where(ph.LOC_MSA_CD.notna(), "ST_" + ph.state)
ph["has_loc"] = ph.PHR_LAT.notna() & ph.PHR_LON.notna()
ph["fills_all"] = ph.fills_all.astype(float); ph["fills_client"] = ph.fills_client.astype(float)
# free room for members new to a pharmacy: the room above its 2025 total volume (its own members are already in that volume)
ph["free_room"] = ph.fills_all * (1/C["CAPACITY_SHARE"] - 1) * (1 - C["HEADROOM_RESERVED"])
ph["netcost"] = float(C["NETWORK_COST_PER_PHARMACY"])
PH = ph.set_index("PHR_NCPDP_ID")
loc_pos = np.where(ph.has_loc.values)[0]                       # positions in ph that have coordinates
tree = cKDTree(xyz(ph.PHR_LAT.values[loc_pos], ph.PHR_LON.values[loc_pos]))
stamp(f"retail pharmacies {len(ph):,} ({int(ph.used_by_client.sum()):,} used by {C['CLIENT']}, {int((~ph.has_loc).sum()):,} without coordinates)")

mem = pd.read_parquet(P(MEM, "fedex_members.parquet")); mem["MBR_ID"] = mem.MBR_ID.astype(str)
dem = pd.read_parquet(P(MEM, "fedex_demand.parquet")); dem["MBR_ID"] = dem.MBR_ID.astype(str); dem["a_nt_gpi"] = dem.a_nt_gpi.astype(str)
use = pd.read_parquet(P(MEM, "fedex_member_pharmacy.parquet")); use["MBR_ID"] = use.MBR_ID.astype(str); use["PHR_NCPDP_ID"] = use.PHR_NCPDP_ID.astype(str)
mf = pd.read_parquet(P(MEM, "fedex_maint_fills.parquet")); mf["MBR_ID"] = mf.MBR_ID.astype(str); mf["PHR_NCPDP_ID"] = mf.PHR_NCPDP_ID.astype(str)
use = use[(use.PHR_DISPNSR_TYPE_DESC == C["RETAIL_TYPE"]) & use.PHR_NCPDP_ID.isin(PH.index)]
dem["cls"] = dem.PHR_DISPNSR_CLASS_DESC.fillna("Independent Pharmacy").astype(str) if C["ASSIGN_BY_CLASS"] else "RETAIL"
use["cls"] = use.PHR_DISPNSR_CLASS_DESC.fillna("Independent Pharmacy").astype(str) if C["ASSIGN_BY_CLASS"] else "RETAIL"
if C["TREND_GPI2"]:
    f = dem.a_nt_gpi.str[:2].map(C["TREND_GPI2"]).fillna(1.0)
    dem["fills"] = dem.fills * f; dem["qty"] = dem.qty * f; dem["days"] = dem.days * f
dem = dem[dem.qty > 0]
stamp(f"members {len(mem):,} | demand rows {len(dem):,} | member x pharmacy rows {len(use):,} | maintenance fills {len(mf):,}")

# ---- member x class pairs: demand, today's pharmacy, visits, adjusted claims
pairs = dem.groupby(["MBR_ID", "cls"]).agg(fills=("fills", "sum"), qty=("qty", "sum"), days=("days", "sum"),
                                           paid_2025=("ingred", "sum"), fee_2025=("fee", "sum"), drugs=("a_nt_gpi", "size")).reset_index()
pairs["fills"] = pairs.fills.astype(float); pairs["adj_claims"] = pairs.days / 30.0
today = use.sort_values(["MBR_ID", "cls", "fills"], ascending=[True, True, False]).drop_duplicates(["MBR_ID", "cls"])
pairs = pairs.merge(today[["MBR_ID", "cls", "PHR_NCPDP_ID"]].rename(columns={"PHR_NCPDP_ID": "today"}), on=["MBR_ID", "cls"], how="left")
vis = use.groupby(["MBR_ID", "cls"]).visits.sum().rename("visits").reset_index()
pairs = pairs.merge(vis, on=["MBR_ID", "cls"], how="left"); pairs["visits"] = pairs.visits.fillna(1.0).clip(lower=1.0).astype(float)
pairs = pairs.merge(mem[["MBR_ID", "MBR_LAT", "MBR_LON"]], on="MBR_ID", how="left")
stamp(f"member x class pairs {len(pairs):,} | without coordinates {int(pairs.MBR_LAT.isna().sum()):,} (kept at today's pharmacy)")

# ---- settings from pharmacy density, access radius per member, nearest retail pharmacy
mloc = mem[mem.MBR_LAT.notna() & mem.MBR_LON.notna()].copy()
mxyz = xyz(mloc.MBR_LAT.values, mloc.MBR_LON.values)
n2 = tree.query_ball_point(mxyz, chord(2.0), return_length=True)
n5 = tree.query_ball_point(mxyz, chord(5.0), return_length=True)
mloc["setting"] = np.where(n2 >= C["URBAN_MIN_WITHIN_2MI"], "URBAN", np.where(n5 >= C["SUBURBAN_MIN_WITHIN_5MI"], "SUBURBAN", "RURAL"))
mloc["radius"] = mloc.setting.map(C["ACCESS_MILES"]).astype(float)
_, nearest = tree.query(mxyz, k=1)
mloc["nearest_ncpdp"] = ph.PHR_NCPDP_ID.values[loc_pos[nearest]]
mem = mem.merge(mloc[["MBR_ID", "setting", "radius", "nearest_ncpdp"]], on="MBR_ID", how="left")
pairs = pairs.merge(mloc[["MBR_ID", "setting", "radius"]], on="MBR_ID", how="left")
stamp("member settings: " + ", ".join(f"{k} {v:,}" for k, v in mloc.setting.value_counts().items()))

# ---- distance to today's pharmacy; metro = metro of today's main retail pharmacy, else of the nearest one
pairs["d0"] = haversine(pairs.MBR_LAT.values, pairs.MBR_LON.values, pairs.today.map(PH.PHR_LAT).values.astype(float), pairs.today.map(PH.PHR_LON).values.astype(float))
mem["metro"] = mem.main_retail_pharmacy.astype("string").map(PH.metro).fillna(mem.nearest_ncpdp.map(PH.metro))
pairs = pairs.merge(mem[["MBR_ID", "metro"]], on="MBR_ID", how="left")
pairs["metro"] = pairs.metro.fillna(pairs.today.map(PH.metro))
pairs = pairs[pairs.metro.notna()].reset_index(drop=True)
MEMBERS = mem.set_index("MBR_ID")

# ================================================================== 2. PRICES ==
con = duckdb.connect()
con.execute(f"SET memory_limit='{C['DUCKDB_MEMORY']}'"); con.execute(f"SET threads={C['THREADS']}"); con.execute("SET preserve_insertion_order=false")
price_file, chain_file, slice_file = P(BOOK, "book_price_table.parquet"), P(BOOK, "book_chain_price_table.parquet"), P(OUT, "price_slice.parquet")
drugs = dem[["a_nt_gpi", "a_irpg_drugtype"]].drop_duplicates(); con.register("drugs", drugs)
near_any = set()                                               # pharmacies within reach of any member: the only possible candidates
for hits in tree.query_ball_point(mxyz, chord(max(C["ACCESS_MILES"].values()))): near_any.update(loc_pos[hits].tolist())
near_any |= set(np.where(ph.used_by_client.astype(bool).values)[0].tolist())
cand_ph = ph.iloc[sorted(near_any)][["PHR_NCPDP_ID"]]; con.register("cand_ph", cand_ph)
if not os.path.exists(slice_file):
    stamp(f"cutting the price list to {len(drugs):,} client drugs and {len(cand_ph):,} pharmacies in reach (one pass over the book, once)")
    con.execute(f"""COPY (SELECT CAST(p.PHR_NCPDP_ID AS VARCHAR) AS PHR_NCPDP_ID, CAST(p.a_nt_gpi AS VARCHAR) AS a_nt_gpi, p.a_irpg_drugtype,
                              p.unit_ingred, p.fee_per_fill, p.price_source
                       FROM read_parquet('{price_file}') p
                       JOIN drugs d ON CAST(p.a_nt_gpi AS VARCHAR) = d.a_nt_gpi AND p.a_irpg_drugtype = d.a_irpg_drugtype
                       JOIN cand_ph c ON CAST(p.PHR_NCPDP_ID AS VARCHAR) = c.PHR_NCPDP_ID
                       WHERE p.unit_ingred IS NOT NULL
                       ORDER BY PHR_NCPDP_ID) TO '{slice_file}' (FORMAT PARQUET)""")
n_slice = con.execute(f"SELECT count(*) FROM read_parquet('{slice_file}')").fetchone()[0]
book = con.execute(f"""SELECT d.a_nt_gpi, d.a_irpg_drugtype,
                              sum(p.ingred_all)/nullif(sum(p.qty_all),0) AS unit_ingred, sum(p.fee_all)/nullif(sum(p.claims_all),0) AS fee_per_fill
                       FROM read_parquet('{price_file}') p JOIN drugs d ON CAST(p.a_nt_gpi AS VARCHAR) = d.a_nt_gpi AND p.a_irpg_drugtype = d.a_irpg_drugtype
                       GROUP BY 1, 2""").df()
chain_p = con.execute(f"""SELECT CAST(c.chain AS VARCHAR) AS chain, d.a_nt_gpi, d.a_irpg_drugtype, c.unit_ingred, c.fee_per_fill
                          FROM read_parquet('{chain_file}') c JOIN drugs d ON CAST(c.a_nt_gpi AS VARCHAR) = d.a_nt_gpi AND c.a_irpg_drugtype = d.a_irpg_drugtype
                          WHERE c.unit_ingred IS NOT NULL""").df()
con.register("book", book); con.register("chain_p", chain_p)
stamp(f"price slice {n_slice:,} rows | chain fallback {len(chain_p):,} rows | book fallback {len(book):,} drugs")

# ================================================================== 3. QUALITY ==
# PDC per member and maintenance drug: days supplied / days from first fill to data end, capped at 1. Attributed to the
# pharmacy that filled it most. Pharmacy score = mean PDC of its pairs (own score needs QUALITY_MIN_PAIRS, else chain, else overall)
mf["FILLED_DT"] = pd.to_datetime(mf.FILLED_DT); data_end = mf.FILLED_DT.max()
g = mf.groupby(["MBR_ID", "a_nt_gpi", "a_irpg_drugtype"]).agg(days=("DAYS_SUPPLY", "sum"), first=("FILLED_DT", "min")).reset_index()
g["period"] = ((data_end - g["first"]).dt.days + 30).clip(lower=30, upper=C["QUALITY_MAX_DAYS"])
g["pdc"] = (g.days.astype(float) / g.period).clip(upper=1.0)
where = (mf.groupby(["MBR_ID", "a_nt_gpi", "a_irpg_drugtype", "PHR_NCPDP_ID"]).size().reset_index(name="n")
           .sort_values("n", ascending=False).drop_duplicates(["MBR_ID", "a_nt_gpi", "a_irpg_drugtype"]))
g = g.merge(where[["MBR_ID", "a_nt_gpi", "a_irpg_drugtype", "PHR_NCPDP_ID"]], on=["MBR_ID", "a_nt_gpi", "a_irpg_drugtype"])
qs = g.groupby("PHR_NCPDP_ID").pdc.agg(["mean", "size"])
q_global = float(g.pdc.mean()) if len(g) else 0.8
own = qs[qs["size"] >= C["QUALITY_MIN_PAIRS"]]["mean"]
ph["q_own"] = ph.PHR_NCPDP_ID.map(own)
ph["quality"] = ph.q_own.fillna(ph.chain.map(ph.dropna(subset=["q_own"]).groupby("chain").q_own.mean())).fillna(q_global)
PH = ph.set_index("PHR_NCPDP_ID")
stamp(f"quality: PDC on {len(g):,} member x drug pairs | {len(own):,} pharmacies with their own score | overall {q_global:.3f}")

# ================================================================== 4. DESIGNS ==
designs = [(0.0, 0.0)] + [(float(a), float(C["LAMBDA_HOLD"])) for a in C["A_SWEEP"]] + [(float(C["A_HOLD"]), float(l)) for l in C["LAMBDA_SWEEP"]]
designs = list(dict.fromkeys(designs))
HOLD = (float(C["A_HOLD"]), float(C["LAMBDA_HOLD"]))
if SMOKE: designs = [(0.0, 0.0), HOLD, (50.0, float(C["LAMBDA_HOLD"]))]
tag = lambda d: f"A={d[0]:g} lam={d[1]:g}"
def band_of(x):
    for lo, hi, nm in C["BANDS"]:
        if lo < x <= hi or (nm == "as close or closer" and x <= 0): return nm
    return C["BANDS"][-1][2]
fixed_nets = {}
for nm, f in C["FIXED_NETWORKS"].items():
    if os.path.exists(f): fixed_nets[nm] = set(pd.read_csv(f, dtype=str).PHR_NCPDP_ID.astype(str))
    else: print(f"fixed network {nm}: file {f} not found, scenario skipped")
footprint = pd.read_csv(C["FOOTPRINT_FILE"]) if C["FOOTPRINT_FILE"] and os.path.exists(C["FOOTPRINT_FILE"]) else None

# ================================================================== 5. ONE METRO ==
def solve_metro(metro, pm, mm, allowed_set, used, chain_fix, scenario):
    """pm: pairs of this metro. mm: members of this metro. used: fills given to each pharmacy by earlier metros, per design.
    chain_fix: (chain, state) decisions made in earlier metros, per design. Returns results per design."""
    radius_max = max(C["ACCESS_MILES"].values())
    pm = pm.reset_index(drop=True)
    m_ok = pm.MBR_LAT.notna().values
    cand_pos = set()
    if m_ok.any():
        for hits in tree.query_ball_point(xyz(pm.MBR_LAT.values[m_ok], pm.MBR_LON.values[m_ok]), chord(radius_max)): cand_pos.update(loc_pos[hits].tolist())
    cand_pos |= set(np.where(ph.PHR_NCPDP_ID.isin(pm.today.dropna()).values)[0].tolist())
    lp_ = ph.iloc[sorted(cand_pos)].reset_index(drop=True)
    if len(lp_) == 0: return None
    lp_loc = lp_[lp_.has_loc].reset_index(drop=True)
    if allowed_set is not None: lp_loc = lp_loc[lp_loc.PHR_NCPDP_ID.isin(allowed_set)].reset_index(drop=True)
    sub = cKDTree(xyz(lp_loc.PHR_LAT.values, lp_loc.PHR_LON.values)) if len(lp_loc) else None
    # ---- candidates per pair: inside the member's radius, same class, nearest SEARCH_NEAREST, plus today's pharmacy
    rows = []
    ball = sub.query_ball_point(xyz(pm.MBR_LAT.fillna(0).values, pm.MBR_LON.fillna(0).values), chord(pm.radius.fillna(0.001).values)) if sub is not None else [[]]*len(pm)
    for i, r in enumerate(pm.itertuples()):
        this = []
        if m_ok[i] and sub is not None:
            hits = ball[i]
            if hits:
                h = lp_loc.iloc[hits]
                if C["ASSIGN_BY_CLASS"]: h = h[h.cls == r.cls]
                if len(h):
                    d = haversine(r.MBR_LAT, r.MBR_LON, h.PHR_LAT.values, h.PHR_LON.values); o = np.argsort(d)[:C["SEARCH_NEAREST"]]
                    this = [(i, h.PHR_NCPDP_ID.iat[k], float(d[k]), int(h.PHR_NCPDP_ID.iat[k] == r.today)) for k in o]
            if allowed_set is not None and not this:                     # named network: nearest in network pharmacy, any distance, own class if it has one
                h = lp_loc[lp_loc.cls == r.cls] if C["ASSIGN_BY_CLASS"] else lp_loc
                if len(h) == 0: h = lp_loc
                if len(h):
                    d = haversine(r.MBR_LAT, r.MBR_LON, h.PHR_LAT.values, h.PHR_LON.values); o = np.argsort(d)[:3]
                    this = [(i, h.PHR_NCPDP_ID.iat[k], float(d[k]), int(h.PHR_NCPDP_ID.iat[k] == r.today)) for k in o]
        if isinstance(r.today, str) and not any(t[1] == r.today for t in this):
            # today's pharmacy is always priced (it is the baseline); it is a candidate when allowed, kept beyond the radius if configured
            this.append((i, r.today, float(r.d0), 1))
        rows += this
    cand = pd.DataFrame(rows, columns=["pi", "ncpdp", "dist", "is_today"]).drop_duplicates(["pi", "ncpdp"])
    if len(cand) == 0: return None
    elig = np.ones(len(cand), dtype=bool)
    if allowed_set is not None: elig &= cand.ncpdp.isin(allowed_set).values
    if not C["KEEP_TODAY_BEYOND_RADIUS"]:
        n_cand = cand.groupby("pi").ncpdp.transform("size").values
        elig &= ~((cand.is_today.values == 1) & (cand.dist.values > pm.radius.fillna(1e9).values[cand.pi.values]) & (n_cand > 1))
    cand["eligible"] = elig.astype(int)
    # ---- price every candidate with the member's own drugs: price slice, then chain, then book average
    cand["MBR_ID"] = pm.MBR_ID.values[cand.pi.values]; cand["cls"] = pm.cls.values[cand.pi.values]; cand["chain"] = cand.ncpdp.map(PH.chain)
    dm = dem[dem.MBR_ID.isin(pm.MBR_ID)][["MBR_ID", "cls", "a_nt_gpi", "a_irpg_drugtype", "fills", "qty"]]
    con.register("cand", cand[["pi", "ncpdp", "MBR_ID", "cls", "chain"]]); con.register("dm", dm)
    cost = con.execute(f"""
        WITH pr AS (SELECT * FROM read_parquet('{slice_file}') WHERE PHR_NCPDP_ID IN (SELECT DISTINCT ncpdp FROM cand))
        SELECT c.pi, c.ncpdp,
               sum(coalesce(p.unit_ingred, ch.unit_ingred, b.unit_ingred) * d.qty
                   + coalesce(p.fee_per_fill, ch.fee_per_fill, b.fee_per_fill) * d.fills)                       AS cost,
               sum(CASE WHEN p.price_source = 'client' THEN p.unit_ingred*d.qty + p.fee_per_fill*d.fills END)    AS ev_client,
               sum(CASE WHEN p.price_source = 'all clients' THEN p.unit_ingred*d.qty + p.fee_per_fill*d.fills END) AS ev_book,
               sum(CASE WHEN p.unit_ingred IS NULL AND ch.unit_ingred IS NOT NULL THEN ch.unit_ingred*d.qty + ch.fee_per_fill*d.fills END) AS ev_chain,
               sum(CASE WHEN p.unit_ingred IS NULL AND ch.unit_ingred IS NULL THEN b.unit_ingred*d.qty + b.fee_per_fill*d.fills END) AS ev_avg
        FROM cand c
        JOIN dm d ON d.MBR_ID = c.MBR_ID AND d.cls = c.cls
        LEFT JOIN pr p ON p.PHR_NCPDP_ID = c.ncpdp AND p.a_nt_gpi = d.a_nt_gpi AND p.a_irpg_drugtype = d.a_irpg_drugtype
        LEFT JOIN chain_p ch ON ch.chain = c.chain AND ch.a_nt_gpi = d.a_nt_gpi AND ch.a_irpg_drugtype = d.a_irpg_drugtype
        LEFT JOIN book b ON b.a_nt_gpi = d.a_nt_gpi AND b.a_irpg_drugtype = d.a_irpg_drugtype
        GROUP BY 1, 2""").df()
    con.unregister("cand"); con.unregister("dm")
    cand = cand.merge(cost, on=["pi", "ncpdp"], how="left"); cand = cand[cand.cost.notna()]
    # ---- prune to what the optimizer sees: nearest, cheapest, today's
    cand["rk_d"] = cand.groupby("pi").dist.rank(method="first"); cand["rk_c"] = cand.groupby("pi").cost.rank(method="first")
    cand = cand[(cand.rk_d <= C["KEEP_NEAREST"]) | (cand.rk_c <= C["KEEP_CHEAPEST"]) | (cand.is_today == 1)].reset_index(drop=True)
    cand["fills"] = pm.fills.values[cand.pi.values]; cand["visits"] = pm.visits.values[cand.pi.values]
    d0 = pm.d0.values[cand.pi.values]; dd = cand.dist.values
    cand["extra"] = np.where(np.isnan(d0) | np.isnan(dd), 0.0, np.maximum(0.0, dd - d0))
    cand["quality"] = cand.ncpdp.map(PH.quality).values
    tc = cand[cand.is_today == 1].drop_duplicates("pi").set_index("pi").cost
    pm["cost_today"] = pm.index.map(tc)
    pm["cost_today"] = pm.cost_today.fillna(pm.paid_2025 + pm.fee_2025)       # no priced today pharmacy: what was paid in 2025
    cand = cand[cand.eligible == 1].reset_index(drop=True)
    if len(cand) == 0: return None

    # ---- the x set: assignment candidates, guarantee candidates, footprint candidates
    xs = list(dict.fromkeys(cand.ncpdp))
    guar = []                                                                   # (member, setting, [normal pharmacies within reach])
    if C["NORMAL_GUARANTEE"] and len(mm):
        need = mm[mm.MBR_LAT.notna() & mm.radius.notna()]
        if C["ASSIGN_BY_CLASS"]:
            cls_used = pm.groupby("MBR_ID").cls.apply(set); normal = set(C["NORMAL_CLASSES"])
            need = need[[not (cls_used.get(m, set()) & normal) for m in need.MBR_ID]]
        else:
            need = need[~need.MBR_ID.isin(pm.MBR_ID)]
        norm_ph = lp_loc[lp_loc.cls.isin(C["NORMAL_CLASSES"])] if C["ASSIGN_BY_CLASS"] else lp_loc
        if len(need):
            if len(norm_ph):
                nt = cKDTree(xyz(norm_ph.PHR_LAT.values, norm_ph.PHR_LON.values))
                for r, hits in zip(need.itertuples(), nt.query_ball_point(xyz(need.MBR_LAT.values, need.MBR_LON.values), chord(need.radius.values))):
                    guar.append((r.MBR_ID, r.setting, [norm_ph.PHR_NCPDP_ID.iat[k] for k in hits]))
            else: guar = [(r.MBR_ID, r.setting, []) for r in need.itertuples()]
    for _, _, lst in guar: xs += [p for p in lst if p not in xs]
    fp_rows = []
    if footprint is not None and len(lp_loc):
        fpm = footprint[footprint.LAT.between(lp_loc.PHR_LAT.min()-0.3, lp_loc.PHR_LAT.max()+0.3) & footprint.LON.between(lp_loc.PHR_LON.min()-0.3, lp_loc.PHR_LON.max()+0.3)]
        for r in fpm.itertuples():
            hits = sub.query_ball_point(xyz([r.LAT], [r.LON])[0], chord(C["ACCESS_MILES"].get(r.SETTING, 5.0)))
            lst = [lp_loc.PHR_NCPDP_ID.iat[k] for k in hits]; fp_rows.append((r.ZIP, r.SETTING, lst)); xs += [p for p in lst if p not in xs]
    xi = {p: k for k, p in enumerate(xs)}; n_x = len(xs); n_a = len(cand); A_ = n_x
    cand["xi"] = cand.ncpdp.map(xi).astype(int)
    chains = sorted({(PH.chain[p], PH.state[p]) for p in xs if PH.chain[p] in C["ALL_OR_NOTHING_CHAINS"]})
    zi = {c: n_x + n_a + k for k, c in enumerate(chains)}; n_var = n_x + n_a + len(chains)
    by_pair = cand.groupby("pi").indices; by_ph = cand.groupby("ncpdp").indices
    incumbents = cand[cand.is_today == 1].groupby("ncpdp").fills.sum()
    today_set = set(pm.today.dropna()); n_today_open = len([p for p in today_set if p in xi])
    is_today = cand.is_today.values.astype(float); fills_v = cand.fills.values; visits_v = cand.visits.values; extra_v = cand.extra.values; cost_v = cand.cost.values
    totf = float(pm.fills.sum())
    q_today = float((pm.fills * pm.today.map(PH.quality).fillna(q_global)).sum()) / totf
    q_best = float((cand.groupby("pi").quality.max().reindex(pm.index).fillna(q_global) * pm.fills).sum()) / totf

    results = {}
    for des in designs:
        A, LAM = des
        used_d = used.setdefault(des, {}); fix_d = chain_fix.setdefault(des, {})
        # ---- capacity for this design: room for new members is what is left after earlier metros; today's members always fit
        cap = {p: max(0.0, float(PH.free_room[p]) - used_d.get(p, 0.0)) for p in xs}
        closed = [p for p in xs if cap[p] < 1.0]                    # no room: keeps its own members, takes no new ones
        lb = np.zeros(n_var); ub = np.ones(n_var); closed_cols = []
        for p in closed:
            if p in by_ph:
                idx = by_ph[p]; cc = idx[is_today[idx] == 0]; ub[A_ + cc] = 0.0; closed_cols += (A_ + cc).tolist()
        for p, k in xi.items():
            if PH.state[p] in C["MANDATED_STATES"] or p in C["MANDATED_NCPDP"]: lb[k] = 1.0
            if allowed_set is not None: lb[k] = ub[k] = (1.0 if p in allowed_set else 0.0)
            key = (PH.chain[p], PH.state[p])
            if key in fix_d: lb[k] = ub[k] = fix_d[key]
        # ---- rows
        ri, ci, vv, rlo, rhi = [], [], [], [], []; nrow = 0
        def add_row(cols, vals, lo, hi):
            nonlocal nrow
            ri.extend([nrow]*len(cols)); ci.extend(cols); vv.extend(vals); rlo.append(lo); rhi.append(hi); nrow += 1
        for pi_, grp in by_pair.items(): add_row((A_ + grp).tolist(), [1.0]*len(grp), 1.0, 1.0)                 # C1
        cap_rows = []
        for p, grp in by_ph.items():                                                                                # C4 new members only
            new_ = grp[is_today[grp] == 0]
            if len(new_): add_row((A_ + new_).tolist() + [xi[p]], fills_v[new_].tolist() + [-cap[p]], -inf, 0.0); cap_rows.append(nrow - 1)
        link = np.arange(n_a) if n_var <= C["TIGHT_LINKING_MAX_VARS"] else np.where(is_today == 1)[0]              # C2 a <= x (all, or incumbents)
        ri.extend(np.repeat(np.arange(nrow, nrow + len(link)), 2).tolist()); ci.extend(np.c_[A_ + link, cand.xi.values[link]].ravel().tolist())
        vv.extend([1.0, -1.0]*len(link)); rlo.extend([-inf]*len(link)); rhi.extend([0.0]*len(link)); nrow += len(link)
        q_row = None
        if C["QUALITY_FLOOR"]:                                                                                      # C7
            QF = min(q_today, q_best) - 1e-6
            add_row((A_ + np.arange(n_a)).tolist(), (fills_v * cand.quality.values).tolist(), QF*totf, inf); q_row = nrow - 1
        guar_rows = []
        for _, _, lst in guar:                                                                                      # C8
            cols = [xi[p] for p in lst if p in xi]
            if cols: add_row(cols, [1.0]*len(cols), 1.0, inf); guar_rows.append(nrow - 1)
        for z, s_, lst in fp_rows:
            cols = [xi[p] for p in lst if p in xi]; need_n = C["ADEQUACY_COUNT"].get(s_, 1)
            if len(cols) >= need_n: add_row(cols, [1.0]*len(cols), float(need_n), inf)
        size_row = None
        if C["NETWORK_SIZE_SHARE"] is not None and C["NETWORK_COST_PER_PHARMACY"] == 0 and allowed_set is None:     # C10
            add_row(list(range(n_x)), [1.0]*n_x, -inf, float(max(1, int(math.ceil(C["NETWORK_SIZE_SHARE"] * max(1, n_today_open)))))); size_row = nrow - 1
        for p, k in xi.items():                                                                                     # C5
            key = (PH.chain[p], PH.state[p])
            if key in zi: add_row([k, zi[key]], [1.0, -1.0], 0.0, 0.0)
        # ---- objective and model
        cost_vec = np.zeros(n_var); cost_vec[:n_x] = [PH.netcost[p] for p in xs]
        cost_vec[A_:A_+n_a] = cost_v + LAM*visits_v*extra_v - A*visits_v*is_today
        M = sparse.coo_matrix((vv, (ri, ci)), shape=(nrow, n_var)).tocsc()
        rlo = np.array(rlo, dtype=float); rhi = np.array(rhi, dtype=float)
        lp = highspy.HighsLp(); lp.num_col_ = n_var; lp.num_row_ = nrow
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = M.indptr.astype(np.int32); lp.a_matrix_.index_ = M.indices.astype(np.int32); lp.a_matrix_.value_ = M.data.astype(float)
        lp.integrality_ = [highspy.HighsVarType.kInteger]*n_var
        relaxed = []; t0 = time.time()
        for attempt in range(5):
            lp.col_cost_ = cost_vec; lp.col_lower_ = lb; lp.col_upper_ = ub; lp.row_lower_ = rlo; lp.row_upper_ = rhi   # arrays are copied in
            h = highspy.Highs(); h.setOptionValue("output_flag", False); h.setOptionValue("threads", int(C["THREADS"]))
            h.setOptionValue("time_limit", float(C["SOLVER_TIME_LIMIT"])); h.setOptionValue("mip_rel_gap", float(C["SOLVER_GAP"]))
            h.passModel(lp); h.run(); st = h.modelStatusToString(h.getModelStatus()); sol = h.getSolution()
            if st in ("Optimal", "Time limit reached") and sol.value_valid: break
            # infeasible: relax size cap, then quality floor, then guarantees, then capacity, and say so in the Metros sheet
            if size_row is not None and "size cap" not in relaxed: rhi[size_row] = inf; relaxed.append("size cap"); continue
            if q_row is not None and "quality floor" not in relaxed: rlo[q_row] = -inf; relaxed.append("quality floor"); continue
            if guar_rows and "guarantees" not in relaxed: rlo[guar_rows] = -inf; relaxed.append("guarantees"); continue
            if (cap_rows or closed_cols) and "capacity" not in relaxed:
                if cap_rows: rhi[cap_rows] = inf
                if closed_cols: ub[closed_cols] = 1.0
                relaxed.append("capacity"); continue
            break
        if not (st in ("Optimal", "Time limit reached") and sol.value_valid):
            print(f"  {metro} {tag(des)}: NOT SOLVED ({st})", flush=True); continue
        v = np.asarray(sol.col_value); a_val = np.round(v[A_:A_+n_a]).astype(int); x_val = np.round(v[:n_x]).astype(int)
        # ---- polish: one pass per pair, take a cheaper open candidate if capacity and the quality floor allow it
        obj_v = cost_v + LAM*visits_v*extra_v - A*visits_v*is_today
        load = {p: float(fills_v[grp][(a_val[grp] == 1) & (is_today[grp] == 0)].sum()) for p, grp in by_ph.items()}   # new members only
        qsum = float((fills_v * cand.quality.values * a_val).sum()); qmin = (min(q_today, q_best) - 1e-6)*totf if C["QUALITY_FLOOR"] and "quality floor" not in relaxed else -inf
        n_pol = 0
        for pi_, grp in by_pair.items():
            cur = grp[a_val[grp] == 1]
            if len(cur) != 1: continue
            cur = int(cur[0]); best = cur
            for j in grp[np.argsort(obj_v[grp])]:
                if obj_v[j] >= obj_v[cur] - 1e-9: break
                if ub[A_ + j] < 0.5 or x_val[cand.xi.iat[j]] != 1: continue
                pj = cand.ncpdp.iat[j]; f_ = float(fills_v[j])
                if is_today[j] == 0 and "capacity" not in relaxed and load[pj] + f_ > cap[pj] + 1e-6: continue
                if qsum - f_*cand.quality.iat[cur] + f_*cand.quality.iat[j] < qmin: continue
                best = int(j); break
            if best != cur:
                a_val[cur] = 0; a_val[best] = 1
                if is_today[cur] == 0: load[cand.ncpdp.iat[cur]] -= float(fills_v[cur])
                if is_today[best] == 0: load[cand.ncpdp.iat[best]] += float(fills_v[best])
                qsum += float(fills_v[best])*(cand.quality.iat[best] - cand.quality.iat[cur]); n_pol += 1
        # ---- read the solution
        chosen = cand[a_val == 1].drop_duplicates("pi").set_index("pi")
        open_set = {p for p, k in xi.items() if x_val[k] == 1}
        used_set = set(chosen.ncpdp)                      # open and actually serving members: the network that matters when network cost is 0
        ph_use = chosen.groupby("ncpdp").agg(pairs=("cls", "size"), fills=("fills", "sum")).reset_index()
        for p, f_ in chosen[chosen.is_today == 0].groupby("ncpdp").fills.sum().items(): used_d[p] = used_d.get(p, 0.0) + float(f_)
        for key, k in zi.items(): fix_d[key] = float(np.round(v[k]))
        rows_ = pm[["MBR_ID", "cls", "setting", "today", "d0", "visits", "adj_claims", "fills", "cost_today", "paid_2025", "fee_2025"]].copy()
        rows_["new"] = chosen.ncpdp.reindex(pm.index).values; rows_["cost_new"] = chosen.cost.reindex(pm.index).values
        rows_["miles_new"] = chosen.dist.reindex(pm.index).values; rows_["extra_miles"] = chosen.extra.reindex(pm.index).values
        for e in ("ev_client", "ev_book", "ev_chain", "ev_avg"): rows_[e] = chosen[e].reindex(pm.index).fillna(0.0).values
        rows_["served"] = rows_.new.notna().astype(int)
        rows_["cost_new"] = rows_.cost_new.fillna(rows_.cost_today)                 # unserved pair: no change, no saving
        rows_["moved"] = ((rows_.new != rows_.today) & (rows_.served == 1)).astype(int)
        rows_["saving"] = rows_.cost_today - rows_.cost_new
        rows_["band"] = [band_of(x) if mv else "not moved" for x, mv in zip(rows_.extra_miles.fillna(0), rows_.moved)]
        rows_["beyond_radius"] = (rows_.miles_new > pm.radius.fillna(1e9).values).astype(int)
        # why a member could land somewhere both dearer and farther: today's pharmacy closed, full, or the quality floor
        t_open = rows_.today.map(lambda p_: (x_val[xi[p_]] == 1) if p_ in xi else False)
        rows_["adverse_reason"] = np.where((rows_.moved == 1) & (rows_.saving < -1) & (rows_.extra_miles > 0),
                                           np.where(~t_open, "today's pharmacy not in the network (size cap, chain or mandate rule)",
                                                    "quality floor" if C["QUALITY_FLOOR"] and "quality floor" not in relaxed else "unexplained"), "")
        rows_["metro"] = metro; rows_["design"] = tag(des); rows_["scenario"] = scenario
        allot = []
        for m_, s_, lst in guar:                                                    # allotted: nearest open normal pharmacy within reach
            opn = [p for p in lst if p in open_set]; r = MEMBERS.loc[m_]
            if opn:
                d = haversine(r.MBR_LAT, r.MBR_LON, PH.PHR_LAT[opn].values, PH.PHR_LON[opn].values); k = int(np.argmin(d))
                allot.append(dict(scenario=scenario, design=tag(des), metro=metro, MBR_ID=m_, setting=s_, allotted=opn[k], miles=round(float(d[k]), 2)))
            else: allot.append(dict(scenario=scenario, design=tag(des), metro=metro, MBR_ID=m_, setting=s_, allotted=None, miles=np.nan))
        res = dict(rows=rows_, open=sorted(open_set), used=sorted(used_set), today_open=sorted(today_set), status=st, gap=float(h.getInfo().mip_gap), solve_s=round(time.time()-t0, 1),
                   relaxed=", ".join(relaxed), n_var=n_var, n_row=nrow, allot=allot, closed=len(closed), polished=n_pol, ph_use=ph_use,
                   netcost_new=float(sum(PH.netcost[p] for p in open_set)), netcost_today=float(sum(PH.netcost[p] for p in today_set if p in PH.index)))
        results[des] = res
        print(f"  {str(metro):>10} {tag(des):<16} {st:<18} {res['solve_s']:>6}s  vars {n_var:>8,}  saving ${rows_.saving.sum():>12,.0f}  "
              f"moved {int(rows_.moved.sum()):>6,} of {len(rows_):,}  used {len(used_set):,} of {n_x:,} open {len(open_set):,}" + (f"  relaxed: {res['relaxed']}" if relaxed else ""), flush=True)
    return results

# ================================================================== 6. ALL METROS ==
def run_scenario(scenario, allowed_set):
    metros = pairs.groupby("metro").size().sort_values(ascending=False)
    if SMOKE: metros = metros.head(C["SMOKE_METROS"])
    stamp(f"scenario {scenario}: {len(metros)} metros, {int(metros.sum()):,} member x class pairs, {len(designs)} designs")
    used, chain_fix, out = {}, {}, {}
    for metro, cnt in metros.items():
        pm_all = pairs[pairs.metro == metro].copy(); mm_all = mem[mem.metro == metro]
        # a very large metro is cut into geographic blocks (lat then lon quantiles) solved one after another
        nb = int(math.ceil(len(pm_all) / C["MAX_PAIRS_PER_BLOCK"]))
        if nb > 1 and pm_all.MBR_LAT.notna().any():
            k1 = int(math.ceil(math.sqrt(nb))); k2 = int(math.ceil(nb / k1))
            lat_b = pd.qcut(pm_all.MBR_LAT.rank(method="first"), k1, labels=False).fillna(0).astype(int)
            lon_b = pm_all.groupby(lat_b).MBR_LON.transform(lambda x: pd.qcut(x.rank(method="first"), k2, labels=False, duplicates="drop")).fillna(0).astype(int)
            pm_all["block"] = lat_b.astype(str) + "." + lon_b.astype(str)
            mm_all = mm_all.merge(pm_all.drop_duplicates("MBR_ID")[["MBR_ID", "block"]], on="MBR_ID", how="left")
            mm_all["block"] = mm_all.block.fillna(pm_all.block.iloc[0])
        else:
            pm_all["block"] = "0"; mm_all = mm_all.assign(block="0")
        for blk in sorted(pm_all.block.unique()):
            label = f"{metro}" if nb == 1 else f"{metro}/{blk}"
            res = solve_metro(label, pm_all[pm_all.block == blk].drop(columns="block"), mm_all[mm_all.block == blk], allowed_set, used, chain_fix, scenario)
            if not res: continue
            for des, r in res.items():
                o = out.setdefault(des, dict(rows=[], open=set(), used=set(), today=set(), allot=[], metros=[], ph_use=[]))
                o["rows"].append(r["rows"]); o["open"] |= set(r["open"]); o["used"] |= set(r["used"]); o["today"] |= set(r["today_open"]); o["allot"] += r["allot"]
                o["ph_use"].append(r["ph_use"])
                o["metros"].append(dict(scenario=scenario, design=tag(des), metro=label, pairs=len(r["rows"]), status=r["status"], gap=round(r["gap"], 5), solve_s=r["solve_s"],
                                        variables=r["n_var"], rows=r["n_row"], relaxed=r["relaxed"], busy_pharmacies_closed_to_new=r["closed"], polished=r["polished"],
                                        saving=round(float(r["rows"].saving.sum())), moved=int(r["rows"].moved.sum()), open=len(r["open"]), used=len(r["used"]),
                                        netcost_new=r["netcost_new"], netcost_today=r["netcost_today"]))
    return out

scenarios = {"custom network": None}; scenarios.update(fixed_nets)
ALL = {nm: run_scenario(nm, allowed) for nm, allowed in scenarios.items()}

# ================================================================== 7. REPORT ==
stamp("reporting")
runs, bands, metros_tab, pharm_rows, adverse, gaps, allots, qual = [], [], [], [], [], [], [], []
held_rows = None
for scen, out in ALL.items():
    for des, o in out.items():
        rows = pd.concat(o["rows"], ignore_index=True)
        B0 = float(rows.cost_today.sum()); new = float(rows.cost_new.sum()); paid = float((rows.paid_2025 + rows.fee_2025).sum())
        mv = rows[rows.moved == 1]
        ev = rows[["ev_client", "ev_book", "ev_chain", "ev_avg"]].sum(); ev_tot = max(1e-9, float(ev.sum()))
        bad = mv[(mv.saving < -1) & (mv.extra_miles > 0)]
        netc_today = sum(m["netcost_today"] for m in o["metros"]); netc_new = sum(m["netcost_new"] for m in o["metros"])
        qt = float((rows.fills*rows.today.map(PH.quality).fillna(q_global)).sum()/rows.fills.sum()); qn = float((rows.fills*rows.new.fillna(rows.today).map(PH.quality).fillna(q_global)).sum()/rows.fills.sum())
        runs.append(dict(scenario=scen, design=tag(des), A=des[0], lam=des[1],
                 paid_2025_actual=round(paid), drug_bill_today=round(B0), drug_bill_new=round(new),
                 gross_saving=round(B0 - new), gross_saving_pct=round(100*(B0-new)/B0, 2) if B0 else 0,
                 network_cost_today=round(netc_today), network_cost_new=round(netc_new), net_saving=round(B0 - new + netc_today - netc_new),
                 members=int(rows.MBR_ID.nunique()), member_class_pairs=len(rows), members_moved=int(mv.MBR_ID.nunique()),
                 members_moved_pct=round(100*mv.MBR_ID.nunique()/max(1, rows.MBR_ID.nunique()), 1),
                 pairs_moved=len(mv), visits_moved=int(mv.visits.sum()), adjusted_claims_moved=round(float(mv.adj_claims.sum())),
                 sent_farther=int(mv[mv.extra_miles > 0].MBR_ID.nunique()), extra_miles_per_year=int((mv.extra_miles.fillna(0)*mv.visits).sum()),
                 avg_miles_today=round(float((rows.d0.fillna(0)*rows.visits).sum()/rows.visits.sum()), 2),
                 avg_miles_new=round(float((rows.miles_new.fillna(rows.d0).fillna(0)*rows.visits).sum()/rows.visits.sum()), 2),
                 beyond_radius=int(rows.beyond_radius.sum()), unserved_pairs=int((rows.served == 0).sum()), quality_today=round(qt, 4), quality_new=round(qn, 4),
                 pharmacies_open=len(o["open"]), pharmacies_used=len(o["used"]), pharmacies_today=len(o["today"]),
                 kept=len(o["used"] & o["today"]), dropped=len(o["today"] - o["used"]), added=len(o["used"] - o["today"]),
                 adverse_moves=len(bad), adverse_cost=round(float(bad.saving.sum())),
                 share_priced_client=round(100*float(ev.ev_client)/ev_tot, 1), share_priced_all_clients=round(100*float(ev.ev_book)/ev_tot, 1),
                 share_priced_chain=round(100*float(ev.ev_chain)/ev_tot, 1), share_priced_book_avg=round(100*float(ev.ev_avg)/ev_tot, 1),
                 allotted=len([a for a in o["allot"] if a["allotted"]]), allot_gaps=len([a for a in o["allot"] if not a["allotted"]]),
                 metros=len(o["metros"]), metros_relaxed=len([m for m in o["metros"] if m["relaxed"]]), solve_s=round(sum(m["solve_s"] for m in o["metros"]), 1)))
        assert abs(float(rows.saving.sum()) - (B0 - new)) < 1.0, "savings do not reconcile"      # headline = sum of member rows
        for lo, hi, nm in C["BANDS"]:
            g_ = mv[mv.band == nm]
            bands.append(dict(scenario=scen, design=tag(des), band=nm, member_class_moves=len(g_), members=int(g_.MBR_ID.nunique()),
                              saving=round(float(g_.saving.sum())), saving_per_move=round(float(g_.saving.sum())/len(g_)) if len(g_) else 0,
                              visits=int(g_.visits.sum()), extra_miles_per_year=int((g_.extra_miles.fillna(0)*g_.visits).sum())))
        assert abs(sum(b["saving"] for b in bands[-len(C["BANDS"]):]) - float(mv.saving.sum())) < 1 + len(C["BANDS"]), "bands do not reconcile"
        metros_tab += o["metros"]
        pu = pd.concat(o["ph_use"], ignore_index=True).groupby("ncpdp").sum() if o["ph_use"] else pd.DataFrame(columns=["pairs", "fills"])
        for p in sorted(o["used"] | o["today"]):
            pharm_rows.append(dict(scenario=scen, design=tag(des), PHR_NCPDP_ID=p, status=("kept" if p in o["used"] and p in o["today"] else "added" if p in o["used"] else "dropped"),
                                   assigned_pairs=int(pu.pairs.get(p, 0)) if len(pu) else 0, assigned_fills=round(float(pu.fills.get(p, 0.0))) if len(pu) else 0))
        if len(bad): adverse.append(bad.assign(scenario=scen, design=tag(des))[["scenario", "design", "metro", "MBR_ID", "cls", "today", "new", "saving", "extra_miles", "visits", "adverse_reason"]])
        allots += o["allot"]
        if scen == "custom network" and des == HOLD:
            held_rows = rows
            gaps.append(rows[rows.beyond_radius == 1][["metro", "MBR_ID", "cls", "setting", "today", "new", "miles_new", "d0"]].assign(reason="served beyond the access radius: today's pharmacy kept, or nothing inside"))
        if scen != "custom network" and (rows.served == 0).any():
            gaps.append(rows[rows.served == 0][["metro", "MBR_ID", "cls", "setting", "today", "new", "miles_new", "d0"]].assign(reason=f"{scen}: no pharmacy of this network within reach"))

RUNS, BANDS, METROS = pd.DataFrame(runs), pd.DataFrame(bands), pd.DataFrame(metros_tab)
PHARM = pd.DataFrame(pharm_rows)
if len(PHARM): PHARM = PHARM.merge(ph[["PHR_NCPDP_ID", "PHR_NM", "chain", "cls", "state", "PHR_CITY_NM", "metro", "fills_all", "fills_client", "quality"]], on="PHR_NCPDP_ID", how="left")
ADVERSE = pd.concat(adverse, ignore_index=True) if adverse else pd.DataFrame(columns=["scenario", "design", "metro", "MBR_ID"])
GAPS = pd.concat(gaps, ignore_index=True) if gaps else pd.DataFrame(); ALLOT = pd.DataFrame(allots)
if held_rows is None: held_rows = pd.concat(next(iter(ALL["custom network"].values()))["rows"], ignore_index=True)
hd = held_rows.copy()
for c_ in ("today", "new"): hd[c_ + "_name"] = hd[c_].map(PH.PHR_NM); hd[c_ + "_chain"] = hd[c_].map(PH.chain)
hd["extra_miles_per_year"] = (hd.extra_miles.fillna(0) * hd.visits).round(0)
WORST = hd[hd.moved == 1].sort_values("extra_miles_per_year", ascending=False).head(200)
# PETL files for the held design: baseline assignment and proposed assignment, one row per member and class
hd[["MBR_ID", "cls", "today"]].rename(columns={"today": "PHR_NCPDP_ID", "cls": "DISPENSER_CLASS"}).to_csv(os.path.join(OUT, "petl_baseline.csv"), index=False)
hd[["MBR_ID", "cls", "new"]].rename(columns={"new": "PHR_NCPDP_ID", "cls": "DISPENSER_CLASS"}).to_csv(os.path.join(OUT, "petl_proposed.csv"), index=False)

readme = pd.DataFrame({"sheet": ["Runs", "Bands", "Moved_members", "Worst_moves", "Pharmacies", "Metros", "Adverse_moves", "Access_gaps", "Allotted", "Config"],
    "what": ["one row per scenario and design: drug bill today and new, gross and net saving, members moved, visits and adjusted claims moved, miles, quality today and new, pharmacies kept, dropped, added, price evidence shares, adverse moves",
             "moved member x class rows by band of extra miles: count, saving, visits, extra miles. Bands add up to the moved saving",
             f"every member x class pair in the held design {tag(HOLD)}: today's and new pharmacy, miles, extra miles, visits, adjusted claims, saving, cost by price source",
             "the 200 members hit hardest in the held design, by extra miles over the year",
             "every pharmacy serving the client today or in the new network, per scenario and design: kept, dropped, added, members and fills assigned, with name, chain, class, state, metro, 2025 volume, quality. With network cost at 0 a pharmacy is in the network when it serves at least one member",
             "per metro and design: solver status, gap, time, size, which rules had to be relaxed, busy pharmacies closed to new members",
             "members assigned to a pharmacy that is both more expensive and farther than today's. Expected zero; any row is explained by capacity, chain or size rules",
             "member x class pairs served beyond their access radius in the held design, with the reason",
             "members with no retail fills or only Government / Alternate sites, and the normal pharmacy within reach allotted to them (zero fills)",
             "every setting used in this run"]})
with pd.ExcelWriter(os.path.join(OUT, "results.xlsx"), engine="openpyxl") as xw:
    readme.to_excel(xw, "README", index=False); RUNS.to_excel(xw, "Runs", index=False); BANDS.to_excel(xw, "Bands", index=False)
    hd.head(300_000).to_excel(xw, "Moved_members", index=False); WORST.to_excel(xw, "Worst_moves", index=False)   # full table in the parquet
    PHARM.to_excel(xw, "Pharmacies", index=False); METROS.to_excel(xw, "Metros", index=False)
    ADVERSE.to_excel(xw, "Adverse_moves", index=False); GAPS.to_excel(xw, "Access_gaps", index=False); ALLOT.to_excel(xw, "Allotted", index=False)
    pd.DataFrame([(k, json.dumps(v) if isinstance(v, (dict, list)) else str(v)) for k, v in C.items()], columns=["setting", "value"]).to_excel(xw, "Config", index=False)
RUNS.to_csv(os.path.join(OUT, "runs.csv"), index=False); hd.to_parquet(os.path.join(OUT, "moved_members_held.parquet"), index=False)

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    r_ = RUNS[RUNS.scenario == "custom network"]
    ra = r_[r_.lam == C["LAMBDA_HOLD"]].sort_values("A"); rl = r_[r_.A == C["A_HOLD"]].sort_values("lam")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ax[0].plot(ra.members_moved, ra.gross_saving/1e6, "-o", color="#002677"); ax[0].set_xlabel("members who change pharmacy"); ax[0].set_ylabel("gross saving, $M a year")
    for _, r in ra.iterrows(): ax[0].annotate(f"A={r.A:g}", (r.members_moved, r.gross_saving/1e6), textcoords="offset points", xytext=(4, -10), fontsize=8)
    ax[0].set_title(f"Savings against members moved, lambda = {C['LAMBDA_HOLD']:g}")
    ax[1].plot(rl.extra_miles_per_year, rl.gross_saving/1e6, "-s", color="#FF612B"); ax[1].set_xlabel("extra member miles a year"); ax[1].set_ylabel("gross saving, $M a year")
    for _, r in rl.iterrows(): ax[1].annotate(f"lam={r.lam:g}", (r.extra_miles_per_year, r.gross_saving/1e6), textcoords="offset points", xytext=(4, -10), fontsize=8)
    ax[1].set_title(f"Savings against extra miles, A = {C['A_HOLD']:g}")
    for a_ in ax: a_.grid(alpha=.25); a_.spines["top"].set_visible(False); a_.spines["right"].set_visible(False)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "tradeoff.png"), dpi=130); plt.close("all")
except Exception as e: print("chart skipped:", e)

print("\n=== RUNS ===")
print(RUNS[["scenario", "design", "drug_bill_today", "drug_bill_new", "gross_saving", "gross_saving_pct", "members_moved", "members_moved_pct", "sent_farther",
            "extra_miles_per_year", "quality_today", "quality_new", "pharmacies_today", "pharmacies_used", "dropped", "added", "adverse_moves", "share_priced_client", "metros_relaxed"]].to_string(index=False))
stamp(f"done. {OUT}: results.xlsx, runs.csv, petl_baseline.csv, petl_proposed.csv, moved_members_held.parquet, tradeoff.png")
