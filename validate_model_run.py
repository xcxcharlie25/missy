"""
validate_model_run.py   Independent check of a Final_Model_Run output against the raw tables. Read only.

Takes model_results (held design = custom network, A=5 lam=2 unless changed below), member_tables, book_tables and the RUCA
ZIP file, and re-derives every rule on its own: distances, settings and radius, who moved, kind of pharmacy, specialty,
frozen cases, money (every case repriced from the price table again), capacity, quality floor, size cap, allotments,
the Runs / Bands / Pharmacies sheets, the PETL files. Nothing from the model is trusted; each number is recomputed.

    python validate_model_run.py                  folders under ~/Downloads (book_tables, member_tables, model_results)
    python validate_model_run.py <folder>         folder holding those three folders

Writes validation_report.xlsx (sheets Summary and one sheet of examples per failed check) and validation_summary.csv
into model_results, and prints the summary. PASS = rule holds on every row. FAIL = at least one row breaks it, examples
in the report. INFO = a number to look at, not a rule.
"""
import os, sys, json, math
import numpy as np, pandas as pd, duckdb
from scipy.spatial import cKDTree

# ================================================================== CONFIG =====
HOME = os.path.expanduser("~")
FOLDER = sys.argv[1] if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]) else os.path.join(HOME, "Downloads")
BOOK, MEM, RES = (os.path.join(FOLDER, d) for d in ("book_tables", "member_tables", "model_results"))
RUCA_FILE = next((p for p in (os.path.join(FOLDER, "RUCA-codes-2020-zipcode.xlsx"), os.path.join(HOME, "Downloads", "RUCA-codes-2020-zipcode.xlsx")) if os.path.exists(p)), None)
PREFIX = "fedex_"
HELD_SCENARIO, HELD_DESIGN = "custom network", "A=5 lam=2"     # the design in moved_members_held.parquet (option 2 files: "A=5 lam=2 netcost=0")
# the rules as agreed (same values as the model CONFIG; change both if one changes)
NORMAL_TYPE = "Community or Retail Pharmacy"
NORMAL_CLASSES = ["Chain Pharmacy", "Independent Pharmacy", "Franchise Pharmacy"]
FROZEN_TYPES = ["Mail Order Pharmacy", "Home Infusion Therapy Provider"]
ACCESS_MILES = dict(URBAN=2.0, SUBURBAN=5.0, RURAL=15.0)
RUCA_SETTING = {1: "URBAN", 2: "SUBURBAN", 3: "SUBURBAN", 4: "RURAL", 5: "RURAL", 6: "RURAL", 7: "RURAL", 8: "RURAL", 9: "RURAL", 10: "RURAL"}
URBAN_MIN_WITHIN_2MI, SUBURBAN_MIN_WITHIN_5MI = 5, 1
CAPACITY_SHARE = 0.90
NETWORK_SIZE_SHARE = 1.0
MONEY_TOL = 1.0          # dollars per case
MILES_TOL = 0.02         # miles
MEMORY = "12GB"
EXAMPLES = 200           # rows kept per failed check

R_EARTH = 3958.7613
def haversine(lat1, lon1, lat2, lon2):
    la1, lo1, la2, lo2 = (np.radians(np.asarray(v, float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((la2-la1)/2)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
    return 2*R_EARTH*np.arcsin(np.sqrt(np.clip(a, 0, 1)))
def xyz(lat, lon):
    la, lo = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    return np.c_[np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la)]
def chord(miles): return 2*np.sin(np.asarray(miles, float)/(2*R_EARTH))
P = lambda d, f: os.path.join(d, f).replace("\\", "/")

SUMMARY, DETAILS = [], {}
def check(name, failures, total, note="", info=False):
    """failures: DataFrame of offending rows (empty = pass). total: rows the rule was tested on"""
    n = len(failures)
    status = "INFO" if info else ("PASS" if n == 0 else "FAIL")
    SUMMARY.append(dict(check=name, status=status, rows_tested=int(total), rows_failing=int(n), note=note))
    if n and not info: DETAILS[name] = failures.head(EXAMPLES)
    print(f"{status:<5} {name:<70} tested {int(total):>9,}  failing {n:>7,}  {note}", flush=True)

# ================================================================== 1. LOAD ====
print(f"folder {FOLDER}\nRUCA file {RUCA_FILE}", flush=True)
con = duckdb.connect(); con.execute(f"SET memory_limit='{MEMORY}'")
rows = pd.read_parquet(P(RES, "moved_members_held.parquet"))
rows["MBR_ID"] = rows.MBR_ID.astype(str)
for c in ("today", "new"): rows[c] = rows[c].astype("string")
RUNS = pd.read_csv(P(RES, "runs.csv"))
CSV = P(RES, "csv")
BANDS = pd.read_csv(P(CSV, "Bands.csv")); PHARM = pd.read_csv(P(CSV, "Pharmacies.csv"), dtype={"PHR_NCPDP_ID": str}); METROS = pd.read_csv(P(CSV, "Metros.csv"))
ALLOT = pd.read_csv(P(CSV, "Allotted.csv"), dtype={"MBR_ID": str, "allotted": str}) if os.path.getsize(P(CSV, "Allotted.csv")) > 5 else pd.DataFrame()
ADV = pd.read_csv(P(CSV, "Adverse_moves.csv"))
petl_b = pd.read_csv(P(RES, "petl_baseline.csv"), dtype=str); petl_p = pd.read_csv(P(RES, "petl_proposed.csv"), dtype=str)
held_tag = rows.design.iloc[0]
if held_tag != HELD_DESIGN: print(f"note: the held design in the file is '{held_tag}', using it", flush=True); HELD_DESIGN = held_tag
run = RUNS[(RUNS.scenario == HELD_SCENARIO) & (RUNS.design == HELD_DESIGN)].iloc[0]
NC_OBJ = bool(run.get("network_cost_in_objective", False)) if "network_cost_in_objective" in run.index else False
print(f"held design {HELD_DESIGN}: {len(rows):,} cases, Runs rows {len(RUNS)}", flush=True)

ph = pd.read_parquet(P(BOOK, "book_pharmacies.parquet")); ph["PHR_NCPDP_ID"] = ph.PHR_NCPDP_ID.astype(str)
ph["ptype"] = ph.PHR_DISPNSR_TYPE_DESC.astype("string").fillna(NORMAL_TYPE).str.strip().astype(str)
ph["cls"] = ph.PHR_DISPNSR_CLASS_DESC.astype("string").fillna("Independent Pharmacy").str.strip().astype(str)
ph["kind"] = ph.ptype + " | " + ph.cls
ph["normal"] = ph.kind.isin({f"{NORMAL_TYPE} | {c}" for c in NORMAL_CLASSES})
ph["frozen_type"] = ph.ptype.isin(FROZEN_TYPES)
ph["chain"] = ph.a_nf_pharmacy.fillna("Independent").astype(str)
for c in ("PHR_LAT", "PHR_LON"): ph[c] = pd.to_numeric(ph[c], errors="coerce")
ph["free_room"] = ph.fills_all.astype(float) * (1/CAPACITY_SHARE - 1)
PH = ph.set_index("PHR_NCPDP_ID")

mem = pd.read_parquet(P(MEM, PREFIX + "members.parquet")); mem["MBR_ID"] = mem.MBR_ID.astype(str)
for c in ("MBR_LAT", "MBR_LON"): mem[c] = pd.to_numeric(mem[c], errors="coerce")
MEMBERS = mem.set_index("MBR_ID")
dem = pd.read_parquet(P(MEM, PREFIX + "demand.parquet")); dem["MBR_ID"] = dem.MBR_ID.astype(str); dem["a_nt_gpi"] = dem.a_nt_gpi.astype(str)
use = pd.read_parquet(P(MEM, PREFIX + "member_pharmacy.parquet")); use["MBR_ID"] = use.MBR_ID.astype(str); use["PHR_NCPDP_ID"] = use.PHR_NCPDP_ID.astype(str)
def kind_claims(df):
    t = df.PHR_DISPNSR_TYPE_DESC.astype("string").fillna(NORMAL_TYPE).str.strip(); c = df.PHR_DISPNSR_CLASS_DESC.astype("string").fillna("Independent Pharmacy").str.strip()
    return (t + " | " + c).astype(str).where(df.mail.fillna(0).astype(int).values != 1, "Mail")
dem["kind"] = kind_claims(dem); use["kind"] = kind_claims(use)
for d_ in (dem, use): d_["specialty"] = d_.specialty.fillna(0).astype(int)
dem = dem[dem.qty > 0]
CASE = ["MBR_ID", "kind", "specialty"]
rows["frozen"] = rows.frozen.astype(int); rows["moved"] = rows.moved.astype(int); rows["specialty"] = rows.specialty.astype(int)
rows["kind_new"] = rows.new.map(PH.kind); rows["kind_today"] = rows.today.map(PH.kind)
rows["normal_new"] = rows.new.map(PH.normal); rows["MBR_LAT"] = rows.MBR_ID.map(MEMBERS.MBR_LAT); rows["MBR_LON"] = rows.MBR_ID.map(MEMBERS.MBR_LON)
mv = rows[rows.moved == 1]; live = rows[rows.frozen == 0]
print(f"book {len(ph):,} pharmacies | members {len(mem):,} | demand rows {len(dem):,}\n", flush=True)

# ================================================================== 2. STRUCTURE ==
# 2.1 one row per case, and the cases are exactly the member x kind x specialty groups in the demand table (with a metro)
exp_cases = dem.groupby(CASE).size().reset_index(name="n")
dup = rows[rows.duplicated(CASE, keep=False)]
check("2.1 one row per case (member x kind x specialty)", dup, len(rows))
missing = exp_cases.merge(rows[CASE], on=CASE, how="left", indicator=True); missing = missing[missing._merge == "left_only"]
check("2.2 every demand case is in the output (cases left out = no metro, see note)", missing[missing.MBR_ID.isin(mem[mem.MBR_LAT.notna()].MBR_ID)], len(exp_cases),
      note=f"{len(missing):,} demand cases not in output; {int((~missing.MBR_ID.isin(mem[mem.MBR_LAT.notna()].MBR_ID)).sum()):,} of them have no coordinates")
extra = rows.merge(exp_cases, on=CASE, how="left"); check("2.3 no output case without demand", extra[extra.n.isna()], len(rows))
# 2.4 pharmacies and members exist
bad = rows[rows.new.notna() & ~rows.new.isin(PH.index)]; check("2.4 every new pharmacy exists in the book", bad, int(rows.new.notna().sum()))
bad = rows[~rows.MBR_ID.isin(MEMBERS.index)]; check("2.5 every member exists in the member table", bad, len(rows))
# 2.6 served cases have a new pharmacy; unserved have no change
bad = rows[(rows.served == 1) & rows.new.isna()]; check("2.6 served cases have a new pharmacy", bad, len(rows))
bad = rows[(rows.served == 0) & ((rows.moved == 1) | (rows.saving.abs() > 0.5))]; check("2.7 unserved cases: not moved, no saving", bad, int((rows.served == 0).sum()))

# ================================================================== 3. DEMAND, VISITS, MONEY IN ==
agg = dem.groupby(CASE).agg(fills_x=("fills", "sum"), paid_x=("ingred", "sum"), fee_x=("fee", "sum"), qty_paid_x=("qty_paid", "sum")).reset_index()
r2 = rows.merge(agg, on=CASE, how="left")
bad = r2[(r2.fills - r2.fills_x).abs() > 1e-6]; check("3.1 case fills = all fills in demand (zero due fills count)", bad[CASE + ["fills", "fills_x"]], len(r2))
bad = r2[((r2.paid_2025 - r2.paid_x).abs() > 0.01) | ((r2.fee_2025 - r2.fee_x).abs() > 0.01)]
check("3.2 paid_2025 and fee_2025 = client money on paid fills in demand", bad[CASE + ["paid_2025", "paid_x", "fee_2025", "fee_x"]], len(r2))
vis = use.groupby(CASE).visits.sum().reset_index().rename(columns={"visits": "visits_x"})
r2 = r2.merge(vis, on=CASE, how="left"); r2["visits_x"] = r2.visits_x.fillna(1).clip(lower=1)
bad = r2[(r2.visits - r2.visits_x).abs() > 1e-6]; check("3.3 visits = distinct fill dates at the case's pharmacies", bad[CASE + ["visits", "visits_x"]], len(r2))
# today's pharmacy = the one with most fills within the case
tdy = use.sort_values(CASE + ["fills"], ascending=[True, True, True, False]).drop_duplicates(CASE)[CASE + ["PHR_NCPDP_ID"]].rename(columns={"PHR_NCPDP_ID": "today_x"})
r2 = r2.merge(tdy, on=CASE, how="left")
bad = r2[r2.today.fillna("") != r2.today_x.fillna("")]; check("3.4 today's pharmacy = most fills within the case", bad[CASE + ["today", "today_x"]], len(r2))
tot_paid = float((mem.ingred_all + mem.fee_all).sum())
check("3.5 paid_2025_actual in Runs = client money in the member table", pd.DataFrame([dict(runs=run.paid_2025_actual, members=tot_paid)]) if abs(run.paid_2025_actual - tot_paid) > max(1.0, 1e-6*tot_paid) else pd.DataFrame(), 1,
      note=f"Runs {run.paid_2025_actual:,.0f} vs member table {tot_paid:,.0f} (cases without a metro explain a small gap)", info=abs(run.paid_2025_actual - tot_paid) <= max(1.0, 1e-4*tot_paid))

# ================================================================== 4. KIND, FROZEN, SPECIALTY ==
bad = mv[(mv.frozen == 0) & (mv.kind_new != mv.kind)]
check("4.1 moved cases: new pharmacy is the same kind (type | class) as the case", bad[CASE + ["today", "new", "kind_new"]], len(mv))
bad = rows[(rows.frozen == 0) & (rows.kind_today.notna()) & (rows.kind_today != rows.kind)]
check("4.2 info: today's pharmacy kind in the book differs from the kind on the claims (today is kept anyway)", bad[CASE + ["today", "kind_today"]], len(rows), info=True)
fz = rows[rows.frozen == 1]
bad = fz[(fz.new.fillna("") != fz.today.fillna("")) | (fz.moved != 0) | (fz.saving.abs() > 0.005) | fz.miles_new.notna()]
check("4.3 frozen cases (mail, home infusion): kept, not moved, no saving, no miles", bad[CASE + ["today", "new", "moved", "saving", "miles_new"]], len(fz))
exp_frozen = rows[(rows.kind == "Mail") | rows.ptype.isin(FROZEN_TYPES)]
bad = pd.concat([exp_frozen[exp_frozen.frozen == 0], rows[(rows.frozen == 1) & ~rows.index.isin(exp_frozen.index)]])
check("4.4 frozen flag = mail kind or a frozen type, nothing else", bad[CASE + ["ptype", "frozen"]], len(rows))
bad = rows[(rows.frozen == 0) & rows.new.map(PH.frozen_type).fillna(False).astype(bool)]
check("4.5 no case that can move is assigned to a mail or home infusion pharmacy", bad[CASE + ["new"]], len(live))
# specialty: every drug of a moved specialty case has a price row with volume at the new pharmacy, and no chain / book fallback was used
sp = mv[(mv.specialty == 1) & (mv.frozen == 0)]
sp_d = dem[(dem.specialty == 1)].merge(sp[CASE + ["new"]], on=CASE)[["MBR_ID", "kind", "specialty", "new", "a_nt_gpi", "a_irpg_drugtype"]]
con.register("sp_d", sp_d)
sp_chk = con.execute(f"""SELECT s.*, p.claims_all, p.qty_all, p.unit_ingred
                         FROM sp_d s LEFT JOIN read_parquet('{P(BOOK, "book_price_table.parquet")}') p
                           ON CAST(p.PHR_NCPDP_ID AS VARCHAR) = s.new AND CAST(p.a_nt_gpi AS VARCHAR) = s.a_nt_gpi AND p.a_irpg_drugtype = s.a_irpg_drugtype""").df()
bad = sp_chk[sp_chk.claims_all.isna() | (sp_chk.claims_all <= 0) | sp_chk.unit_ingred.isna()]
check("4.6 specialty: new pharmacy has a price row with volume for EVERY drug of the case", bad, len(sp_chk), note=f"{len(sp):,} specialty cases moved")
bad = sp[(sp.ev_chain.abs() > 0.005) | (sp.ev_avg.abs() > 0.005)]
check("4.7 specialty: no chain or book average price used", bad[CASE + ["new", "ev_chain", "ev_avg"]], len(rows[rows.specialty == 1]))
check("4.8 info: specialty cases, moved, saving", pd.DataFrame(), len(rows[rows.specialty == 1]), info=True,
      note=f"{int(rows.specialty.sum()):,} specialty cases, {len(sp):,} moved, saving {float(sp.saving.sum()):,.0f} of {float(mv.saving.sum()):,.0f}")

# ================================================================== 5. DISTANCES, SETTINGS, RADIUS ==
d_new = haversine(rows.MBR_LAT, rows.MBR_LON, rows.new.map(PH.PHR_LAT).astype(float), rows.new.map(PH.PHR_LON).astype(float))
d_tod = haversine(rows.MBR_LAT, rows.MBR_LON, rows.today.map(PH.PHR_LAT).astype(float), rows.today.map(PH.PHR_LON).astype(float))
rows["miles_new_x"] = np.where(rows.frozen == 1, np.nan, d_new); rows["d0_x"] = np.where(rows.frozen == 1, np.nan, d_tod)
ok = live[live.miles_new.notna()].index
bad = rows.loc[ok][(rows.loc[ok].miles_new - rows.loc[ok].miles_new_x).abs() > MILES_TOL]
check("5.1 miles to the new pharmacy recomputed from coordinates", bad[CASE + ["new", "miles_new", "miles_new_x"]], len(ok))
ok = live[live.d0.notna()].index
bad = rows.loc[ok][(rows.loc[ok].d0 - rows.loc[ok].d0_x).abs() > MILES_TOL]
check("5.2 miles to today's pharmacy recomputed from coordinates", bad[CASE + ["today", "d0", "d0_x"]], len(ok))
ex = np.where(mv.frozen == 1, 0.0, np.maximum(0.0, mv.miles_new.fillna(0) - mv.d0.fillna(0)))
bad = mv[(mv.extra_miles.fillna(0) - ex).abs() > MILES_TOL]
check("5.3 extra miles = max(0, new miles - today miles) on moved cases", bad[CASE + ["d0", "miles_new", "extra_miles"]], len(mv))
bad = rows[(rows.moved == 0) & (rows.extra_miles.fillna(0).abs() > 1e-9)]; check("5.4 not moved: zero extra miles", bad[CASE + ["extra_miles"]], int((rows.moved == 0).sum()))
# settings from the RUCA file, then density fallback with normal pharmacies only
def load_ruca(path):
    if not path: return None
    xl = pd.ExcelFile(path)
    for sh in xl.sheet_names:
        head = pd.read_excel(xl, sheet_name=sh, header=None, nrows=6, dtype=str)
        for hr in range(len(head)):
            cols = [str(c).strip() for c in head.iloc[hr].tolist()]
            if "ZIPCode" in cols and "PrimaryRUCA" in cols:
                df = pd.read_excel(xl, sheet_name=sh, header=hr, dtype=str); df.columns = [str(c).strip() for c in df.columns]
                z = df["ZIPCode"].astype("string").str.replace(r"\D", "", regex=True).str.zfill(5)
                s = pd.Series(pd.to_numeric(df["PrimaryRUCA"], errors="coerce").map(RUCA_SETTING).values, index=z.values)
                s = s[s.notna() & (pd.Series(s.index).str.len() == 5).values]
                return s[~s.index.duplicated(keep="first")]
    return None
ruca = load_ruca(RUCA_FILE)
m2 = mem.copy(); m2["zip5"] = m2.MBR_ZIP.astype("string").str.replace(r"\D", "", regex=True).str.slice(0, 5).str.zfill(5)
m2["setting_ruca"] = m2.zip5.map(ruca) if ruca is not None else pd.NA
dens = ph[ph.normal & ph.PHR_LAT.notna() & ph.PHR_LON.notna()]
dt = cKDTree(xyz(dens.PHR_LAT.values, dens.PHR_LON.values))
ml = m2[m2.MBR_LAT.notna() & m2.MBR_LON.notna()]
n2 = dt.query_ball_point(xyz(ml.MBR_LAT.values, ml.MBR_LON.values), chord(2.0), return_length=True)
n5 = dt.query_ball_point(xyz(ml.MBR_LAT.values, ml.MBR_LON.values), chord(5.0), return_length=True)
m2.loc[ml.index, "setting_density"] = np.where(n2 >= URBAN_MIN_WITHIN_2MI, "URBAN", np.where(n5 >= SUBURBAN_MIN_WITHIN_5MI, "SUBURBAN", "RURAL"))
m2["setting_x"] = m2.setting_ruca.astype(object).where(m2.setting_ruca.notna(), m2.get("setting_density"))
rs = rows.drop_duplicates("MBR_ID")[["MBR_ID", "setting"]].merge(m2[["MBR_ID", "MBR_ZIP", "setting_ruca", "setting_density", "setting_x"]], on="MBR_ID", how="left")
bad = rs[rs.setting.fillna("") != rs.setting_x.fillna("")]
check("5.5 member setting = RUCA by ZIP (1 urban, 2-3 suburban, 4-10 rural), else density of normal pharmacies", bad, len(rs),
      note=f"RUCA {int(rs.setting_ruca.notna().sum()):,} members, density {int((rs.setting_ruca.isna() & rs.setting_density.notna()).sum()):,}, none {int(rs.setting_x.isna().sum()):,}")
rows["radius_x"] = rows.setting.map(ACCESS_MILES)
bad = mv[(mv.frozen == 0) & (mv.miles_new > mv.setting.map(ACCESS_MILES) + 1e-6)]
check("5.6 moved cases land inside the member's access radius (2 / 5 / 15 miles)", bad[CASE + ["setting", "miles_new"]], len(mv))
br = rows[(rows.frozen == 0) & (rows.miles_new > rows.radius_x + 1e-6)]
bad = pd.concat([br[br.beyond_radius != 1], rows[(rows.beyond_radius == 1) & ~rows.index.isin(br.index) & (rows.frozen == 0)]])
check("5.7 beyond_radius flag = served farther than the radius (today's pharmacy kept)", bad[CASE + ["setting", "miles_new", "beyond_radius"]], len(live))
check("5.8 info: cases kept beyond their radius (today's pharmacy, by rule)", pd.DataFrame(), len(live), info=True, note=f"{len(br):,} cases, {int(br.MBR_ID.nunique()):,} members")

# ================================================================== 6. MONEY: every case repriced again ==
# unit price at a pharmacy for a drug: the pharmacy's own row; a case that is not specialty falls back to the chain row, then the book average
cases = rows[CASE + ["today", "new", "frozen", "cost_today", "cost_new", "paid_2025", "fee_2025", "saving", "moved", "served"]].copy()
cases["case_id"] = np.arange(len(cases))
need = pd.concat([cases[["case_id", "today"]].rename(columns={"today": "ncpdp"}), cases[["case_id", "new"]].rename(columns={"new": "ncpdp"})]).dropna()
need["ncpdp"] = need.ncpdp.astype(str); need["side"] = ["today"]*int(cases.today.notna().sum()) + ["new"]*int(cases.new.notna().sum())
need = need.merge(cases[["case_id"] + CASE + ["frozen"]], on="case_id"); need = need[need.frozen == 0]
need["chain"] = need.ncpdp.map(PH.chain)
dmv = dem[["MBR_ID", "kind", "specialty", "a_nt_gpi", "a_irpg_drugtype", "qty_paid", "fills_paid"]]
con.register("need", need[["case_id", "side", "ncpdp", "chain"] + CASE]); con.register("dmv", dmv)
drugs = dem[["a_nt_gpi", "a_irpg_drugtype"]].drop_duplicates(); con.register("drugs", drugs)
price_file, chain_file = P(BOOK, "book_price_table.parquet"), P(BOOK, "book_chain_price_table.parquet")
con.execute(f"""CREATE OR REPLACE TABLE book AS
    SELECT d.a_nt_gpi, d.a_irpg_drugtype, sum(p.ingred_all)/nullif(sum(p.qty_all),0) AS bu, sum(p.fee_all)/nullif(sum(p.claims_all),0) AS bf
    FROM read_parquet('{price_file}') p JOIN drugs d ON CAST(p.a_nt_gpi AS VARCHAR) = d.a_nt_gpi AND p.a_irpg_drugtype = d.a_irpg_drugtype GROUP BY 1, 2""")
con.execute(f"""CREATE OR REPLACE TABLE pr AS
    SELECT CAST(PHR_NCPDP_ID AS VARCHAR) AS ncpdp, CAST(a_nt_gpi AS VARCHAR) AS a_nt_gpi, a_irpg_drugtype, unit_ingred AS pu, fee_per_fill AS pf
    FROM read_parquet('{price_file}') WHERE CAST(PHR_NCPDP_ID AS VARCHAR) IN (SELECT DISTINCT ncpdp FROM need) AND unit_ingred IS NOT NULL""")
con.execute(f"""CREATE OR REPLACE TABLE ch AS
    SELECT CAST(chain AS VARCHAR) AS chain, CAST(a_nt_gpi AS VARCHAR) AS a_nt_gpi, a_irpg_drugtype, unit_ingred AS cu, fee_per_fill AS cf
    FROM read_parquet('{chain_file}') WHERE unit_ingred IS NOT NULL""")
rep = con.execute("""
    WITH lines AS (
        SELECT n.case_id, n.side, d.qty_paid AS q, d.fills_paid AS f,
               CASE WHEN n.specialty = 1 THEN p.pu ELSE coalesce(p.pu, c.cu, b.bu) END AS u,
               CASE WHEN n.specialty = 1 THEN p.pf ELSE coalesce(p.pf, c.cf, b.bf) END AS fee
        FROM need n
        JOIN dmv d ON d.MBR_ID = n.MBR_ID AND d.kind = n.kind AND d.specialty = n.specialty
        LEFT JOIN pr p ON p.ncpdp = n.ncpdp AND p.a_nt_gpi = d.a_nt_gpi AND p.a_irpg_drugtype = d.a_irpg_drugtype
        LEFT JOIN ch c ON c.chain = n.chain AND c.a_nt_gpi = d.a_nt_gpi AND c.a_irpg_drugtype = d.a_irpg_drugtype
        LEFT JOIN book b ON b.a_nt_gpi = d.a_nt_gpi AND b.a_irpg_drugtype = d.a_irpg_drugtype)
    SELECT case_id, side, CASE WHEN sum(CASE WHEN u IS NULL THEN 1 ELSE 0 END) > 0 THEN NULL ELSE sum(u*q + coalesce(fee, 0)*f) END AS cost
    FROM lines GROUP BY 1, 2""").df()
rep = rep.pivot(index="case_id", columns="side", values="cost").reindex(cases.case_id)
cases["cost_today_x"] = rep["today"].values if "today" in rep.columns else np.nan
cases["cost_new_x"] = rep["new"].values if "new" in rep.columns else np.nan
actual = cases.paid_2025 + cases.fee_2025
# frozen: counted at what was paid. Today's pharmacy unpriced: what was paid. New = today: same as today
cases["cost_today_x"] = np.where(cases.frozen == 1, actual, cases.cost_today_x.fillna(actual))
same = cases.new.fillna("") == cases.today.fillna("")
cases["cost_new_x"] = np.where(cases.frozen == 1, cases.cost_today_x, np.where(same, cases.cost_today_x, cases.cost_new_x))
bad = cases[(cases.cost_today - cases.cost_today_x).abs() > MONEY_TOL]
check("6.1 cost today = the case repriced at today's pharmacy (paid fills x unit price + paid fills x fee)", bad[CASE + ["today", "cost_today", "cost_today_x"]], len(cases))
bad = cases[(cases.cost_new - cases.cost_new_x).abs() > MONEY_TOL]
check("6.2 cost new = the case repriced at the new pharmacy, same rule", bad[CASE + ["new", "cost_new", "cost_new_x"]], len(cases))
bad = cases[(cases.frozen == 0) & (cases.moved == 1) & cases.cost_new_x.isna()]
check("6.3 no case moved to a pharmacy where one of its drugs has no price", bad[CASE + ["new"]], int(((cases.frozen == 0) & (cases.moved == 1)).sum()))
bad = cases[(cases.saving - (cases.cost_today - cases.cost_new)).abs() > 0.01]; check("6.4 saving = cost today - cost new", bad[CASE + ["saving", "cost_today", "cost_new"]], len(cases))
bad = cases[(cases.moved == 0) & (cases.saving.abs() > 0.005)]; check("6.5 not moved: no saving", bad[CASE + ["saving"]], int((cases.moved == 0).sum()))
nz = cases.merge(agg[CASE + ["qty_paid_x"]], on=CASE, how="left")
bad = nz[(nz.frozen == 0) & (nz.qty_paid_x <= 0) & (nz.cost_new.abs() > 0.005)]
check("6.6 cases with no paid fills carry no money (zero due fills are not priced)", bad[CASE + ["cost_today", "cost_new"]], int(((nz.frozen == 0) & (nz.qty_paid_x <= 0)).sum()))
check("6.7 info: client bill today vs what was actually paid in 2025 (repricing at today's pharmacy)", pd.DataFrame(), len(cases), info=True,
      note=f"repriced {float(cases.cost_today.sum()):,.0f} vs paid {float(actual.sum()):,.0f} ({100*(float(cases.cost_today.sum())/float(actual.sum())-1):+.2f}%)")

# ================================================================== 7. CAPACITY, QUALITY, SIZE CAP ==
newload = mv[mv.frozen == 0].groupby("new").fills.sum()
relaxed_cap = set(METROS[(METROS.design == HELD_DESIGN) & (METROS.scenario == HELD_SCENARIO) & METROS.relaxed.fillna("").str.contains("capacity")].metro)
m_of_ph = mv[mv.frozen == 0].groupby("new").metro.agg(lambda s: set(s))
cap = pd.DataFrame(dict(new_fills=newload, free_room=newload.index.map(PH.free_room))); cap["relaxed_metro"] = [bool(m_of_ph[p] & relaxed_cap) for p in cap.index]
bad = cap[(cap.new_fills > cap.free_room + 1e-6) & ~cap.relaxed_metro]
check("7.1 capacity: fills moved INTO a pharmacy <= its free room (2025 volume / 0.9 - 2025 volume)", bad, len(cap), note=f"{len(relaxed_cap)} metro blocks had capacity relaxed")
bad = cap[(cap.free_room < 1.0) & (cap.new_fills > 0) & ~cap.relaxed_metro]
check("7.2 a pharmacy with no free room takes no new members", bad, len(cap))
# quality floor per metro block: fill weighted quality of the new assignment >= today's, unless relaxed
q = PHARM[(PHARM.design == HELD_DESIGN) & (PHARM.scenario == HELD_SCENARIO)].drop_duplicates("PHR_NCPDP_ID").set_index("PHR_NCPDP_ID").quality
qg = float(q.mean())
lv = live.copy(); lv["q_t"] = lv.today.map(q).fillna(qg); lv["q_n"] = lv.new.fillna(lv.today).map(q).fillna(qg)
qm = lv.groupby("metro").apply(lambda g: pd.Series(dict(q_today=(g.fills*g.q_t).sum()/g.fills.sum(), q_new=(g.fills*g.q_n).sum()/g.fills.sum())), include_groups=False).reset_index()
relaxed_q = set(METROS[(METROS.design == HELD_DESIGN) & (METROS.scenario == HELD_SCENARIO) & METROS.relaxed.fillna("").str.contains("quality")].metro)
bad = qm[(qm.q_new < qm.q_today - 1e-4) & ~qm.metro.isin(relaxed_q)]
check("7.3 quality floor per metro: fill weighted PDC of the new network >= today's", bad, len(qm), note=f"overall today {float((lv.fills*lv.q_t).sum()/lv.fills.sum()):.4f} new {float((lv.fills*lv.q_n).sum()/lv.fills.sum()):.4f}")
# size cap per metro block: pharmacies switched on <= pharmacies the client uses today in that block (option 1 only)
if not NC_OBJ:
    tod = live.groupby("metro").today.nunique().rename("today_count")
    mt = METROS[(METROS.design == HELD_DESIGN) & (METROS.scenario == HELD_SCENARIO)].set_index("metro")[["open", "network", "relaxed"]].join(tod)
    bad = mt[(mt.open > np.ceil(NETWORK_SIZE_SHARE * mt.today_count)) & ~mt.relaxed.fillna("").str.contains("size")]
    check("7.4 size cap per metro: open pharmacies <= pharmacies used today (unless relaxed)", bad.reset_index(), len(mt), note=f"{int(mt.relaxed.fillna('').str.contains('size').sum())} blocks relaxed the cap")

# ================================================================== 8. NORMAL PHARMACY FOR EVERYONE, ALLOTTED ==
has_normal_case = set(rows[rows.is_normal.astype(bool) & (rows.served == 1)].MBR_ID)
al = ALLOT[(ALLOT.design == HELD_DESIGN) & (ALLOT.scenario == HELD_SCENARIO)] if len(ALLOT) else pd.DataFrame(columns=["MBR_ID", "allotted", "miles", "setting"])
al = al.copy(); al["MBR_ID"] = al.MBR_ID.astype(str)
in_rows = set(rows.MBR_ID)
need_all = mem[mem.MBR_ID.isin(in_rows) & ~mem.MBR_ID.isin(has_normal_case) & mem.MBR_LAT.notna()]
need_all = need_all[need_all.MBR_ID.map(rows.drop_duplicates("MBR_ID").set_index("MBR_ID").setting).notna()]
bad = need_all[~need_all.MBR_ID.isin(set(al.MBR_ID))]
check("8.1 every member without a normal case (with a home and a setting) has an Allotted row", bad[["MBR_ID", "MBR_ZIP"]], len(need_all))
bad = al[al.MBR_ID.isin(has_normal_case)]; check("8.2 no member with a normal case is also allotted", bad, len(al))
ala = al[al.allotted.notna()].copy()
bad = ala[~ala.allotted.map(PH.normal).fillna(False).astype(bool)]; check("8.3 allotted pharmacy is a normal pharmacy", bad, len(ala))
ala["miles_x"] = haversine(ala.MBR_ID.map(MEMBERS.MBR_LAT), ala.MBR_ID.map(MEMBERS.MBR_LON), ala.allotted.map(PH.PHR_LAT).astype(float), ala.allotted.map(PH.PHR_LON).astype(float))
bad = ala[(ala.miles_x > ala.setting.map(ACCESS_MILES) + 1e-6) | ((ala.miles - ala.miles_x).abs() > MILES_TOL)]
check("8.4 allotted pharmacy is inside the member's radius, miles recomputed", bad, len(ala))
net = PHARM[(PHARM.design == HELD_DESIGN) & (PHARM.scenario == HELD_SCENARIO) & PHARM.status.isin(["kept", "added"])].PHR_NCPDP_ID.astype(str)
bad = ala[~ala.allotted.isin(set(net))]; check("8.5 allotted pharmacy is in the network", bad, len(ala))
check("8.6 info: members allotted, allotment gaps", pd.DataFrame(), len(al), info=True, note=f"{len(ala):,} allotted, {int(al.allotted.isna().sum()):,} with no normal pharmacy within reach")

# ================================================================== 9. SHEETS RECONCILE ==
served = rows[rows.served == 1]
net_x = set(served[served.frozen == 0].new.dropna()) | set(ala.allotted)
tod_x = set(live.today.dropna())
exp = dict(client_drug_bill_today=rows.cost_today.sum(), client_drug_bill_new=rows.cost_new.sum(), client_saving=rows.saving.sum(),
           members=rows.MBR_ID.nunique(), members_with_retail_cases=live.MBR_ID.nunique(), cases=len(rows), cases_specialty=int(rows.specialty.sum()), cases_frozen=int(rows.frozen.sum()),
           members_moved=mv.MBR_ID.nunique(), cases_moved=len(mv), visits_total=live.visits.sum(), visits_moved=mv.visits.sum(),
           visits_moved_pct=100*mv.visits.sum()/live.visits.sum(), members_moved_pct=100*mv.MBR_ID.nunique()/live.MBR_ID.nunique(),
           sent_farther=mv[mv.extra_miles > 0].MBR_ID.nunique(), extra_miles_per_year=(mv.extra_miles.fillna(0)*mv.visits).sum(),
           beyond_radius=int(rows.beyond_radius.sum()), unserved_cases=int((rows.served == 0).sum()),
           pharmacies_today=len(tod_x), pharmacies_network_new=len(net_x), kept=len(net_x & tod_x), dropped=len(tod_x - net_x), added=len(net_x - tod_x),
           adverse_moves=len(mv[(mv.saving < -1) & (mv.extra_miles > 0)]))
diff = []
for k, v in exp.items():
    got = float(run[k]); tol = 1.0 if k in ("client_drug_bill_today", "client_drug_bill_new", "client_saving", "extra_miles_per_year") else (0.06 if k.endswith("_pct") else 0.5)
    if abs(got - float(v)) > tol: diff.append(dict(column=k, runs=got, recomputed=round(float(v), 2)))
check("9.1 Runs sheet (held design) recomputed from the case rows", pd.DataFrame(diff), len(exp))
bd = BANDS[(BANDS.design == HELD_DESIGN) & (BANDS.scenario == HELD_SCENARIO)]
diff = []
for _, b in bd.iterrows():
    g_ = mv[mv.band == b.band]
    if abs(len(g_) - b.case_moves) > 0.5 or abs(float(g_.saving.sum()) - b.saving) > 1.0 or abs(float(g_.visits.sum()) - b.visits) > 0.5: diff.append(dict(band=b.band, sheet_moves=b.case_moves, moves=len(g_), sheet_saving=b.saving, saving=round(float(g_.saving.sum()))))
check("9.2 Bands sheet = moved cases grouped by extra miles band", pd.DataFrame(diff), len(bd))
bad = mv[(mv.moved == 1) & ((mv.band == "not moved") | ((mv.extra_miles <= 0) & (mv.band != "as close or closer")) | ((mv.extra_miles > 15) & (mv.band != "over 15")))]
check("9.3 band label matches extra miles", bad[CASE + ["extra_miles", "band"]], len(mv))
pm = PHARM[(PHARM.design == HELD_DESIGN) & (PHARM.scenario == HELD_SCENARIO)].copy(); pm["PHR_NCPDP_ID"] = pm.PHR_NCPDP_ID.astype(str)
pm["status_x"] = np.where(pm.PHR_NCPDP_ID.isin(net_x) & pm.PHR_NCPDP_ID.isin(tod_x), "kept", np.where(pm.PHR_NCPDP_ID.isin(net_x), "added", "dropped"))
bad = pm[pm.status != pm.status_x]; check("9.4 Pharmacies sheet status (kept / added / dropped) recomputed", bad[["PHR_NCPDP_ID", "status", "status_x"]], len(pm))
miss = (net_x | tod_x) - set(pm.PHR_NCPDP_ID); check("9.5 Pharmacies sheet lists every pharmacy used today or in the new network", pd.DataFrame({"PHR_NCPDP_ID": sorted(miss)}), len(net_x | tod_x))
bad = pm[pm.ptype.isin(FROZEN_TYPES)]; check("9.6 Pharmacies sheet has no mail or home infusion pharmacies", bad[["PHR_NCPDP_ID", "ptype", "status"]], len(pm))
adv_x = mv[(mv.saving < -1) & (mv.extra_miles > 0)]
ad = ADV[(ADV.design == HELD_DESIGN) & (ADV.scenario == HELD_SCENARIO)] if len(ADV) and "design" in ADV.columns else pd.DataFrame()
check("9.7 Adverse_moves sheet = moved cases both dearer and farther", pd.DataFrame([dict(sheet=len(ad), recomputed=len(adv_x))]) if len(ad) != len(adv_x) else pd.DataFrame(), len(adv_x),
      note=f"{len(adv_x):,} adverse moves, reasons: {dict(adv_x.adverse_reason.value_counts()) if len(adv_x) else {}}")
# monotone trade off across designs
r_ = RUNS[RUNS.scenario == HELD_SCENARIO].copy()
if NC_OBJ and "network_cost_per_pharmacy" in r_.columns: r_ = r_[r_.network_cost_per_pharmacy == r_.network_cost_per_pharmacy.iloc[0]]
ra = r_[r_.lam == r_.lam.mode().iloc[0]].sort_values("A"); rl = r_[r_.A == r_.A.mode().iloc[0]].sort_values("lam")
bad = pd.concat([ra[ra.members_moved.diff() > 0][["design", "members_moved"]], rl[rl.extra_miles_per_year.diff() > 0][["design", "extra_miles_per_year"]]])
check("9.8 trade off: members moved falls as A rises; extra miles fall as lambda rises", bad, len(ra) + len(rl), note=f"saving from {r_.client_saving.min():,.0f} to {r_.client_saving.max():,.0f}")
bad = r_[r_.quality_new < r_.quality_today - 1e-4][["design", "quality_today", "quality_new"]]; check("9.9 every design keeps quality at or above today", bad, len(r_))
bad = r_[(r_.optum_network_cost_today.abs() + r_.optum_network_cost_new.abs() > 0) & (r_.client_saving != (r_.client_drug_bill_today - r_.client_drug_bill_new))][["design"]] if "optum_network_cost_today" in r_.columns else pd.DataFrame()
check("9.10 client saving = drug bill today - drug bill new, network cost never inside it", pd.concat([bad, r_[(r_.client_saving - (r_.client_drug_bill_today - r_.client_drug_bill_new)).abs() > 1][["design"]]]), len(r_))

# ================================================================== 10. PETL ==
key = ["MBR_ID", "DISPENSER_TYPE", "DISPENSER_CLASS", "SPECIALTY", "MAIL"]
bad = petl_b[petl_b.duplicated(key, keep=False)]; check("10.1 PETL baseline: one row per case", bad, len(petl_b))
bad = petl_p[petl_p.duplicated(key, keep=False)]; check("10.2 PETL proposed: one row per case", bad, len(petl_p))
j = petl_b.merge(petl_p, on=key, how="outer", suffixes=("_b", "_p"), indicator=True)
check("10.3 PETL baseline and proposed have the same cases", j[j._merge != "both"], len(j))
rr = rows.copy(); rr["MAIL"] = (rr.kind == "Mail").astype(int).astype(str); rr["SPECIALTY"] = rr.specialty.astype(str)
rr = rr.rename(columns={"ptype": "DISPENSER_TYPE", "cls": "DISPENSER_CLASS"})
j2 = rr[key + ["today", "new"]].merge(petl_b, on=key, how="left").merge(petl_p, on=key, how="left", suffixes=("_b", "_p"))
bad = j2[(j2.today.fillna("") != j2.PHR_NCPDP_ID_b.fillna("")) | (j2.new.fillna("") != j2.PHR_NCPDP_ID_p.fillna(""))]
check("10.4 PETL baseline = today's pharmacy, proposed = new pharmacy, per case", bad, len(j2))
bad = petl_p[petl_p.PHR_NCPDP_ID.notna() & ~petl_p.PHR_NCPDP_ID.isin(PH.index)]; check("10.5 PETL proposed pharmacies exist in the book", bad, len(petl_p))

# ================================================================== REPORT ==
S = pd.DataFrame(SUMMARY)
S.to_csv(P(RES, "validation_summary.csv"), index=False)
try:
    with pd.ExcelWriter(P(RES, "validation_report.xlsx"), engine="openpyxl") as xw:
        S.to_excel(xw, sheet_name="Summary", index=False)
        for i, (name, df_) in enumerate(DETAILS.items()):
            df_.to_excel(xw, sheet_name=f"{name.split(' ')[0].replace('.', '_')}_examples"[:31], index=False)
except Exception as e: print(f"validation_report.xlsx not written ({e}); validation_summary.csv is there")
n_fail = int((S.status == "FAIL").sum())
print(f"\n{int((S.status == 'PASS').sum())} passed, {n_fail} failed, {int((S.status == 'INFO').sum())} info. Report: {P(RES, 'validation_report.xlsx')}")
