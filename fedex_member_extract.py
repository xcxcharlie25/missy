"""
fedex_member_extract.py

Member side data for one client (CARRIER_ID = CLIENT), everything the model run needs.

Rules (agreed Nov 2026):
    rows      the client's approved claims: amount due >= 0 (reversals and negatives out), client price type not blank
              and not ZERO, fill date in YEAR. The files are cut by submission date, so older fills are dropped here.
    repeats   rows identical on member, pharmacy, drug (GPI + G/B), fill date, quantity, days supply, ingredient cost
              and fee are one fill (a claim submitted several times). One is kept, across all files.
    money     client ingredient cost + client dispensing fee, on the rows the client actually paid (amount due > 0).
              Fills with amount due = 0 (the member paid the whole fill) still count as fills, units, days and visits,
              but add no money and are flagged so the model does not price them anywhere.
    kind      PHR_DISPNSR_TYPE_DESC and PHR_DISPNSR_CLASS_DESC together. Nothing is filtered by type or class.
    specialty a_nf_specialty_flag = Specialty. Carried on the demand and member x pharmacy rows: specialty fills are
              their own case in the model.
    mail      type Mail Order Pharmacy, or MAIL_TODAY = 1. Carried the same way: mail is its own category in the model.
    zip       the member's home ZIP from ZIP_FILE (MBR_ID, mbr_std_zip), for the urban / suburban / rural standard.

Reads the parquet files ONE AT A TIME with DuckDB, keeps only the client's rows, and combines at the end.
Finished files are recorded, so if the run stops you run it again and it continues.

Outputs (in OUT folder):
    per_file/<file>_members.parquet / .xlsx         that file's members (repeats removed within the file)
    per_file/<file>_demand.parquet / _check.xlsx    that file's member x type x class x specialty x mail x GPI x G/B rows
    per_file/<file>_member_pharmacy.parquet / .xlsx that file's member x pharmacy rows
    fedex_members.parquet / .xlsx                   one row per member: location, ZIP, group, totals, main pharmacy
    fedex_demand.parquet / _check.xlsx              member x type x class x specialty x mail x GPI x G/B: fills, qty, days, money
    fedex_member_pharmacy.parquet / .xlsx           member x pharmacy used today: fills, visits, money
    fedex_maint_fills.parquet                       maintenance fills for the adherence (quality) score
    member_stage.duckdb                             working database, delete when done

Run:  python fedex_member_extract.py
"""
import re
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

# ------------------------------- CONFIG --------------------------------------
FOLDER    = Path(__file__).parent              # the folder this script is in (with the parquet files)
PATTERN   = "*2025_*.parquet"
OUT       = FOLDER / "member_tables"
CLIENT    = "FEDEX"                            # CARRIER_ID of the client
YEAR      = 2025                               # fills outside this year are dropped
MEMORY    = "16GB"                             # a few GB below the RAM of the machine
THREADS   = 6
MBR_LAT   = None                               # None = detect automatically, or the column name
MBR_LON   = None
ZIP_FILE  = FOLDER / "member_zipcodes.xlsx"    # MBR_ID, mbr_std_zip. Optional: skipped with a message if missing
XLSX_ROWS = 200_000
PER_FILE  = True
# the normal kind of pharmacy: every member must end up with one of these (the model does that, this script only labels it)
NORMAL_TYPE    = "Community or Retail Pharmacy"
NORMAL_CLASSES = ["Chain Pharmacy", "Independent Pharmacy", "Franchise Pharmacy"]
MAIL_TYPE      = "Mail Order Pharmacy"

C = dict(  # column names in the parquet files; change here if yours differ
    mbr="MBR_ID", carrier="CARRIER_ID", account="ACCOUNT_ID", group="EMPLOYER_GROUP_ID",
    ncpdp="PHR_NCPDP_ID", ptype="PHR_DISPNSR_TYPE_DESC", pclass="PHR_DISPNSR_CLASS_DESC",
    gpi="a_nt_gpi", gb="a_irpg_drugtype", qty="a_nf_drg_qty", days="DAYS_SUPPLY",
    maint="a_nt_maintenance_flag", spec="a_nf_specialty_flag",
    ingred="m_nf_clt_ingred_cost_paid", fee="m_nf_clt_dispensing_fee",
    due="m_nf_app_due_amt", prctyp="a_nf_clt_prc_typ", filled="FILLED_DT",
)
MAIL_FLAG = "MAIL_TODAY"                       # optional column: 1 = mail fill. Mail is also recognised by the pharmacy type
# -----------------------------------------------------------------------------

if len(sys.argv) > 1:
    FOLDER = Path(sys.argv[1])
    OUT = FOLDER / "member_tables"
    ZIP_FILE = FOLDER / "member_zipcodes.xlsx"

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "duck_tmp").mkdir(exist_ok=True)
PER = OUT / "per_file"
PER.mkdir(exist_ok=True)

files = sorted(FOLDER.glob(PATTERN))
if not files:
    sys.exit(f"No files matching {PATTERN} in {FOLDER}")
print(f"{len(files)} files in {FOLDER}")

con = duckdb.connect(str(OUT / "member_stage.duckdb"))
con.execute(f"SET memory_limit='{MEMORY}'")
con.execute(f"SET threads={THREADS}")
con.execute(f"SET temp_directory='{(OUT / 'duck_tmp').as_posix()}'")
con.execute("SET preserve_insertion_order=false")


def q(sql):
    return con.execute(sql)


def columns_of(path):
    return [r[0] for r in q(f"DESCRIBE SELECT * FROM read_parquet('{path.as_posix()}')").fetchall()]


def table_exists(name):
    return q(f"SELECT count(*) FROM information_schema.tables WHERE table_name='{name}'").fetchone()[0] > 0


def xlsx(df, path):
    """the xlsx copies are only for looking at the data; if Excel writing fails the run goes on (the parquet is what the model reads)"""
    try:
        df.to_excel(path, sheet_name="data", index=False)
    except Exception as e:
        print(f"   {path.name}: not written ({type(e).__name__}: {e}), parquet is fine")


def export(sql, stem, xlsx_rows=None):
    """parquet with everything, xlsx with everything or the first xlsx_rows rows"""
    q(f"COPY ({sql}) TO '{(stem.with_suffix('.parquet')).as_posix()}' (FORMAT PARQUET)")
    if xlsx_rows is not None:
        xlsx(q(f"{sql} LIMIT {xlsx_rows}").df(), stem.with_suffix(".xlsx"))


# ---- check columns on the first file -----------------------------------------
cols = columns_of(files[0])
missing = [v for v in C.values() if v not in cols]
if missing:
    sys.exit(f"Columns not found in {files[0].name}: {missing}\nColumns present: {cols}")
HAS_MAIL_FLAG = MAIL_FLAG in cols
if not HAS_MAIL_FLAG:
    print(f"no {MAIL_FLAG} column: mail is recognised by the pharmacy type only")

if MBR_LAT is None or MBR_LON is None:
    lat = [c for c in cols if re.search(r"(MBR|MEM).*LAT", c, re.I) and not re.search(r"PH(R|ARM)", c, re.I)]
    lon = [c for c in cols if re.search(r"(MBR|MEM).*(LON|LNG)", c, re.I) and not re.search(r"PH(R|ARM)", c, re.I)]
    MBR_LAT = MBR_LAT or (lat[0] if lat else None)
    MBR_LON = MBR_LON or (lon[0] if lon else None)
print("member coordinates:", MBR_LAT, MBR_LON)
lat_col = f"TRY_CAST({MBR_LAT} AS DOUBLE)" if MBR_LAT else "NULL::DOUBLE"     # coordinates are text in the files
lon_col = f"TRY_CAST({MBR_LON} AS DOUBLE)" if MBR_LON else "NULL::DOUBLE"

# the client's approved claims of the year: one row per claim row as it is in the file (repeats removed later)
IS_MAINT = f"CASE WHEN upper(trim(CAST({C['maint']} AS VARCHAR))) IN ('Y', 'YES', '1', 'TRUE', 'MAINTENANCE') THEN 1 ELSE 0 END"
IS_SPEC  = f"CASE WHEN upper(trim(CAST({C['spec']} AS VARCHAR))) IN ('SPECIALTY', 'Y', 'YES', '1', 'TRUE') THEN 1 ELSE 0 END"
IS_MAIL  = (f"CASE WHEN {C['ptype']} = '{MAIL_TYPE}'" + (f" OR CAST({MAIL_FLAG} AS VARCHAR) IN ('1', 'Y', 'TRUE', 'true')" if HAS_MAIL_FLAG else "")
            + " THEN 1 ELSE 0 END")
ROWS_SQL = f"""
    SELECT '{{file}}' AS file,
           CAST({C['mbr']} AS VARCHAR) AS mbr, CAST({C['account']} AS VARCHAR) AS account, CAST({C['group']} AS VARCHAR) AS grp,
           {lat_col} AS lat, {lon_col} AS lon,
           CAST({C['ncpdp']} AS VARCHAR) AS ncpdp, {C['ptype']} AS ptype, {C['pclass']} AS pclass,
           CAST({C['gpi']} AS VARCHAR) AS gpi, {C['gb']} AS gb,
           TRY_CAST({C['filled']} AS DATE) AS filled,
           {C['qty']} AS qty, {C['days']} AS days,
           {C['ingred']} AS ingred, {C['fee']} AS fee,
           CASE WHEN {C['due']} > 0 THEN 1 ELSE 0 END AS paid,
           {IS_MAINT} AS maint, {IS_SPEC} AS spec, {IS_MAIL} AS mail
    FROM read_parquet('{{path}}')
    WHERE CAST({C['carrier']} AS VARCHAR) = '{CLIENT}'
      AND {C['due']} >= 0
      AND coalesce(trim(CAST({C['prctyp']} AS VARCHAR)), '') NOT IN ('', 'ZERO')
      AND TRY_CAST({C['filled']} AS DATE) BETWEEN DATE '{YEAR}-01-01' AND DATE '{YEAR}-12-31'
"""
# repeats: one row per identical fill. Other columns take the first value seen
KEY = "mbr, ncpdp, gpi, gb, filled, qty, days, ingred, fee"
DEDUPE = f"""
    SELECT {KEY}, any_value(file) AS file, any_value(account) AS account, any_value(grp) AS grp,
           any_value(lat) AS lat, any_value(lon) AS lon, any_value(ptype) AS ptype, any_value(pclass) AS pclass,
           max(paid) AS paid, max(maint) AS maint, max(spec) AS spec, max(mail) AS mail
    FROM {{src}} GROUP BY {KEY}
"""
# sums on a fills table: money only on fills the client paid
MONEY = "CASE WHEN paid = 1 THEN ingred ELSE 0 END", "CASE WHEN paid = 1 THEN fee ELSE 0 END"
IS_NORMAL = f"CASE WHEN ptype = '{NORMAL_TYPE}' AND pclass IN ({', '.join(repr(c) for c in NORMAL_CLASSES)}) AND mail = 0 THEN 1 ELSE 0 END"


def demand_sql(src):
    return f"""
    SELECT mbr AS {C['mbr']}, ptype AS {C['ptype']}, pclass AS {C['pclass']}, spec AS specialty, mail, gpi AS {C['gpi']}, gb AS {C['gb']},
           count(*)::BIGINT AS fills, sum(qty) AS qty, sum(days) AS days,
           sum(paid)::BIGINT AS fills_paid, sum(CASE WHEN paid = 1 THEN qty ELSE 0 END) AS qty_paid,
           sum({MONEY[0]}) AS ingred, sum({MONEY[1]}) AS fee, max(maint) AS maint
    FROM {src} GROUP BY 1, 2, 3, 4, 5, 6, 7"""


def usage_sql(src):
    return f"""
    SELECT mbr AS {C['mbr']}, ncpdp AS {C['ncpdp']}, spec AS specialty, mail,
           mode(ptype) AS {C['ptype']}, mode(pclass) AS {C['pclass']},
           count(*)::BIGINT AS fills, count(DISTINCT filled)::BIGINT AS visits, sum(paid)::BIGINT AS fills_paid,
           sum({MONEY[0]}) AS ingred, sum({MONEY[1]}) AS fee
    FROM {src} GROUP BY 1, 2, 3, 4"""


def members_sql(src, zip_join=""):
    return f"""
    WITH m AS (
        SELECT mbr, mode(account) AS account, mode(grp) AS grp, median(lat) AS lat, median(lon) AS lon,
               count(*)::BIGINT AS fills_all, sum({IS_NORMAL})::BIGINT AS fills_normal, sum(paid)::BIGINT AS fills_paid,
               sum({MONEY[0]}) AS ingred_all, sum({MONEY[1]}) AS fee_all,
               sum(CASE WHEN {IS_NORMAL} = 1 THEN {MONEY[0]} END) AS ingred_normal,
               sum(CASE WHEN {IS_NORMAL} = 1 THEN {MONEY[1]} END) AS fee_normal,
               min(filled) AS first_fill, max(filled) AS last_fill
        FROM {src} GROUP BY mbr
    ),
    u AS (
        SELECT mbr, count(DISTINCT ncpdp) AS pharmacies_used,
               arg_max(ncpdp, n) AS main_pharmacy,
               arg_max(ncpdp, n) FILTER (WHERE is_normal = 1) AS main_normal_pharmacy
        FROM (SELECT mbr, ncpdp, count(*) AS n, max({IS_NORMAL}) AS is_normal FROM {src} GROUP BY 1, 2) GROUP BY 1
    ),
    d AS (
        SELECT mbr, count(DISTINCT kind) AS kinds_used, string_agg(DISTINCT kind, ' ; ') AS kinds,
               count(DISTINCT (gpi, gb)) AS drugs, count(DISTINCT CASE WHEN maint = 1 THEN (gpi, gb) END) AS maint_drugs,
               max(spec) AS has_specialty, max(mail) AS has_mail
        FROM (SELECT *, CASE WHEN mail = 1 THEN 'Mail' ELSE ptype || ' | ' || pclass END AS kind FROM {src}) GROUP BY 1
    )
    SELECT m.mbr AS {C['mbr']}, m.account AS {C['account']}, m.grp AS {C['group']},
           m.lat AS MBR_LAT, m.lon AS MBR_LON{zip_join and ', z.MBR_ZIP' or ', NULL::VARCHAR AS MBR_ZIP'},
           m.fills_all, m.fills_normal, m.fills_paid, m.ingred_all, m.fee_all, m.ingred_normal, m.fee_normal,
           m.first_fill, m.last_fill,
           u.pharmacies_used, u.main_pharmacy, u.main_normal_pharmacy,
           d.kinds_used, d.kinds, d.drugs, d.maint_drugs, d.has_specialty, d.has_mail,
           m.fills_normal > 0 AS has_normal
    FROM m LEFT JOIN u USING (mbr) LEFT JOIN d USING (mbr){zip_join}"""


# ---- member ZIP codes --------------------------------------------------------
ZIP_JOIN = ""
if ZIP_FILE.exists():
    z = pd.read_excel(ZIP_FILE, dtype=str)
    z.columns = [c.strip() for c in z.columns]
    id_col = next((c for c in z.columns if c.upper() == "MBR_ID"), z.columns[0])
    zip_col = next((c for c in z.columns if "zip" in c.lower()), z.columns[1])
    z = z[[id_col, zip_col]].rename(columns={id_col: "mbr", zip_col: "MBR_ZIP"})
    z["mbr"] = z.mbr.astype(str).str.strip()
    z["MBR_ZIP"] = z.MBR_ZIP.astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.replace(r"\D", "", regex=True)  # 98003-1234 -> 980031234
    z = z[z.MBR_ZIP.str.match(r"^\d{3,9}$")]                                    # drops null, blank, text
    z["MBR_ZIP"] = z.MBR_ZIP.map(lambda s: s.zfill(5) if len(s) <= 5 else s.zfill(9)[:5])   # 2333 -> 02333, ZIP+4 -> first five
    z = z.drop_duplicates("mbr", keep="last")
    con.register("zips", z)
    q("CREATE OR REPLACE TABLE member_zip AS SELECT * FROM zips")
    ZIP_JOIN = " LEFT JOIN member_zip z USING (mbr)"
    print(f"member ZIP codes: {len(z):,} members with a usable ZIP from {ZIP_FILE.name}")
else:
    print(f"no {ZIP_FILE.name}: members get no ZIP (the model falls back to the pharmacy density rule)")

# ---- stage: one file at a time ----------------------------------------------
if table_exists("rows_stage") and "paid" not in [r[0] for r in q("DESCRIBE rows_stage").fetchall()]:
    sys.exit(f"{OUT / 'member_stage.duckdb'} was made by an older version of this script. Delete the member_tables folder and run again.")
if table_exists("demand_stage") and not table_exists("rows_stage"):
    sys.exit(f"{OUT / 'member_stage.duckdb'} was made by an older version of this script. Delete the member_tables folder and run again.")
q("CREATE TABLE IF NOT EXISTS done_files (file VARCHAR)")
done = {r[0] for r in q("SELECT file FROM done_files").fetchall()}
if table_exists("rows_stage"):
    q("DELETE FROM rows_stage WHERE file NOT IN (SELECT file FROM done_files)")   # leftovers of a file that was cut off mid way

t_all = time.time()
for i, f in enumerate(files, 1):
    if f.name in done:
        print(f"[{i}/{len(files)}] {f.name}  already done, skipped")
        continue
    t = time.time()
    q(f"CREATE OR REPLACE TEMP TABLE cur_raw AS {ROWS_SQL.format(path=f.as_posix(), file=f.name)}")
    q(f"CREATE OR REPLACE TEMP TABLE cur_rows AS {DEDUPE.format(src='cur_raw')}")     # repeats within the file removed
    n_raw, = q("SELECT count(*) FROM cur_raw").fetchone()
    con.begin()                                                                  # all or nothing
    if table_exists("rows_stage"):
        q("INSERT INTO rows_stage SELECT * FROM cur_rows")
    else:
        q("CREATE TABLE rows_stage AS SELECT * FROM cur_rows")
    q(f"INSERT INTO done_files VALUES ('{f.name}')")
    con.commit()
    if PER_FILE:
        export(members_sql("cur_rows", ZIP_JOIN) + " ORDER BY fills_all DESC", PER / f"{f.stem}_members", 1_000_000)
        export(demand_sql("cur_rows") + f" ORDER BY {C['mbr']}, fills DESC", PER / f"{f.stem}_demand", None)
        xlsx(q(demand_sql("cur_rows") + f" ORDER BY {C['mbr']}, fills DESC LIMIT {XLSX_ROWS}").df(), PER / f"{f.stem}_demand_check.xlsx")
        export(usage_sql("cur_rows") + f" ORDER BY {C['mbr']}, fills DESC", PER / f"{f.stem}_member_pharmacy", 1_000_000)
    n_m, n_c, n_paid, n_n = q("SELECT count(DISTINCT mbr), count(*), sum(paid), sum(" + IS_NORMAL + ") FROM cur_rows").fetchone()
    q("DROP TABLE cur_raw"); q("DROP TABLE cur_rows")
    print(f"[{i}/{len(files)}] {f.name}  {time.time()-t:,.0f}s  {n_m or 0:,} {CLIENT} members  {n_raw or 0:,} rows -> {n_c or 0:,} fills "
          f"({n_paid or 0:,} paid by {CLIENT}, {n_n or 0:,} at normal pharmacies)" + ("   <- no rows for this client, check CLIENT" if not n_m else ""))

print(f"all files done in {(time.time()-t_all)/60:,.1f} min, combining ...")

# ---- combine: repeats across files removed, then the four tables -------------
q(f"CREATE OR REPLACE TABLE fills AS {DEDUPE.format(src='rows_stage')}")
n_stage, = q("SELECT count(*) FROM rows_stage").fetchone(); n_fills, = q("SELECT count(*) FROM fills").fetchone()
print(f"fills: {n_stage:,} after the within file dedupe, {n_fills:,} after the across files dedupe")

q(f"CREATE OR REPLACE TABLE demand AS {demand_sql('fills')}")
q(f"CREATE OR REPLACE TABLE member_pharmacy AS {usage_sql('fills')}")
q(f"CREATE OR REPLACE TABLE members AS {members_sql('fills', ZIP_JOIN)}")
q(f"""CREATE OR REPLACE TABLE maint_fills AS
      SELECT mbr AS {C['mbr']}, ncpdp AS {C['ncpdp']}, gpi AS {C['gpi']}, gb AS {C['gb']}, filled AS {C['filled']}, days AS {C['days']},
             spec AS specialty, mail, paid
      FROM fills WHERE maint = 1""")

# ---- write -------------------------------------------------------------------
export("SELECT * FROM members ORDER BY fills_all DESC", OUT / "fedex_members", 1_000_000)
export(f"SELECT * FROM demand ORDER BY {C['mbr']}, fills DESC", OUT / "fedex_demand", None)
xlsx(q(f"SELECT * FROM demand ORDER BY {C['mbr']}, fills DESC LIMIT {XLSX_ROWS}").df(), OUT / "fedex_demand_check.xlsx")
export(f"SELECT * FROM member_pharmacy ORDER BY {C['mbr']}, fills DESC", OUT / "fedex_member_pharmacy", 1_000_000)
export(f"SELECT * FROM maint_fills ORDER BY {C['mbr']}, {C['gpi']}, {C['filled']}", OUT / "fedex_maint_fills", None)

n_m, n_norm, n_zip, n_spec, n_mail, paid_total = q("""SELECT count(*), sum(has_normal::INT), sum((MBR_ZIP IS NOT NULL)::INT), sum(has_specialty),
                                                             sum(has_mail), sum(ingred_all + fee_all) FROM members""").fetchone()
n_d, n_u, n_mf = (q("SELECT count(*) FROM demand").fetchone()[0], q("SELECT count(*) FROM member_pharmacy").fetchone()[0],
                  q("SELECT count(*) FROM maint_fills").fetchone()[0])
n_f, n_paid = q("SELECT count(*), sum(paid) FROM fills").fetchone()
kinds = q(f"SELECT CASE WHEN mail = 1 THEN 'Mail' ELSE {C['ptype']} || ' | ' || {C['pclass']} END AS kind, sum(fills) FROM demand GROUP BY 1 ORDER BY 2 DESC").fetchall()
n_norm, n_zip, n_spec, n_mail, paid_total, n_paid = (int(n_norm or 0), int(n_zip or 0), int(n_spec or 0), int(n_mail or 0), float(paid_total or 0), int(n_paid or 0))
print(f"""
members                        {n_m:>12,}   with a normal pharmacy: {n_norm:,}   with ZIP: {n_zip:,}   with specialty fills: {n_spec:,}   with mail fills: {n_mail:,}
fills                          {n_f:>12,}   paid by {CLIENT}: {n_paid:,}   money paid by {CLIENT}: {paid_total:,.0f}
member x kind x drug           {n_d:>12,}
member x pharmacy              {n_u:>12,}
maintenance fills              {n_mf:>12,}
fills by kind of pharmacy: """ + ", ".join(f"{k} {int(n):,}" for k, n in kinds) + f"""
written to {OUT}
""")
con.close()
