"""
fedex_member_extract.py

Member side data for one client (CARRIER_ID = CLIENT), everything the model run needs.

Reads the 2025 parquet files ONE AT A TIME with DuckDB, keeps only the client's rows,
sums them per member, and combines at the end. Finished files are recorded, so if the
run stops you run it again and it continues.

Outputs (in OUT folder):
    per_file/<file>_members.parquet / .xlsx         that file's members
    per_file/<file>_demand.parquet / _check.xlsx    that file's member x class x GPI x G/B rows
    per_file/<file>_member_pharmacy.parquet / .xlsx that file's member x pharmacy rows
    per_file/<file>_maint_fills.parquet             that file's maintenance fills, one row per claim
    fedex_members.parquet / .xlsx                   one row per member: location, group, totals, main pharmacy
    fedex_demand.parquet / _check.xlsx              member x dispenser class x GPI x G/B: fills, qty, days, cost today
    fedex_member_pharmacy.parquet / .xlsx           member x pharmacy used today: fills, visits, cost
    fedex_maint_fills.parquet                       maintenance fills for the adherence (quality) score
    member_stage.duckdb                             working database, delete when done

Run:  python fedex_member_extract.py
"""
import re
import sys
import time
from pathlib import Path

import duckdb

# ------------------------------- CONFIG --------------------------------------
FOLDER    = Path(__file__).parent              # the folder this script is in (with the parquet files)
PATTERN   = "*2025_*.parquet"
OUT       = FOLDER / "member_tables"
CLIENT    = "FEDEX"                            # CARRIER_ID of the client
MEMORY    = "12GB"
THREADS   = 6
MBR_LAT   = None                               # None = detect automatically, or the column name
MBR_LON   = None
XLSX_ROWS = 200_000
PER_FILE  = True
RETAIL    = "Community or Retail Pharmacy"     # phase 1 pharmacy type

C = dict(  # column names in the parquet files; change here if yours differ
    mbr="MBR_ID", carrier="CARRIER_ID", account="ACCOUNT_ID", group="EMPLOYER_GROUP_ID",
    ncpdp="PHR_NCPDP_ID", ptype="PHR_DISPNSR_TYPE_DESC", pclass="PHR_DISPNSR_CLASS_DESC",
    gpi="a_nt_gpi", gb="a_irpg_drugtype", qty="a_nf_drg_qty", days="DAYS_SUPPLY",
    maint="a_nt_maintenance_flag", ingred="m_nf_clt_ingred_cost_paid", fee="m_nf_clt_dispensing_fee",
    due="m_nf_app_due_amt", prctyp="a_nf_clt_prc_typ", filled="FILLED_DT",
)
OPTIONAL = dict(  # member address columns if the file has them, else leave None
    mbr_zip=None, mbr_state=None, mbr_msa=None,
)
# -----------------------------------------------------------------------------

if len(sys.argv) > 1:
    FOLDER = Path(sys.argv[1])
    OUT = FOLDER / "member_tables"

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


# ---- check columns on the first file -----------------------------------------
cols = columns_of(files[0])
missing = [v for v in list(C.values()) + [v for v in OPTIONAL.values() if v] if v not in cols]
if missing:
    sys.exit(f"Columns not found in {files[0].name}: {missing}\nColumns present: {cols}")

if MBR_LAT is None or MBR_LON is None:
    lat = [c for c in cols if re.search(r"(MBR|MEM).*LAT", c, re.I) and not re.search(r"PH(R|ARM)", c, re.I)]
    lon = [c for c in cols if re.search(r"(MBR|MEM).*(LON|LNG)", c, re.I) and not re.search(r"PH(R|ARM)", c, re.I)]
    MBR_LAT = MBR_LAT or (lat[0] if lat else None)
    MBR_LON = MBR_LON or (lon[0] if lon else None)
print("member coordinates:", MBR_LAT, MBR_LON)
lat_expr = f"median({MBR_LAT})" if MBR_LAT else "NULL::DOUBLE"
lon_expr = f"median({MBR_LON})" if MBR_LON else "NULL::DOUBLE"
opt_stage = "".join(f", mode({v}) AS {k}" for k, v in OPTIONAL.items() if v)
opt_final = "".join(f", mode({k}) AS {v}" for k, v in OPTIONAL.items() if v)
opt_out   = "".join(f", {k} AS {v}" for k, v in OPTIONAL.items() if v)
opt_mbr   = "".join(f", m.{k} AS {v}" for k, v in OPTIONAL.items() if v)

# client's paid claims only
ROWS = f"""
    FROM read_parquet('{{path}}')
    WHERE CAST({C['carrier']} AS VARCHAR) = '{CLIENT}'
      AND {C['due']} > 0
      AND coalesce(trim(CAST({C['prctyp']} AS VARCHAR)), '') NOT IN ('', 'ZERO')
"""
IS_RETAIL = f"CASE WHEN {C['ptype']} = '{RETAIL}' THEN 1 ELSE 0 END"
IS_MAINT  = f"CASE WHEN upper(trim(CAST({C['maint']} AS VARCHAR))) IN ('Y', 'YES', '1', 'TRUE') THEN 1 ELSE 0 END"

MBR_SQL = f"""
    SELECT '{{file}}' AS file,
           {C['mbr']} AS mbr,
           mode({C['account']}) AS account, mode({C['group']}) AS grp,
           {lat_expr} AS lat, {lon_expr} AS lon{opt_stage},
           count(*)            AS fills_all,
           sum({IS_RETAIL})    AS fills_retail,
           sum({C['ingred']})  AS ingred_all,
           sum({C['fee']})     AS fee_all,
           sum(CASE WHEN {IS_RETAIL} = 1 THEN {C['ingred']} END) AS ingred_retail,
           sum(CASE WHEN {IS_RETAIL} = 1 THEN {C['fee']} END)    AS fee_retail,
           min({C['filled']}) AS first_fill, max({C['filled']}) AS last_fill
    {ROWS}
    GROUP BY {C['mbr']}
"""

DEMAND_SQL = f"""
    SELECT '{{file}}' AS file,
           {C['mbr']} AS mbr, {C['pclass']} AS pclass, {C['gpi']} AS gpi, {C['gb']} AS gb,
           count(*)           AS fills,
           sum({C['qty']})    AS qty,
           sum({C['days']})   AS days,
           sum({C['ingred']}) AS ingred,
           sum({C['fee']})    AS fee,
           max({IS_MAINT})    AS maint
    {ROWS}
      AND {C['ptype']} = '{RETAIL}'
    GROUP BY {C['mbr']}, {C['pclass']}, {C['gpi']}, {C['gb']}
"""

USAGE_SQL = f"""
    SELECT '{{file}}' AS file,
           {C['mbr']} AS mbr, {C['ncpdp']} AS ncpdp,
           mode({C['ptype']})  AS ptype,
           mode({C['pclass']}) AS pclass,
           count(*)                    AS fills,
           count(DISTINCT {C['filled']}) AS visits,
           sum({C['ingred']}) AS ingred,
           sum({C['fee']})    AS fee
    {ROWS}
    GROUP BY {C['mbr']}, {C['ncpdp']}
"""

MAINT_SQL = f"""
    SELECT '{{file}}' AS file,
           {C['mbr']} AS mbr, {C['ncpdp']} AS ncpdp, {C['gpi']} AS gpi, {C['gb']} AS gb,
           {C['filled']} AS filled, {C['days']} AS days
    {ROWS}
      AND {C['ptype']} = '{RETAIL}'
      AND {IS_MAINT} = 1
"""


def table_exists(name):
    return q(f"SELECT count(*) FROM information_schema.tables WHERE table_name='{name}'").fetchone()[0] > 0


def add(table, sql):
    if table_exists(table):
        q(f"INSERT INTO {table} {sql}")
    else:
        q(f"CREATE TABLE {table} AS {sql}")


def export(sql, stem, xlsx_rows=None):
    """parquet with everything, xlsx with everything or the first xlsx_rows rows"""
    q(f"COPY ({sql}) TO '{(stem.with_suffix('.parquet')).as_posix()}' (FORMAT PARQUET)")
    if xlsx_rows is not None:
        q(f"{sql} LIMIT {xlsx_rows}").df().to_excel(stem.with_suffix(".xlsx"), index=False)


q("CREATE TABLE IF NOT EXISTS done_files (file VARCHAR)")
done = {r[0] for r in q("SELECT file FROM done_files").fetchall()}

# ---- one file at a time ------------------------------------------------------
t_all = time.time()
for i, f in enumerate(files, 1):
    if f.name in done:
        print(f"[{i}/{len(files)}] {f.name}  already done, skipped")
        continue
    t = time.time()
    fill = dict(path=f.as_posix(), file=f.name)
    add("mbr_stage", MBR_SQL.format(**fill))
    add("demand_stage", DEMAND_SQL.format(**fill))
    add("usage_stage", USAGE_SQL.format(**fill))
    add("maint_stage", MAINT_SQL.format(**fill))
    q(f"INSERT INTO done_files VALUES ('{f.name}')")
    con.commit()
    if PER_FILE:
        w = f"WHERE file = '{f.name}'"
        export(f"""SELECT mbr AS {C['mbr']}, account AS {C['account']}, grp AS {C['group']},
                          lat AS MBR_LAT, lon AS MBR_LON{opt_out},
                          fills_all::BIGINT AS fills_all, fills_retail::BIGINT AS fills_retail,
                          ingred_all, fee_all, ingred_retail, fee_retail, first_fill, last_fill
                   FROM mbr_stage {w} ORDER BY fills_all DESC""",
               PER / f"{f.stem}_members", 1_000_000)
        export(f"""SELECT mbr AS {C['mbr']}, pclass AS {C['pclass']}, gpi AS {C['gpi']}, gb AS {C['gb']},
                          fills::BIGINT AS fills, qty, days, ingred, fee, maint
                   FROM demand_stage {w} ORDER BY mbr, fills DESC""",
               PER / f"{f.stem}_demand", None)
        q(f"""SELECT mbr AS {C['mbr']}, pclass AS {C['pclass']}, gpi AS {C['gpi']}, gb AS {C['gb']},
                     fills::BIGINT AS fills, qty, days, ingred, fee, maint
              FROM demand_stage {w} ORDER BY mbr, fills DESC LIMIT {XLSX_ROWS}""").df() \
            .to_excel(PER / f"{f.stem}_demand_check.xlsx", index=False)
        export(f"""SELECT mbr AS {C['mbr']}, ncpdp AS {C['ncpdp']}, ptype AS {C['ptype']}, pclass AS {C['pclass']},
                          fills::BIGINT AS fills, visits::BIGINT AS visits, ingred, fee
                   FROM usage_stage {w} ORDER BY mbr, fills DESC""",
               PER / f"{f.stem}_member_pharmacy", 1_000_000)
        export(f"""SELECT mbr AS {C['mbr']}, ncpdp AS {C['ncpdp']}, gpi AS {C['gpi']}, gb AS {C['gb']},
                          filled AS {C['filled']}, days AS {C['days']}
                   FROM maint_stage {w} ORDER BY mbr, gpi, filled""",
               PER / f"{f.stem}_maint_fills", None)
    n_m, n_c, n_r = q(f"SELECT count(*), sum(fills_all), sum(fills_retail) FROM mbr_stage WHERE file='{f.name}'").fetchone()
    print(f"[{i}/{len(files)}] {f.name}  {time.time()-t:,.0f}s  "
          f"{n_m:,} {CLIENT} members  {n_c:,} paid claims  {n_r:,} retail")

print(f"all files done in {(time.time()-t_all)/60:,.1f} min, combining ...")

# ---- combine -----------------------------------------------------------------
q(f"""
CREATE OR REPLACE TABLE member_pharmacy AS
SELECT mbr AS {C['mbr']}, ncpdp AS {C['ncpdp']},
       mode(ptype) AS {C['ptype']}, mode(pclass) AS {C['pclass']},
       sum(fills)::BIGINT AS fills, sum(visits)::BIGINT AS visits,
       sum(ingred) AS ingred, sum(fee) AS fee
FROM usage_stage
GROUP BY mbr, ncpdp
""")

q(f"""
CREATE OR REPLACE TABLE demand AS
SELECT mbr AS {C['mbr']}, pclass AS {C['pclass']}, gpi AS {C['gpi']}, gb AS {C['gb']},
       sum(fills)::BIGINT AS fills, sum(qty) AS qty, sum(days) AS days,
       sum(ingred) AS ingred, sum(fee) AS fee, max(maint) AS maint
FROM demand_stage
GROUP BY mbr, pclass, gpi, gb
""")

q(f"""
CREATE OR REPLACE TABLE members AS
WITH m AS (
    SELECT mbr, mode(account) AS account, mode(grp) AS grp,
           median(lat) AS lat, median(lon) AS lon{opt_final},
           sum(fills_all)::BIGINT AS fills_all, sum(fills_retail)::BIGINT AS fills_retail,
           sum(ingred_all) AS ingred_all, sum(fee_all) AS fee_all,
           sum(ingred_retail) AS ingred_retail, sum(fee_retail) AS fee_retail,
           min(first_fill) AS first_fill, max(last_fill) AS last_fill
    FROM mbr_stage GROUP BY mbr
),
u AS (
    SELECT {C['mbr']} AS mbr,
           count(*) AS pharmacies_used,
           count(*) FILTER (WHERE {C['ptype']} = '{RETAIL}') AS retail_pharmacies_used,
           arg_max({C['ncpdp']}, fills) FILTER (WHERE {C['ptype']} = '{RETAIL}') AS main_retail_pharmacy
    FROM member_pharmacy GROUP BY 1
),
d AS (
    SELECT {C['mbr']} AS mbr,
           count(DISTINCT {C['pclass']}) AS retail_classes_used,
           string_agg(DISTINCT {C['pclass']}, ' | ') AS retail_classes,
           count(*) AS retail_drugs,
           sum(maint)::BIGINT AS maint_drugs
    FROM demand GROUP BY 1
)
SELECT m.mbr AS {C['mbr']}, m.account AS {C['account']}, m.grp AS {C['group']},
       m.lat AS MBR_LAT, m.lon AS MBR_LON{opt_mbr},
       m.fills_all, m.fills_retail, m.ingred_all, m.fee_all, m.ingred_retail, m.fee_retail,
       m.first_fill, m.last_fill,
       coalesce(u.pharmacies_used, 0) AS pharmacies_used,
       coalesce(u.retail_pharmacies_used, 0) AS retail_pharmacies_used,
       u.main_retail_pharmacy,
       coalesce(d.retail_classes_used, 0) AS retail_classes_used,
       d.retail_classes,
       coalesce(d.retail_drugs, 0) AS retail_drugs,
       coalesce(d.maint_drugs, 0) AS maint_drugs,
       m.fills_retail > 0 AS phase1_member
FROM m LEFT JOIN u USING (mbr) LEFT JOIN d USING (mbr)
""")

# ---- write -------------------------------------------------------------------
export("SELECT * FROM members ORDER BY fills_retail DESC", OUT / "fedex_members", 1_000_000)
export(f"SELECT * FROM demand ORDER BY {C['mbr']}, fills DESC", OUT / "fedex_demand", None)
q(f"SELECT * FROM demand ORDER BY {C['mbr']}, fills DESC LIMIT {XLSX_ROWS}").df() \
    .to_excel(OUT / "fedex_demand_check.xlsx", index=False)
export(f"SELECT * FROM member_pharmacy ORDER BY {C['mbr']}, fills DESC", OUT / "fedex_member_pharmacy", 1_000_000)
export(f"""SELECT mbr AS {C['mbr']}, ncpdp AS {C['ncpdp']}, gpi AS {C['gpi']}, gb AS {C['gb']},
                  filled AS {C['filled']}, days AS {C['days']}
           FROM maint_stage ORDER BY mbr, gpi, filled""", OUT / "fedex_maint_fills", None)

n_m, n_p1, n_multi = q("""SELECT count(*), sum(phase1_member::INT), sum((retail_classes_used > 1)::INT)
                          FROM members""").fetchone()
n_d, n_u, n_mf = (q("SELECT count(*) FROM demand").fetchone()[0],
                  q("SELECT count(*) FROM member_pharmacy").fetchone()[0],
                  q("SELECT count(*) FROM maint_stage").fetchone()[0])
print(f"""
members                 {n_m:>12,}   with retail fills: {n_p1:,}   using more than one retail class: {n_multi:,}
member x class x drug   {n_d:>12,}
member x pharmacy       {n_u:>12,}
maintenance fills       {n_mf:>12,}
written to {OUT}
""")
con.close()
