"""
book_pharmacy_extract.py

Pharmacy level data and pharmacy x drug price data from the whole book of business.

Reads the 2025 parquet files ONE AT A TIME with DuckDB, keeps only summed results
per pharmacy and per pharmacy x drug, and combines them at the end. The raw claims
never sit in memory. Finished files are recorded, so if the run stops you just run
it again and it continues with the files that are left.

Outputs (in OUT folder):
    per_file/<file>_pharmacies.parquet / .xlsx   that file's pharmacies, all rows
    per_file/<file>_price.parquet                that file's pharmacy x GPI x G/B rows
    per_file/<file>_price_check.xlsx             first XLSX_ROWS of it, client rows first
    book_pharmacies.parquet / .xlsx              all files combined, one row per pharmacy
    book_price_table.parquet                     all files combined, one row per pharmacy x GPI x G/B
    book_price_table_check.xlsx                  first XLSX_ROWS of it, client rows first
    book_chain_price_table.parquet               one row per chain x GPI x G/B (fallback)
    book_stage.duckdb                            working database, delete when done

Run:  python book_pharmacy_extract.py
"""
import re
import sys
import time
from pathlib import Path

import duckdb

# ------------------------------- CONFIG --------------------------------------
FOLDER    = Path.home() / "Downloads"        # where the parquet files are
PATTERN   = "*2025_*.parquet"                 # jan2025_first.parquet, feb2025_third.parquet ...
OUT       = FOLDER / "book_tables"            # outputs go here
CLIENT    = "FEDEX"                           # CARRIER_ID of the client
MEMORY    = "12GB"                            # a few GB below the RAM of the machine
THREADS   = 6
PHR_LAT   = None                              # None = detect automatically, or the column name
PHR_LON   = None
XLSX_ROWS = 200_000                           # rows of a price table written to xlsx for checking (Excel max 1,048,576)
PER_FILE  = True                              # also save each parquet file's own pharmacy and price tables
CAPACITY_SHARE = 0.90                         # 2025 fills = 90% of what a pharmacy can take

C = dict(  # column names in the parquet files; change here if yours differ
    ncpdp="PHR_NCPDP_ID", npi="PHR_NPI_NO", name="PHR_NM", affl="PHR_AFFL_NM", chain="a_nf_pharmacy",
    ptype="PHR_DISPNSR_TYPE_DESC", pclass="PHR_DISPNSR_CLASS_DESC",
    state="PHR_ST_CD", msa="LOC_MSA_CD", zip="PHR_ZIP_CD", city="PHR_CITY_NM", addr="PHR_ADDR1_TXT",
    carrier="CARRIER_ID", gpi="a_nt_gpi", gb="a_irpg_drugtype", qty="a_nf_drg_qty",
    ingred="m_nf_clt_ingred_cost_paid", fee="m_nf_clt_dispensing_fee",
    due="m_nf_app_due_amt", prctyp="a_nf_clt_prc_typ", filled="FILLED_DT",
)
# -----------------------------------------------------------------------------

if len(sys.argv) > 1:                         # optional: python book_pharmacy_extract.py <folder>
    FOLDER = Path(sys.argv[1])
    OUT = FOLDER / "book_tables"

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "duck_tmp").mkdir(exist_ok=True)
PER = OUT / "per_file"
PER.mkdir(exist_ok=True)

files = sorted(FOLDER.glob(PATTERN))
if not files:
    sys.exit(f"No files matching {PATTERN} in {FOLDER}")
print(f"{len(files)} files in {FOLDER}")

con = duckdb.connect(str(OUT / "book_stage.duckdb"))
con.execute(f"SET memory_limit='{MEMORY}'")
con.execute(f"SET threads={THREADS}")
con.execute(f"SET temp_directory='{(OUT / 'duck_tmp').as_posix()}'")
con.execute("SET preserve_insertion_order=false")


def q(sql):
    return con.execute(sql)


def columns_of(path):
    return [r[0] for r in q(f"DESCRIBE SELECT * FROM read_parquet('{path.as_posix()}')").fetchall()]


# ---- check columns on the first file before spending hours -------------------
cols = columns_of(files[0])
missing = [v for v in C.values() if v not in cols]
if missing:
    sys.exit(f"Columns not found in {files[0].name}: {missing}\nColumns present: {cols}")

if PHR_LAT is None or PHR_LON is None:
    lat = [c for c in cols if re.search(r"(PHR|PHARM).*LAT", c, re.I)]
    lon = [c for c in cols if re.search(r"(PHR|PHARM).*(LON|LNG)", c, re.I)]
    PHR_LAT = PHR_LAT or (lat[0] if lat else None)
    PHR_LON = PHR_LON or (lon[0] if lon else None)
print("pharmacy coordinates:", PHR_LAT, PHR_LON)
lat_expr = f"median({PHR_LAT})" if PHR_LAT else "NULL::DOUBLE"
lon_expr = f"median({PHR_LON})" if PHR_LON else "NULL::DOUBLE"

# paid claims only, with the client flag
PAID = f"""
    FROM read_parquet('{{path}}')
    WHERE {C['due']} > 0
      AND coalesce(trim(CAST({C['prctyp']} AS VARCHAR)), '') NOT IN ('', 'ZERO')
"""
IS_CLIENT = f"CASE WHEN CAST({C['carrier']} AS VARCHAR) = '{CLIENT}' THEN 1 ELSE 0 END"

PHARM_SQL = f"""
    SELECT '{{file}}' AS file,
           {C['ncpdp']}  AS ncpdp,
           mode({C['npi']})    AS npi,   mode({C['name']})  AS name,  mode({C['affl']})  AS affl,
           mode({C['chain']})  AS chain, mode({C['ptype']}) AS ptype, mode({C['pclass']}) AS pclass,
           mode({C['state']})  AS state, mode({C['msa']})   AS msa,   mode({C['zip']})   AS zip,
           mode({C['city']})   AS city,  mode({C['addr']})  AS addr,
           {lat_expr} AS lat, {lon_expr} AS lon,
           count(*)                                        AS fills_all,
           sum({IS_CLIENT})                                AS fills_client,
           sum({C['ingred']})                              AS ingred_all,
           sum({C['fee']})                                 AS fee_all,
           sum(CASE WHEN {IS_CLIENT} = 1 THEN {C['ingred']} END) AS ingred_client,
           sum(CASE WHEN {IS_CLIENT} = 1 THEN {C['fee']} END)    AS fee_client,
           min({C['filled']}) AS first_fill, max({C['filled']}) AS last_fill
    {PAID}
    GROUP BY {C['ncpdp']}
"""

PRICE_SQL = f"""
    SELECT '{{file}}' AS file,
           {C['ncpdp']} AS ncpdp, {C['gpi']} AS gpi, {C['gb']} AS gb,
           sum(CASE WHEN {IS_CLIENT} = 1 THEN {C['ingred']} END) AS ingred_client,
           sum(CASE WHEN {IS_CLIENT} = 1 THEN {C['fee']} END)    AS fee_client,
           sum(CASE WHEN {IS_CLIENT} = 1 THEN {C['qty']} END)    AS qty_client,
           sum({IS_CLIENT})                                      AS claims_client,
           sum({C['ingred']}) AS ingred_all,
           sum({C['fee']})    AS fee_all,
           sum({C['qty']})    AS qty_all,
           count(*)           AS claims_all
    {PAID}
    GROUP BY {C['ncpdp']}, {C['gpi']}, {C['gb']}
"""


def table_exists(name):
    return q(f"SELECT count(*) FROM information_schema.tables WHERE table_name='{name}'").fetchone()[0] > 0


def add(table, sql):
    if table_exists(table):
        q(f"INSERT INTO {table} {sql}")
    else:
        q(f"CREATE TABLE {table} AS {sql}")


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
    add("pharm_stage", PHARM_SQL.format(**fill))
    add("price_stage", PRICE_SQL.format(**fill))
    q(f"INSERT INTO done_files VALUES ('{f.name}')")
    con.commit()
    if PER_FILE:                                   # this file's own tables, small enough to open
        where = f"WHERE file = '{f.name}'"
        ph_sql = f"""SELECT ncpdp AS PHR_NCPDP_ID, npi AS PHR_NPI_NO, name AS PHR_NM, affl AS PHR_AFFL_NM,
                            chain AS a_nf_pharmacy, ptype AS PHR_DISPNSR_TYPE_DESC, pclass AS PHR_DISPNSR_CLASS_DESC,
                            state AS PHR_ST_CD, msa AS LOC_MSA_CD, zip AS PHR_ZIP_CD, city AS PHR_CITY_NM,
                            addr AS PHR_ADDR1_TXT, lat AS PHR_LAT, lon AS PHR_LON,
                            fills_all::BIGINT AS fills_all, fills_client::BIGINT AS fills_client,
                            ingred_all, fee_all, ingred_client, fee_client, first_fill, last_fill
                     FROM pharm_stage {where} ORDER BY fills_client DESC, fills_all DESC"""
        pr_sql = f"""SELECT ncpdp AS PHR_NCPDP_ID, gpi AS a_nt_gpi, gb AS a_irpg_drugtype,
                            ingred_client, fee_client, qty_client, claims_client::BIGINT AS claims_client,
                            ingred_all, fee_all, qty_all, claims_all::BIGINT AS claims_all,
                            ingred_client / nullif(qty_client, 0) AS unit_ingred_client,
                            ingred_all    / nullif(qty_all, 0)    AS unit_ingred_all
                     FROM price_stage {where} ORDER BY claims_client DESC, claims_all DESC"""
        q(f"COPY ({ph_sql}) TO '{(PER / f'{f.stem}_pharmacies.parquet').as_posix()}' (FORMAT PARQUET)")
        q(f"COPY ({pr_sql}) TO '{(PER / f'{f.stem}_price.parquet').as_posix()}' (FORMAT PARQUET)")
        q(ph_sql).df().to_excel(PER / f"{f.stem}_pharmacies.xlsx", index=False)
        q(f"{pr_sql} LIMIT {XLSX_ROWS}").df().to_excel(PER / f"{f.stem}_price_check.xlsx", index=False)
    n_ph, n_cl, n_fx = q(f"""SELECT count(*), sum(fills_all), sum(fills_client)
                             FROM pharm_stage WHERE file='{f.name}'""").fetchone()
    print(f"[{i}/{len(files)}] {f.name}  {time.time()-t:,.0f}s  "
          f"{n_ph:,} pharmacies  {n_cl:,} paid claims  {n_fx:,} {CLIENT} claims")

print(f"all files done in {(time.time()-t_all)/60:,.1f} min, combining ...")

# ---- combine: pharmacy table -------------------------------------------------
q(f"""
CREATE OR REPLACE TABLE pharmacies AS
SELECT ncpdp        AS PHR_NCPDP_ID,
       mode(npi)    AS PHR_NPI_NO,
       mode(name)   AS PHR_NM,
       mode(affl)   AS PHR_AFFL_NM,
       mode(chain)  AS a_nf_pharmacy,
       mode(ptype)  AS PHR_DISPNSR_TYPE_DESC,
       mode(pclass) AS PHR_DISPNSR_CLASS_DESC,
       mode(state)  AS PHR_ST_CD,
       mode(msa)    AS LOC_MSA_CD,
       mode(zip)    AS PHR_ZIP_CD,
       mode(city)   AS PHR_CITY_NM,
       mode(addr)   AS PHR_ADDR1_TXT,
       median(lat)  AS PHR_LAT,
       median(lon)  AS PHR_LON,
       sum(fills_all)::BIGINT     AS fills_all,
       sum(fills_client)::BIGINT  AS fills_client,
       sum(ingred_all)    AS ingred_all,
       sum(fee_all)       AS fee_all,
       sum(ingred_client) AS ingred_client,
       sum(fee_client)    AS fee_client,
       min(first_fill)    AS first_fill,
       max(last_fill)     AS last_fill,
       sum(fills_client) > 0                                   AS used_by_client,
       ceil(sum(fills_all) / {CAPACITY_SHARE})::BIGINT         AS capacity_fills,
       mode(ptype) = 'Community or Retail Pharmacy'            AS phase1_retail
FROM pharm_stage
GROUP BY ncpdp
""")

# ---- combine: pharmacy x drug price table ------------------------------------
q("""
CREATE OR REPLACE TABLE price_table AS
WITH s AS (
    SELECT ncpdp, gpi, gb,
           sum(ingred_client) AS ingred_client, sum(fee_client) AS fee_client,
           sum(qty_client)    AS qty_client,    sum(claims_client)::BIGINT AS claims_client,
           sum(ingred_all)    AS ingred_all,    sum(fee_all)    AS fee_all,
           sum(qty_all)       AS qty_all,       sum(claims_all)::BIGINT AS claims_all
    FROM price_stage
    GROUP BY ncpdp, gpi, gb
)
SELECT ncpdp AS PHR_NCPDP_ID, gpi AS a_nt_gpi, gb AS a_irpg_drugtype,
       ingred_client, fee_client, qty_client, claims_client,
       ingred_all, fee_all, qty_all, claims_all,
       CASE WHEN claims_client > 0 THEN ingred_client / nullif(qty_client, 0)
            ELSE ingred_all / nullif(qty_all, 0) END               AS unit_ingred,
       CASE WHEN claims_client > 0 THEN fee_client / claims_client
            ELSE fee_all / claims_all END                           AS fee_per_fill,
       CASE WHEN claims_client > 0 THEN 'client' ELSE 'all clients' END AS price_source
FROM s
""")

# ---- combine: chain x drug fallback ------------------------------------------
q("""
CREATE OR REPLACE TABLE chain_price_table AS
SELECT p.a_nf_pharmacy AS chain, s.a_nt_gpi, s.a_irpg_drugtype,
       sum(s.ingred_all) AS ingred_all, sum(s.fee_all) AS fee_all,
       sum(s.qty_all)    AS qty_all,    sum(s.claims_all)::BIGINT AS claims_all,
       sum(s.ingred_all) / nullif(sum(s.qty_all), 0)    AS unit_ingred,
       sum(s.fee_all)    / nullif(sum(s.claims_all), 0) AS fee_per_fill
FROM price_table s JOIN pharmacies p USING (PHR_NCPDP_ID)
GROUP BY 1, 2, 3
""")

# ---- write -------------------------------------------------------------------
for t in ("pharmacies", "price_table", "chain_price_table"):
    q(f"COPY {t} TO '{(OUT / f'book_{t}.parquet').as_posix()}' (FORMAT PARQUET)")

q("SELECT * FROM pharmacies ORDER BY fills_client DESC, fills_all DESC").df() \
    .to_excel(OUT / "book_pharmacies.xlsx", index=False)
q(f"SELECT * FROM price_table ORDER BY claims_client DESC, claims_all DESC LIMIT {XLSX_ROWS}").df() \
    .to_excel(OUT / "book_price_table_check.xlsx", index=False)

n_ph, n_fx, n_ret = q("SELECT count(*), sum(used_by_client::INT), sum(phase1_retail::INT) FROM pharmacies").fetchone()
n_pr, n_pr_fx = q("SELECT count(*), sum((price_source='client')::INT) FROM price_table").fetchone()
n_ch = q("SELECT count(*) FROM chain_price_table").fetchone()[0]
print(f"""
pharmacies            {n_ph:>12,}   used by {CLIENT}: {n_fx:,}   retail: {n_ret:,}
pharmacy x drug rows  {n_pr:>12,}   priced from {CLIENT} claims: {n_pr_fx:,}
chain x drug rows     {n_ch:>12,}
written to {OUT}
""")
con.close()
