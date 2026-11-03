"""
check_duplicates.py   Repeated fills among FedEx paid rows: within each file and across all files.
"""
import duckdb
con = duckdb.connect()
con.execute("SET memory_limit='16GB'")
K = "MBR_ID, PHR_NCPDP_ID, a_nt_gpi, a_irpg_drugtype, FILLED_DT, a_nf_drg_qty, DAYS_SUPPLY, m_nf_clt_ingred_cost_paid, m_nf_clt_dispensing_fee"
W = """WHERE CARRIER_ID = 'FEDEX' AND m_nf_app_due_amt > 0
         AND coalesce(trim(CAST(a_nf_clt_prc_typ AS VARCHAR)), '') NOT IN ('', 'ZERO')"""

per_file = con.execute(f"""SELECT filename, count(*) AS rows_, count(DISTINCT ({K})) AS distinct_fills
                           FROM read_parquet('*2025_*.parquet', filename = true) {W} GROUP BY 1 ORDER BY 1""").df()
per_file["repeats_within_file"] = per_file.rows_ - per_file.distinct_fills
print("per file"); print(per_file.to_string(index=False))

across = con.execute(f"SELECT count(*) FROM (SELECT DISTINCT {K} FROM read_parquet('*2025_*.parquet') {W})").fetchone()[0]
rows, within = int(per_file.rows_.sum()), int(per_file.distinct_fills.sum())
print(f"""
all files
  rows                         {rows:>12,}
  distinct within each file    {within:>12,}   repeats inside a file: {rows - within:,}
  distinct across all files    {across:>12,}   repeats that cross files: {within - across:,}
""")
