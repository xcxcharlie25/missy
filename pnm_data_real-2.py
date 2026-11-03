import duckdb
con = duckdb.connect()
con.execute("SET memory_limit='16GB'")
FEDEX = """FROM read_parquet('*2025_*.parquet', filename = true)
           WHERE CARRIER_ID = 'FEDEX' AND m_nf_app_due_amt > 0
             AND coalesce(trim(CAST(a_nf_clt_prc_typ AS VARCHAR)), '') NOT IN ('', 'ZERO')"""

print("1. total paid, specialty part, fills, members")
print(con.execute(f"""SELECT round(sum(m_nf_clt_ingred_cost_paid + m_nf_clt_dispensing_fee)) AS paid,
                             round(sum(CASE WHEN a_nf_specialty_flag = 'Y' THEN m_nf_clt_ingred_cost_paid + m_nf_clt_dispensing_fee END)) AS specialty_part,
                             count(*) AS fills, count(DISTINCT MBR_ID) AS members {FEDEX}""").df().to_string(index=False))

print("\n2. duplicates: rows should equal claims")
print(con.execute(f"SELECT count(*) AS rows_, count(DISTINCT CLAIM_ID) AS claims, min(FILLED_DT) AS first, max(FILLED_DT) AS last {FEDEX}").df().to_string(index=False))

print("\n3. date range per file: ranges should not overlap")
print(con.execute(f"SELECT filename, min(FILLED_DT) AS first, max(FILLED_DT) AS last, count(*) AS fills {FEDEX} GROUP BY 1 ORDER BY 2").df().to_string(index=False))
