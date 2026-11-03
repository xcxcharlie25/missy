import duckdb
con = duckdb.connect(); con.execute("SET memory_limit='16GB'")
K = "MBR_ID, PHR_NCPDP_ID, a_nt_gpi, a_irpg_drugtype, FILLED_DT, a_nf_drg_qty, DAYS_SUPPLY, m_nf_clt_ingred_cost_paid, m_nf_clt_dispensing_fee"
W = """WHERE CARRIER_ID = 'FEDEX' AND m_nf_app_due_amt > 0
         AND coalesce(trim(CAST(a_nf_clt_prc_typ AS VARCHAR)), '') NOT IN ('', 'ZERO')"""
con.execute(f"""CREATE TABLE r AS SELECT {K}, a_nf_clm_sk, SBM_DT, m_nf_app_due_amt, filename,
                       count(*) OVER (PARTITION BY {K}) AS copies
                FROM read_parquet('*2025_*.parquet', filename = true) {W}""")
def show(t, s): print(f"\n{t}"); print(con.execute(s).df().to_string(index=False))

show("1. money and fills: kept vs dropped by the dedupe",
     f"""SELECT 'all rows' AS what, count(*) AS fills, round(sum(m_nf_clt_ingred_cost_paid + m_nf_clt_dispensing_fee)) AS paid FROM r
         UNION ALL SELECT 'one per fill', count(*), round(sum(m_nf_clt_ingred_cost_paid + m_nf_clt_dispensing_fee)) FROM (SELECT DISTINCT {K} FROM r)""")

show("2. repeated groups: same claim key or different, same submission date or different",
     f"""SELECT CASE WHEN n_keys = 1 THEN 'same claim key' ELSE 'different claim keys' END AS claim_key,
                CASE WHEN n_sbm = 1 THEN 'same submission date' ELSE 'different submission dates' END AS submitted,
                count(*) AS groups, sum(copies) AS rows_
         FROM (SELECT {K}, count(DISTINCT a_nf_clm_sk) AS n_keys, count(DISTINCT SBM_DT) AS n_sbm, count(*) AS copies FROM r GROUP BY {K} HAVING count(*) > 1)
         GROUP BY 1, 2 ORDER BY 4 DESC""")

show("3. how many copies a repeated fill has",
     f"SELECT copies, count(*) AS rows_ FROM r GROUP BY 1 ORDER BY 1")

show("4. ten examples: every row of five repeated fills",
     f"""SELECT * EXCLUDE (copies) FROM r WHERE ({K}) IN (SELECT ({K}) FROM r WHERE copies BETWEEN 2 AND 4 ORDER BY MBR_ID LIMIT 5)
         ORDER BY MBR_ID, FILLED_DT, SBM_DT""")
