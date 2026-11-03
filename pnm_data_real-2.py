import duckdb
con = duckdb.connect()
con.execute("SET memory_limit='16GB'")
MONEY = "(m_nf_clt_ingred_cost_paid + m_nf_clt_dispensing_fee)"
FEDEX = """FROM read_parquet('*2025_*.parquet', filename = true)
           WHERE CARRIER_ID = 'FEDEX' AND m_nf_app_due_amt > 0
             AND coalesce(trim(CAST(a_nf_clt_prc_typ AS VARCHAR)), '') NOT IN ('', 'ZERO')"""
def show(title, sql):
    print(f"\n{title}"); print(con.execute(sql).df().to_string(index=False))

show("1. specialty flag values", f"SELECT a_nf_specialty_flag, count(*) AS fills, round(sum({MONEY})) AS paid {FEDEX} GROUP BY 1 ORDER BY 3 DESC")
show("2. MAIL_TODAY values", f"SELECT MAIL_TODAY, count(*) AS fills, round(sum({MONEY})) AS paid {FEDEX} GROUP BY 1 ORDER BY 3 DESC")
show("3. duplicates: rows should equal distinct fills",
     f"SELECT count(*) AS rows_, count(DISTINCT (MBR_ID, PHR_NCPDP_ID, a_nt_gpi, FILLED_DT, a_nf_drg_qty, m_nf_clt_ingred_cost_paid)) AS distinct_fills, min(FILLED_DT) AS first, max(FILLED_DT) AS last {FEDEX}")
show("4. date range per file", f"SELECT filename, min(FILLED_DT) AS first, max(FILLED_DT) AS last, count(*) AS fills {FEDEX} GROUP BY 1 ORDER BY 2")
show("5. money per fill bands", f"""SELECT CASE WHEN {MONEY} < 50 THEN 'a. under 50' WHEN {MONEY} < 200 THEN 'b. 50 to 200' WHEN {MONEY} < 1000 THEN 'c. 200 to 1,000'
                                            WHEN {MONEY} < 5000 THEN 'd. 1,000 to 5,000' ELSE 'e. over 5,000' END AS band,
                                       count(*) AS fills, round(sum({MONEY})) AS paid, round(100 * sum({MONEY}) / (SELECT sum({MONEY}) {FEDEX}), 1) AS pct_of_paid
                                {FEDEX} GROUP BY 1 ORDER BY 1""")
show("6. top 15 drugs by money", f"SELECT a_nt_gpi, a_irpg_drugtype, a_nf_specialty_flag, count(*) AS fills, round(sum({MONEY})) AS paid, round(sum({MONEY}) / count(*)) AS per_fill {FEDEX} GROUP BY 1, 2, 3 ORDER BY 5 DESC LIMIT 15")
