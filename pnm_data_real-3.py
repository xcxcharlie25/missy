import duckdb
con = duckdb.connect()
con.execute("SET memory_limit='16GB'")
FEDEX = """FROM read_parquet('*2025_*.parquet')
           WHERE CARRIER_ID = 'FEDEX' AND m_nf_app_due_amt > 0
             AND coalesce(trim(CAST(a_nf_clt_prc_typ AS VARCHAR)), '') NOT IN ('', 'ZERO')"""
MONEY = "(m_nf_clt_ingred_cost_paid + m_nf_clt_dispensing_fee)"
FILL, SBM = "TRY_CAST(FILLED_DT AS DATE)", "TRY_CAST(SBM_DT AS DATE)"
def show(title, sql):
    print(f"\n{title}"); print(con.execute(sql).df().to_string(index=False))

show("1. fills by year of fill date",
     f"SELECT year({FILL}) AS fill_year, count(*) AS fills, round(sum({MONEY})) AS paid {FEDEX} GROUP BY 1 ORDER BY 1")

show("2. fills by month of fill date, 2025 only",
     f"SELECT strftime({FILL}, '%Y-%m') AS fill_month, count(*) AS fills {FEDEX} AND year({FILL}) = 2025 GROUP BY 1 ORDER BY 1")

show("3. submission date range: what the file is really cut by",
     f"SELECT min({SBM}) AS first_submitted, max({SBM}) AS last_submitted, min({FILL}) AS first_filled, max({FILL}) AS last_filled {FEDEX}")

show("4. days from fill to submission",
     f"""SELECT CASE WHEN d <= 1 THEN 'a. same day or next' WHEN d <= 7 THEN 'b. within a week' WHEN d <= 30 THEN 'c. within a month'
                     WHEN d <= 365 THEN 'd. within a year' ELSE 'e. over a year' END AS lag, count(*) AS fills, round(sum(money)) AS paid
         FROM (SELECT date_diff('day', {FILL}, {SBM}) AS d, {MONEY} AS money {FEDEX}) GROUP BY 1 ORDER BY 1""")

show("5. the old fills: top price types and pharmacy types",
     f"SELECT a_nf_clt_prc_typ, PHR_DISPNSR_TYPE_DESC, count(*) AS fills, round(sum({MONEY})) AS paid {FEDEX} AND year({FILL}) < 2025 GROUP BY 1, 2 ORDER BY 3 DESC LIMIT 10")
