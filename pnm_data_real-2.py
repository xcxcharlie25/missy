"""
check_spend.py   Read only checks on the FedEx claims, straight from the parquet files. Run from the folder with the files:
    python3.14 check_spend.py
"""
import duckdb
con = duckdb.connect()
con.execute("SET memory_limit='16GB'")
MONEY = "(m_nf_clt_ingred_cost_paid + m_nf_clt_dispensing_fee)"
FEDEX = """FROM read_parquet('*2025_*.parquet', filename = true)
           WHERE CARRIER_ID = 'FEDEX' AND m_nf_app_due_amt > 0
             AND coalesce(trim(CAST(a_nf_clt_prc_typ AS VARCHAR)), '') NOT IN ('', 'ZERO')"""
ALL_FEDEX = "FROM read_parquet('*2025_*.parquet') WHERE CARRIER_ID = 'FEDEX'"


def show(title, sql):
    print(f"\n{title}")
    print(con.execute(sql).df().to_string(index=False))


show("1. total paid (client ingredient cost + client fee), fills, members: the rows the model uses",
     f"SELECT round(sum({MONEY})) AS paid, count(*) AS fills, count(DISTINCT MBR_ID) AS members, min(FILLED_DT) AS first, max(FILLED_DT) AS last {FEDEX}")

show("2. specialty flag values",
     f"SELECT a_nf_specialty_flag, count(*) AS fills, round(sum({MONEY})) AS paid {FEDEX} GROUP BY 1 ORDER BY 3 DESC")

show("3. MAIL_TODAY values",
     f"SELECT MAIL_TODAY, count(*) AS fills, round(sum({MONEY})) AS paid {FEDEX} GROUP BY 1 ORDER BY 3 DESC")

show("4. duplicates: rows should equal distinct fills (member, pharmacy, drug, date, quantity, cost)",
     f"SELECT count(*) AS rows_, count(DISTINCT (MBR_ID, PHR_NCPDP_ID, a_nt_gpi, FILLED_DT, a_nf_drg_qty, m_nf_clt_ingred_cost_paid)) AS distinct_fills {FEDEX}")

show("5. date range per file: ranges should not overlap",
     f"SELECT filename, min(FILLED_DT) AS first, max(FILLED_DT) AS last, count(*) AS fills {FEDEX} GROUP BY 1 ORDER BY 2")

show("6. money per fill bands",
     f"""SELECT CASE WHEN {MONEY} < 50 THEN 'a. under 50' WHEN {MONEY} < 200 THEN 'b. 50 to 200' WHEN {MONEY} < 1000 THEN 'c. 200 to 1,000'
                     WHEN {MONEY} < 5000 THEN 'd. 1,000 to 5,000' ELSE 'e. over 5,000' END AS band,
                count(*) AS fills, round(sum({MONEY})) AS paid,
                round(100 * sum({MONEY}) / (SELECT sum({MONEY}) {FEDEX}), 1) AS pct_of_paid
         {FEDEX} GROUP BY 1 ORDER BY 1""")

show("7. top 15 drugs by money",
     f"""SELECT a_nt_gpi, a_irpg_drugtype, a_nf_specialty_flag, count(*) AS fills, round(sum({MONEY})) AS paid, round(sum({MONEY}) / count(*)) AS per_fill
         {FEDEX} GROUP BY 1, 2, 3 ORDER BY 5 DESC LIMIT 15""")

show("8. what the amount due filter removes (all FedEx rows, no filter)",
     f"""SELECT CASE WHEN m_nf_app_due_amt IS NULL THEN 'missing' WHEN m_nf_app_due_amt = 0 THEN 'zero' WHEN m_nf_app_due_amt < 0 THEN 'negative' ELSE 'positive' END AS amount_due,
                CASE WHEN m_nf_clt_ingred_cost_paid > 0 THEN 'client ingredient paid' ELSE 'client ingredient zero or empty' END AS ingredient,
                CASE WHEN coalesce(trim(CAST(a_nf_clt_prc_typ AS VARCHAR)), '') = '' THEN 'blank' ELSE a_nf_clt_prc_typ END AS client_price_type,
                count(*) AS rows_
         {ALL_FEDEX} GROUP BY 1, 2, 3 ORDER BY 1, 4 DESC""")

show("9. empty client cost: what those rows look like (all FedEx rows, no filter)",
     f"""SELECT CASE WHEN m_nf_clt_ingred_cost_paid IS NULL THEN 'client cost empty' ELSE 'client cost present' END AS client_cost,
                CASE WHEN coalesce(trim(CAST(a_nf_clt_prc_typ AS VARCHAR)), '') = '' THEN 'blank' ELSE a_nf_clt_prc_typ END AS client_price_type,
                CASE WHEN m_nf_app_due_amt > 0 THEN 'due > 0' ELSE 'due 0 or less' END AS amount_due,
                count(*) AS rows_,
                round(sum(m_nf_app_ingred_cost_paid + m_nf_app_dispensing_fee)) AS pharmacy_money,
                round(sum(m_nf_clt_ingred_cost_paid + m_nf_clt_dispensing_fee)) AS client_money,
                round(sum(abs(m_nf_clt_ingred_cost_paid - m_nf_app_ingred_cost_paid))) AS difference_where_both
         {ALL_FEDEX} GROUP BY 1, 2, 3 ORDER BY 1, 4 DESC""")

show("10. empty client fee: same look (all FedEx rows, no filter)",
     f"""SELECT CASE WHEN m_nf_clt_dispensing_fee IS NULL THEN 'client fee empty' ELSE 'client fee present' END AS client_fee,
                CASE WHEN m_nf_app_due_amt > 0 THEN 'due > 0' ELSE 'due 0 or less' END AS amount_due,
                count(*) AS rows_,
                round(sum(m_nf_app_dispensing_fee)) AS pharmacy_fee, round(sum(m_nf_clt_dispensing_fee)) AS client_fee_money,
                round(sum(abs(m_nf_clt_dispensing_fee - m_nf_app_dispensing_fee))) AS difference_where_both
         {ALL_FEDEX} GROUP BY 1, 2 ORDER BY 1, 3 DESC""")
