import duckdb
con = duckdb.connect()
con.execute("SET memory_limit='16GB'")
MONEY = "(m_nf_clt_ingred_cost_paid + m_nf_clt_dispensing_fee)"
SPEC = """FROM read_parquet('*2025_*.parquet')
          WHERE CARRIER_ID = 'FEDEX' AND m_nf_app_due_amt > 0
            AND coalesce(trim(CAST(a_nf_clt_prc_typ AS VARCHAR)), '') NOT IN ('', 'ZERO')
            AND a_nf_specialty_flag = 'Specialty'"""
def show(title, sql):
    print(f"\n{title}"); print(con.execute(sql).df().to_string(index=False))

show("1. specialty fills by kind of pharmacy",
     f"""SELECT PHR_DISPNSR_TYPE_DESC, PHR_DISPNSR_CLASS_DESC, count(*) AS fills, round(sum({MONEY})) AS paid,
                count(DISTINCT PHR_NCPDP_ID) AS pharmacies, count(DISTINCT MBR_ID) AS members
         {SPEC} GROUP BY 1, 2 ORDER BY 4 DESC""")

show("2. specialty drugs by class (first 2 digits of GPI): 27 = antidiabetics incl. GLP-1, 21 = cancer, 66 = immunology ...",
     f"""SELECT substr(CAST(a_nt_gpi AS VARCHAR), 1, 2) AS gpi_class, count(*) AS fills, round(sum({MONEY})) AS paid,
                round(sum({MONEY}) / count(*)) AS per_fill, count(DISTINCT PHR_NCPDP_ID) AS pharmacies
         {SPEC} GROUP BY 1 ORDER BY 3 DESC""")

show("3. top 25 specialty drugs: how many pharmacies dispense each one",
     f"""SELECT a_nt_gpi, a_irpg_drugtype, count(*) AS fills, round(sum({MONEY})) AS paid, round(sum({MONEY}) / count(*)) AS per_fill,
                count(DISTINCT PHR_NCPDP_ID) AS pharmacies, count(DISTINCT MBR_ID) AS members,
                count(DISTINCT CASE WHEN PHR_DISPNSR_TYPE_DESC = 'Community or Retail Pharmacy' THEN PHR_NCPDP_ID END) AS community_pharmacies
         {SPEC} GROUP BY 1, 2 ORDER BY 4 DESC LIMIT 25""")

show("4. GLP-1 (GPI 2717...): specialty flag split and where filled",
     f"""SELECT a_nf_specialty_flag, PHR_DISPNSR_TYPE_DESC, count(*) AS fills, round(sum({MONEY})) AS paid, count(DISTINCT PHR_NCPDP_ID) AS pharmacies
         FROM read_parquet('*2025_*.parquet')
         WHERE CARRIER_ID = 'FEDEX' AND m_nf_app_due_amt > 0 AND coalesce(trim(CAST(a_nf_clt_prc_typ AS VARCHAR)), '') NOT IN ('', 'ZERO')
           AND substr(CAST(a_nt_gpi AS VARCHAR), 1, 4) = '2717'
         GROUP BY 1, 2 ORDER BY 4 DESC""")
