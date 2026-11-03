python3.14 -c "import duckdb; print('\n'.join(r[0] + '  ' + r[1] for r in duckdb.sql(\"DESCRIBE SELECT * FROM read_parquet('apr2025_first.parquet')\").fetchall()))"
