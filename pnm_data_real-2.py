# =============================================================================
# PHARMACY NETWORK MODEL  ·  PART 1 (REAL DATA)  ·  REAL FILLS, SYNTHETIC FOR WHAT IS STILL TO COME
# Pranshi Saxena  ·  Optum Rx Quantum Initiative
#
# Run:   python pnm_data_real.py               builds the same tables as pnm_data.py, prints the data report,
#                                              writes exports_real/
# Quick: PNM_SMOKE=1 python pnm_data_real.py   300 members only
# Model: PNM_DATA=real python pnm_model.py     results go to exports_real/ and charts_real/, keep them in house
# In this folder:
#   networkOptVisitLevel.xlsx (or .csv)   the fill file. Other name or place: set PNM_REAL_FILE
#   pharmacy_list.csv (or .xlsx)          Angelica's pharmacy list, when it comes. Optional. PNM_PHARMACY_FILE
#   zip_centroids.csv                     public ZIP center points: turns a ZIP into a point on the map
#   pnm_data.py                           config, the geography rules and the data report come from there
#
# Every column asked from Angelica is filled in under its exact name. In the file: used as is.
# Not in the file: a synthetic stand in. When the real column arrives it takes over, no code change.
#   fill file       CLM_SK, CLAIM_NBR, m_nf_clt_ingred_cost_paid, m_nf_clt_dispensing_fee, m_nf_app_ingred_cost_paid,
#                   m_nf_app_dispensing_fee, m_irpg_contrate, m_irpg_contfee, a_irpg_drugtype,
#                   a_nf_pricing_type_combined, a_nf_specialty_flag, a_nt_gpi, a_nt_maintenance_flag
#   pharmacy list   PHR_NCPDP_ID, PHR_NPI_NO, PHR_NM, PHR_AFFL_NM, PHR_ADDR1_TXT, PHR_CITY_NM, PHR_ST_CD,
#                   PHR_ZIP_CD, PHR_CNTY_NM, DLVRY_CHANL, PHR_DISPNSR_TYPE_DESC
# The report says which columns are real and which are synthetic.
#
# Logic
#   location   each member sits at their main retail pharmacy (its PHR_ZIP_CD), so the 10 mile wall runs from it
#   drugs      specialty if a_nf_specialty_flag is Y, else generic or brand from a_irpg_drugtype.
#              a_nt_maintenance_flag Y: held at mail for mail members. a_nt_gpi: drug class for the growth trend
#   rates      paid over AWP, per pharmacy and bucket. Too few fills: the chain's rate, then the type's
#   synthetic, not asked for: capacity, quality, accepts new patients, admin cost
# =============================================================================
import os, re, zlib
from decimal import Decimal, InvalidOperation
import numpy as np, pandas as pd
from pnm_data import CONFIG, KIND, SMOKE, stamp, tier_of, haversine_miles, report   # haversine_miles: for pnm_model.py

CONFIG.update({
    'REAL_FILE'    : os.environ.get('PNM_REAL_FILE', 'networkOptVisitLevel.xlsx'),
    'PHARMACY_FILE': os.environ.get('PNM_PHARMACY_FILE', 'pharmacy_list.csv'),
    'ZIP_FILE'     : os.environ.get('PNM_ZIP_FILE', 'zip_centroids.csv'),
    'REAL_SHEET'   : 0,                                  # first sheet
    # fill file headers we have today. If one differs, change the right side only
    'FILL_COLS'    : dict(MBR='MBR_ID', NDC='a_nt_ndc', DATE='FILLED_DT', KIND='KIND', DAYS='DAYS_SUPPLY',
                          AWP='POST_AWP', AWP_DAY='AWP_DAY', PHR_NAME='PHARMACY_NAME', NCPDP='PHARMACY_NCPDP_ID',
                          NPI='PHARMACY_NPI_NO', MAIL_FLAG='MAIL_TODAY', NINETY_FLAG='NINETY_TODAY'),
    'MEMBER_KEY_EXTRA': ['CARRIER_ID', 'ACCOUNT_ID', 'EMPLOYER_GROUP_ID'],   # joined to MBR_ID when the file has them
    # words in KIND, DLVRY_CHANL or PHR_DISPNSR_TYPE_DESC and the kind of place they mean. Anything else is left out
    'CHANNEL_WORDS': [('infusion', 'INFUSION'), ('special', 'SPECIALTY'), ('mail', 'MAIL'), ('home deliv', 'MAIL'),
                      ('retail', 'RETAIL'), ('community', 'RETAIL')],
    # a_irpg_drugtype values, read by first letter: G generic, B brand, and Medi-Span Y generic, M N O brand
    'DRUG_TYPE_LETTERS': dict(G='GENERIC', Y='GENERIC', B='BRAND', M='BRAND', N='BRAND', O='BRAND'),
    'AREA_ZIP3'    : [],                 # e.g. ['554']: only members whose main pharmacy is in these 3 digit ZIPs
    'DROP_EXACT_DUPLICATES': True,       # one row per claim (CLAIM_NBR), or per identical row without it
    'MAX_MEMBERS'  : 300 if SMOKE else 10_000,   # above this, a repeatable sample of members
    'ANNUALIZE'    : True,               # the file covers part of a year: scale volumes to a year
    'TREND_GPI2'   : {},                 # growth by the first 2 digits of a_nt_gpi, e.g. {'27': 1.03}. Empty: none
    'MIN_FILLS_FOR_RATE': 5,             # fewer fills in a bucket: the chain's rate, then the type's
    # synthetic stand ins
    'GENERIC_FILL_SHARE': 0.85,          # cheapest drugs by AWP per day, this share of fills, are generic
    'CHAIN_MIN_STORES'  : 3,             # a pharmacy name with this many stores is a chain
    'SYNTH_CANDIDATE_SHARE': 0.40,       # pharmacies no member uses today, share of retail. Only with no pharmacy list
    'CAPACITY_VS_TODAY' : (1.3, 2.5),    # capacity = today's load times this
    'CAPACITY_FLOOR'    : 200,           # fills a year
    'SEED_REAL'    : 23,
    'BASELINE_REALISM': 'REAL CLAIMS',
    'EXPORT_DIR'   : 'exports_real',     # kept apart from the synthetic exports
})
FILL_ASKED = ['CLM_SK', 'CLAIM_NBR', 'm_nf_clt_ingred_cost_paid', 'm_nf_clt_dispensing_fee', 'm_nf_app_ingred_cost_paid',
              'm_nf_app_dispensing_fee', 'm_irpg_contrate', 'm_irpg_contfee', 'a_irpg_drugtype',
              'a_nf_pricing_type_combined', 'a_nf_specialty_flag', 'a_nt_gpi', 'a_nt_maintenance_flag']
PHARMACY_ASKED = ['PHR_NCPDP_ID', 'PHR_NPI_NO', 'PHR_NM', 'PHR_AFFL_NM', 'PHR_ADDR1_TXT', 'PHR_CITY_NM', 'PHR_ST_CD',
                  'PHR_ZIP_CD', 'PHR_CNTY_NM', 'DLVRY_CHANL', 'PHR_DISPNSR_TYPE_DESC']
MONEY = ['m_nf_clt_ingred_cost_paid', 'm_nf_clt_dispensing_fee', 'm_nf_app_ingred_cost_paid', 'm_nf_app_dispensing_fee',
         'm_irpg_contrate', 'm_irpg_contfee']
CHAN = {'RETAIL': 'Retail', 'MAIL': 'MAIL', 'SPECIALTY': 'SPECIALTY', 'INFUSION': 'MEDICAL'}   # pnm_data.py labels
FILE_CHAN = {'RETAIL': 'Retail', 'MAIL': 'Mail', 'SPECIALTY': 'Specialty', 'INFUSION': 'Home Infusion'}
TYPE_DESC = dict(RETAIL='COMMUNITY/RETAIL PHARMACY', MAIL='MAIL ORDER PHARMACY', SPECIALTY='SPECIALTY PHARMACY',
                 INFUSION='HOME INFUSION THERAPY PROVIDER')
BUCKETS = ('GENERIC', 'BRAND', 'SPECIALTY')
RATE_COL = {b: f'CLT_RATE_{b}' for b in BUCKETS}
TIER_SHARE = dict(RETAIL=CONFIG['RETAIL_PER_1000'], SPECIALTY=CONFIG['SPEC_PER_1000'], INFUSION=CONFIG['INF_PER_1000'],
                  MAIL=dict(URBAN=1, SUBURBAN=1, RURAL=1))
QUALITY = dict(RETAIL=(3.5, 5.0), SPECIALTY=(4.2, 5.0), INFUSION=(4.0, 5.0), MAIL=(4.3, 4.9))
GPI2 = dict(MAINT=['27', '36', '39', '58', '44', '33', '34', '49'], ACUTE=['01', '02', '03', '66', '43', '65', '41'],
            SPEC=['21', '66', '12'])       # synthetic drug groups only
STOP = {'THE', 'A', 'AN', 'OF', 'PHARMACY', 'PHARMACIES', 'DRUG', 'DRUGS', 'RX', 'COMMUNITY', 'FAMILY', 'MEDICAL',
        'CENTER', 'HEALTH', 'CARE', 'APOTHECARY', 'STORE', 'STORES', 'INC', 'LLC', 'CO', 'CORP'}

# =============================================================================
# SMALL HELPERS
# =============================================================================
def rng_for(tag, key):                 # same key, same numbers, every run
    return np.random.default_rng([CONFIG['SEED_REAL'], zlib.crc32(f'{tag}|{key}'.encode())])

def num(s): return pd.to_numeric(s, errors='coerce')
def yes(s): return s.fillna('').astype(str).str.strip().str.upper().isin(['Y', 'YES', '1', '1.0', 'TRUE', 'T'])

def as_id(s, width=None):
    """IDs as text, undoing what Excel does: 12345.0, 1.2E+14, dropped leading zeros."""
    def one(x):
        if x is None: return None
        x = str(x).strip()
        if x.lower() in ('', 'nan', 'none', 'null', 'nat', '-'): return None
        if re.fullmatch(r'\d+\.0+', x): x = x.split('.')[0]
        elif re.search(r'[eE]\+?\d+$', x):
            try: x = format(Decimal(x), 'f').split('.')[0]
            except InvalidOperation: pass
        return x.zfill(width) if (width and x.isdigit()) else x
    return s.map(one)

def as_zip(s):
    def one(x):
        if x is None or str(x).strip().lower() in ('', 'nan', 'none', 'null'): return None
        d = re.sub(r'\D', '', str(x).strip().split('.')[0])
        if not d: return None
        return d.zfill(5) if len(d) <= 5 else d.zfill(9)[:5]       # 55401, 554011234, 55401-1234, 2134
    return s.map(one)

def kind_of(v):
    v = str(v).lower()
    for w, k in CONFIG['CHANNEL_WORDS']:
        if w in v: return k
    return 'OTHER'

def drug_type_bucket(v):
    v = '' if v is None else str(v).strip().upper()
    if not v or v in ('NAN', 'NONE'): return None
    if 'GENERIC' in v: return 'GENERIC'
    if 'BRAND' in v: return 'BRAND'
    return CONFIG['DRUG_TYPE_LETTERS'].get(v[:1])

def name_key(nm):
    words = re.sub(r'[^A-Z ]', ' ', re.sub(r"['\-.&]", '', str(nm).upper())).split()
    words = [w for w in words if w not in STOP]
    return words[0] if words else 'UNKNOWN'

def miles(la1, lo1, la2, lo2):          # same formula as haversine_miles in pnm_data.py, for whole columns
    la1, lo1, la2, lo2 = (np.radians(np.asarray(x, float)) for x in (la1, lo1, la2, lo2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 3958.8 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def most_used(c, by, site):
    """The site used most; ties go to the most recent fill, then the lower site id."""
    n = (c.groupby(by + [site]).agg(n=('FILLED_DT', 'size'), last=('FILLED_DT', 'max')).reset_index()
          .sort_values(by + ['n', 'last', site], ascending=[True] * len(by) + [False, False, True]))
    return n.drop_duplicates(by)[by + [site]].reset_index(drop=True)

def read_table(f):
    return pd.read_csv(f, dtype=str) if f.lower().endswith('.csv') else pd.read_excel(f, sheet_name=CONFIG['REAL_SHEET'], dtype=str)

def header_map(raw):                    # case blind, and phr. or md. in front is fine
    return {str(c).split('.')[-1].strip().lower(): c for c in raw.columns}

# =============================================================================
# ZIPs. Public ZIP center points
# =============================================================================
def load_zips():
    f = CONFIG['ZIP_FILE']
    if not os.path.exists(f): raise SystemExit(f"cannot find {f}. It turns ZIPs into points on the map: put it in this folder")
    z = pd.read_csv(f, dtype={'ZIP': str})
    z['ZIP'] = z.ZIP.str.zfill(5)
    return z.drop_duplicates('ZIP').set_index('ZIP')

def area_zips(zips, center, radius):
    s = zips[(zips.ZIP_TYPE == 'STANDARD') & (zips.ACTIVE == 1)]
    s = s.assign(D=miles(center[0], center[1], s.LAT, s.LON))
    return s[s.D <= radius]

def draw_zip(tag, key, kind, az):
    """SYNTHETIC: a real ZIP near the center, urban, suburban or rural the way pnm_data.py spreads its sites."""
    r = rng_for(tag, key); sh = TIER_SHARE[kind]; tiers = list(sh)
    tier = tiers[r.choice(len(tiers), p=np.array([sh[t] for t in tiers], float) / sum(sh.values()))]
    tm = CONFIG['TIER_MILES']; lo = {'URBAN': 0.0, 'SUBURBAN': tm['URBAN'], 'RURAL': tm['SUBURBAN']}[tier]
    pool = az[(az.D >= lo) & (az.D < tm[tier])]
    if not len(pool): pool = az.nsmallest(20, 'D')
    return pool.index[int(r.integers(len(pool)))]

def locate(zipcol, zips):
    """ZIP to point. A ZIP not in the list falls back to the middle of its 3 digit area."""
    z3 = zips.groupby(zips.index.str[:3])[['LAT', 'LON']].mean()
    lat = zipcol.map(zips.LAT); lon = zipcol.map(zips.LON); src = pd.Series('ZIP', index=zipcol.index)
    miss = lat.isna() & zipcol.notna()
    lat[miss] = zipcol[miss].str[:3].map(z3.LAT); lon[miss] = zipcol[miss].str[:3].map(z3.LON); src[miss] = 'ZIP3'
    src[lat.isna()] = 'NONE'
    return lat, lon, src

# =============================================================================
# READ the fill file, and the pharmacy list when it is there
# =============================================================================
def read_fills():
    f = CONFIG['REAL_FILE']
    if not os.path.exists(f): raise SystemExit(f"cannot find {f}. Put it in this folder or set PNM_REAL_FILE")
    raw = read_table(f); heads = header_map(raw); C = CONFIG['FILL_COLS']
    def col(name, required=False):
        c = heads.get(str(name).strip().lower())
        if c is None and required:
            raise SystemExit(f"column {name} not found. Headers in the file: {list(raw.columns)}. Fix CONFIG['FILL_COLS']")
        return c
    keys = [c for c in (col(k) for k in CONFIG['MEMBER_KEY_EXTRA']) if c] + [col(C['MBR'], True)]
    mbr = as_id(raw[keys[0]])
    for k in keys[1:]: mbr = mbr.str.cat(as_id(raw[k]), sep='|')
    if not (col(C['NCPDP']) or col(C['NPI'])): raise SystemExit('the fill file needs a pharmacy NCPDP or NPI column')
    none = pd.Series(None, index=raw.index, dtype=object)
    ncpdp = as_id(raw[col(C['NCPDP'])], 7) if col(C['NCPDP']) else none
    npi = as_id(raw[col(C['NPI'])], 10) if col(C['NPI']) else none
    days = num(raw[col(C['DAYS'], True)])
    awp = num(raw[col(C['AWP'])]) if col(C['AWP']) else pd.Series(np.nan, index=raw.index)
    if col(C['AWP_DAY']): awp = awp.fillna(num(raw[col(C['AWP_DAY'])]) * days)
    ch = raw[col(C['KIND'], True)].fillna('').astype(str).str.strip()
    df = pd.DataFrame(dict(MBR_SK=mbr, NDC11=as_id(raw[col(C['NDC'], True)], 11),
                           FILLED_DT=pd.to_datetime(raw[col(C['DATE'], True)], errors='coerce').dt.normalize(),
                           CHANNEL_IN_FILE=ch, CH=ch.map(kind_of), DAYS=days, AWP=awp,
                           PHR_KEY=ncpdp.where(ncpdp.notna(), npi.map(lambda x: f'NPI{x}' if x else None)),
                           NCPDP=ncpdp, NPI=npi, PHR_NM=raw[col(C['PHR_NAME'])].fillna('') if col(C['PHR_NAME']) else ''))
    src = {}
    for c in FILL_ASKED:
        h = col(c); src[c] = 'REAL' if h else 'SYNTHETIC'
        if h: df[c] = raw[h]
    for c in ('CLM_SK', 'CLAIM_NBR'):
        if c in df: df[c] = as_id(df[c])
    if 'a_nt_gpi' in df: df['a_nt_gpi'] = as_id(df['a_nt_gpi'], 14)
    for c in MONEY:
        if c in df: df[c] = num(df[c])
    notes = dict(file=f, rows=len(df), member_key=keys, dropped={}, fill_source=src)
    out = pd.Series(False, index=df.index)
    for why, bad in [('no member', df.MBR_SK.isna()), ('no drug', df.NDC11.isna()), ('no fill date', df.FILLED_DT.isna()),
                     ('no pharmacy', df.PHR_KEY.isna()), ('days supply not above 0', ~(df.DAYS > 0)),
                     ('no AWP', ~(df.AWP >= 0))]:
        notes['dropped'][why] = int((bad & ~out).sum()); out |= bad
    notes['other_channels'] = df[~out & (df.CH == 'OTHER')].CHANNEL_IN_FILE.value_counts().to_dict()
    out |= df.CH == 'OTHER'
    df = df[~out]
    if CONFIG['DROP_EXACT_DUPLICATES']:
        by_claim = 'CLAIM_NBR' in df and df.CLAIM_NBR.notna().all()
        dup = df.duplicated(['MBR_SK', 'CLAIM_NBR'] if by_claim else ['MBR_SK', 'NDC11', 'FILLED_DT', 'PHR_KEY', 'DAYS', 'AWP'])
        notes['dropped']['same claim twice' if by_claim else 'exact copies of another row'] = int(dup.sum()); df = df[~dup]
    flags = {k: col(C[k]) for k in ('MAIL_FLAG', 'NINETY_FLAG')}
    notes['_file_flags'] = pd.DataFrame({'MBR_SK': mbr, **{k: num(raw[c]) for k, c in flags.items() if c}})
    return df.reset_index(drop=True), notes

def read_pharmacy_list():
    f = CONFIG['PHARMACY_FILE']
    if not os.path.exists(f): return None
    raw = read_table(f); heads = header_map(raw)
    pl = pd.DataFrame({c: raw[heads[c.lower()]] for c in PHARMACY_ASKED if c.lower() in heads})
    if 'PHR_ZIP_CD' not in pl or not ({'PHR_NCPDP_ID', 'PHR_NPI_NO'} & set(pl)):
        raise SystemExit(f"{f} needs PHR_ZIP_CD and PHR_NCPDP_ID or PHR_NPI_NO. Headers in it: {list(raw.columns)}")
    if 'PHR_NCPDP_ID' in pl: pl['PHR_NCPDP_ID'] = as_id(pl.PHR_NCPDP_ID, 7)
    if 'PHR_NPI_NO' in pl: pl['PHR_NPI_NO'] = as_id(pl.PHR_NPI_NO, 10)
    pl['PHR_ZIP_CD'] = as_zip(pl.PHR_ZIP_CD)
    return pl.reset_index(drop=True)

def sample_members(df, notes):
    mem = sorted(df.MBR_SK.unique())
    notes['members_in_file'] = len(mem)
    if len(mem) > CONFIG['MAX_MEMBERS']:
        keep = set(sorted(mem, key=lambda k: (zlib.crc32(k.encode()), k))[:CONFIG['MAX_MEMBERS']])
        df = df[df.MBR_SK.isin(keep)].reset_index(drop=True)
    return df

def pharmacy_types(df, notes):
    """One type per pharmacy, its main channel. Each fill takes the type of the pharmacy it was filled at."""
    n = (df.groupby(['PHR_KEY', 'CH']).size().rename('n').reset_index()
           .sort_values(['PHR_KEY', 'n', 'CH'], ascending=[True, False, True]))
    ptype = n.drop_duplicates('PHR_KEY').set_index('PHR_KEY').CH
    new = df.PHR_KEY.map(ptype)
    notes['fills_relabelled'] = int((new != df.CH).sum())
    return df.assign(CH=new), ptype

# =============================================================================
# DRUG FIELDS. a_nf_specialty_flag, a_irpg_drugtype, a_nf_pricing_type_combined, a_nt_maintenance_flag, a_nt_gpi
# =============================================================================
def drug_fields(df, notes):
    """SYNTHETIC only where the file has no such column:
       a_nf_specialty_flag     Y if the drug mostly goes through specialty pharmacies
       a_irpg_drugtype         GENERIC for the cheapest drugs by AWP per day, GENERIC_FILL_SHARE of fills, else BRAND
       a_nf_pricing_type_combined   MAC for generic, AWP otherwise
       a_nt_maintenance_flag   Y if the drug goes to mail, is filled for 84+ days, or a member refills it
       a_nt_gpi                a made up 14 digit code with a plausible drug group
    Then the price bucket: SPECIALTY if a_nf_specialty_flag is Y, else GENERIC or BRAND from a_irpg_drugtype."""
    d = df.assign(AWPD=df.AWP / df.DAYS, SPEC=(df.CH == 'SPECIALTY').astype(float),
                  LONG=((df.CH == 'MAIL') | (df.DAYS >= 84)).astype(float))
    per = d.groupby('NDC11').agg(fills=('AWPD', 'size'), awpd=('AWPD', 'median'), spec=('SPEC', 'mean'), long=('LONG', 'max'))
    per['refill'] = d.groupby(['NDC11', 'MBR_SK']).size().groupby('NDC11').max() >= 2
    spec_g = per.spec >= 0.5
    rest = per[~spec_g].reset_index().sort_values(['awpd', 'NDC11'])
    before = rest.fills.cumsum().shift(fill_value=0) / max(1, rest.fills.sum())
    type_g = pd.Series('BRAND', index=per.index)
    type_g[rest.NDC11[before < CONFIG['GENERIC_FILL_SHARE']].to_numpy()] = 'GENERIC'
    maint_g = (per.long > 0) | per.refill
    src = notes['fill_source']
    if src['a_nf_specialty_flag'] == 'SYNTHETIC': df['a_nf_specialty_flag'] = df.NDC11.map(spec_g.map({True: 'Y', False: 'N'}))
    if src['a_irpg_drugtype'] == 'SYNTHETIC': df['a_irpg_drugtype'] = df.NDC11.map(type_g)
    if src['a_nf_pricing_type_combined'] == 'SYNTHETIC':
        df['a_nf_pricing_type_combined'] = np.where(df.a_irpg_drugtype.map(drug_type_bucket) == 'GENERIC', 'MAC', 'AWP')
    if src['a_nt_maintenance_flag'] == 'SYNTHETIC': df['a_nt_maintenance_flag'] = df.NDC11.map(maint_g.map({True: 'Y', False: 'N'}))
    if src['a_nt_gpi'] == 'SYNTHETIC':
        def gpi(n):
            r = rng_for('GPI', n); pool = GPI2['SPEC'] if spec_g[n] else GPI2['MAINT'] if maint_g[n] else GPI2['ACUTE']
            return pool[int(r.integers(len(pool)))] + ''.join(str(x) for x in r.integers(0, 10, 12))
        df['a_nt_gpi'] = df.NDC11.map({n: gpi(n) for n in per.index})
    spec = yes(df.a_nf_specialty_flag)
    dt = df.a_irpg_drugtype.map(drug_type_bucket)
    notes['drugtype_unread'] = int((dt.isna() & ~spec).sum())
    notes['drugtype_values'] = df.a_irpg_drugtype.fillna('blank').value_counts().head(6).to_dict()
    df['PRICE_BUCKET'] = np.where(spec, 'SPECIALTY', dt.fillna(df.NDC11.map(type_g)))
    return df

# =============================================================================
# PHARMACY LIST. One row per pharmacy, exact columns. Angelica's list when it is in the folder, synthetic for the rest
# =============================================================================
def pharmacy_list(df, ptype, zips, notes):
    g = df.sort_values('FILLED_DT').groupby('PHR_KEY')
    pl = pd.DataFrame({'PHR_NCPDP_ID': g.NCPDP.last(), 'PHR_NPI_NO': g.NPI.last(), 'PHR_NM': g.PHR_NM.last(),
                       'DLVRY_CHANL': g.CHANNEL_IN_FILE.agg(lambda s: s.value_counts().index[0])})
    for c in PHARMACY_ASKED:
        if c not in pl: pl[c] = None
    pl['PHR_TYPE_CD'] = ptype; pl['IN_FILL_FILE'] = 1; pl['ROW_SOURCE'] = 'FILL FILE'
    src = {c: 'SYNTHETIC' for c in PHARMACY_ASKED}
    src.update({c: 'FILL FILE' for c in ('PHR_NCPDP_ID', 'PHR_NPI_NO', 'PHR_NM', 'DLVRY_CHANL')})
    real = read_pharmacy_list(); zip_src = pd.Series('SYNTHETIC', index=pl.index)
    if real is not None:
        real['_row'] = range(len(real)); hit = {}
        for key_col, mine in (('PHR_NCPDP_ID', pl.PHR_NCPDP_ID), ('PHR_NPI_NO', pl.PHR_NPI_NO)):
            if key_col not in real: continue
            look = real.dropna(subset=[key_col]).drop_duplicates(key_col).set_index(key_col)._row
            for k, v in mine.items():
                if k not in hit and v in look.index: hit[k] = int(look[v])
        for c in PHARMACY_ASKED:
            if c in real and c not in ('PHR_NCPDP_ID', 'PHR_NPI_NO'):
                vals = pd.Series({k: real.at[r, c] for k, r in hit.items()}, dtype=object)
                if len(vals): pl.loc[vals.index, c] = vals.where(vals.notna(), pl.loc[vals.index, c])
                src[c] = 'PHARMACY LIST' if len(hit) == len(pl) else f'PHARMACY LIST for {len(hit)} of {len(pl)}, rest synthetic'
        zip_src[list(hit)] = 'REAL'
        extra = real[~real._row.isin(hit.values())].copy()      # pharmacies no member of ours used: candidates
        extra.index = [n if isinstance(n, str) else f'NPI{p}' for n, p in
                       zip(extra.get('PHR_NCPDP_ID', pd.Series(None, index=extra.index)), extra.get('PHR_NPI_NO', pd.Series(None, index=extra.index)))]
        extra = extra[~extra.index.duplicated() & ~extra.index.isin(pl.index) & (extra.index != 'NPINone')]
        t = extra.get('DLVRY_CHANL', pd.Series('', index=extra.index)).map(kind_of)
        t = t.where(t != 'OTHER', extra.get('PHR_DISPNSR_TYPE_DESC', pd.Series('', index=extra.index)).map(kind_of))
        notes['list_dropped_other'] = int((t == 'OTHER').sum())
        extra = extra[t != 'OTHER'].assign(PHR_TYPE_CD=t[t != 'OTHER'], IN_FILL_FILE=0, ROW_SOURCE='PHARMACY LIST')
        pl = pd.concat([pl, extra[[c for c in pl.columns if c in extra]]])
        zip_src = pd.concat([zip_src, pd.Series('REAL', index=extra.index)])
        notes['pharmacy_list'] = CONFIG['PHARMACY_FILE']
    # affiliation: SYNTHETIC from the name where missing. A name with CHAIN_MIN_STORES stores is a chain
    key = pl.PHR_NM.map(name_key)
    stores = key[pl.PHR_TYPE_CD == 'RETAIL'].value_counts()
    chains = set(stores[stores >= CONFIG['CHAIN_MIN_STORES']].index) - {'UNKNOWN'}
    guess = pd.Series(np.where((pl.PHR_TYPE_CD == 'RETAIL') & ~key.isin(chains), 'INDEPENDENT', key), index=pl.index)
    pl['PHR_AFFL_NM'] = pl.PHR_AFFL_NM.where(pl.PHR_AFFL_NM.notna(), guess)
    # SYNTHETIC candidates, pharmacies in the area that no member uses today. Only without Angelica's list
    if real is None and CONFIG['SYNTH_CANDIDATE_SHARE'] > 0:
        aff = pl[pl.PHR_TYPE_CD == 'RETAIL'].PHR_AFFL_NM.value_counts(normalize=True); rows = {}
        for k in range(int(round(CONFIG['SYNTH_CANDIDATE_SHARE'] * (pl.PHR_TYPE_CD == 'RETAIL').sum()))):
            r = rng_for('CAND', k); a = aff.index[r.choice(len(aff), p=aff.to_numpy())]; sid = f'SYN{k:04d}'
            rows[sid] = dict(PHR_NCPDP_ID=sid, PHR_NPI_NO=f'SYN{k:07d}', PHR_AFFL_NM=a, DLVRY_CHANL='Retail',
                             PHR_NM=(f'SYNTHETIC INDEPENDENT PHARMACY {sid}' if a == 'INDEPENDENT' else f'{a} PHARMACY {sid}'),
                             PHR_TYPE_CD='RETAIL', IN_FILL_FILE=0, ROW_SOURCE='SYNTHETIC')
        if rows:
            pl = pd.concat([pl, pd.DataFrame.from_dict(rows, orient='index')])
            zip_src = pd.concat([zip_src, pd.Series('SYNTHETIC', index=list(rows))])
    # ZIP: SYNTHETIC where missing. With Angelica's list: a ZIP from her list, so it stays in the same area
    pl['PHR_ZIP_CD'] = as_zip(pl.PHR_ZIP_CD.astype(object))
    need = pl.PHR_ZIP_CD.isna()
    if need.any():
        if real is not None:
            pool = sorted(set(pl.PHR_ZIP_CD.dropna()) & set(zips.index))
            pl.loc[need, 'PHR_ZIP_CD'] = [pool[int(rng_for('ZIP', k).integers(len(pool)))] for k in pl.index[need]]
        else:
            az = area_zips(zips, CONFIG['CENTER'], CONFIG['TIER_MILES']['RURAL'])
            pl.loc[need, 'PHR_ZIP_CD'] = [draw_zip('ZIP', k, t, az) for k, t in zip(pl.index[need], pl.PHR_TYPE_CD[need])]
        zip_src[need[need].index] = 'SYNTHETIC'
    pl['ZIP_SOURCE'] = zip_src.reindex(pl.index)
    pl['PHR_LAT'], pl['PHR_LON'], pl['LOCATION_SOURCE'] = locate(pl.PHR_ZIP_CD, zips)
    notes['pharmacies_no_location'] = int((pl.LOCATION_SOURCE == 'NONE').sum())
    pl = pl[pl.LOCATION_SOURCE != 'NONE']
    for c, z in (('PHR_CITY_NM', 'CITY'), ('PHR_ST_CD', 'STATE'), ('PHR_CNTY_NM', 'COUNTY')):
        pl[c] = pl[c].where(pl[c].notna(), pl.PHR_ZIP_CD.map(zips[z]))
    pl['PHR_ADDR1_TXT'] = pl.PHR_ADDR1_TXT.where(pl.PHR_ADDR1_TXT.notna(), 'SYNTHETIC ADDRESS')
    pl['DLVRY_CHANL'] = pl.DLVRY_CHANL.where(pl.DLVRY_CHANL.notna(), pl.PHR_TYPE_CD.map(FILE_CHAN))
    pl['PHR_DISPNSR_TYPE_DESC'] = pl.PHR_DISPNSR_TYPE_DESC.where(pl.PHR_DISPNSR_TYPE_DESC.notna(), pl.PHR_TYPE_CD.map(TYPE_DESC))
    for c in ('PHR_CITY_NM', 'PHR_ST_CD', 'PHR_CNTY_NM'):
        if src[c] == 'SYNTHETIC': src[c] = 'FROM THE ZIP'
    notes['pharmacy_source'] = src
    pl['PHR_CHAIN_NM'] = np.where(pl.PHR_TYPE_CD != 'RETAIL', pl.PHR_AFFL_NM,
                                  np.where(pl.PHR_AFFL_NM.fillna('').str.upper().isin(['', 'INDEPENDENT', 'NONE']),
                                           'Independent', pl.PHR_AFFL_NM))
    pl['PHR_SITE_OF_CARE_CD'] = np.where(pl.PHR_TYPE_CD == 'INFUSION', 'HOME_INFUSION', 'NA')   # pharmacy claims
    pl['SETTING_FACTOR'] = [CONFIG['SETTING'].get(s if t == 'INFUSION' else t, 1.0) for t, s in zip(pl.PHR_TYPE_CD, pl.PHR_SITE_OF_CARE_CD)]
    return pl

# =============================================================================
# PAID AMOUNTS AND RATES
# =============================================================================
def rate_card(key):
    """SYNTHETIC rate card, same ranges as pnm_data.py. A chain has one card for all its stores."""
    r = rng_for('RATE', key); c = {b: round(float(r.uniform(*CONFIG['RATE_RANGE'][b])), 3) for b in BUCKETS}
    c['FEE'] = round(float(r.uniform(*CONFIG['FEE_RANGE'])), 2)
    for b in BUCKETS: c['P_' + b] = round(float(min(0.985, c[b] + r.uniform(*CONFIG['PHR_RATE_DEEPER']))), 3)
    c['P_FEE'] = round(float(r.uniform(0.55, 1.60)), 2)
    return c

def paid_fields(df, pl, notes):
    """SYNTHETIC only where the file has no such column: the six paid and contract columns, from the rate cards,
    and CLM_SK and CLAIM_NBR as SYN numbers."""
    src = notes['fill_source']
    card_key = pd.Series(np.where((pl.PHR_TYPE_CD == 'RETAIL') & (pl.PHR_CHAIN_NM != 'Independent'), pl.PHR_CHAIN_NM, pl.index), index=pl.index)
    cards = {k: rate_card(ck) for k, ck in card_key.items()}
    s = df.PHR_KEY.map(pl.SETTING_FACTOR)
    get = lambda f: np.array([f(cards[p], b) for p, b in zip(df.PHR_KEY, df.PRICE_BUCKET)])
    make = {'m_nf_clt_ingred_cost_paid': lambda: (df.AWP * (1 - get(lambda c, b: c[b])) * s).round(2),
            'm_nf_clt_dispensing_fee'  : lambda: get(lambda c, b: c['FEE']),
            'm_nf_app_ingred_cost_paid': lambda: (df.AWP * (1 - get(lambda c, b: c['P_' + b])) * s).round(2),
            'm_nf_app_dispensing_fee'  : lambda: get(lambda c, b: c['P_FEE']),
            'm_irpg_contrate'          : lambda: get(lambda c, b: c['P_' + b]),
            'm_irpg_contfee'           : lambda: get(lambda c, b: c['P_FEE']),
            'CLM_SK'                   : lambda: [f'SYN{i:09d}' for i in range(len(df))],
            'CLAIM_NBR'                : lambda: [f'SYN{i:012d}' for i in range(len(df))]}
    for c, f in make.items():
        if src[c] == 'SYNTHETIC': df[c] = f()
    return df

def derive_rates(df, pl, notes):
    """Rates the model prices with, from paid over AWP: 1 minus paid / (AWP x setting), the way derive() in pnm_data.py
    works back from paid. Per pharmacy and bucket when it has MIN_FILLS_FOR_RATE fills, else its chain's, else its type's."""
    m = CONFIG['MIN_FILLS_FOR_RATE']
    f = df[df.AWP > 0].assign(S=df.PHR_KEY.map(pl.SETTING_FACTOR), CHAIN=df.PHR_KEY.map(pl.PHR_CHAIN_NM),
                              TY=df.PHR_KEY.map(pl.PHR_TYPE_CD))
    f['AS'] = f.AWP * f.S
    levels = [('own', ['PHR_KEY']), ('chain', ['CHAIN', 'TY']), ('type', ['TY'])]
    def table(paid, per_bucket=True):
        x = f[f[paid].notna()]; out = {}
        for lvl, keys in levels:
            x2 = x[x.CHAIN != 'Independent'] if lvl == 'chain' else x
            g = x2.groupby(keys + (['PRICE_BUCKET'] if per_bucket else [])).agg(n=('AS', 'size'), a=('AS', 'sum'), p=(paid, 'sum'))
            g = g[g.n >= m]
            out[lvl] = (1 - g.p / g.a) if per_bucket else (g.p / g.n)
        return out
    who = pl.assign(CHAIN=pl.PHR_CHAIN_NM, TY=pl.PHR_TYPE_CD)
    used = {}
    def pick(tab, keyf, default):
        vals, how = [], []
        for p, row in who.iterrows():
            for lvl, keys in levels:
                k = keyf(p, row, keys)
                if lvl == 'chain' and row.CHAIN == 'Independent': continue
                if k in tab[lvl].index: vals.append(float(tab[lvl][k])); how.append(lvl); break
            else: vals.append(default); how.append('default')
        return vals, how
    key_b = lambda b: (lambda p, row, keys: tuple([p] if keys == ['PHR_KEY'] else [row[k] for k in keys]) + (b,))
    key_n = lambda p, row, keys: p if keys == ['PHR_KEY'] else (tuple(row[k] for k in keys) if len(keys) > 1 else row[keys[0]])
    clt, phr = table('m_nf_clt_ingred_cost_paid'), table('m_nf_app_ingred_cost_paid')
    for b in BUCKETS:
        mid = float(np.mean(CONFIG['RATE_RANGE'][b]))
        pl[f'CLT_RATE_{b}'], h = pick(clt, key_b(b), mid); used[f'client {b.lower()}'] = pd.Series(h).value_counts().to_dict()
        pl[f'PHR_RATE_{b}'], _ = pick(phr, key_b(b), mid + float(np.mean(CONFIG['PHR_RATE_DEEPER'])))
    # own level tables are keyed by a 1 tuple for rates; fees use plain keys
    cf, pf = table('m_nf_clt_dispensing_fee', False), table('m_nf_app_dispensing_fee', False)
    pl['CLIENT_CONTRACT_DIS_FEE'], h = pick(cf, key_n, float(np.mean(CONFIG['FEE_RANGE']))); used['client fee'] = pd.Series(h).value_counts().to_dict()
    pl['PHR_DISP_FEE_CTRCTD'], _ = pick(pf, key_n, 1.0)
    for c in [f'CLT_RATE_{b}' for b in BUCKETS] + [f'PHR_RATE_{b}' for b in BUCKETS]: pl[c] = pl[c].round(4)
    for c in ('CLIENT_CONTRACT_DIS_FEE', 'PHR_DISP_FEE_CTRCTD'): pl[c] = pl[c].round(2)
    notes['rate_from'] = used

# =============================================================================
# MEMBERS. Location = their main retail pharmacy
# =============================================================================
def members_and_center(df, pl, zips, notes):
    anchor = most_used(df[df.CH == 'RETAIL'], ['MBR_SK'], 'PHR_KEY').set_index('MBR_SK').PHR_KEY
    anchor = anchor[anchor.isin(pl.index)]
    if CONFIG['AREA_ZIP3']:
        keep = anchor[anchor.map(pl.PHR_ZIP_CD).str[:3].isin(CONFIG['AREA_ZIP3'])].index
        notes['area_dropped_members'] = int(df.MBR_SK.nunique() - len(keep))
        df = df[df.MBR_SK.isin(keep)].reset_index(drop=True); anchor = anchor[keep]
        pl = pl[pl.index.isin(df.PHR_KEY) | pl.PHR_ZIP_CD.str[:3].isin(CONFIG['AREA_ZIP3'])]
    notes['members_kept'] = df.MBR_SK.nunique(); notes['admin_scale'] = notes['members_kept'] / notes['members_in_file']
    notes['top_zip3'] = anchor.map(pl.PHR_ZIP_CD).str[:3].value_counts().head(5).to_dict()
    pts = pl.loc[anchor.to_numpy(), ['PHR_LAT', 'PHR_LON']] if len(anchor) else pl[pl.PHR_TYPE_CD == 'RETAIL'][['PHR_LAT', 'PHR_LON']]
    CONFIG['CENTER'] = (round(float(pts.PHR_LAT.median()), 4), round(float(pts.PHR_LON.median()), 4))
    az = area_zips(zips, CONFIG['CENTER'], CONFIG['TIER_MILES']['RURAL'])
    rows = []
    for m in sorted(df.MBR_SK.unique()):
        a = anchor.get(m)
        if a is not None:
            rows.append(dict(MBR_SK=m, MBR_ZIP_CD=pl.at[a, 'PHR_ZIP_CD'], MBR_LAT=pl.at[a, 'PHR_LAT'], MBR_LON=pl.at[a, 'PHR_LON'],
                             MBR_STATE_CD=pl.at[a, 'PHR_ST_CD'], ANCHOR_SITE=a, MBR_LOC_SOURCE='MAIN RETAIL PHARMACY'))
        else:
            z = draw_zip('MBR', m, 'RETAIL', az)
            rows.append(dict(MBR_SK=m, MBR_ZIP_CD=z, MBR_LAT=zips.at[z, 'LAT'], MBR_LON=zips.at[z, 'LON'],
                             MBR_STATE_CD=zips.at[z, 'STATE'], ANCHOR_SITE=None, MBR_LOC_SOURCE='SYNTHETIC ZIP, NO RETAIL FILLS'))
    me = pd.DataFrame(rows)
    me['MBR_URBAN_RURAL_CD'] = [tier_of(a, b) for a, b in zip(me.MBR_LAT, me.MBR_LON)]
    return df, pl, me, az

def footprint_from_zips(az, members):
    """Territory footprint: every standard ZIP within the rural radius of the center, as the model's grid was."""
    fp = pd.DataFrame({'ZIP': az.index, 'ZIP_LAT': az.LAT.to_numpy(), 'ZIP_LON': az.LON.to_numpy(), 'CITY': az.CITY.to_numpy()})
    fp['SETTING'] = [tier_of(a, b) for a, b in zip(fp.ZIP_LAT, fp.ZIP_LON)]
    fp['MEMBERS_TODAY'] = fp.ZIP.map(members.MBR_ZIP_CD.value_counts()).fillna(0).astype(int)
    return fp.reset_index(drop=True)

# =============================================================================
# MODEL TABLES in pnm_data.py column names
# =============================================================================
def site_facts(pl, notes):
    """SYNTHETIC, not asked for: accepts new patients, quality, admin cost. Every pharmacy in the fill file is today's network."""
    rows = []
    for p in pl.itertuples():
        s = rng_for('SITE', p.Index)
        adm = (('RETAIL_INDEP' if p.PHR_CHAIN_NM == 'Independent' else 'RETAIL_CHAIN') if p.PHR_TYPE_CD == 'RETAIL' else p.PHR_TYPE_CD)
        rows.append(dict(PHR_ACCEPT_NEW_PATIENT=int(s.random() > CONFIG['CLOSED_TO_NEW_SHARE']),
                         PHR_QUALITY_SCORE=round(float(s.uniform(*QUALITY[p.PHR_TYPE_CD])), 1),
                         PHR_ADMIN_COST=int(s.uniform(*CONFIG['ADMIN_RANGE'][adm]) * notes['admin_scale'])))
    pl = pl.join(pd.DataFrame(rows, index=pl.index))
    pl['PHR_IN_CURRENT_NTWK'] = pl.IN_FILL_FILE.astype(int)
    return pl

def build_claims(df, pl):
    rate = np.select([df.PRICE_BUCKET == b for b in BUCKETS], [df.PHR_KEY.map(pl[RATE_COL[b]]) for b in BUCKETS])
    est = df.AWP * (1 - rate) * df.PHR_KEY.map(pl.SETTING_FACTOR)
    return pd.DataFrame(dict(
        CLM_SK=df.CLM_SK, MBR_SK=df.MBR_SK, PHR_PMT_CNTR_SK=df.PHR_KEY, GPI_SK=df.NDC11, NDC11=df.NDC11,
        FILLED_DT=df.FILLED_DT, DLVRY_CHANL=df.CH.map(CHAN), CHANNEL_IN_FILE=df.CHANNEL_IN_FILE,
        CLAIM_STAT_ID='PAID', REJ_CNT=0, REVERSAL_IND='N', PRORATED_DAYS_SUPPLY=df.DAYS,
        BRND_TRADE_NM_FLAG=np.where(df.PRICE_BUCKET == 'GENERIC', 'G', 'B'), PRICE_BUCKET=df.PRICE_BUCKET,
        MAIL_ORDR_CORP_IND=(df.CH == 'MAIL').astype(int), RETL_90_CORP_IND=((df.CH == 'RETAIL') & (df.DAYS >= 84)).astype(int),
        SPECIALTY_DRUG_PRG_MBR_IND=((df.PRICE_BUCKET == 'SPECIALTY') | (df.CH == 'INFUSION')).astype(int),
        AWP=df.AWP.round(2), CAL_INGRED_COST_PAID=df.m_nf_clt_ingred_cost_paid.fillna(est).round(2),
        CAL_DISPENSING_FEE=df.m_nf_clt_dispensing_fee.fillna(df.PHR_KEY.map(pl.CLIENT_CONTRACT_DIS_FEE)),
        BENEFIT_TYPE_CD=np.where(df.CH == 'INFUSION', 'MEDICAL', 'RX'))).reset_index(drop=True)

def build_drugs(df):
    mode = lambda c: df.groupby('NDC11')[c].agg(lambda s: s.fillna('').value_counts().index[0])
    d = df.groupby('NDC11').agg(FILLS=('DAYS', 'size'), A=('AWP', 'sum'), D=('DAYS', 'sum'),
                                MEDICAL=('CH', lambda s: 'MEDICAL' if (s == 'INFUSION').any() else 'RX'))
    for c in ('a_nt_gpi', 'a_irpg_drugtype', 'a_nf_pricing_type_combined', 'a_nf_specialty_flag', 'a_nt_maintenance_flag', 'PRICE_BUCKET'):
        d[c] = mode(c)
    d = d.reset_index()
    return pd.DataFrame(dict(GPI_SK=d.NDC11, NDC11=d.NDC11, a_nt_gpi=d.a_nt_gpi, GPI_CLASS_CD=d.a_nt_gpi.str[:2],
                             a_irpg_drugtype=d.a_irpg_drugtype, a_nf_pricing_type_combined=d.a_nf_pricing_type_combined,
                             a_nf_specialty_flag=d.a_nf_specialty_flag, a_nt_maintenance_flag=d.a_nt_maintenance_flag,
                             PRICE_BUCKET=d.PRICE_BUCKET, MAINT_IND=yes(d.a_nt_maintenance_flag).astype(int),
                             SPECIALTY_DRUG_PRG_IND=(d.PRICE_BUCKET == 'SPECIALTY').astype(int), BENEFIT_TYPE_CD=d.MEDICAL,
                             FILLS=d.FILLS, AWP_DAY=(d.A / d.D).round(4)))

def set_capacity(ph, dv):
    """SYNTHETIC: capacity from today's load, so a store can take on more members. A pharmacy nobody uses today gets
    the middle capacity of its type. Mail and home infusion unlimited."""
    load = dv['visits'].merge(dv['today'], on=['MBR_SK', 'KIND']).groupby('TODAY_SITE').FILLS.sum()
    cap = {}
    for p, t, soc in zip(ph.PHR_PMT_CNTR_SK, ph.PHR_TYPE_CD, ph.PHR_SITE_OF_CARE_CD):
        if t == 'MAIL' or soc == 'HOME_INFUSION': cap[p] = 2_000_000
        elif p in load.index: cap[p] = int(max(CONFIG['CAPACITY_FLOOR'], np.ceil(load[p] * rng_for('CAP', p).uniform(*CONFIG['CAPACITY_VS_TODAY']))))
    typ = ph.set_index('PHR_PMT_CNTR_SK').PHR_TYPE_CD; mid = pd.Series(cap).groupby(typ).median()
    ph['PHR_CAPACITY_SCRIPTS'] = [cap.get(p, int(mid.get(t, CONFIG['CAPACITY_FLOOR']))) for p, t in zip(ph.PHR_PMT_CNTR_SK, ph.PHR_TYPE_CD)]

# =============================================================================
# DERIVATIONS. Same rules as derive() in pnm_data.py: visits = distinct fill dates, fills = count, today's site = the
# one used most, mail and 90 day flags = any such fill, growth trend by drug class. Two changes because the data is
# real: AWP per day is read from the file, and SUPPLY is the member's own days per fill. Volumes are scaled to a year.
# =============================================================================
def derive_real(members, ph, claims, drugs, annual):
    tr = claims.GPI_SK.map(drugs.set_index('GPI_SK').GPI_CLASS_CD).map(CONFIG['TREND_GPI2']).fillna(1.0)
    c = claims.assign(KIND=claims.DLVRY_CHANL.map(KIND), TREND=tr.to_numpy())
    g = (c.assign(DT=c.PRORATED_DAYS_SUPPLY * c.TREND).groupby(['MBR_SK', 'GPI_SK', 'PRICE_BUCKET', 'KIND'])
          .agg(DT=('DT', 'sum'), D=('PRORATED_DAYS_SUPPLY', 'sum'), A=('AWP', 'sum'), N=('CLM_SK', 'size')).reset_index())
    demand = g.assign(DAYS=g.DT * annual, AWP_DAY=g.A / g.D, SUPPLY=g.D / g.N)[
        ['MBR_SK', 'GPI_SK', 'PRICE_BUCKET', 'DAYS', 'AWP_DAY', 'KIND', 'SUPPLY']]
    visits = (c.groupby(['MBR_SK', 'KIND']).agg(VISITS=('FILLED_DT', 'nunique'), FILLS=('CLM_SK', 'size'), TREND=('TREND', 'mean'))
                .reset_index())
    visits['VISITS'] = (visits.VISITS * visits.TREND * annual).round(1); visits['FILLS'] = (visits.FILLS * visits.TREND * annual).round(1)
    today = most_used(c, ['MBR_SK', 'KIND'], 'PHR_PMT_CNTR_SK').rename(columns={'PHR_PMT_CNTR_SK': 'TODAY_SITE'})
    me, pl = members.set_index('MBR_SK'), ph.set_index('PHR_PMT_CNTR_SK')
    d0 = miles(today.MBR_SK.map(me.MBR_LAT), today.MBR_SK.map(me.MBR_LON), today.TODAY_SITE.map(pl.PHR_LAT), today.TODAY_SITE.map(pl.PHR_LON))
    today['D0'] = np.where(today.TODAY_SITE.map(pl.PHR_TYPE_CD) == 'MAIL', 0.0, d0)
    flags = c.groupby('MBR_SK').agg(MAIL_TODAY=('MAIL_ORDR_CORP_IND', 'max'), NINETY_TODAY=('RETL_90_CORP_IND', 'max')).reset_index()
    flags['NINETY_TODAY'] = ((flags.NINETY_TODAY == 1) | (flags.MAIL_TODAY == 1)).astype(int)
    multi = c.groupby(['MBR_SK', 'KIND']).PHR_PMT_CNTR_SK.nunique().rename('n_sites').reset_index()
    return dict(demand=demand, visits=visits, today=today, flags=flags, multi=multi)

# =============================================================================
# BUILD, REPORT, EXPORT
# =============================================================================
def build():
    zips = load_zips()
    stamp(f"reading {CONFIG['REAL_FILE']}")
    df, notes = read_fills()
    df = sample_members(df, notes)
    first, last = df.FILLED_DT.min(), df.FILLED_DT.max()
    notes.update(first=first, last=last, period_days=int((last - first).days) + 1)
    annual = 365.0 / notes['period_days'] if CONFIG['ANNUALIZE'] else 1.0
    notes['annual'] = annual
    stamp('drugs, pharmacies, members')
    df, ptype = pharmacy_types(df, notes)
    df = drug_fields(df, notes)
    pl = pharmacy_list(df, ptype, zips, notes)
    lost = ~df.PHR_KEY.isin(pl.index)
    notes['dropped']['pharmacy with no location'] = int(lost.sum()); df = df[~lost].reset_index(drop=True)
    df, pl, members, az = members_and_center(df, pl, zips, notes)
    df = paid_fields(df, pl, notes)
    derive_rates(df, pl, notes)
    pl = site_facts(pl, notes)
    ph = pl.rename_axis('PHR_PMT_CNTR_SK').reset_index().rename(columns={'PHR_NPI_NO': 'PHR_NPI'})
    ph['PHR_NPI_NO'] = ph.PHR_NPI; ph['PHR_STATE_CD'] = ph.PHR_ST_CD
    claims = build_claims(df, pl)
    drugs = build_drugs(df)
    footprint = footprint_from_zips(az, members)
    stamp('deriving from claims')
    dv = derive_real(members, ph, claims, drugs, annual)
    set_capacity(ph, dv)
    ff = notes.pop('_file_flags')
    if len(ff.columns) > 1:
        ff = ff[ff.MBR_SK.isin(members.MBR_SK)].groupby('MBR_SK').max()
        mine = dv['flags'].set_index('MBR_SK').loc[ff.index]
        notes['flag_agree'] = {n: round(100 * float((ff[k].fillna(0).astype(int) == mine[n]).mean()), 1)
                               for k, n in (('MAIL_FLAG', 'MAIL_TODAY'), ('NINETY_FLAG', 'NINETY_TODAY')) if k in ff}
    me = members.set_index('MBR_SK')
    df['Distance_Placeholder'] = miles(df.MBR_SK.map(me.MBR_LAT), df.MBR_SK.map(me.MBR_LON),
                                       df.PHR_KEY.map(pl.PHR_LAT), df.PHR_KEY.map(pl.PHR_LON)).round(2)
    order = ['PHR_PMT_CNTR_SK', 'PHR_TYPE_CD', 'PHR_CHAIN_NM', 'IN_FILL_FILE', 'ROW_SOURCE'] + PHARMACY_ASKED + [
             'ZIP_SOURCE', 'LOCATION_SOURCE', 'PHR_LAT', 'PHR_LON', 'PHR_STATE_CD', 'PHR_NPI']
    ph = ph[order + [c for c in ph.columns if c not in order]]
    return dict(drugs=drugs, members=members, pharmacies=ph, claims=claims, footprint=footprint, fills_in=df,
                notes=notes, **dv)

def report_real(D):
    n, ph, me, cl = D['notes'], D['pharmacies'], D['members'], D['claims']
    print("\n================ REAL DATA ================")
    print(f"file {n['file']} | rows {n['rows']:,} | fills kept {len(cl):,} | member key {' + '.join(n['member_key'])}")
    left = {k: v for k, v in n['dropped'].items() if v}
    if left: print('  left out: ' + ' | '.join(f'{k} {v:,}' for k, v in left.items()))
    if n['other_channels']: print(f"  channels left out, not retail, mail, specialty or infusion: {n['other_channels']}")
    print(f"fills dated {n['first']:%d %b %Y} to {n['last']:%d %b %Y}, {n['period_days']} days: "
          + (f"volumes scaled x{n['annual']:.2f} to a year" if n['annual'] != 1 else 'volumes as in the file'))
    print(f"members {len(me):,}" + (f" of the {n['members_in_file']:,} in the file (sample or area). Admin cost scaled "
                                    f"x{n['admin_scale']:.2f} to match" if n['admin_scale'] < 1 else ', all in the file'))
    if n.get('area_dropped_members'): print(f"  outside AREA_ZIP3 {CONFIG['AREA_ZIP3']}: {n['area_dropped_members']:,} members left out")
    print(f"area center {CONFIG['CENTER']} | members by main pharmacy's 3 digit ZIP: {n['top_zip3']}")
    print(f"pharmacies {len(ph):,} {ph.PHR_TYPE_CD.value_counts().to_dict()} | used by our members {int(ph.IN_FILL_FILE.sum()):,} "
          f"| others, candidates to add {int((ph.IN_FILL_FILE == 0).sum()):,} {ph[ph.IN_FILL_FILE == 0].ROW_SOURCE.value_counts().to_dict()}")
    print(f"  fills moved to their pharmacy's main channel {n['fills_relabelled']:,} | ZIP source {ph.ZIP_SOURCE.value_counts().to_dict()}"
          + (f" | no location, left out {n['pharmacies_no_location']}" if n['pharmacies_no_location'] else ''))
    print(f"members per retail pharmacy used {len(me) / max(1, int(((ph.PHR_TYPE_CD == 'RETAIL') & (ph.IN_FILL_FILE == 1)).sum())):.1f} "
          f"(the synthetic design has about {3000 / sum(CONFIG['RETAIL_PER_1000'].values()):.0f})")
    rc = ph[(ph.PHR_TYPE_CD == 'RETAIL') & (ph.IN_FILL_FILE == 1)].PHR_CHAIN_NM.value_counts()
    print(f"retail chains: {rc.drop('Independent', errors='ignore').head(8).to_dict()} | independents {int(rc.get('Independent', 0))}")
    print(f"member location: {me.MBR_LOC_SOURCE.value_counts().to_dict()}")
    print(f"price buckets, % of fills: {cl.PRICE_BUCKET.value_counts(normalize=True).mul(100).round(1).to_dict()} | "
          f"a_irpg_drugtype values {n['drugtype_values']}" + (f" | unreadable, bucket guessed from AWP per day: {n['drugtype_unread']:,}"
                                                             if n['drugtype_unread'] else ''))
    print(f"rates from own fills, the chain's, the type's or the default: {n['rate_from']}")
    if n.get('flag_agree'): print(f"file's member flags vs the code's rules, % of members agreeing: {n['flag_agree']}")
    fs, ps = n['fill_source'], n['pharmacy_source']
    print(f"fill file, REAL:      {[c for c in FILL_ASKED if fs[c] == 'REAL'] or 'none yet'}")
    print(f"fill file, SYNTHETIC: {[c for c in FILL_ASKED if fs[c] == 'SYNTHETIC'] or 'none'}")
    print(f"pharmacy list{' ' + n['pharmacy_list'] if n.get('pharmacy_list') else ', not in the folder yet'}: {ps}")
    print("always SYNTHETIC: capacity, quality, accepts new patients, admin cost")

def export(D):
    d = CONFIG['EXPORT_DIR']; os.makedirs(d, exist_ok=True)
    for k in ('members', 'pharmacies', 'drugs', 'claims'): D[k].to_csv(f'{d}/{k}.csv', index=False)
    D['footprint'].to_csv(f'{d}/footprint_zips.csv', index=False)
    for k in ('demand', 'visits', 'today', 'flags'): D[k].to_csv(f'{d}/derived_{k}.csv', index=False)
    # the inputs with every asked column filled in, exact names. The pharmacy list doubles as the template for Angelica
    C, f = CONFIG['FILL_COLS'], D['fills_in']
    fills = pd.DataFrame({C['MBR']: f.MBR_SK, C['NDC']: f.NDC11, C['DATE']: f.FILLED_DT, C['KIND']: f.CHANNEL_IN_FILE,
                          C['DAYS']: f.DAYS, C['AWP']: f.AWP, C['PHR_NAME']: f.PHR_NM, C['NCPDP']: f.NCPDP, C['NPI']: f.NPI,
                          'Distance_Placeholder': f.Distance_Placeholder, **{c: f[c] for c in FILL_ASKED}})
    fills.to_csv(f'{d}/input_fills_completed.csv', index=False)
    D['pharmacies'][['ROW_SOURCE', 'ZIP_SOURCE'] + PHARMACY_ASKED].to_csv(f'{d}/input_pharmacy_list.csv', index=False)
    n = D['notes']
    pd.DataFrame([dict(table='fill file', column=c, source=n['fill_source'][c]) for c in FILL_ASKED] +
                 [dict(table='pharmacy list', column=c, source=n['pharmacy_source'][c]) for c in PHARMACY_ASKED]
                 ).to_csv(f'{d}/column_sources.csv', index=False)
    stamp(f'exports written to {d}/')

def load_data(verbose=True):
    D = build()
    if verbose: report_real(D); report(D)   # report() from pnm_data.py also sets each footprint ZIP's adequacy rule
    return D

if __name__ == '__main__':
    D = load_data(); export(D)
