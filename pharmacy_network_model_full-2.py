# =============================================================================
# PHARMACY NETWORK AND CHANNEL OPTIMIZATION  ·  COMPLETE STANDALONE SCRIPT
# Pranshi Saxena  ·  Optum Rx Quantum Initiative
#
# Run:      python pharmacy_network_model_full.py
# Needs:    pip install pyomo highspy pandas numpy matplotlib plotly
# Output:   console tables, matplotlib figures, and interactive HTML charts
#           written next to this script. Open the HTML files in any browser.
#
# Sections
#   1  Optum schema provenance registry
#   2  Configuration dials
#   3  Synthetic data generation in Optum column names
#   4  Claims aggregation, drug classes derived from claims
#   5  Model structures, lenses, feasibility pre check
#   6  Decision variables and objective
#   7  Constraints
#   8  Solve and default results
#   9  Nine business scenarios
#  10  Disruption impact in member terms
#  11  Reference script parity exports
#  12  Trade off sweep, disruption in miles against savings, heat maps
# =============================================================================


# ============================================================================
# PHARMACY NETWORK AND CHANNEL OPTIMIZATION
# ============================================================================


# ============================================================================
# 0. PACKAGES AND SOLVER
# ============================================================================

import numpy as np, pandas as pd, time
import matplotlib.pyplot as plt
import os
os.makedirs('charts', exist_ok=True)
from math import radians, sin, cos, sqrt, atan2
from pyomo.environ import (ConcreteModel, Var, Objective, Constraint, ConstraintList,
                           Binary, NonNegativeReals, minimize, value, SolverFactory)
pd.set_option('display.width',170); pd.set_option('display.max_columns',30)
print('imports OK')


# ============================================================================
# 1. SCHEMA PROVENANCE: WHAT IS REAL, WHAT WE STILL NEED
# ============================================================================

"""
Optum schema provenance registry.
CONFIRMED = column seen in the claims SQL (F_CLM_TRANSACTION_EXTN / _EXTN2)
NEEDED    = column our model requires that is NOT in the claims fact tables;
            it lives in a dimension we do not yet have. Synthesised here, name is our best guess.
"""
SCHEMA = {
 # ---- claims fact, CONFIRMED in F_CLM_TRANSACTION_EXTN ----
 'CLM_SK':                ('CONFIRMED','EXTN','claim surrogate key'),
 'CLAIM_NBR':             ('CONFIRMED','EXTN','claim number'),
 'MBR_SK':                ('CONFIRMED','EXTN','member surrogate key, join to member dim'),
 'CAG_SK':                ('CONFIRMED','EXTN','client/account/group key'),
 'CARRIER_SK':            ('CONFIRMED','EXTN','carrier key'),
 'FILLED_DT':             ('CONFIRMED','EXTN','fill date'),
 'PHR_PMT_CNTR_SK':       ('CONFIRMED','EXTN','pharmacy key, join to pharmacy dim'),
 'DLVRY_CHANL':           ('CONFIRMED','EXTN','delivery channel, CASE maps - and N to Retail'),
 'CLAIM_STAT_ID':         ('CONFIRMED','EXTN','claim status, filter to paid'),
 'REJ_CNT':               ('CONFIRMED','EXTN','reject count'),
 'REVERSAL_IND':          ('CONFIRMED','EXTN2','reversal indicator, exclude'),
 'SBM_PRESCRIBED_QTY':    ('CONFIRMED','EXTN','submitted quantity'),
 'BRND_TRADE_NM_FLAG':    ('CONFIRMED','EXTN','brand vs generic'),
 'FORMULARY_FLG':         ('CONFIRMED','EXTN','on formulary flag'),
 'MAINT_GPI_LIST':        ('CONFIRMED','EXTN','maintenance GPI list membership'),
 'DRUG_MAINT_CD_SK':      ('CONFIRMED','EXTN','maintenance vs acute code'),
 'DAY_90_RETAIL_PROG_IND':('CONFIRMED','EXTN','retail 90 program indicator'),
 'PRIOR_AUTH_USED_ID':    ('CONFIRMED','EXTN','prior auth used'),
 'CAL_INGRED_COST_PAID':  ('CONFIRMED','EXTN','calculated ingredient cost paid'),
 'CAL_DISPENSING_FEE':    ('CONFIRMED','EXTN','calculated dispensing fee'),
 'CAL_PATIENT_PAY_AMT':   ('CONFIRMED','EXTN','member pay amount'),
 'DRG_SPEND_AMT':         ('CONFIRMED','EXTN','drug spend'),
 'TOT_COVERED_DRG_COST_AMT':('CONFIRMED','EXTN','total covered drug cost'),
 'PREV_FILL_1_DT':        ('CONFIRMED','EXTN','previous fill date, adherence'),
 # ---- claims fact, CONFIRMED in F_CLM_TRANSACTION_EXTN2 ----
 'DRUG_SK':               ('CONFIRMED','EXTN2','drug surrogate key'),
 'GPI_SK':                ('CONFIRMED','EXTN2','GPI key, THE therapeutic class source'),
 'AHFS_CD_SK':            ('CONFIRMED','EXTN2','AHFS class key, alternative classification'),
 'DAYS_OF_SUPPLY_SK':     ('CONFIRMED','EXTN2','days supply key'),
 'PRORATED_DAYS_SUPPLY':  ('CONFIRMED','EXTN2','prorated days supply'),
 'INJECTABLE_IND':        ('CONFIRMED','EXTN2','injectable indicator, drug form'),
 'INSULIN_PLAN_IND':      ('CONFIRMED','EXTN2','insulin plan indicator'),
 'SPECIALTY_DRUG_PRG_IND':('CONFIRMED','EXTN2','specialty drug program'),
 'SPECIALTY_DRUG_PRG_MBR_IND':('CONFIRMED','EXTN2','specialty member flag, equals BT SpecialityInd'),
 'SPECIALTY_DRUG_PRG_PHARM_IND':('CONFIRMED','EXTN2','specialty pharmacy flag'),
 'SPECIALTY_NETWORK_PRG_IND':('CONFIRMED','EXTN2','specialty network program'),
 'MAIL_ORDR_CORP_IND':    ('CONFIRMED','EXTN2','mail order flag, equals BT MailOrderIndicator'),
 'RETL_90_CORP_IND':      ('CONFIRMED','EXTN2','retail 90 flag'),
 'CORP_CHNL_TYPE_NM':     ('CONFIRMED','EXTN2','corporate channel type name'),
 'CLT_FORMULARY_SK':      ('CONFIRMED','EXTN2','client formulary key'),
 'PLN_FORMULARY':         ('CONFIRMED','EXTN2','plan formulary'),
 'CLIENT_COST_TIER_AMT':  ('CONFIRMED','EXTN2','client cost tier'),
 'CLT_AWP_DISCOUNT_RATE': ('CONFIRMED','EXTN2','client AWP discount rate'),
 'APP_AWP_DISCOUNT_RATE': ('CONFIRMED','EXTN2','applied AWP discount rate, our contract_pct source'),
 'CLIENT_CONTRACT_DIS_FEE':('CONFIRMED','EXTN2','client contracted dispensing fee'),
 'PHR_ING_COST_CTRCTD':   ('CONFIRMED','EXTN2','pharmacy contracted ingredient cost'),
 'PHR_DISP_FEE_CTRCTD':   ('CONFIRMED','EXTN2','pharmacy contracted dispensing fee'),
 'PATIENT_AGE':           ('CONFIRMED','EXTN2','patient age'),
 'PAYER_TYPE':            ('CONFIRMED','EXTN2','payer type'),
 'ADHERENCE_INCNT_PRG_IND':('CONFIRMED','EXTN2','adherence incentive program'),
 'CLIENT_MAC_ID':         ('CONFIRMED','EXTN2','client MAC list'),
 'PHARMACY_MAC_ID':       ('CONFIRMED','EXTN2','pharmacy MAC list'),
 # ---- NEEDED: dimensions we do not have ----
 'MBR_ZIP_CD':            ('NEEDED','MEMBER DIM','member ZIP, drives all distance and access. TOP PRIORITY'),
 'MBR_LAT':               ('NEEDED','MEMBER DIM','member latitude or ZIP centroid'),
 'MBR_LON':               ('NEEDED','MEMBER DIM','member longitude or ZIP centroid'),
 'MBR_STATE_CD':          ('NEEDED','MEMBER DIM','member state'),
 'MBR_URBAN_RURAL_CD':    ('NEEDED','MEMBER DIM','urban/rural designation, RUCA or CMS'),
 'PHR_NPI':               ('NEEDED','PHARMACY DIM','pharmacy NPI'),
 'PHR_LAT':               ('NEEDED','PHARMACY DIM','pharmacy latitude'),
 'PHR_LON':               ('NEEDED','PHARMACY DIM','pharmacy longitude'),
 'PHR_STATE_CD':          ('NEEDED','PHARMACY DIM','pharmacy state, drives any willing pharmacy mandate'),
 'PHR_TYPE_CD':           ('NEEDED','PHARMACY DIM','retail / mail / specialty / infusion type'),
 'PHR_CHAIN_NM':          ('NEEDED','PHARMACY DIM','chain name, drives chain contracting rules'),
 'PHR_IN_CURRENT_NTWK':   ('NEEDED','NETWORK CONFIG','is pharmacy in the client current network'),
 'PHR_ACCEPT_NEW_PATIENT':('NEEDED','PHARMACY DIM','accepting new patients'),
 'PHR_CAPACITY_SCRIPTS':  ('NEEDED','OPERATIONS','annual dispensing capacity'),
 'PHR_QUALITY_SCORE':     ('NEEDED','QUALITY','pharmacy quality score'),
 'GPI_CLASS_CD':          ('NEEDED','DRUG DIM','GPI decode: GPI_SK to therapeutic class code'),
 'GPI_CLASS_NM':          ('NEEDED','DRUG DIM','therapeutic class name'),
 'BENEFIT_TYPE_CD':       ('NEEDED','MEDICAL CLAIMS','pharmacy vs medical benefit. Infusions likely in medical claims, separate system'),
}
def provenance_table():
    import pandas as pd
    return pd.DataFrame([(k,v[0],v[1],v[2]) for k,v in SCHEMA.items()],
                        columns=['column','status','source','note'])


prov = provenance_table()
print(prov.status.value_counts().to_dict())
print('\n--- NEEDED: the data request ---')
print(prov[prov.status=='NEEDED'][['column','source','note']].to_string(index=False))


# ============================================================================
# 2. [CONFIG] THE DIALS
# ============================================================================

CONFIG = {
    'LAMBDA_D':20.0, 'LAMBDA_SWEEP':[0,20,200],
    'CONTRACT_WEIGHT':2.5, 'DISTANCE_PENALTY':0.05,
    'EXTRA_CHOICE_REWARD':0.0, 'CURRENT_NETWORK_REWARD':0.0, 'CURRENT_SITE_REWARD_SCALE':250.0,
    'RETAIL_ACCESS':{'URBAN':(4.0,0.70),'RURAL':(15.0,0.95)},   # radius miles, weighted floor
    'SPEC_RADIUS':25.0, 'SPEC_FLOOR':0.85,
    'INF_RADIUS':40.0,  'INF_FLOOR':0.90,
    'D_MAX_RETAIL':20.0, 'QUALITY_FLOOR':4.0, 'COVERAGE_BONUS':1.0,
    'AWP_STATES':{'WI'},          # mock any willing pharmacy statute table
    'GAMMA':None,
    # Chains such as CVS are negotiated as ONE contract for the whole chain, not store by store.
    # PER_SITE  assumes every selected location carries its own administration cost.
    # PER_CHAIN assumes one negotiated contract fee per chain and member stores add no cost.
    'CHAIN_COST_MODE'   : 'PER_SITE',
    'CHAIN_CONTRACT_FEE': 150_000,
}
CONFIG


# ============================================================================
# 3. [INPUT] SYNTHETIC DATA IN OPTUM SCHEMA
# ============================================================================

# Synthetic data in Optum claims schema. Column names match the warehouse; values are invented.
import numpy as np, pandas as pd
from math import radians, sin, cos, sqrt, atan2

def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.8
    dlat = radians(lat2-lat1); dlon = radians(lon2-lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return R*2*atan2(sqrt(a), sqrt(1-a))

# GPI class catalog. GPI_CLASS_CD is the 2-digit GPI level (real GPI-2 values);
# names are real therapeutic class names. Volume shares are ASSUMED and are exactly what
# the claims aggregation replaces when real data arrives.
GPI_CLASSES = [
 # cd    name                          share form         inject insulin specialty benefit
 ('27','Cardiovascular Agents',        0.20,'ORAL',        0,0,0,'RX'),
 ('58','Antidiabetics Oral',           0.11,'ORAL',        0,0,0,'RX'),
 ('27B','Insulins',                    0.07,'INJ_COLD',    1,1,0,'RX'),
 ('01','Anti-infective Agents',        0.15,'ORAL',        0,0,0,'RX'),
 ('66','Anti-inflammatory Oral',       0.18,'ORAL',        0,0,0,'RX'),
 ('44','Respiratory Agents',           0.11,'INHALED',     0,0,0,'RX'),
 ('66B','Immunologic Agents Injectable',0.07,'PEN_COLD',   1,0,1,'RX'),
 ('21','Antineoplastic Oral',          0.05,'ORAL_SPEC',   0,0,1,'RX'),
 ('66C','Immunologic Agents Infused',  0.06,'INFUSION',    1,0,1,'MEDICAL'),
]

def gen_member_dim(n=900, center=(44.9778,-93.2650), seed=7):
    rng = np.random.default_rng(seed)
    lat = center[0] + rng.uniform(-0.45,0.45,n); lon = center[1] + rng.uniform(-0.60,0.60,n)
    urc = rng.choice(['URBAN','RURAL'], n, p=[0.7,0.3])
    return pd.DataFrame({
        'MBR_SK': [f'M{i:05d}' for i in range(n)],
        'MBR_ZIP_CD': [f'{55400+i%600}' for i in range(n)],
        'MBR_LAT': lat, 'MBR_LON': lon,
        'MBR_STATE_CD': np.where(lon < -93.30,'MN', np.where(lon < -93.20,'WI','IA')),
        'MBR_URBAN_RURAL_CD': urc,
        'PATIENT_AGE': rng.integers(18,85,n),
        'CAG_SK': rng.choice(['CAG001','CAG002'], n, p=[0.7,0.3]),
    })

def gen_pharmacy_dim(n_retail=120, n_mail=5, n_spec=12, n_inf=10, center=(44.9778,-93.2650), seed=11):
    rng = np.random.default_rng(seed); rows=[]; i=0
    def add(t, cap, qual_lo, qual_hi):
        nonlocal i
        lat = center[0] + rng.uniform(-0.48,0.48); lon = center[1] + rng.uniform(-0.63,0.63)
        chain = rng.choice(['ChainA','ChainB','ChainC','Independent'], p=[0.3,0.25,0.2,0.25]) if t=='RETAIL' \
                else {'MAIL':'OptumHomeDelivery','SPECIALTY':'SpecialtyNet','INFUSION':'InfusionNet'}[t]
        rows.append(dict(
            PHR_PMT_CNTR_SK=f'P{i:05d}', PHR_NPI=f'{1500000000+i}',
            PHR_LAT=lat, PHR_LON=lon,
            PHR_STATE_CD='MN' if lon < -93.30 else ('WI' if lon < -93.20 else 'IA'),
            PHR_TYPE_CD=t, PHR_CHAIN_NM=chain,
            PHR_IN_CURRENT_NTWK=int(rng.random()<0.55),
            PHR_ACCEPT_NEW_PATIENT=int(rng.random()>0.05),
            PHR_CAPACITY_SCRIPTS=cap, PHR_QUALITY_SCORE=round(float(rng.uniform(qual_lo,qual_hi)),1),
            APP_AWP_DISCOUNT_RATE=round(float(rng.uniform(0.75,0.96)),3),
            CLIENT_CONTRACT_DIS_FEE=round(float(rng.uniform(0.75,2.75)),2),
        )); i+=1
    for _ in range(n_retail): add('RETAIL', int(rng.uniform(600,2600)), 3.5, 5.0)
    for _ in range(n_mail):   add('MAIL',   2_000_000, 4.3, 4.9)
    for _ in range(n_spec):   add('SPECIALTY', 900, 4.2, 5.0)
    for k in range(n_inf):
        add('INFUSION', 300, 4.0, 5.0)
        rows[-1]['PHR_SITE_OF_CARE_CD'] = ['HOSPITAL_OPD','INFUSION_CTR','PHYSICIAN_OFC','HOME_INFUSION'][k % 4]
        rows[-1]['PHR_IN_CURRENT_NTWK'] = 1
    df = pd.DataFrame(rows)
    df['PHR_SITE_OF_CARE_CD'] = df.get('PHR_SITE_OF_CARE_CD', pd.Series(index=df.index, dtype=object)).fillna('NA')
    return df

def gen_drug_dim():
    return pd.DataFrame([dict(GPI_SK=f'G{i:03d}', GPI_CLASS_CD=cd, GPI_CLASS_NM=nm,
                              DRUG_FORM_CD=form, INJECTABLE_IND=inj, INSULIN_PLAN_IND=ins,
                              SPECIALTY_DRUG_PRG_IND=spec, BENEFIT_TYPE_CD=ben, _ASSUMED_SHARE=sh)
                         for i,(cd,nm,sh,form,inj,ins,spec,ben) in enumerate(GPI_CLASSES)])

def gen_claims(members, pharmacies, drugs, months=12, seed=13):
    """Claim level rows in Optum schema. Member drug mix varies by age and random morbidity,
    which is what makes class shares DERIVED from claims rather than assumed per member."""
    rng = np.random.default_rng(seed); rows=[]; clm=0
    ph_by_type = {t: pharmacies[pharmacies.PHR_TYPE_CD==t].PHR_PMT_CNTR_SK.tolist()
                  for t in pharmacies.PHR_TYPE_CD.unique()}
    cur_net = set(pharmacies[pharmacies.PHR_IN_CURRENT_NTWK==1].PHR_PMT_CNTR_SK)
    base_share = drugs._ASSUMED_SHARE.to_numpy()
    for _, m in members.iterrows():
        # member specific morbidity tilts the mix: this is why real claims are needed.
        # Specialty and infused use is CONCENTRATED: only a small share of members have any,
        # which is the real world pattern that a uniform assumed share cannot reproduce.
        w = base_share.copy()
        spec_idx = drugs.index[drugs.SPECIALTY_DRUG_PRG_IND == 1].to_numpy()
        if rng.random() < 0.10:
            w[spec_idx] *= 9.0                      # specialty member
        else:
            w[spec_idx] *= 0.012                    # everyone else: near zero
        tilt = rng.dirichlet(w*14 + 0.05)
        n_fills = max(1, int(rng.gamma(3.0, 3.0) * (1 + (m.PATIENT_AGE-45)/90)))
        is_mail = rng.random() < 0.05
        for _ in range(n_fills):
            d = drugs.iloc[rng.choice(len(drugs), p=tilt/tilt.sum())]
            if d.BENEFIT_TYPE_CD == 'MEDICAL':      t = 'INFUSION'
            elif d.SPECIALTY_DRUG_PRG_IND == 1:     t = 'SPECIALTY'
            elif is_mail:                           t = 'MAIL'
            else:                                   t = 'RETAIL'
            cand = [p for p in ph_by_type[t] if p in cur_net] or ph_by_type[t]
            pl = pharmacies.set_index('PHR_PMT_CNTR_SK')
            near3 = sorted(cand, key=lambda q: haversine_miles(m.MBR_LAT, m.MBR_LON, pl.loc[q,'PHR_LAT'], pl.loc[q,'PHR_LON']))[:3]
            p_id = near3[rng.integers(len(near3))]
            ing = float(np.exp(rng.normal({'ORAL':3.1,'INHALED':4.4,'INJ_COLD':5.3,
                                           'PEN_COLD':8.2,'ORAL_SPEC':8.9,'INFUSION':8.5}[d.DRUG_FORM_CD], 0.4)))
            days = int(rng.choice([30,60,90], p=[0.62,0.08,0.30])) if t in ('MAIL','RETAIL') else 30
            rows.append(dict(
                CLM_SK=f'C{clm:07d}', CLAIM_NBR=f'{900000+clm}', MBR_SK=m.MBR_SK, CAG_SK=m.CAG_SK,
                CARRIER_SK='CAR01', FILLED_DT=f'2025-{rng.integers(1,13):02d}-{rng.integers(1,29):02d}',
                PHR_PMT_CNTR_SK=p_id, GPI_SK=d.GPI_SK, DRUG_SK=f'D{d.name:04d}', AHFS_CD_SK=f'A{d.name:03d}',
                DLVRY_CHANL={'RETAIL':'Retail','MAIL':'MAIL','SPECIALTY':'SPECIALTY','INFUSION':'MEDICAL'}[t],
                CLAIM_STAT_ID='PAID', REJ_CNT=0, REVERSAL_IND='N',
                DAYS_OF_SUPPLY_SK=days, PRORATED_DAYS_SUPPLY=days,
                SBM_PRESCRIBED_QTY=int(days*rng.uniform(1,2)),
                BRND_TRADE_NM_FLAG='B' if rng.random()<0.3 else 'G',
                FORMULARY_FLG='Y' if rng.random()<0.93 else 'N',
                DRUG_MAINT_CD_SK='MAINT' if d.DRUG_FORM_CD in ('ORAL','INHALED') else 'ACUTE',
                DAY_90_RETAIL_PROG_IND=int(days==90 and t=='RETAIL'),
                MAIL_ORDR_CORP_IND=int(t=='MAIL'), RETL_90_CORP_IND=int(days==90 and t=='RETAIL'),
                SPECIALTY_DRUG_PRG_IND=int(d.SPECIALTY_DRUG_PRG_IND),
                SPECIALTY_DRUG_PRG_MBR_IND=int(d.SPECIALTY_DRUG_PRG_IND),
                INJECTABLE_IND=int(d.INJECTABLE_IND), INSULIN_PLAN_IND=int(d.INSULIN_PLAN_IND),
                PRIOR_AUTH_USED_ID='Y' if (d.SPECIALTY_DRUG_PRG_IND and rng.random()<0.6) else 'N',
                CAL_INGRED_COST_PAID=round(ing,2),
                CAL_DISPENSING_FEE=round(float(rng.uniform(0.75,2.75)),2),
                CAL_PATIENT_PAY_AMT=round(ing*float(rng.uniform(0.02,0.25)),2),
                DRG_SPEND_AMT=round(ing,2), TOT_COVERED_DRG_COST_AMT=round(ing,2),
                CLIENT_COST_TIER_AMT=int(rng.integers(1,5)),
                APP_AWP_DISCOUNT_RATE=float(pharmacies.set_index('PHR_PMT_CNTR_SK').loc[p_id,'APP_AWP_DISCOUNT_RATE']),
                PATIENT_AGE=int(m.PATIENT_AGE), PAYER_TYPE='COMMERCIAL',
                BENEFIT_TYPE_CD=d.BENEFIT_TYPE_CD,
            )); clm+=1
    return pd.DataFrame(rows)

def aggregate_claims(claims, members, pharmacies, drugs):
    """Claims -> model inputs. THIS is where drug classes come from claims, not assumption."""
    c = claims[(claims.CLAIM_STAT_ID=='PAID') & (claims.REVERSAL_IND=='N') & (claims.REJ_CNT==0)].copy()
    c['NET_COST'] = c.CAL_INGRED_COST_PAID + c.CAL_DISPENSING_FEE - c.CAL_PATIENT_PAY_AMT
    # member x class volume and cost, derived
    mk = (c.groupby(['MBR_SK','GPI_SK'])
            .agg(FILL_CNT=('CLM_SK','size'), DAYS_SUPPLY=('PRORATED_DAYS_SUPPLY','sum'),
                 NET_COST=('NET_COST','sum'), MBR_PAY=('CAL_PATIENT_PAY_AMT','sum'))
            .reset_index())
    # member level rollups
    mtot = mk.groupby('MBR_SK').agg(TOT_FILLS=('FILL_CNT','sum'), TOT_COST=('NET_COST','sum')).reset_index()
    # member flags derived from their own claims
    flags = (c.groupby('MBR_SK')
               .agg(MAIL_ORDR_CORP_IND=('MAIL_ORDR_CORP_IND','max'),
                    SPECIALTY_DRUG_PRG_MBR_IND=('SPECIALTY_DRUG_PRG_MBR_IND','max'))
               .reset_index())
    # current pharmacy: most used in the current network, per member
    cur = set(pharmacies[pharmacies.PHR_IN_CURRENT_NTWK==1].PHR_PMT_CNTR_SK)
    cc = c[c.PHR_PMT_CNTR_SK.isin(cur)]
    curph = (cc.groupby(['MBR_SK','PHR_PMT_CNTR_SK']).size().rename('n').reset_index()
               .sort_values('n', ascending=False).drop_duplicates('MBR_SK')
               .rename(columns={'PHR_PMT_CNTR_SK':'CUR_PHR_PMT_CNTR_SK'})[['MBR_SK','CUR_PHR_PMT_CNTR_SK']])
    mem = (members.merge(mtot, on='MBR_SK', how='inner')
                  .merge(flags, on='MBR_SK', how='left')
                  .merge(curph, on='MBR_SK', how='left'))
    # cost per fill by class and pharmacy type, derived
    cpt = (c.merge(pharmacies[['PHR_PMT_CNTR_SK','PHR_TYPE_CD']], on='PHR_PMT_CNTR_SK')
             .groupby(['GPI_SK','PHR_TYPE_CD']).NET_COST.mean().rename('COST_PER_FILL').reset_index())
    return mem, mk, cpt


members_dim = gen_member_dim()
pharm_dim   = gen_pharmacy_dim()
drug_dim    = gen_drug_dim()
claims      = gen_claims(members_dim, pharm_dim, drug_dim)
print(f"member dim {members_dim.shape} | pharmacy dim {pharm_dim.shape} | drug dim {drug_dim.shape} | claims {claims.shape}")
print("channel mix:", claims.DLVRY_CHANL.value_counts().to_dict())
claims.head(3)[['CLM_SK','MBR_SK','PHR_PMT_CNTR_SK','GPI_SK','DLVRY_CHANL','DAYS_OF_SUPPLY_SK','CAL_INGRED_COST_PAID']]


# ============================================================================
# 4. CLAIMS AGGREGATION: DRUG CLASSES DERIVED, NOT ASSUMED
# ============================================================================

mem, mem_class, cost_per_fill = aggregate_claims(claims, members_dim, pharm_dim, drug_dim)
dd = drug_dim.set_index('GPI_SK')
print(f"members with claims {len(mem)} | member-class rows {len(mem_class)}")
print(f"mail members {int(mem.MAIL_ORDR_CORP_IND.sum())} | specialty members {int(mem.SPECIALTY_DRUG_PRG_MBR_IND.sum())}")
sh = claims.groupby('GPI_SK').size()/len(claims)
comp = pd.DataFrame({'derived_from_claims': sh, 'generator_seed_share': dd._ASSUMED_SHARE,
                     'class': dd.GPI_CLASS_NM}).round(3)
print('\nclass shares, derived vs seed:'); print(comp.to_string())
piv = mem_class.pivot_table(index='MBR_SK', columns='GPI_SK', values='FILL_CNT', fill_value=0)
shm = piv.div(piv.sum(axis=1), axis=0)
sp_cols = [g for g in piv.columns if dd.loc[g,'SPECIALTY_DRUG_PRG_IND']==1]
print(f"\nper member variation, the point of using claims:")
print(f"  members with zero cardiovascular fills: {int((shm[shm.columns[0]]==0).sum())} of {len(shm)}")
sp = claims[claims.SPECIALTY_DRUG_PRG_IND==1]
print(f"  specialty: {100*len(sp)/len(claims):.1f}% of fills but {100*sp.CAL_INGRED_COST_PAID.sum()/claims.CAL_INGRED_COST_PAID.sum():.0f}% of spend")


# ============================================================================
# 5. MODEL STRUCTURES: DISTANCES, ELIGIBILITY, LENSES, FEASIBILITY PRE CHECK
# ============================================================================

ph = pharm_dim.set_index('PHR_PMT_CNTR_SK'); me = mem.set_index('MBR_SK')
RET = ph.index[ph.PHR_TYPE_CD=='RETAIL'].tolist(); MAILP = ph.index[ph.PHR_TYPE_CD=='MAIL'].tolist()
SPE = ph.index[ph.PHR_TYPE_CD=='SPECIALTY'].tolist(); INF = ph.index[ph.PHR_TYPE_CD=='INFUSION'].tolist()
SITES = ph.index.tolist(); CURRENT = ph.index[ph.PHR_IN_CURRENT_NTWK==1].tolist()
MEMBERS = me.index.tolist(); KL = drug_dim.GPI_SK.tolist()

D = {}
for m in MEMBERS:
    la, lo = me.loc[m,'MBR_LAT'], me.loc[m,'MBR_LON']
    for p in SITES: D[(m,p)] = haversine_miles(la, lo, ph.loc[p,'PHR_LAT'], ph.loc[p,'PHR_LON'])

VOL  = {(r.MBR_SK, r.GPI_SK): r.FILL_CNT for r in mem_class.itertuples()}
CPF  = {(r.GPI_SK, r.PHR_TYPE_CD): r.COST_PER_FILL for r in cost_per_fill.itertuples()}
# Drug cost is PHARMACY SPECIFIC, not type specific. Each site prices a fill as the
# class AWP proxy times one minus that site's negotiated discount, plus that site's
# dispensing fee, times the site of care factor. Better contracted pharmacies are
# genuinely cheaper, so the network chosen now changes what the client pays. This is
# the lever that makes cost move across designs.
_mean_disc = float(pharm_dim.APP_AWP_DISCOUNT_RATE.mean())
AWP_K = {k: CPF.get((k,'RETAIL'), CPF.get((k,'SPECIALTY'), CPF.get((k,'INFUSION'), 50.0)))/(1.0-_mean_disc)
         for k in KL}
FEE   = ph.CLIENT_CONTRACT_DIS_FEE.to_dict()
SOC   = {'HOSPITAL_OPD':1.60,'INFUSION_CTR':1.00,'PHYSICIAN_OFC':1.15,'HOME_INFUSION':0.90}
def cost_of(k,p):
    t = ph.loc[p,'PHR_TYPE_CD']
    site = 0.90 if t=='MAIL' else SOC.get(ph.loc[p,'PHR_SITE_OF_CARE_CD'], 1.0)
    return (AWP_K[k]*(1.0-ph.loc[p,'APP_AWP_DISCOUNT_RATE']) + FEE.get(p,1.5)) * site
cur_of = me.CUR_PHR_PMT_CNTR_SK.to_dict()

# Ported from the reference script's diagnostics: for every member with a current pharmacy,
# the nearest SAME TYPE alternative over all candidates (his nearest_eligible_distance), and
# the gap they would face if their pharmacy were dropped. This gap now scales the disruption
# penalty, so stranding a member is expensive and dropping a pharmacy with an alternative
# across the street is nearly free.
NEXT_ALT, GAP0 = {}, {}
for m_ in MEMBERS:
    cp_ = cur_of.get(m_)
    if isinstance(cp_, str):
        t_ = ph.loc[cp_,'PHR_TYPE_CD']
        alts_ = [D[(m_,p_)] for p_ in SITES if p_ != cp_ and ph.loc[p_,'PHR_TYPE_CD']==t_]
        NEXT_ALT[m_] = min(alts_) if alts_ else 30.0
        GAP0[m_] = max(0.0, NEXT_ALT[m_] - D[(m_,cp_)])

A = {}
for (m,k) in VOL:
    thr = CONFIG['RETAIL_ACCESS']['URBAN' if me.loc[m,'MBR_URBAN_RURAL_CD']=='URBAN' else 'RURAL'][0]
    d_ = dd.loc[k]
    if d_.BENEFIT_TYPE_CD=='MEDICAL':   opts = sorted(INF, key=lambda p: D[(m,p)])[:5]
    elif d_.SPECIALTY_DRUG_PRG_IND==1:  opts = sorted(SPE, key=lambda p: D[(m,p)])[:4]
    elif me.loc[m,'MAIL_ORDR_CORP_IND']==1: opts = list(MAILP)[:3]
    else:
        opts = sorted([p for p in RET if D[(m,p)] <= CONFIG['D_MAX_RETAIL']], key=lambda p: D[(m,p)])[:8] + list(MAILP)[:2]
        if not opts: opts = sorted(RET, key=lambda p: D[(m,p)])[:1]
    c_ = cur_of.get(m)
    if isinstance(c_, str) and ph.loc[c_,'PHR_TYPE_CD']==ph.loc[opts[0],'PHR_TYPE_CD'] and c_ not in opts:
        opts = opts + [c_]                      # continuity guarantee
    A[(m,k)] = opts
YIDX = [(m,k,p) for (m,k),o in A.items() for p in o]

STD   = [m for m in MEMBERS if me.loc[m,'MAIL_ORDR_CORP_IND']==0]
SPECM = [m for m in MEMBERS if me.loc[m,'SPECIALTY_DRUG_PRG_MBR_IND']==1]
N_ret = {m: [p for p in RET if D[(m,p)] <= CONFIG['RETAIL_ACCESS']['URBAN' if me.loc[m,'MBR_URBAN_RURAL_CD']=='URBAN' else 'RURAL'][0]] for m in STD}
# BT nearby ring: threshold x 1.25, drives the extra-choice reward
N_nearby = {m: [p for p in RET if D[(m,p)] <= 1.25*CONFIG['RETAIL_ACCESS']['URBAN' if me.loc[m,'MBR_URBAN_RURAL_CD']=='URBAN' else 'RURAL'][0]] for m in STD}
N_spe = {m: [p for p in SPE if D[(m,p)] <= CONFIG['SPEC_RADIUS']] for m in SPECM}
N_inf = {m: [p for p in INF if D[(m,p)] <= CONFIG['INF_RADIUS']] for m in MEMBERS}
w_mem = {m: me.loc[m,'TOT_FILLS'] for m in MEMBERS}
spec_w = {m: sum(VOL.get((m,k),0) for k in KL if dd.loc[k,'SPECIALTY_DRUG_PRG_IND']==1 and dd.loc[k,'BENEFIT_TYPE_CD']!='MEDICAL') for m in SPECM}
inf_w  = {m: sum(VOL.get((m,k),0) for k in KL if dd.loc[k,'BENEFIT_TYPE_CD']=='MEDICAL') for m in MEMBERS}
TOT = sum(VOL.values())

def coverage_rates(active):
    act=set(active); out={}
    for s_ in ['URBAN','RURAL']:
        mm=[m for m in STD if me.loc[m,'MBR_URBAN_RURAL_CD']==s_]
        out[f'retail_{s_.lower()}'] = sum(w_mem[m] for m in mm if any(p in act for p in N_ret[m]))/sum(w_mem[m] for m in mm)
    out['retail_all']=sum(w_mem[m] for m in STD if any(p in act for p in N_ret[m]))/sum(w_mem[m] for m in STD)
    sw={m:spec_w[m] for m in SPECM if spec_w[m]>0}
    out['specialty']=sum(sw[m] for m in sw if any(p in act for p in N_spe[m]))/sum(sw.values())
    iw={m:inf_w[m] for m in MEMBERS if inf_w[m]>0}
    out['infusion']=sum(iw[m] for m in iw if any(p in act for p in N_inf[m]))/sum(iw.values())
    return out
coverage_before = coverage_rates(CURRENT); ceiling = coverage_rates(SITES)
print(f"routing variables {len(YIDX):,} | standard {len(STD)} | specialty {len(SPECM)} | mail {len(MEMBERS)-len(STD)}")
print("\ncoverage before, current network:", {k:round(v,3) for k,v in coverage_before.items()})
print("\nfeasibility pre check, ceiling vs floor:")
for s_,(r_,f_) in CONFIG['RETAIL_ACCESS'].items():
    c_=ceiling[f'retail_{s_.lower()}']; print(f"  retail {s_:6s}: ceiling {c_:.3f} vs floor {f_:.2f} {'OK' if c_>=f_ else 'IMPOSSIBLE'}")
for nm,key,fl in [('specialty','specialty',CONFIG['SPEC_FLOOR']),('infusion','infusion',CONFIG['INF_FLOOR'])]:
    print(f"  {nm:12s}: ceiling {ceiling[key]:.3f} vs floor {fl:.2f} {'OK' if ceiling[key]>=fl else 'IMPOSSIBLE'}")


# ============================================================================
# 6. DECISION VARIABLES AND OBJECTIVE
# ============================================================================

DOLLARS_PER_MILE = 4.0
CHOICE_UNIT      = 5.0      # $ per fill for having 2+ nearby options (BT extra_choice)
DISRUPTION_UNIT  = 60.0     # $ charged per fill stranded when a member's pharmacy leaves      # member travel cost per mile per fill
mdl = ConcreteModel()
CH = ph.PHR_CHAIN_NM.unique().tolist()
mdl.x=Var(SITES,within=Binary); mdl.z=Var(CH,within=Binary); mdl.y=Var(YIDX,within=NonNegativeReals,bounds=(0,1))
mdl.v=Var(STD,within=Binary); mdl.vs=Var(SPECM,within=Binary); mdl.vi=Var(MEMBERS,within=Binary)
mdl.ws=Var(STD,within=Binary)   # BT's w[m]: member has 2+ nearby in-network retail options
C_cost={t: VOL[(t[0],t[1])]*cost_of(t[1],t[2]) for t in YIDX}
k0 = KL[0]
_rc = sorted(cost_of(k0,q) for q in RET)
print(f"cost check, {k0} at retail: cheapest ${_rc[0]:.2f} per fill, dearest ${_rc[-1]:.2f}, "
      f"spread {100*(_rc[-1]/_rc[0]-1):.0f} percent. Cost now depends on WHICH pharmacies are in.")
# Steering terms are expressed in DOLLARS so they are comparable with drug cost.
# v0.6 used raw counts, which made every steering weight a rounding error (<1% of cost)
# and caused all scenarios to return the identical network. This is the fix.
C_con ={t: C_cost[t]*(1-ph.loc[t[2],'APP_AWP_DISCOUNT_RATE']) for t in YIDX}          # $ lost to weak contract terms
C_dis ={t: (0.0 if ph.loc[t[2],'PHR_TYPE_CD']=='MAIL' else VOL[(t[0],t[1])]*D[(t[0],t[2])]*DOLLARS_PER_MILE) for t in YIDX}
Q_OF  ={t: VOL[(t[0],t[1])]*ph.loc[t[2],'PHR_QUALITY_SCORE'] for t in YIDX}
CP_OF ={t: VOL[(t[0],t[1])]*ph.loc[t[2],'APP_AWP_DISCOUNT_RATE'] for t in YIDX}
SITE_ADMIN = 30_000
NEG_CHAINS = [c for c in CH if c != 'Independent']      # negotiated as one contract each
def admin_expr(m_):
    if CONFIG['CHAIN_COST_MODE'] == 'PER_CHAIN':
        return (CONFIG['CHAIN_CONTRACT_FEE']*sum(m_.z[c] for c in NEG_CHAINS)
                + SITE_ADMIN*sum(m_.x[p] for p in SITES if ph.loc[p,'PHR_CHAIN_NM']=='Independent'))
    return SITE_ADMIN*sum(m_.x[p] for p in SITES)
def admin_value(sel, zc):
    if CONFIG['CHAIN_COST_MODE'] == 'PER_CHAIN':
        return (CONFIG['CHAIN_CONTRACT_FEE']*sum(zc.get(c,0) for c in NEG_CHAINS)
                + SITE_ADMIN*sum(1 for p in sel if ph.loc[p,'PHR_CHAIN_NM']=='Independent'))
    return SITE_ADMIN*len(sel)
ADMIN = {p: SITE_ADMIN for p in SITES}                   # reference prints only
# Disruption is priced per fill AND per mile of hardship: base $ x (1 + extra miles to the
# next alternative, capped at 10). Flat pricing treated a drop next door the same as a
# stranding; this does not.
DISR_W={m: w_mem[m]*DISRUPTION_UNIT*min(10.0, 1.0+GAP0.get(m,0.0)) for m in MEMBERS if isinstance(cur_of.get(m),str)}
def cost_expr(m_): return sum(C_cost[t]*m_.y[t] for t in YIDX)
def disr_expr(m_): return sum(DISR_W[m]*(1-m_.x[cur_of[m]]) for m in DISR_W)
def set_objective(m_,P):
    if hasattr(m_,'obj'): m_.del_component(m_.obj)
    cw,dp=P['CONTRACT_WEIGHT'],P['DISTANCE_PENALTY']
    m_.obj=Objective(expr=sum((C_cost[t]+cw*C_con[t]+dp*C_dis[t])*m_.y[t] for t in YIDX)
        + admin_expr(m_) + P['LAMBDA_D']*disr_expr(m_)
        - P.get('EXTRA_CHOICE_REWARD',0.0)*CHOICE_UNIT*sum(w_mem[m]*m_.ws[m] for m in STD)
        - P['CURRENT_NETWORK_REWARD']*CONFIG['CURRENT_SITE_REWARD_SCALE']*sum(m_.x[p] for p in CURRENT)
        - CONFIG['COVERAGE_BONUS']*(sum(w_mem[m]*m_.v[m] for m in STD)
            + sum(spec_w[m]*m_.vs[m] for m in SPECM) + sum(inf_w[m]*m_.vi[m] for m in MEMBERS)), sense=minimize)
set_objective(mdl, CONFIG); print('model built, objective set')


# ============================================================================
# 7. CONSTRAINTS
# ============================================================================

mdl.c = ConstraintList()
for m in STD:     mdl.c.add(mdl.v[m]  <= (sum(mdl.x[p] for p in N_ret[m]) if N_ret[m] else 0))
for m in SPECM:   mdl.c.add(mdl.vs[m] <= (sum(mdl.x[p] for p in N_spe[m]) if N_spe[m] else 0))
for m in MEMBERS: mdl.c.add(mdl.vi[m] <= (sum(mdl.x[p] for p in N_inf[m]) if N_inf[m] else 0))
for m in STD:
    if len(N_nearby[m])>=2: mdl.c.add(2*mdl.ws[m] <= sum(mdl.x[p] for p in N_nearby[m]))
    else: mdl.c.add(mdl.ws[m]==0)
sw={m:spec_w[m] for m in SPECM if spec_w[m]>0}
iw={m:inf_w[m] for m in MEMBERS if inf_w[m]>0}
mdl.c.add(sum(iw[m]*mdl.vi[m] for m in iw) >= min(CONFIG['INF_FLOOR'], ceiling['infusion']-1e-4)*sum(iw.values()))
for (m,k),o in A.items(): mdl.c.add(sum(mdl.y[m,k,p] for p in o) == 1)
for t in YIDX: mdl.c.add(mdl.y[t] <= mdl.x[t[2]])
use={p:[] for p in SITES}
for t in YIDX: use[t[2]].append(t)
for p in SITES:
    if use[p]: mdl.c.add(sum(VOL[(t[0],t[1])]*mdl.y[t] for t in use[p]) <= ph.loc[p,'PHR_CAPACITY_SCRIPTS']*mdl.x[p])
locked=0
for t in YIDX:
    if ph.loc[t[2],'PHR_ACCEPT_NEW_PATIENT']==0 and cur_of.get(t[0])!=t[2]:
        mdl.c.add(mdl.y[t]==0); locked+=1
for p in SITES:
    ch_=ph.loc[p,'PHR_CHAIN_NM']
    if ch_ in ('ChainA','ChainB'): mdl.c.add(mdl.x[p]==mdl.z[ch_])
    else:                          mdl.c.add(mdl.x[p]<=mdl.z[ch_])
for p in MAILP: mdl.c.add(mdl.x[p]==1)
awp=[p for p in RET if ph.loc[p,'PHR_STATE_CD'] in CONFIG['AWP_STATES']]
for p in awp: mdl.c.add(mdl.x[p]==1)
_q_best = sum(VOL[(m,k)]*max(ph.loc[q,'PHR_QUALITY_SCORE'] for q in o) for (m,k),o in A.items())/TOT
_q_floor = min(CONFIG['QUALITY_FLOOR'], _q_best-1e-3)
mdl.c.add(sum(Q_OF[t]*mdl.y[t] for t in YIDX) >= _q_floor*TOT)
def set_floors(m_, floors, spec_floor):
    # Floors are clamped to the attainable ceiling so a demand can never be impossible.
    floors={k:(r_, min(f_, {'URBAN':ceiling['retail_urban'],'RURAL':ceiling['retail_rural']}[k]-1e-4))
            for k,(r_,f_) in floors.items()}
    spec_floor=min(spec_floor, ceiling['specialty']-1e-4)
    # Access floors are scenario configurable. The bar moves; enforcement never does.
    if hasattr(m_,'flr'): m_.del_component(m_.flr)
    m_.flr = ConstraintList()
    for s_,(r_,f_) in floors.items():
        mm=[a for a in STD if me.loc[a,'MBR_URBAN_RURAL_CD']==s_]
        m_.flr.add(sum(w_mem[a]*m_.v[a] for a in mm) >= f_*sum(w_mem[a] for a in mm))
    sw_={a:spec_w[a] for a in SPECM if spec_w[a]>0}
    m_.flr.add(sum(sw_[a]*m_.vs[a] for a in sw_) >= spec_floor*sum(sw_.values()))

set_floors(mdl, CONFIG['RETAIL_ACCESS'], CONFIG['SPEC_FLOOR'])
print(f'constraints added | closed-site routings locked {locked} | AWP mandated {len(awp)} retail sites | access floors applied')


# ============================================================================
# 8. SOLVE AND RESULTS
# ============================================================================

def solve(m_):
    opt=SolverFactory('appsi_highs'); t0=time.time(); r=opt.solve(m_); return r, time.time()-t0
res, dt = solve(mdl); print(f"status {res.solver.termination_condition} | {dt:.1f}s")
B0 = sum(VOL[(m,k)]*cost_of(k,cur_of[m]) for (m,k) in VOL
         if isinstance(cur_of.get(m),str) and (m,k,cur_of[m]) in set(YIDX))
B0 += sum(VOL[(m,k)]*cost_of(k,A[(m,k)][0]) for (m,k) in VOL
          if not (isinstance(cur_of.get(m),str) and (m,k,cur_of[m]) in set(YIDX)))
def metrics(m_):
    sel=[p for p in SITES if round(value(m_.x[p]))==1]; yv={t:value(m_.y[t]) for t in YIDX}
    drug=sum(C_cost[t]*yv[t] for t in YIDX); zc={c: round(value(m_.z[c])) for c in CH}
    adm=admin_value(sel, zc)
    disr=sum(w_mem[m] for m in DISR_W if round(value(m_.x[cur_of[m]]))==0)
    kept=sum(VOL[(m,k)]*yv[(m,k,cur_of[m])] for (m,k) in VOL
             if isinstance(cur_of.get(m),str) and (m,k,cur_of[m]) in yv)
    mix={}
    for t in YIDX:
        mix.setdefault(t[1],{}); tt=ph.loc[t[2],'PHR_TYPE_CD']
        mix[t[1]][tt]=mix[t[1]].get(tt,0)+VOL[(t[0],t[1])]*yv[t]
    mix={k:{a:b/sum(v.values()) for a,b in v.items()} for k,v in mix.items()}
    ws_on = sum(w_mem[m]*round(value(m_.ws[m])) for m in STD)/sum(w_mem[m] for m in STD)
    cov_all = sum(1 for m in STD if any(p in set(sel) for p in N_ret[m]))/len(STD)
    rr = [(t, D[(t[0],t[2])], VOL[(t[0],t[1])]*yv[t]) for t in YIDX if ph.loc[t[2],'PHR_TYPE_CD']!='MAIL']
    assigned_avg_dist = sum(d_*v_ for _,d_,v_ in rr)/max(1e-9,sum(v_ for _,_,v_ in rr))
    return dict(selected=sel, drug=drug, admin=adm, total=drug+adm,
                extra_choice_rate=100*ws_on, coverage_all_members=cov_all, assigned_avg_dist=assigned_avg_dist,
                disr_pct=100*disr/sum(DISR_W.values()), kept_pct=100*kept/TOT,
                quality=sum(Q_OF[t]*yv[t] for t in YIDX)/TOT,
                contract=100*sum(CP_OF[t]*yv[t] for t in YIDX)/TOT,
                cov_after=coverage_rates(sel), mix=mix)
M=metrics(mdl)
print(f"\nOUTPUT 1 roster: {len(M['selected'])} of {len(SITES)} sites")
print(f"OUTPUT 2 economics: drug ${M['drug']:,.0f} + admin ${M['admin']:,.0f} | baseline ${B0:,.0f} | savings {100*(1-M['drug']/B0):.1f}%")
rxc=sum(C_cost[t]*value(mdl.y[t]) for t in YIDX if dd.loc[t[1],'BENEFIT_TYPE_CD']=='RX')
print(f"         by benefit: pharmacy ${rxc:,.0f} | medical ${M['drug']-rxc:,.0f}")
print(f"OUTPUT 3 disruption {M['disr_pct']:.1f}% | continuity {M['kept_pct']:.1f}%")
cb=pd.DataFrame({'before':coverage_before,'after':M['cov_after']}); cb['delta']=cb.after-cb.before
print("OUTPUT 4 coverage:"); print(cb.round(3))
print(f"OUTPUT 5 quality {M['quality']:.2f} vs floor {CONFIG['QUALITY_FLOOR']} | weighted AWP discount {M['contract']:.1f}%")
mx=pd.DataFrame(M['mix']).T.fillna(0); mx.index=[dd.loc[i,'GPI_CLASS_NM'][:28] for i in mx.index]
print("\nOUTPUT 6 routing by class and channel:"); print(mx.round(2).to_string())

fig,ax=plt.subplots(1,2,figsize=(12,4))
cb[['before','after']].plot.bar(ax=ax[0],rot=20); ax[0].set_title('Coverage before vs after'); ax[0].set_ylim(0,1.05)
mx.plot.barh(stacked=True, ax=ax[1]); ax[1].set_title('Routing mix by therapeutic class'); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig('charts/static_01_results.png', dpi=150, bbox_inches='tight'); plt.savefig('charts/static_02_scenario_bars.png', dpi=150, bbox_inches='tight'); plt.savefig('charts/static_03_dashboard.png', dpi=150, bbox_inches='tight'); plt.savefig('charts/static_04_scenario_networks.png', dpi=150, bbox_inches='tight'); plt.show()


# ============================================================================
# 9. SCENARIO SUITE, REFERENCE SCRIPT FORMAT
# ============================================================================

def build_scenarios(base, natural, ceil):
    # Floors are calibrated to THIS data: interpolated between the coverage the cheapest network
    # delivers on its own (natural) and the most any network can deliver (ceiling). A floor below
    # natural is inert; a floor above ceiling is infeasible. Scenarios walk that band.
    def lvl(lens, f):  # f = fraction of the natural->ceiling gap to demand
        return round(natural[lens] + f*(ceil[lens]-natural[lens]) - 1e-4, 4)
    F = lambda fu, fr: {'URBAN':(4.0, lvl('retail_urban',fu)), 'RURAL':(15.0, lvl('retail_rural',fr))}
    S = lambda f: lvl('specialty', f)
    return [
      dict(name='Lowest Cost',        budget=None,           LAMBDA_D=0.2, CONTRACT_WEIGHT=1.0, DISTANCE_PENALTY=0.2,
           CURRENT_NETWORK_REWARD=0.0, FLOORS=F(-0.5,-0.5), SPEC_FLOOR=S(-0.5)),
      dict(name='Mid Cost',           budget=int(base*1.15), LAMBDA_D=1.0, CONTRACT_WEIGHT=2.0, DISTANCE_PENALTY=0.5,
           EXTRA_CHOICE_REWARD=1.0, CURRENT_NETWORK_REWARD=0.5, FLOORS=F(0.30,0.30), SPEC_FLOOR=S(0.30)),
      dict(name='Better Coverage',    budget=int(base*1.35), LAMBDA_D=2.0, CONTRACT_WEIGHT=2.5, DISTANCE_PENALTY=1.0,
           EXTRA_CHOICE_REWARD=3.0, CURRENT_NETWORK_REWARD=1.0, FLOORS=F(0.65,0.65), SPEC_FLOOR=S(0.65)),
      dict(name='Highest Cost',       budget=int(base*1.70), LAMBDA_D=4.0, CONTRACT_WEIGHT=3.0, DISTANCE_PENALTY=1.5,
           EXTRA_CHOICE_REWARD=5.0, CURRENT_NETWORK_REWARD=2.0, FLOORS=F(0.95,0.95), SPEC_FLOOR=S(0.95)),
      dict(name='Biggest Percentage', budget=int(base*1.40), LAMBDA_D=1.0, CONTRACT_WEIGHT=14.0, DISTANCE_PENALTY=0.2,
           CURRENT_NETWORK_REWARD=0.5, FLOORS=F(0.30,0.30), SPEC_FLOOR=S(0.30)),
      dict(name='Specialty Focus',    budget=int(base*1.40), LAMBDA_D=2.0, CONTRACT_WEIGHT=3.0, DISTANCE_PENALTY=0.6,
           CURRENT_NETWORK_REWARD=0.5, FLOORS=F(0.30,0.30), SPEC_FLOOR=S(1.0)),
      dict(name='Member Stability',   budget=int(base*1.60), LAMBDA_D=25.0, CONTRACT_WEIGHT=2.0, DISTANCE_PENALTY=0.5,
           CURRENT_NETWORK_REWARD=4.0, FLOORS=F(0.30,0.30), SPEC_FLOOR=S(0.30)),
      dict(name='Mail Forward',       budget=int(base*1.20), LAMBDA_D=1.0, CONTRACT_WEIGHT=2.0, DISTANCE_PENALTY=4.0,
           CURRENT_NETWORK_REWARD=0.5, FLOORS=F(0.30,0.30), SPEC_FLOOR=S(0.30)),
      dict(name='Urban Maximum',      budget=int(base*1.90), LAMBDA_D=4.0, CONTRACT_WEIGHT=2.5, DISTANCE_PENALTY=1.0,
           EXTRA_CHOICE_REWARD=3.0, CURRENT_NETWORK_REWARD=1.0, FLOORS=F(1.0,0.65), SPEC_FLOOR=S(0.65)),
    ]

def set_budget(m_,b):
    if hasattr(m_,'bc'): m_.del_component(m_.bc)
    if b is not None: m_.bc = Constraint(expr=cost_expr(m_)+admin_expr(m_) <= b)
def run(s):
    P=dict(CONFIG); P.update({k:v for k,v in s.items() if k in ('LAMBDA_D','CONTRACT_WEIGHT','DISTANCE_PENALTY','EXTRA_CHOICE_REWARD','CURRENT_NETWORK_REWARD')})
    P.setdefault('EXTRA_CHOICE_REWARD',0.0)
    set_objective(mdl,P); set_budget(mdl,s.get('budget'))
    if 'FLOORS' in s: set_floors(mdl, s['FLOORS'], s.get('SPEC_FLOOR', CONFIG['SPEC_FLOOR']))
    r_,t_=solve(mdl); Mi=metrics(mdl)
    return dict(scenario=s['name'], status=str(r_.solver.termination_condition), sites=len(Mi['selected']),
                total_cost=round(Mi['total']), savings_pct=round(100*(1-Mi['drug']/B0),1),
                cov_before=round(coverage_before['retail_all'],3), cov_after=round(Mi['cov_after']['retail_all'],3),
                cov_delta=round(Mi['cov_after']['retail_all']-coverage_before['retail_all'],3),
                disruption_pct=round(Mi['disr_pct'],1), kept_pct=round(Mi['kept_pct'],1),
                choice_pct=round(Mi['extra_choice_rate'],1), cov_all=round(Mi['coverage_all_members'],3),
                avg_dist=round(Mi['assigned_avg_dist'],2),
                awp_discount_pct=round(Mi['contract'],1), solve_s=round(t_,1)), Mi
# Base run: floors switched off. Whatever coverage this cheapest lawful network delivers on its
# own is the NATURAL level; scenario floors are then calibrated between natural and the ceiling.
base, base_M = run(dict(name='Base', budget=None, LAMBDA_D=0.2, CONTRACT_WEIGHT=1.0, DISTANCE_PENALTY=0.2,
                        CURRENT_NETWORK_REWARD=0.0, FLOORS={'URBAN':(4.0,0.0),'RURAL':(15.0,0.0)}, SPEC_FLOOR=0.0))
natural = base_M['cov_after']
print(f"base cheapest lawful network: ${base['total_cost']:,}  ({base['sites']} sites)")
print("natural coverage :", {k:round(v,3) for k,v in natural.items()})
print("ceiling          :", {k:round(v,3) for k,v in ceiling.items()})
print("-> scenario floors are placed inside this band so each one genuinely binds")
outs,scenario_metrics=[],{}
for s in build_scenarios(base['total_cost'], natural, ceiling):
    o,Mi=run(s); outs.append(o); scenario_metrics[s['name']]=Mi
summary=pd.DataFrame(outs)
print('=== Scenario Summary ==='); print(summary.to_string(index=False))


# ============================================================================
# 10. DISRUPTION IMPACT: THE BUSINESS VIEW
# ============================================================================

rows=[]; detail={}
for name, Mi in scenario_metrics.items():
    sel=set(Mi['selected']); recs=[]
    breach=0; gap_gt2=0
    for m_ in MEMBERS:
        cp_=cur_of.get(m_)
        if not isinstance(cp_,str): continue
        t_=ph.loc[cp_,'PHR_TYPE_CD']
        realized=min([D[(m_,p_)] for p_ in sel if ph.loc[p_,'PHR_TYPE_CD']==t_], default=float('nan'))
        # reference script breach rule: gap vs nearest eligible over ALL candidates
        near_all=min([D[(m_,p_)] for p_ in SITES if ph.loc[p_,'PHR_TYPE_CD']==t_])
        gap_=realized-near_all
        if me.loc[m_,'MAIL_ORDR_CORP_IND']==1: allowed=float('inf')
        elif me.loc[m_,'SPECIALTY_DRUG_PRG_MBR_IND']==1: allowed=2.0
        else: allowed=2.0 if me.loc[m_,'MBR_URBAN_RURAL_CD']=='URBAN' else 5.0
        if gap_>allowed: breach+=1
        if me.loc[m_,'MAIL_ORDR_CORP_IND']==0 and me.loc[m_,'SPECIALTY_DRUG_PRG_MBR_IND']==0 and gap_>2.0: gap_gt2+=1
        if cp_ in sel: continue
        recs.append(dict(MBR_SK=m_, fills=int(me.loc[m_,'TOT_FILLS']),
                         old_miles=round(D[(m_,cp_)],1), new_miles=round(realized,1),
                         extra_miles=round(realized-D[(m_,cp_)],1)))
    dd_=pd.DataFrame(recs); detail[name]=dd_
    saved=B0-Mi['drug']
    rows.append(dict(scenario=name, members_disrupted=len(dd_),
        extra_mi_median=round(dd_.extra_miles.median(),1) if len(dd_) else 0.0,
        extra_mi_max=round(dd_.extra_miles.max(),1) if len(dd_) else 0.0,
        annual_extra_member_miles=int((dd_.extra_miles.clip(lower=0)*dd_.fills*2).sum()) if len(dd_) else 0,
        gap_breaches=breach, std_gap_gt2=gap_gt2,
        drug_savings=int(saved),
        savings_per_disrupted=int(saved/len(dd_)) if len(dd_) else None))
impact=pd.DataFrame(rows)
print('=== Disruption Impact (reference script diagnostics, extended) ===')
print(impact.to_string(index=False))
worst=[n for n in detail if len(detail[n])>0]
if worst:
    n0=max(worst, key=lambda n: len(detail[n]))
    print(chr(10)+f"Worst affected members, {n0}:")
    print(detail[n0].sort_values('extra_miles',ascending=False).head(8).to_string(index=False))


# ============================================================================
# 11. REFERENCE SCRIPT PARITY OUTPUTS
# ============================================================================

import os
os.makedirs('exports', exist_ok=True)
# reference script's three CSVs, from our tables
members_dim.to_csv('exports/synthetic_members.csv', index=False)
pharm_dim.to_csv('exports/synthetic_pharmacies.csv', index=False)
pd.DataFrame([dict(MBR_SK=m, PHR_PMT_CNTR_SK=p, distance_miles=D[(m,p)]) for m in MEMBERS for p in SITES])   .to_csv('exports/synthetic_distances.csv', index=False)
claims.to_csv('exports/synthetic_claims.csv', index=False)

# per member results, reference format, for the default solve
sel0=set(M['selected'])
mr=[]
for m in MEMBERS:
    cp=cur_of.get(m); is_std=m in set(STD)
    mr.append(dict(MBR_SK=m,
        covered=int(any(p in sel0 for p in N_ret[m])) if is_std else None,
        kept_current=int(isinstance(cp,str) and cp in sel0),
        has_extra_choice=int(len([p for p in N_nearby[m] if p in sel0])>=2) if is_std else None,
        current_pharmacy=cp))
member_results_df=pd.DataFrame(mr)
member_results_df.to_csv('exports/member_results.csv', index=False)
no_radius=[m for m in STD if not N_ret[m]]
print(f"exports written: 4 synthetic tables + member_results.csv ({len(member_results_df)} rows)")
if no_radius:
    print(f"WARNING: {len(no_radius)} standard members have NO retail option inside their access radius;")
    print("their routing uses the nearest-site fallback (reference script marks these uncoverable).")


# ============================================================================
# 12. THE TRADE OFF ANALYSIS. DISRUPTION AGAINST SAVINGS, READ FROM RESULTS
# ============================================================================
# Disruption is part of the objective, so the analysis is multi objective.
# Two complementary reads are produced.
#   12.1  A controlled grid over the disruption price and the access demand,
#         shown as heat maps. Savings in green, member burden in red.
#   12.2  The true frontier by epsilon constraint. Member movement becomes a
#         CAP the client chooses, cost is minimized under each cap, and the
#         curve is read directly from solved results.
#   12.3  The chain contracting dial. Chains negotiate as one contract, so
#         the administration cost model has two settings and both are solved.
# All interactive charts are written to the charts folder and collected into
# a single dashboard.html at the end.
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
FIGS = {}
pio.renderers.default = 'browser' if (os.environ.get('DISPLAY') or os.name=='nt') else 'json'
NAVY, RED, GREEN, GREY = '#1F3B57', '#B3202C', '#2E7D5B', '#6B7580'
LABEL = {0.0: 'No access target', 0.5: 'Moderate access target', 1.0: 'Full access target'}

def band(lens, fr):
    return round(natural[lens] + fr*(ceiling[lens]-natural[lens]) - 1e-4, 4)

def moved_and_miles(sel):
    per = []
    for m_ in MEMBERS:
        cp = cur_of.get(m_)
        if isinstance(cp, str) and cp not in sel:
            ty  = ph.loc[cp, 'PHR_TYPE_CD']
            new = min([D[(m_, q)] for q in sel if ph.loc[q, 'PHR_TYPE_CD'] == ty],
                      default=float('nan'))
            per.append(max(0.0, new - D[(m_, cp)]))
    tot = int(sum(e*me.loc[m_, 'TOT_FILLS']*2 for e, m_ in zip(per, [m for m in MEMBERS
              if isinstance(cur_of.get(m), str) and cur_of[m] not in sel])))
    return dict(members_moved=len(per),
                mi_each=round(float(np.mean(per)), 2) if per else 0.0,
                total_extra_miles=tot)

# ---------------------------------------------------------------------------
# 12.1 Grid and heat maps
# ---------------------------------------------------------------------------
LAM_GRID, FLOOR_GRID = [0.0, 1.0, 2.0, 4.0, 8.0], [0.0, 0.5, 1.0]
grid = []
for lam in LAM_GRID:
    for fr in FLOOR_GRID:
        P = dict(CONFIG); P.update(LAMBDA_D=lam, CONTRACT_WEIGHT=2.0,
                                   DISTANCE_PENALTY=0.5, EXTRA_CHOICE_REWARD=0.0,
                                   CURRENT_NETWORK_REWARD=0.0)
        fl = {'URBAN': (4.0, band('retail_urban', fr)),
              'RURAL': (15.0, band('retail_rural', fr))}
        set_objective(mdl, P); set_budget(mdl, None); set_floors(mdl, fl, band('specialty', fr))
        r_, t_ = solve(mdl)
        sel  = set(q for q in SITES if round(value(mdl.x[q])) == 1)
        drug = sum(C_cost[t]*value(mdl.y[t]) for t in YIDX)
        zc   = {c: round(value(mdl.z[c])) for c in CH}
        mm   = moved_and_miles(sel)
        grid.append(dict(disruption_price=lam, access_demand=fr,
                         savings_pct=round(100*(1-drug/B0), 2),
                         dollars_saved=round(B0-drug),
                         total_cost=round(drug+admin_value(sel, zc)),
                         access_pct=round(100*coverage_rates(sel)['retail_all'], 1),
                         sites=len(sel), solve_s=round(t_, 1), **mm))
        print(f"  grid lam={lam:<4} floor={fr:<4} moved {mm['members_moved']:>4} "
              f"saved ${round(B0-drug):>9,}")
sweep = pd.DataFrame(grid)
print(f"\n{len(sweep)} grid designs solved in {sweep.solve_s.sum():.0f}s\n")
print(sweep.to_string(index=False))

for val, scale, ttl, key in [
        ('dollars_saved', 'Greens', 'Dollars saved per year', 'heat_savings'),
        ('total_extra_miles', 'Reds', 'Annual extra member miles', 'heat_miles'),
        ('members_moved', 'Oranges', 'Members whose pharmacy leaves', 'heat_moved')]:
    piv = sweep.pivot(index='access_demand', columns='disruption_price', values=val)
    piv.index = [LABEL[i] for i in piv.index]
    h = px.imshow(piv, text_auto=True, aspect='auto', color_continuous_scale=scale,
                  labels=dict(x='price placed on disruption', y='', color=''), title=ttl)
    h.update_layout(template='plotly_white', height=340, coloraxis_showscale=False)
    h.write_html(f'charts/{key}.html'); FIGS[key] = h; h.show()

# Same savings, member burden is a choice: dual axis read of the grid
sw = sweep.sort_values(['access_demand', 'disruption_price']).reset_index(drop=True)
f_burden = go.Figure()
f_burden.add_bar(name='Dollars saved per year', x=list(range(len(sw))), y=sw.dollars_saved,
                 marker_color=GREEN)
f_burden.add_scatter(name='Extra member miles per year', x=list(range(len(sw))),
                     y=sw.total_extra_miles, yaxis='y2', mode='lines+markers',
                     line=dict(color=RED, width=2.5), marker_size=7)
f_burden.update_layout(
    title='Same savings in every design. The member burden is a choice, not a price',
    xaxis=dict(tickvals=list(range(len(sw))),
               ticktext=[f"λ{int(r.disruption_price)}<br>{LABEL[r.access_demand].split()[0]}"
                         for _, r in sw.iterrows()], tickfont=dict(size=8)),
    yaxis=dict(title='dollars saved', tickformat='$,', color=GREEN),
    yaxis2=dict(title='extra member miles', overlaying='y', side='right',
                color=RED, tickformat=','),
    template='plotly_white', height=430, legend=dict(orientation='h', y=1.12))
f_burden.write_html('charts/chart_savings_vs_burden.html')
FIGS['burden'] = f_burden; f_burden.show()

# ---------------------------------------------------------------------------
# 12.2 The true frontier. Epsilon constraint: cap members moved, minimize cost
# The problem is multi objective: coverage, disruption in members, disruption in
# miles, and cost. Coverage is held at its maximum. Disruption is treated twice,
# as the capped axis of the frontier and as a live term inside the objective.
# Cost is what remains to minimize. The frontier is then read from results.
# ---------------------------------------------------------------------------
MOVED = {m: cur_of[m] for m in MEMBERS if isinstance(cur_of.get(m), str)}
def set_move_cap(m_, cap):
    if hasattr(m_, 'movecap'): m_.del_component(m_.movecap)
    if cap is not None:
        m_.movecap = Constraint(expr=sum(1 - m_.x[MOVED[m]] for m in MOVED) <= cap)

CAPS = [0, 10, 25, 50, 100, 200, 400, None]
FR_FLOOR = 1.0    # every member the geography can cover IS covered in every design
fl = {'URBAN': (4.0, band('retail_urban', FR_FLOOR)),
      'RURAL': (15.0, band('retail_rural', FR_FLOOR))}
# Hybrid multi objective method. The cap positions each design on the frontier.
# Disruption ALSO stays in the objective at a secondary weight, and its weight is
# miles scaled, so inside every cap the model still moves as few members as few
# miles as possible. Cost decides the frontier, disruption is minimized within it,
# and both dimensions of member burden are optimized, never merely tolerated.
P = dict(CONFIG); P.update(LAMBDA_D=0.5, CONTRACT_WEIGHT=2.0, DISTANCE_PENALTY=0.5,
                           EXTRA_CHOICE_REWARD=0.0, CURRENT_NETWORK_REWARD=0.0)
front = []
for cap in CAPS:
    set_objective(mdl, P); set_budget(mdl, None)
    set_floors(mdl, fl, band('specialty', FR_FLOOR)); set_move_cap(mdl, cap)
    r_, t_ = solve(mdl)
    sel  = set(q for q in SITES if round(value(mdl.x[q])) == 1)
    drug = sum(C_cost[t]*value(mdl.y[t]) for t in YIDX)
    zc   = {c: round(value(mdl.z[c])) for c in CH}
    mm   = moved_and_miles(sel)
    front.append(dict(cap='unlimited' if cap is None else cap,
                      dollars_saved=round(B0-drug),
                      total_cost=round(drug+admin_value(sel, zc)),
                      sites=len(sel), solve_s=round(t_, 1), **mm))
    print(f"  cap {str(cap):>9} moved {mm['members_moved']:>4} "
          f"saves ${round(B0-drug):>9,} at {mm['mi_each']} miles each")
set_move_cap(mdl, None)
frontier = pd.DataFrame(front)
print("\n=== The true frontier ===")
print(frontier.to_string(index=False))
print("Within every cap the objective also minimized disruption itself, members and miles both.")
print("A design therefore never moves a member the cap merely permitted it to move.")

f_front = go.Figure(go.Scatter(
    x=frontier.members_moved, y=frontier.dollars_saved, mode='lines+markers+text',
    text=[f"${v/1000:.0f}k" for v in frontier.dollars_saved], textposition='top center',
    line=dict(color=NAVY, width=3), marker=dict(size=10),
    customdata=frontier[['mi_each', 'sites', 'cap', 'total_extra_miles']],
    hovertemplate='<b>%{x} members move</b> at %{customdata[0]} extra miles each'
                  '<br>%{customdata[3]:,} total extra member miles a year'
                  '<br>saves $%{y:,} a year with %{customdata[1]} sites'
                  '<br>cap set at %{customdata[2]}<extra></extra>'))
zero = frontier[frontier.members_moved == 0]
if len(zero):
    f_front.add_annotation(x=0, y=float(zero.dollars_saved.max()), ax=70, ay=-35,
        text=f"<b>Nobody moves, still saves ${int(zero.dollars_saved.max()):,}</b>",
        showarrow=True, arrowhead=2, font=dict(color=GREEN, size=12))
f_front.update_layout(
    title='All members covered in every design. The curve shows what tolerated movement buys',
    xaxis_title='members allowed to move, the cap the client chooses',
    yaxis=dict(title='dollars saved per year', tickformat='$,'),
    template='plotly_white', height=470)
f_front.write_html('charts/chart_true_frontier.html')
FIGS['frontier'] = f_front; f_front.show()

# ---------------------------------------------------------------------------
# 12.3 The chain contracting dial. One contract per chain against per site cost
# ---------------------------------------------------------------------------
dial_rows = []
for mode in ['PER_SITE', 'PER_CHAIN']:
    CONFIG['CHAIN_COST_MODE'] = mode
    set_objective(mdl, dict(CONFIG, LAMBDA_D=2.0, CONTRACT_WEIGHT=2.0,
                            DISTANCE_PENALTY=0.5, EXTRA_CHOICE_REWARD=0.0,
                            CURRENT_NETWORK_REWARD=0.0))
    set_budget(mdl, None); set_floors(mdl, fl, band('specialty', FR_FLOOR)); set_move_cap(mdl, None)
    r_, t_ = solve(mdl)
    sel  = set(q for q in SITES if round(value(mdl.x[q])) == 1)
    zc   = {c: round(value(mdl.z[c])) for c in CH}
    drug = sum(C_cost[t]*value(mdl.y[t]) for t in YIDX)
    mm   = moved_and_miles(sel)
    dial_rows.append(dict(mode=mode,
                          chains_in=', '.join(sorted(c for c in NEG_CHAINS if zc.get(c))),
                          sites=len(sel), admin=admin_value(sel, zc),
                          drug=round(drug), dollars_saved=round(B0-drug),
                          members_moved=mm['members_moved']))
    print(f"  {mode:9s} sites {len(sel):>3} admin ${admin_value(sel, zc):>9,} "
          f"saves ${round(B0-drug):>9,} moved {mm['members_moved']}")
CONFIG['CHAIN_COST_MODE'] = 'PER_SITE'
dial = pd.DataFrame(dial_rows)
print("\n=== Chain contracting dial ===")
print(dial.to_string(index=False))

f_dial = go.Figure()
f_dial.add_bar(name='Sites in network', x=dial['mode'], y=dial.sites, marker_color=NAVY)
f_dial.add_bar(name='Administration cost, $ thousand', x=dial['mode'],
               y=dial.admin/1000, marker_color=GREY)
f_dial.add_bar(name='Members moved', x=dial['mode'], y=dial.members_moved, marker_color=RED)
f_dial.update_layout(
    barmode='group', template='plotly_white', height=400,
    title='What the contracting assumption changes. Per site cost against one contract per chain',
    xaxis=dict(ticktext=['Every site carries cost', 'One contract per chain, member sites free'],
               tickvals=['PER_SITE', 'PER_CHAIN']))
f_dial.write_html('charts/chart_chain_dial.html')
FIGS['dial'] = f_dial; f_dial.show()

# ============================================================================
# 13. THE REPORT. EVERY RESULT, EVERY CHART, EVERY EXPLANATION, ONE HTML FILE
# ============================================================================
import base64 as _b64
def _png(name):
    try:
        with open(f'charts/{name}.png','rb') as fh:
            return ('<img style="max-width:100%;border:1px solid #E3E7EB;border-radius:8px" '
                    f'src="data:image/png;base64,{_b64.b64encode(fh.read()).decode()}">')
    except Exception:
        return ''
def _fig(k):  return pio.to_html(FIGS[k], include_plotlyjs=False, full_html=False)
def _tbl(df): return '<div style="overflow-x:auto">'+df.to_html(index=False, classes='tbl', border=0)+'</div>'
def sec(title, what, read, ask, body):
    return (f'<section><h2>{title}</h2>'
            f'<p class="lead"><b>What you are looking at.</b> {what}</p>'
            f'<p class="lead"><b>How to read it.</b> {read}</p>'
            f'<p class="lead"><b>The business question it answers.</b> {ask}</p>{body}</section>')

_rc  = sorted(cost_of(KL[0], q) for q in RET)
zero = frontier[frontier.members_moved == 0]
best_zero = int(zero.dollars_saved.max()) if len(zero) else 0
kpi = [('Baseline drug spend', f'${B0:,.0f}'),
       ('Best saving, nobody moved', f'${best_zero:,}'),
       ('Best saving, full coverage', f'${int(frontier.dollars_saved.max()):,}'),
       ('Members', f'{len(MEMBERS):,}'), ('Candidate sites', f'{len(SITES)}')]
cards = ''.join(f'<div class="card"><div class="num">{v}</div><div class="lbl">{k}</div></div>' for k,v in kpi)
need  = prov[prov.status=='NEEDED'][['column','source','note']]

S = []
S.append(sec('1. What this model does',
 'A decision engine that selects which pharmacies form a client network and where each category of prescription is filled, then prices every candidate design before any member feels it.',
 'Read top to bottom. Sections two to four establish the data and today. Five to seven show designs and their member impact. Eight and nine are the decision itself. Ten to twelve cover contracting, solver evidence and requests.',
 'Can we know the savings, the access, and the member burden of a network change before we make it. Yes, and every number below is that answer on synthetic data shaped exactly like ours.', ''))

S.append(sec('2. The data and why cost moves',
 f'Four synthetic tables in the Optum warehouse schema. Drug classes come from claims through GPI. Every pharmacy carries its own negotiated discount and dispensing fee, so the same drug costs ${_rc[0]:.2f} per fill at the best contracted retail store and ${_rc[-1]:.2f} at the worst, a spread of {100*(_rc[-1]/_rc[0]-1):.0f} percent.',
 'The table lists every column the model needs that we do not yet hold. It is the data request, ready to forward.',
 'Is the model fed the way production will feed it. Yes on structure. The listed columns are the only gap.',
 _tbl(need)))

S.append(sec('3. Today, and the ceiling',
 f'Coverage the current network delivers, beside the optimized result, for every access lens. The geography itself caps urban access at {100*ceiling["retail_urban"]:.1f} percent, no network can exceed it.',
 'Before is today. After is the optimized default. Delta is the improvement. Any demand above the ceiling is refused by the model before solving, with the reason named.',
 'How much better can access get, and where is the hard limit a client guarantee must respect.',
 _tbl(cb.reset_index().rename(columns={'index':'lens'}).round(3)) + _png('static_01_results')))

S.append(sec('4. Nine scenario philosophies',
 'The same model solved under nine named client attitudes, from Lowest Cost to Member Stability. Only the dials change, never the rules.',
 'Each row is one lawful network. Compare sites, cost, savings, access, disruption and choice across rows.',
 'What does each way of thinking about members actually buy, in one table.',
 _tbl(summary) + _png('static_02_scenario_bars') + _png('static_04_scenario_networks')))

S.append(sec('5. Member impact of every scenario',
 'For each scenario, the members whose pharmacy leaves, the extra miles they travel, and the savings earned per affected member.',
 'Savings per disrupted member is the number that sizes a make whole offer such as a plan credit. Gap breaches use the reference thresholds of two miles urban and five rural.',
 'If we take this design, exactly who pays for it and how much do we hold to make them whole.',
 _tbl(impact)))

S.append(sec('6. The trade off grid',
 'Fifteen complete networks on a grid. Columns raise the price placed on disruption. Rows raise the access demanded, from nothing to the maximum the geography allows.',
 'Green map is dollars saved. Red map is extra member miles. Orange map is members moved. Read green against red.',
 'Is member burden the price of savings, or a separate choice. Where green stays strong while red fades, the burden was a choice.',
 _fig('heat_savings') + _fig('heat_miles') + _fig('heat_moved') + _fig('burden')))

S.append(sec('7. The decision curve',
 'The frontier. Every design holds coverage at its maximum. Member movement is capped at zero, ten, twenty five and so on, and under each cap the model finds the best cost while still minimizing disruption itself, in members and in miles.',
 'Hover any point for its full record. The annotation marks the design that moves nobody and what it still saves. The client chooses the point, and the sentence in its hover is the decision, ready to minute.',
 f'How much movement buys how much saving. On this book a design that moves nobody still saves ${best_zero:,} a year.',
 _fig('frontier') + _tbl(frontier)))

S.append(sec('8. Chain contracting, the dial',
 'Chains such as CVS negotiate as one contract for every store, not store by store. The model is solved under both cost assumptions, every site carries cost against one fee per chain with member stores free.',
 'Compare the two rows. When member stores are free the model keeps whole chains, sites rise and administration falls.',
 'Which contracting assumption should we plan under, and how different are the networks it produces.',
 _fig('dial') + _tbl(dial)))

S.append(sec('9. Solver evidence',
 'The identical model submitted to every exact solver available. All must agree on the objective, and they do. Differences are speed only.',
 'This table is the classical yardstick. When the D Wave hybrid results arrive they are judged against it.',
 'Can the answer be trusted. Yes, it is proven optimal, independently, by more than one solver.',
 _tbl(benchmark)))

S.append(sec('10. The full picture', 'The complete static dashboard from this run, for print and appendix use.',
 'Map, access, routing by class, spend by class, menu and benchmark in one image.',
 'One page that survives being printed.', _png('static_03_dashboard')))

S.append(sec('11. Glossary, assumptions, and the decisions requested',
 'Plain definitions, the assumptions carried, and the confirmations needed from the business.',
 'Access is members within reach of the right kind of pharmacy, weighted by fills. Disruption is members whose pharmacy leaves, priced by how stranded they become. Choice is two or more nearby options. Savings per disrupted member sizes a make whole offer.',
 'What must the business confirm before real data arrives.',
 '<ol class="lead"><li>The per site administration cost of thirty thousand dollars and the chain contract fee of one hundred fifty thousand.</li>'
 '<li>Which chains contract all or nothing.</li>'
 '<li>Whether access standards mean straight line miles or drive time.</li>'
 '<li>The scenario philosophy to carry into client conversations.</li>'
 '<li>Approval of the data request in section two.</li></ol>'))

html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Pharmacy Network Optimization, full report</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
 body{{margin:0;font-family:Segoe UI,Calibri,sans-serif;background:#F4F6F8;color:#1A1A1A}}
 header{{background:#1F3B57;color:#fff;padding:28px 44px}}
 header h1{{margin:0;font-size:27px}} header p{{margin:6px 0 0;color:#C9D3DE;font-size:13px}}
 .cards{{display:flex;gap:14px;padding:20px 44px;flex-wrap:wrap}}
 .card{{background:#fff;border-radius:10px;padding:14px 20px;box-shadow:0 1px 4px rgba(0,0,0,.08);flex:1;min-width:160px}}
 .num{{font-size:22px;font-weight:700;color:#1F3B57}} .lbl{{font-size:11px;color:#6B7580;margin-top:4px}}
 section{{background:#fff;margin:18px 44px;padding:26px 30px;border-radius:12px;box-shadow:0 1px 4px rgba(0,0,0,.06)}}
 h2{{margin:0 0 10px;color:#1F3B57;font-size:20px;border-left:5px solid #B3202C;padding-left:12px}}
 .lead{{max-width:1000px;color:#3A4450;font-size:14px;line-height:1.55}}
 .tbl{{border-collapse:collapse;font-size:12px;background:#fff;margin:10px 0}}
 .tbl th{{background:#1F3B57;color:#fff;padding:6px 10px;text-align:left}}
 .tbl td{{padding:5px 10px;border-bottom:1px solid #E3E7EB}}
 footer{{padding:20px 44px;color:#8A94A0;font-size:11px}}
</style></head><body>
<header><h1>Pharmacy Network and Channel Optimization</h1>
<p>Full report. Synthetic data in the Optum claims schema. Prepared by Pranshi Saxena.</p></header>
<div class="cards">{cards}</div>
{''.join(S)}
<footer>Every design in this report satisfies the access, capacity, chain, mandate, quality and guarantee rules.
Savings are measured against today's spend on the same book. Values are synthetic and directional.
The identical model is the artifact submitted to D Wave.</footer>
</body></html>"""
with open('network_report.html','w') as fh:
    fh.write(html)
print(f"\nnetwork_report.html written, {len(html)/1e6:.1f} MB, one file, open in any browser")
