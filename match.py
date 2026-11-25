#!/usr/bin/env python3
"""PeriBank PE/control matching, with pregnancy-level exclusion audit."""
import argparse
import csv
import sys
import re
from pathlib import Path
import pandas as pd

csv.field_size_limit(sys.maxsize)

def read(path):
    with open(path, encoding='utf-8-sig') as f:
        first = f.readline().strip()
        directive = first.lower().startswith('sep=')
        header = f.readline() if directive else first
    sep = first[4:] if directive else max([',', '|', '\t'], key=header.count)
    if sep == r'\t': sep = '\t'
    df = pd.read_csv(path, sep=sep, skiprows=int(directive), engine='python',
                     quoting=csv.QUOTE_NONE if sep == '|' else csv.QUOTE_MINIMAL,
                     dtype=str, encoding='utf-8-sig')
    df.columns = [' '.join(c.strip().lower().split()) for c in df.columns]
    if df.columns.duplicated().any():
        raise ValueError('Duplicate headers after normalization: ' + str(df.columns[df.columns.duplicated()].tolist()))
    print(f'{path}: {len(df)} rows; delimiter={sep!r}')
    return df

def id_column(df, override):
    if override:
        c = ' '.join(override.lower().split())
        if c not in df: raise ValueError(f'Missing ID column {c!r}')
        return c
    cols = [c for c in df if 'preg' in c and 'id' in c]
    if len(cols) != 1: raise ValueError(f'Ambiguous pregnancy ID columns: {cols}; specify --pe-id or --control-id')
    return cols[0]

def ids(s):
    return s.astype('string').str.strip().str.replace(r'^(\d+)\.0+$', r'\1', regex=True).replace('', pd.NA)

def screen(df):
    """Only simple, documented clinical exclusions; missing values pass."""
    def val(c):
        if c not in df:
            return pd.Series('', index=df.index, dtype='string')
        return df[c].astype('string').fillna('').str.strip().str.lower()
    def yes(*cols):
        mask = pd.Series(False, index=df.index)
        for c in cols:
            mask |= val(c).isin(['yes', 'true', '1', '1.0', 'positive', 'present'])
        return mask
    def text(cols, pattern):
        mask = pd.Series(False, index=df.index)
        for c in cols:
            mask |= val(c).str.contains(pattern, regex=True, na=False)
        return mask
    def num(c): return pd.to_numeric(val(c), errors='coerce')
    pe_cols = ['preeclampsia/gestational hypertension_x',
               'preeclampsia/gestational hypertension_y']
    pe = yes(*pe_cols, 'preeclampsia', 'found_hellp', 'eclampsia',
             'hypertension/preeclampsia', 'significant findings - eclampsia',
             'significant intrapartum findings - eclampsia_x',
             'significant intrapartum findings - eclampsia_y') | text(
             pe_cols + ['preeclampsia', 'delivery route cesarean indication',
             'delivery route cesarean indication (other)', 'admit reason details_y',
             'indication for induced or d&e/d&c (other)'],
             r'\b(?:pre[ -]?eclampsia|hellp|eclampsia)\b')
    diabetes_cols = ['chronic diabetes', 'gestational diabetes_x',
                     'gestational diabetes_y', 'gestational diabetes (type)']
    diabetes = yes('type i diabetes_x', 'type ii diabetes_x', *diabetes_cols) | text(
        diabetes_cols, r'\b(?:gdm(?:a[12])?|a[12](?:dm)?|a\s*[12]|type\s*(?:i{1,2}|[12])|gestational diabetes|insulin|metformin|glyburide|diet[ -]?controlled)\b')
    bmi = num('pp bmi')
    height = 12*num('pp height (ft)') + num('pp height (in)')
    calculated = 703*num('pp weight') / height.pow(2)
    bmi = bmi.where(bmi.gt(0)).fillna(calculated.where(height.gt(0) & num('pp weight').gt(0)))
    df['screening_pp_bmi'] = bmi
    flags = pd.DataFrame({
        'PE_or_HELLP_control_only': pe,
        'multiple_pregnancy': num('number of babies').gt(1) | yes('multiples - dichorionic diamniotic', 'multiples - monochorionicity') | text(['type of pregnancy'], r'\b(?:twins?|triplets?|quadruplets?|multiple gestation)\b'),
        'chronic_hypertension': yes('hypertension_x'),
        'any_diabetes': diabetes,
        'prepregnancy_BMI_gt35': bmi.gt(35),
        'not_documented_no_labor': ~val('type of labor').eq('no labor'),
    }, index=df.index).fillna(False).astype(bool)
    return flags, pd.DataFrame(index=df.index)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pe',default='preeclampsia.csv')
    p.add_argument('--control',default='PBDBfinal_wterms.csv')
    p.add_argument('--out',default='PE_vs_Control_matched.csv')
    p.add_argument('--pe-id');p.add_argument('--control-id')
    p.add_argument('--audit',default='exclusion_audit.csv')
    p.add_argument('--allow-reuse',action='store_true',help='Allow a control to be used for different PE cases; never twice within a set')
    a=p.parse_args()
    pe,db=read(a.pe),read(a.control)
    pi,di=id_column(pe,a.pe_id),id_column(db,a.control_id)
    pe['_id']=ids(pe[pi]); db['_id']=ids(db[di])
    blocked=set(pe['_id'].dropna()); print(f'Unique PE list IDs: {len(blocked)}; missing PE IDs: {pe["_id"].isna().sum()}')
    db=db.dropna(subset=['_id']).copy()
    flags,review=screen(db)
    required=['type of labor','infant gender','ga at delivery (wks)','ga at delivery (d)']
    if any(c not in db for c in required): raise ValueError(f'Missing required columns: {[c for c in required if c not in db]}')
    age=pd.to_numeric(db['matage'],errors='coerce') if 'matage' in db else pd.Series(float('nan'),index=db.index)
    if {'date of birth_x','date of delivery'}.issubset(db):
        dob=pd.to_datetime(db['date of birth_x'],format='%m/%d/%Y',errors='coerce')
        delivery=pd.to_datetime(db['date of delivery'],format='%m/%d/%Y',errors='coerce')
        age=age.fillna((delivery-dob).dt.days/365.2425)
    db['MatAge']=age.where(age.between(10,65))
    db['Infant gender']=db['infant gender'].str.strip().str.lower()
    w=pd.to_numeric(db['ga at delivery (wks)'],errors='coerce'); d=pd.to_numeric(db['ga at delivery (d)'],errors='coerce')
    db['GA_days']=(7*w+d).where(w.between(15,45)&d.between(0,6)&w.mod(1).eq(0)&d.mod(1).eq(0))
    db['GA']=db['GA_days']/7
    flags['missing_or_invalid_matching_data']=db['MatAge'].isna()|db['GA_days'].isna()|~db['Infant gender'].isin(['male','female'])
    # Conservatively combine flags across duplicate database rows before selecting
    # a single representative. Conflicting matching values require review.
    keys=db['_id']; f=flags.groupby(keys,sort=False).any(); r=review.groupby(keys,sort=False).any()
    dupconf=db.groupby('_id',sort=False)[['MatAge','GA_days','Infant gender']].nunique(dropna=True).gt(1).any(axis=1)
    f['conflicting_duplicate_matching_data']=dupconf
    db=db.drop_duplicates('_id').set_index('_id',drop=False)
    common=[c for c in f if c!='PE_or_HELLP_control_only']
    is_case=f.index.isin(blocked)
    f['listed_PE_control_only']=is_case
    excluded_control=f[common+['PE_or_HELLP_control_only','listed_PE_control_only']].any(axis=1)
    excluded_case=f[common].any(axis=1)
    audit=f.join(r);audit['listed_PE']=is_case
    audit['excluded_as_control']=excluded_control; audit['excluded_as_PE']=excluded_case & is_case
    audit['status']= ['eligible_PE' if c and not ec else 'excluded_PE' if c else 'excluded_control' if ex else 'eligible_control' for c,ec,ex in zip(is_case,excluded_case,excluded_control)]
    audit['exclusion_reasons']=f.apply(lambda row:'; '.join(c for c in f if row[c] and (c not in ['PE_or_HELLP_control_only','listed_PE_control_only'] or not row['listed_PE_control_only'])),axis=1)
    missing_ids=blocked-set(db.index)
    if missing_ids:
        extra=pd.DataFrame({'listed_PE':True,'excluded_as_PE':True,'excluded_as_control':True,'status':'PE_not_in_database','exclusion_reasons':'PE ID not found in database'},index=sorted(missing_ids))
        audit=pd.concat([audit,extra])
    audit.index.name='pregnancy_id';audit.to_csv(a.audit)
    print('\nExclusion counts (overlapping reasons, unique pregnancies):')
    for c in f: print(f'{c}: controls={int(f.loc[~is_case,c].sum())}; listed PE={int(f.loc[is_case,c].sum())}')
    pool=db.loc[~excluded_control].copy()
    eligible=set(f.index[is_case & ~excluded_case]); order=list(dict.fromkeys(pe['_id'].dropna()))
    rows=[]; counts=[0]*6; pair_id=0
    pe=pe.dropna(subset=['_id']).drop_duplicates('_id').set_index('_id')
    for key in order:
        if key not in eligible:
            continue
        pair_id+=1
        case=db.loc[key]
        candidates=pool[(pool['Infant gender']==case['Infant gender']) & ((pool['GA_days']-case['GA_days']).abs()<=7)].copy()
        candidates['age_diff']=(candidates['MatAge']-case['MatAge']).abs()
        candidates['GA_diff_days']=(candidates['GA_days']-case['GA_days']).abs()
        best=candidates.sort_values(['age_diff','GA_diff_days'],kind='stable').head(5)
        counts[len(best)]+=1
        # Keep database fields for both groups; append PE-list-only fields.
        case_data=case.to_dict(); case_data.update({c:v for c,v in pe.loc[key].items() if c not in case_data})
        status='complete' if len(best)==5 else 'partial' if len(best) else 'no_eligible_control'
        rows.append(dict(case_data,pregnancy_id=key,group='PE',pair_id=pair_id,match_rank=0,age_diff=0,GA_diff_days=0,match_status=status,controls_found=len(best)))
        for rank,(cid,control) in enumerate(best.iterrows(),1):
            rows.append(dict(control,pregnancy_id=cid,group='Control',pair_id=pair_id,match_rank=rank,match_status=status,controls_found=len(best)))
        if not a.allow_reuse: pool=pool.drop(index=best.index)
    out=pd.DataFrame(rows).drop(columns=['_id'],errors='ignore') if rows else pd.DataFrame(columns=['pregnancy_id','group','pair_id','match_rank','age_diff','GA_diff_days'])
    front=['pair_id','group','match_rank','pregnancy_id','MatAge','Infant gender','GA_days','age_diff','GA_diff_days','controls_found','match_status','match_note']
    out=out.reindex(columns=front+[c for c in out if c not in front])
    out.to_csv(a.out,index=False)
    print(f'Output rows: {len(out)}; PE rows: {pair_id}; control rows: {len(out)-pair_id}; control reuse: {a.allow_reuse}')
    print(f'\nPE IDs missing from database: {len(missing_ids)}; eligible PE: {len(eligible)}; initial eligible controls: {int((~excluded_control).sum())}')
    print(f'Excluded PE cases (audit only): {len(order)-len(eligible)}')
    for i,n in enumerate(counts): print(f'{i} controls: {n} PE cases')
    print(f'Matched output: {a.out}; screening audit: {a.audit}')

if __name__=='__main__': main()
