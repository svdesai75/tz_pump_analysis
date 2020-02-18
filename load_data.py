import pandas as pd
from approx_coords import add_approx_coords

import re

train_values_fname = 'https://s3.amazonaws.com/drivendata/data/7/public/4910797b-ee55-40a7-8668-10efd5c1b960.csv'
train_labels_fname = 'https://s3.amazonaws.com/drivendata/data/7/public/0bf8bc6e-30d0-4c50-956a-603fc693d966.csv'
test_values_fname  = 'https://s3.amazonaws.com/drivendata/data/7/public/702ddfc5-68cd-4d1d-a0de-f5f566f76d91.csv'

def remove_whitespace(s):
    return re.sub("[\s+,/]","_",s)

def make_counts(d,field_name):
    counts = d.groupby(field_name).agg(count=("id","count")).reset_index()
    count_name = "{}_count".format(field_name)
    return counts.rename(columns={"count":count_name})

def correct_installer(i):
    inew=i.lower()
    inew = re.sub('adra\s?/\s?','adra/',inew)
    inew = re.sub('angli(can church)?','anglican church',inew)
    inew = re.sub('anglican?( church)?','anglican church',inew)
    inew = re.sub('arabs?','arab',inew)
    inew = re.sub('arabs?','arab',inew)
    inew = re.sub('ce[bn]tr?al?','central',inew)
    inew = re.sub('church?','church',inew)
    inew = re.sub('comm?unity?','community',inew)
    inew = re.sub('concer?n?\s?','concern',inew)
    inew = re.sub('consul?tant eng','consulting eng',inew)
    inew = re.sub('consul?ting eng','consulting eng',inew)
    inew = re.sub('govt','government',inew)
    inew = re.sub('gover[nm](ment)?','government',inew)
    inew = re.sub('gover[nm]e(nt)?','government',inew)
    inew = re.sub('government?','government',inew)
    inew = re.sub('ch?ristan','christian',inew)
    inew = re.sub('council?','council',inew)
    inew = re.sub('depar(tment)?','department',inew)
    inew = re.sub('dr\.\s?mato[bm]ola','dr. matabola',inew)
    inew = re.sub('^howard.*','howard and humfrey consultants',inew)
    inew = re.sub('^humphreys.*','howard and humfrey consultants',inew)
    inew = re.sub('wells?','wells',inew)
    inew = re.sub('\s+', ' ', inew)
    inew = re.sub('^african? m.*','african muslim agency',inew)

    return inew

def load_data(subsample):
    print("Load data, {}".format(subsample))
    if subsample=="training":
        values_df=pd.read_csv(train_values_fname,
                              parse_dates=["date_recorded"])
        labels_df=pd.read_csv(train_labels_fname)
        tmpdf=values_df.merge(labels_df, on="id").replace(" ","_")
    elif subsample=="testing":
        tmpdf = pd.read_csv(test_values_fname,
                            parse_dates=["date_recorded"])

    else:
        raise ValueError("Unknown subsample {}".format(subsample))

    tmpdf[        "is_dry"] = tmpdf.quantity=='dry'
    tmpdf[     "other_wpt"] = tmpdf.waterpoint_type=='other'

    tmpdf["month_recorded"] = tmpdf.date_recorded.dt.month
    tmpdf[ "year_recorded"] = tmpdf.date_recorded.dt.year

    if subsample=='training':
        tmpdf.status_group      = tmpdf.status_group.apply(remove_whitespace)
        
    tmpdf.source_type       = tmpdf.source_type.apply(remove_whitespace)
    tmpdf.source            = tmpdf.source.apply(remove_whitespace)
#    tmpdf.installer         = tmpdf.installer.str.lower().fillna('na')
    tmpdf.installer         = tmpdf.installer.fillna('na').apply(correct_installer)
    tmpdf.funder            = tmpdf.funder.fillna('na').str.lower()

    tmpdf.scheme_management.fillna('na', inplace=True)

    installer_counts = make_counts(tmpdf,"installer")
    funder_counts    = make_counts(tmpdf,"funder")
    tmpdf = tmpdf.merge(installer_counts,on="installer").merge(funder_counts,on="funder")
    
    return add_approx_coords(tmpdf)
 
