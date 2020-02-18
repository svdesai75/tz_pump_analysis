from haversine import haversine
import pandas as pd

def calc_precision(row):
    p1=(row.latitude_mean,
        row.longitude_mean)
    p2=(row.latitude_mean + row.latitude_std,
        row.longitude_mean + row.longitude_std)

    return haversine(p1,p2) #default return unit is km

def make_geo_map(ref_df, region_fields, do_plots=False):
    geo_map=ref_df.groupby(region_fields).\
                   agg(npumps=("id","count"),
                       latitude_mean=("latitude","mean"),
                       longitude_mean=("longitude","mean"),
                       latitude_std=("latitude","std"),
                       longitude_std=("longitude","std")).\
                   reset_index()
    
    geo_map["precision"] = geo_map.apply(calc_precision, axis=1)
    
    err_map=geo_map.query("npumps>1")
    geo_map["precision"].fillna(err_map.precision.mean(), inplace=True)
    return geo_map

def get_keys(m):
    data_fields=["npumps","precision","longitude_mean","longitude_std","latitude_mean","latitude_std"]
    keys = [c for c in m.columns if c not in data_fields]
    return keys




def calc_approx_coords(pump, geo_maps):
    pump=pd.DataFrame(pump).transpose()
    best_match=None
    for m in geo_maps:
        keys=get_keys(m)
        match=pump.merge(m,on=keys)
        if match.shape[0]>1:
            print ("ERROR multiple matches for ", keys)
        elif match.shape[0]==0:
            continue
        match=match.loc[0]
        
        if best_match is None:
            best_match=match
        elif match.precision < best_match.precision:
            best_match=match


            
    result_cols=["id","latitude_mean","longitude_mean","precision"]
    rename_map={"latitude_mean":"approx_latitude",
                "longitude_mean":"approx_longitude",
                "precision":"loc_precision"}
    results=best_match[result_cols].rename(rename_map)
    return results

def add_approx_coords(d):

    ref     = d.query("longitude > 10")
    missing = d.query("longitude < 10")

    geo_maps=[]
    geo_maps.append(make_geo_map(ref, ["region","ward","subvillage"]))
    geo_maps.append(make_geo_map(ref,       ["ward","district_code"]))
    geo_maps.append(make_geo_map(ref,                        ["lga"]))
    geo_maps.append(make_geo_map(ref,                     ["region"]))

    approx_coords=missing.apply(lambda p: calc_approx_coords(p, geo_maps), axis=1)
    out_df = d.merge(approx_coords, on="id", how="left")
    out_df.approx_latitude  = out_df.approx_latitude.combine_first(out_df.latitude)
    out_df.approx_longitude = out_df.approx_longitude.combine_first(out_df.longitude)
    out_df.loc_precision    = out_df.loc_precision.fillna(0)
    return out_df
