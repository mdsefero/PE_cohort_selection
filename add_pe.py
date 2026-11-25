import pandas as pd, argparse, csv, sys
csv.field_size_limit(sys.maxsize)

p=argparse.ArgumentParser()
p.add_argument("--wterms", required=True)  # PBDBfinal_wterms.csv (sep=|, first line sep=|)
p.add_argument("--pe", required=True)      # preeclampsia.csv (comma)
p.add_argument("--out", default="preeclampsia_wterms.csv")
a=p.parse_args()

w=pd.read_csv(a.wterms, sep="|", skiprows=1, engine="python", quoting=3)
pe=pd.read_csv(a.pe)

id_w = next((c for c in w.columns if "preg" in c.lower() and "id" in c.lower()), w.columns[0])
id_pe= next((c for c in pe.columns if "preg" in c.lower() and "id" in c.lower()), pe.columns[0])

term_cols=[c for c in w.columns if c.lower().startswith("found_")]
out=pe.merge(w[[id_w]+term_cols], left_on=id_pe, right_on=id_w, how="left").drop(columns=[id_w])

out.to_csv(a.out, index=False)
print(a.out)
