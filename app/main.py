# app/main.py
import os
import re
from typing import Optional
from collections import defaultdict

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ----------------- Config -----------------
DATA_DIR = os.getenv("DATA_DIR", "data")
DATE_COLS = ["Date"]

# ----------------- Helpers -----------------
def pick_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Retourne la 1re colonne trouvée parmi candidates."""
    if df is None or df.empty:
        return None
    for c in candidates:
        if c in df.columns:
            return c
    return None

# ----------------- App init -----------------
app = FastAPI(title="BI API – Ratings/Restaurants", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # adapte si besoin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Data loading -----------------
def _read_csv(name: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        return pd.DataFrame()
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    parse = [c for c in DATE_COLS if c in cols]
    return pd.read_csv(path, parse_dates=parse, dayfirst=True)

def load_frames():
    restaurants = _read_csv("restaurants.csv")
    ratings = _read_csv("ratings.csv")
    cuisines = _read_csv("restaurant_cuisines.csv")
    consumers = _read_csv("consumers.csv")
    cons_prefs = _read_csv("consumer_preferences.csv")

    # Join principal (ratings + restaurants)
    if not ratings.empty and not restaurants.empty \
       and "Restaurant_ID" in ratings.columns and "Restaurant_ID" in restaurants.columns:
        df = ratings.merge(restaurants, on="Restaurant_ID", how="left")
    else:
        df = ratings

    # Joindre cuisines si dispo
    if not df.empty and not cuisines.empty \
       and "Restaurant_ID" in cuisines.columns and "Restaurant_ID" in df.columns:
        df = df.merge(cuisines, on="Restaurant_ID", how="left")

    return restaurants, ratings, cuisines, consumers, cons_prefs, df

RESTAURANTS, RATINGS, CUISINES, CONSUMERS, CONS_PREFS, DF = load_frames()

# colonnes détectées une fois pour toutes
RATING_COL = pick_col(RATINGS, ["Overall_Rating", "Rating", "overall_rating"])
RID_COL = pick_col(RATINGS, ["Restaurant_ID", "Rest_ID", "RID"])
CITY_COL_REST = pick_col(RESTAURANTS, ["City", "city"])
CITY_COL_CONS = pick_col(CONSUMERS, ["City", "city"])
PRICE_COL = pick_col(RESTAURANTS, ["Price", "Price_Range"])
ALCO_COL = pick_col(RESTAURANTS, ["Alcohol_Service", "alcohol_service", "Alcohol", "Alcohol service"])
DRINK_LVL_COL = pick_col(CONSUMERS, ["Drink_Level", "drink_level"])
CONS_ID_COL = pick_col(CONSUMERS, ["Consumer_ID", "consumer_id"])
RATINGS_CONS_ID_COL = pick_col(RATINGS, ["Consumer_ID", "consumer_id"])
AGE_COL = pick_col(CONSUMERS, ["Age", "age"])
BUDGET_COL = pick_col(CONSUMERS, ["Budget", "Budget_Level", "Budget_Category", "budget"])

def _filter_df(df: pd.DataFrame, city: Optional[str], price: Optional[str], alcohol: Optional[str]):
    work = df.copy()
    if city and CITY_COL_REST and CITY_COL_REST in work.columns:
        work = work[work[CITY_COL_REST].astype(str).str.lower() == city.lower()]
    if price and PRICE_COL and PRICE_COL in work.columns:
        work = work[work[PRICE_COL] == price]
    if alcohol and "Alcohol_Clean" in work.columns:
        work = work[work["Alcohol_Clean"] == alcohol]
    return work

# ----------------- Endpoints de base -----------------
@app.get("/health")
def health():
    return {"ok": True, "rows": int(DF.shape[0]) if DF is not None else 0}

@app.get("/metrics/kpis")
def get_kpis(city: Optional[str] = None, price: Optional[str] = None, alcohol: Optional[str] = None):
    if DF is None or DF.empty:
        return {"total_reviews": 0, "avg_rating": None, "restaurants": 0, "alcohol_clean_pct": None}
    work = _filter_df(DF, city, price, alcohol)
    total_reviews = int(work.shape[0])
    restaurants = int(work["Restaurant_ID"].nunique()) if "Restaurant_ID" in work.columns else 0
    avg = float(work[RATING_COL].mean()) if RATING_COL and RATING_COL in work.columns and total_reviews else None
    alc = None
    if ALCO_COL and ALCO_COL in RESTAURANTS and not RESTAURANTS.empty:
        alc_series = RESTAURANTS[ALCO_COL].astype(str).str.lower()
        alc = round(100 * alc_series.str.contains("no|none|not served|no alcohol").mean(), 1)
    return {
        "total_reviews": total_reviews,
        "avg_rating": round(avg, 2) if avg is not None else None,
        "restaurants": restaurants,
        "alcohol_clean_pct": alc
    }

@app.get("/metrics/top-restaurants")
def top_restaurants(n: int = 10, city: Optional[str] = None, price: Optional[str] = None, alcohol: Optional[str] = None):
    if DF is None or DF.empty or not RATING_COL or not RID_COL:
        return []
    work = _filter_df(DF, city, price, alcohol)
    if RID_COL not in work.columns: 
        return []
    keep = [RID_COL] + [c for c in ["Name", RATING_COL] if c in work.columns]
    g = (work[keep]
         .groupby([RID_COL] + ([c for c in ["Name"] if c in keep]), dropna=False)
         .agg(Rating_Count=(RATING_COL, "count"),
              Avg_Rating=(RATING_COL, "mean"))
         .reset_index()
         .sort_values(["Rating_Count","Avg_Rating"], ascending=[False, False])
         .head(n))
    if "Avg_Rating" in g.columns: 
        g["Avg_Rating"] = g["Avg_Rating"].round(2)
    return g.to_dict(orient="records")

@app.get("/metrics/by-city")
def by_city():
    if DF is None or DF.empty or not CITY_COL_REST or not RATING_COL:
        return []
    city = (DF.groupby(CITY_COL_REST, dropna=False)[RATING_COL]
            .mean().reset_index().rename(columns={RATING_COL: "Avg_Rating"})
            .sort_values("Avg_Rating", ascending=False))
    city["Avg_Rating"] = city["Avg_Rating"].round(2)
    return city.to_dict(orient="records")

# ----------------- Analytics -----------------
@app.get("/analytics/price-by-city")
def price_by_city():
    if RESTAURANTS is None or RESTAURANTS.empty or not CITY_COL_REST or not PRICE_COL:
        return {"cities": [], "prices": [], "matrix": []}
    df = RESTAURANTS[[CITY_COL_REST, PRICE_COL]].astype(str)
    df[CITY_COL_REST] = df[CITY_COL_REST].str.strip().str.title()
    df[PRICE_COL] = df[PRICE_COL].str.strip().str.title()
    pv = df.value_counts([CITY_COL_REST, PRICE_COL]).rename("count").reset_index()
    cities = sorted(pv[CITY_COL_REST].unique().tolist())
    prices = sorted(pv[PRICE_COL].unique().tolist())
    grid = defaultdict(lambda: defaultdict(int))
    for _, r in pv.iterrows():
        grid[r[CITY_COL_REST]][r[PRICE_COL]] = int(r["count"])
    matrix = [[grid[c].get(p, 0) for p in prices] for c in cities]
    return {"cities": cities, "prices": prices, "matrix": matrix}

@app.get("/analytics/cuisine-share")
def cuisine_share():
    source, col = None, None
    if CONS_PREFS is not None and not CONS_PREFS.empty:
        col = pick_col(CONS_PREFS, ["Preferred_Cuisine","Cuisine"])
        if col: source = CONS_PREFS
    if source is None and CUISINES is not None and not CUISINES.empty:
        col = pick_col(CUISINES, ["Cuisine","Cuisines"])
        if col: source = CUISINES
    if source is None or col is None:
        return {"labels": [], "values": []}

    s = source[col].astype(str)
    if s.str.contains(r"[,\|]").any():
        s = s.str.split(r"[,\|]").explode()
    s = (s.str.strip().str.title()
           .replace({"": np.nan, "Nan": np.nan, "Null": np.nan, "None": np.nan})
           .dropna())
    vc = s.value_counts()
    if vc.empty: 
        return {"labels": [], "values": []}
    top5 = vc.head(5)
    other = int(vc.iloc[5:].sum()) if len(vc) > 5 else 0
    labels = top5.index.tolist() + (["Other"] if other>0 else [])
    values = top5.astype(int).tolist() + ([other] if other>0 else [])
    return {"labels": labels, "values": values}

@app.get("/analytics/alcohol-service")
def alcohol_service():
    if RESTAURANTS is None or RESTAURANTS.empty or not ALCO_COL:
        return {"labels": [], "values": []}
    s = RESTAURANTS[ALCO_COL].astype(str)

    if s.str.contains(r"[,\|]").any():
        s = s.str.split(r"[,\|]").explode()

    s = (s.str.strip().str.replace(r"\s+"," ", regex=True).str.title()
           .replace({"": np.nan, "Nan": np.nan, "Null": np.nan, "None": np.nan, "N/A": np.nan})
           .dropna())

    def normalize(v: str) -> str:
        v = v.lower().strip()
        if v in {"no","non","none","false","0","not served","no alcohol"}: return "No Alcohol"
        if "byo" in v: return "BYO"
        if any(k in v for k in ["full bar","cocktail","spirits","liquor"]): return "Full Bar"
        if any(k in v for k in ["beer & wine","beer and wine","wine & beer","beer wine","beer","wine"]): return "Beer & Wine"
        if v in {"unknown","na","n/a","unspecified","yes","true","1"}: return "Unknown / Yes"
        if re.search(r"(serve|alcohol|bar)", v): return "Unknown / Yes"
        return "Other"

    norm = s.map(normalize)
    vc = norm.value_counts()
    if vc.empty: 
        return {"labels": [], "values": []}

    order = ["No Alcohol","Beer & Wine","Full Bar","BYO","Unknown / Yes","Other"]
    vc = vc.reindex([x for x in order if x in vc.index]).fillna(0).astype(int)
    return {"labels": vc.index.tolist(), "values": vc.tolist()}

@app.get("/analytics/ratings-by-restaurant")
def ratings_by_restaurant(top: int = 12):
    if RATINGS is None or RATINGS.empty or not RID_COL:
        return {"labels": [], "values": []}
    counts = RATINGS[RID_COL].dropna().value_counts().head(int(top))
    labels = counts.index.astype(str).tolist()
    if RESTAURANTS is not None and not RESTAURANTS.empty and RID_COL in RESTAURANTS.columns:
        name_col = pick_col(RESTAURANTS, ["Restaurant_Name","Name","Restaurant","Rest_Name"])
        if name_col:
            id_to_name = RESTAURANTS.set_index(RID_COL)[name_col].astype(str).to_dict()
            labels = [id_to_name.get(x, f"ID {x}") for x in counts.index]
    return {"labels": labels, "values": counts.astype(int).tolist()}

@app.get("/analytics/age-histogram")
def age_hist(bins: int = 20):
    if CONSUMERS is None or CONSUMERS.empty or not AGE_COL:
        return {"bins": [], "counts": []}
    s = pd.to_numeric(CONSUMERS[AGE_COL], errors="coerce").dropna()
    if s.empty: 
        return {"bins": [], "counts": []}
    counts, edges = np.histogram(s, bins=int(bins))
    centers = ((edges[:-1] + edges[1:]) / 2).tolist()
    return {"bins": centers, "counts": counts.astype(int).tolist()}

@app.get("/analytics/avg-vs-count")
def avg_vs_count(top: int = 10):
    if RATINGS is None or RATINGS.empty or not RID_COL or not RATING_COL:
        return []
    agg = (RATINGS.groupby(RID_COL)
           .agg(ratings_count=(RATING_COL,"count"), avg_overall=(RATING_COL,"mean"))
           .sort_values("ratings_count", ascending=False)
           .head(int(top))
           .reset_index())
    if RESTAURANTS is not None and not RESTAURANTS.empty and RID_COL in RESTAURANTS.columns:
        name_col = pick_col(RESTAURANTS, ["Restaurant_Name","Name","Restaurant","Rest_Name"])
        if name_col:
            id_to_name = RESTAURANTS.set_index(RID_COL)[name_col].astype(str).to_dict()
            agg["label"] = agg[RID_COL].map(lambda x: id_to_name.get(x, f"ID {x}"))
        else:
            agg["label"] = agg[RID_COL].astype(str)
    else:
        agg["label"] = agg[RID_COL].astype(str)

    agg["avg_overall"] = agg["avg_overall"].round(2)
    return agg[["label","ratings_count","avg_overall"]].to_dict(orient="records")

@app.get("/analytics/drink-levels-by-city")
def drink_levels_by_city():
    if CONSUMERS is None or CONSUMERS.empty or not CITY_COL_CONS or not DRINK_LVL_COL:
        return {"cities": [], "levels": [], "matrix": []}
    df = CONSUMERS[[CITY_COL_CONS, DRINK_LVL_COL]].astype(str)
    df[CITY_COL_CONS] = df[CITY_COL_CONS].str.strip().str.title()
    df[DRINK_LVL_COL] = df[DRINK_LVL_COL].str.strip().str.title()
    pv = df.value_counts([CITY_COL_CONS, DRINK_LVL_COL]).rename("count").reset_index()
    cities = sorted(pv[CITY_COL_CONS].unique().tolist())
    levels = sorted(pv[DRINK_LVL_COL].unique().tolist())
    grid = defaultdict(lambda: defaultdict(int))
    for _, r in pv.iterrows():
        grid[r[CITY_COL_CONS]][r[DRINK_LVL_COL]] = int(r["count"])
    matrix = [[grid[c].get(l,0) for l in levels] for c in cities]
    return {"cities": cities, "levels": levels, "matrix": matrix}

@app.get("/analytics/budget-by-occupation")
def budget_by_occupation():
    if CONSUMERS is None or CONSUMERS.empty:
        return {"occupations": [], "buckets": [], "matrix": []}
    occ_col = pick_col(CONSUMERS, ["Occupation"])
    bud_col = BUDGET_COL
    if not occ_col or not bud_col:
        return {"occupations": [], "buckets": [], "matrix": []}
    df = CONSUMERS[[occ_col, bud_col]].astype(str)
    df[occ_col] = df[occ_col].str.strip().str.title()
    df[bud_col] = df[bud_col].str.strip().str.title()
    pv = df.value_counts([occ_col, bud_col]).rename("count").reset_index()
    occs = sorted(pv[occ_col].unique().tolist())
    buckets = sorted(pv[bud_col].unique().tolist())
    grid = defaultdict(lambda: defaultdict(int))
    for _, r in pv.iterrows():
        grid[r[occ_col]][r[bud_col]] = int(r["count"])
    matrix = [[grid[o].get(b,0) for b in buckets] for o in occs]
    return {"occupations": occs, "buckets": buckets, "matrix": matrix}

# --------- Nouveaux endpoints requis par ta page RestaurantRatings ---------
@app.get("/analytics/budget-by-age")
def budget_by_age(binsize: int = 1):
    """
    Retourne un objet:
    { labels:[ages], series:{ High:[], Medium:[], Low:[] } }
    """
    if CONSUMERS is None or CONSUMERS.empty or not AGE_COL or not BUDGET_COL:
        return {"labels": [], "series": {"High": [], "Medium": [], "Low": []}}

    df = CONSUMERS[[AGE_COL, BUDGET_COL]].copy()
    df[AGE_COL] = pd.to_numeric(df[AGE_COL], errors="coerce")
    df = df.dropna(subset=[AGE_COL])

    # normalisation budget
    df[BUDGET_COL] = df[BUDGET_COL].astype(str).str.strip().str.lower()
    map_budget = {
        "high": "High", "élevé": "High", "eleve": "High",
        "medium": "Medium", "moyen": "Medium", "moyenne": "Medium",
        "low": "Low", "faible": "Low"
    }
    df["Budget_norm"] = df[BUDGET_COL].map(map_budget).fillna(df[BUDGET_COL].str.title())

    xmin, xmax = int(df[AGE_COL].min()), int(df[AGE_COL].max())
    if xmin >= xmax:
        return {"labels": [], "series": {"High": [], "Medium": [], "Low": []}}

    edges = np.arange(xmin, xmax + binsize, binsize)
    labels = ((edges[:-1] + edges[1:]) / 2).astype(int).tolist()

    out = {"High": [], "Medium": [], "Low": []}
    for cat in ["High", "Medium", "Low"]:
        sub = df.loc[df["Budget_norm"] == cat, AGE_COL].to_numpy()
        counts, _ = np.histogram(sub, bins=edges) if sub.size else (np.zeros(len(labels)), None)
        out[cat] = counts.astype(int).tolist()

    return {"labels": labels, "series": out}

@app.get("/analytics/drink-satisfaction")
def drink_satisfaction():
    """
    Retourne: { labels:[Abstemious, Casual Drinker, Social Drinker], values:[moyennes] }
    """
    if (CONSUMERS is None or CONSUMERS.empty or
        RATINGS is None or RATINGS.empty or
        not CONS_ID_COL or not RATINGS_CONS_ID_COL or not RATING_COL or not DRINK_LVL_COL):
        return {"labels": [], "values": []}

    cons = CONSUMERS[[CONS_ID_COL, DRINK_LVL_COL]].copy()
    cons[DRINK_LVL_COL] = cons[DRINK_LVL_COL].astype(str).str.strip().str.lower()

    merged = RATINGS.merge(cons, left_on=RATINGS_CONS_ID_COL, right_on=CONS_ID_COL, how="left")
    if merged.empty:
        return {"labels": [], "values": []}

    g = merged.groupby(DRINK_LVL_COL, dropna=True)[RATING_COL].mean().dropna()
    if g.empty:
        return {"labels": [], "values": []}

    order_raw = ["abstemious", "casual drinker", "social drinker"]
    ordered = [x for x in order_raw if x in g.index]
    g = g.reindex(ordered)

    labels = [x.title() for x in g.index]
    vals = g.round(2).tolist()
    return {"labels": labels, "values": vals}

# ----------------- Investment (reste identique, avec robustesse) -----------------
@app.get("/investor/kpis")
def investor_kpis(city: str | None = None):
    if RESTAURANTS.empty or CONSUMERS.empty or RATINGS.empty:
        return {"restaurants": 0, "consumers": 0, "cities": 0, "avg_rating": None}

    rest_f = RESTAURANTS.copy()
    cons_f = CONSUMERS.copy()
    rate_f = RATINGS.copy()

    if city:
        if CITY_COL_REST and CITY_COL_REST in rest_f:
            rest_f = rest_f[rest_f[CITY_COL_REST].astype(str).str.lower() == city.lower()]
        if CITY_COL_CONS and CITY_COL_CONS in cons_f:
            cons_f = cons_f[cons_f[CITY_COL_CONS].astype(str).str.lower() == city.lower()]

    kpi_restaurants = len(rest_f)
    kpi_consumers = len(cons_f)
    kpi_cities = RESTAURANTS[CITY_COL_REST].nunique() if CITY_COL_REST in RESTAURANTS else 0
    kpi_avg_rating = rate_f[RATING_COL].mean() if RATING_COL in rate_f else None

    return {
        "restaurants": int(kpi_restaurants),
        "consumers": int(kpi_consumers),
        "cities": int(kpi_cities),
        "avg_rating": round(float(kpi_avg_rating), 2) if kpi_avg_rating is not None else None,
    }

@app.get("/analytics/price-count-stacked")
def price_count_stacked(city: str | None = None):
    if RESTAURANTS.empty or not PRICE_COL:
        return {"labels": [], "values": []}
    df = RESTAURANTS.copy()
    if city and CITY_COL_REST in df:
        df = df[df[CITY_COL_REST].astype(str).str.lower() == city.lower()]
    counts = df[PRICE_COL].astype(str).str.title().value_counts().reindex(["Low","Medium","High"]).fillna(0)
    return {"city": city or "All", "labels": counts.index.tolist(), "values": counts.astype(int).tolist()}

@app.get("/analytics/top-cuisines")
def top_cuisines(city: str | None = None):
    if CUISINES.empty:
        return {"labels": [], "values": []}
    df = CUISINES.copy()
    if city and CITY_COL_REST and CITY_COL_REST in RESTAURANTS:
        rids = RESTAURANTS.loc[RESTAURANTS[CITY_COL_REST].astype(str).str.lower() == city.lower(), "Restaurant_ID"]
        if "Restaurant_ID" in df:
            df = df[df["Restaurant_ID"].isin(rids)]
    if "Cuisine" not in df:
        return {"labels": [], "values": []}
    top = df["Cuisine"].value_counts().head(5)
    return {"labels": top.index.tolist(), "values": top.values.tolist()}

@app.get("/analytics/franchise-count")
def franchise_count(city: str | None = None):
    if RESTAURANTS.empty or "FranchiseNorm" not in RESTAURANTS:
        return {"labels": [], "values": []}
    df = RESTAURANTS.copy()
    if city and CITY_COL_REST in df:
        df = df[df[CITY_COL_REST].astype(str).str.lower() == city.lower()]
    counts = df["FranchiseNorm"].value_counts().head(10)
    return {"labels": counts.index.tolist(), "values": counts.astype(int).tolist()}

@app.get("/analytics/top-cuisines-by-price-100")
def top_cuisines_by_price_100(city: str | None = None):
    if CUISINES.empty or RESTAURANTS.empty or not PRICE_COL:
        return {"cuisines": [], "buckets": [], "matrix_pct": []}
    tmp = CUISINES.merge(RESTAURANTS, on="Restaurant_ID", how="left")
    if city and CITY_COL_REST in tmp.columns:
        tmp = tmp[tmp[CITY_COL_REST].astype(str).str.lower() == city.lower()]
    if "Cuisine" not in tmp or PRICE_COL not in tmp:
        return {"cuisines": [], "buckets": [], "matrix_pct": []}
    top_cuisines = tmp["Cuisine"].value_counts().head(5).index
    tmp = tmp[tmp["Cuisine"].isin(top_cuisines)]
    ctab = pd.crosstab(tmp["Cuisine"], tmp[PRICE_COL]).reindex(columns=["Low", "Medium", "High"]).fillna(0)
    ctab_pct = ctab.div(ctab.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    return {"cuisines": ctab_pct.index.tolist(), "buckets": ctab_pct.columns.tolist(), "matrix_pct": ctab_pct.values.tolist()}

@app.get("/kpi/avg-rating")
def kpi_avg_rating():
    if RATINGS is None or RATINGS.empty or not RATING_COL:
        return {"value": None, "label": "Moyenne de la note des restaurants"}
    s = pd.to_numeric(RATINGS[RATING_COL], errors="coerce").dropna()
    val = round(float(s.mean()), 2) if not s.empty else None
    return {"value": val, "label": "Moyenne de la note des restaurants", "icon": "star"}

@app.get("/analytics/consumers-occupation-budget")
def consumers_occupation_budget():
    if CONSUMERS is None or CONSUMERS.empty:
        return {"occupations": [], "budgets": [], "matrix": []}

    occ_col = pick_col(CONSUMERS, ["Occupation","job","Job","Employment_Status","employment","occupation"])
    bud_col = pick_col(CONSUMERS, ["Budget","Spending_Level","Budget_Category","budget","Spending"])
    if not occ_col or not bud_col:
        return {"occupations": [], "budgets": [], "matrix": []}

    df = CONSUMERS[[occ_col, bud_col]].dropna().copy()
    df[occ_col] = df[occ_col].astype(str).str.strip().str.title()
    df[bud_col] = df[bud_col].astype(str).str.strip().str.title()

    occ_order = ["Student", "Employed", "Unemployed"]
    bud_order = ["High", "Low", "Medium"]

    ctab = (pd.crosstab(df[occ_col], df[bud_col])
              .reindex(index=occ_order, columns=bud_order)
              .fillna(0).astype(int))
    return {"occupations": ctab.index.tolist(),
            "budgets": ctab.columns.tolist(),
            "matrix": ctab.values.tolist()}

@app.get("/analytics/ratings-count-top10")
def ratings_count_top10():
    if RATINGS is None or RATINGS.empty or RESTAURANTS is None or RESTAURANTS.empty:
        return {"labels": [], "values": [], "percents": []}

    rid_r = pick_col(RATINGS,     ["Restaurant_ID","Rest_ID","RID","restaurant_id"])
    rid_s = pick_col(RESTAURANTS, ["Restaurant_ID","Rest_ID","RID","restaurant_id"])
    name_c = pick_col(RESTAURANTS,["Restaurant Name","Name","Restaurant","Rest_Name","rest_name"])
    if not all([rid_r, rid_s, name_c]):
        return {"labels": [], "values": [], "percents": []}

    df = RATINGS.merge(RESTAURANTS[[rid_s, name_c]], left_on=rid_r, right_on=rid_s, how="inner")
    df[name_c] = df[name_c].astype(str).str.strip()

    counts = (df.groupby(name_c).size()
                .reset_index(name="RatingCount")
                .sort_values("RatingCount", ascending=False)
                .head(10))
    total = int(counts["RatingCount"].sum()) or 1
    perc = (counts["RatingCount"] / total * 100).round(2).tolist()

    return {"labels": counts[name_c].tolist(),
            "values": counts["RatingCount"].astype(int).tolist(),
            "percents": perc}

@app.get("/analytics/ratings-qual-by-name")
def ratings_quality_by_name():
    if RATINGS is None or RATINGS.empty or RESTAURANTS is None or RESTAURANTS.empty:
        return {"names": [], "categories": [], "matrix": []}

    rid_r = pick_col(RATINGS,     ["Restaurant_ID","Rest_ID","RID","restaurant_id"])
    rid_s = pick_col(RESTAURANTS, ["Restaurant_ID","Rest_ID","RID","restaurant_id"])
    name_c = pick_col(RESTAURANTS,["Restaurant Name","Name","Restaurant","Rest_Name","rest_name"])
    if not all([rid_r, rid_s, name_c, RATING_COL]):
        return {"names": [], "categories": [], "matrix": []}

    ratings = RATINGS[[rid_r, RATING_COL]].copy()
    ratings[RATING_COL] = pd.to_numeric(ratings[RATING_COL], errors="coerce")

    # Buckets qualité (équivalent SWITCH PBI)
    def to_bucket(x: float) -> str | None:
        if pd.isna(x): return None
        if x <= 0.99:  return "Faible"
        if x <= 1.99:  return "Moyenne"
        return "Élevée"

    ratings["Rating Quality Text"] = ratings[RATING_COL].map(to_bucket)
    df = ratings.merge(RESTAURANTS[[rid_s, name_c]], left_on=rid_r, right_on=rid_s, how="inner")

    agg = (df.dropna(subset=[name_c, "Rating Quality Text"])
             .groupby([name_c, "Rating Quality Text"]).size()
             .rename("Ratings Count").reset_index())

    qual_order = ["Élevée", "Faible", "Moyenne"]
    pv = (agg.pivot(index=name_c, columns="Rating Quality Text", values="Ratings Count")
             .fillna(0).reindex(columns=qual_order))

    top = pv.sum(axis=1).sort_values(ascending=False).head(10).index
    pv = pv.loc[top]

    return {"names": pv.index.astype(str).tolist(),
            "categories": pv.columns.tolist(),
            "matrix": pv.values.astype(int).tolist()}

@app.get("/analytics/pref-cuisine-top10")
def pref_cuisine_top10():
    if CUISINES is None or CUISINES.empty:
        return {"labels": [], "values": []}
    col = pick_col(CUISINES, ["Cuisine","Cuisines","Preferred_Cuisine","Type_Cuisine"])
    if not col:
        return {"labels": [], "values": []}

    vc = (CUISINES[col].dropna().astype(str).str.strip()
           .value_counts().head(10))
    return {"labels": vc.index.tolist(), "values": vc.astype(int).tolist()}

@app.get("/analytics/consumers-age-bins")
def consumers_age_bins():
    if CONSUMERS is None or CONSUMERS.empty:
        return {"labels": [], "values": []}

    age_col = pick_col(CONSUMERS, ["Age","age","AGE"])
    if not age_col:
        return {"labels": [], "values": []}

    df = CONSUMERS[[age_col]].copy()
    df[age_col] = pd.to_numeric(df[age_col], errors="coerce")
    df = df.dropna()
    df = df[df[age_col] > 0]

    bins = list(range(0, 90, 10))          # [0-10, 10-20, ..., 80-90)
    labels = [f"{b}" for b in bins[:-1]]   # "0","10","20",...
    cats = pd.cut(df[age_col], bins=bins, labels=labels, right=False)
    vc = cats.value_counts().sort_index()

    return {"labels": vc.index.astype(str).tolist(),
            "values": vc.astype(int).tolist()}

@app.get("/kpi/average-age")
def kpi_average_age():
    """Retourne l’âge moyen des consommateurs"""
    if CONSUMERS.empty or AGE_COL not in CONSUMERS:
        return {"indicator": "Average Age", "value": None}

    avg_age = pd.to_numeric(CONSUMERS[AGE_COL], errors="coerce").dropna().mean()
    val = round(float(avg_age), 2) if not np.isnan(avg_age) else None
    return {"indicator": "Average Age", "value": val}


@app.get("/kpi/average-budget-score")
def kpi_average_budget_score():
    """Score moyen du budget (Low=1, Medium=2, High=3)"""
    if CONSUMERS.empty or BUDGET_COL not in CONSUMERS:
        return {"indicator": "Average Budget Score", "value": None}

    df = CONSUMERS[[BUDGET_COL]].copy()
    df[BUDGET_COL] = df[BUDGET_COL].astype(str).str.strip().str.lower()
    score_map = {"low": 1, "medium": 2, "high": 3}
    df["Score"] = df[BUDGET_COL].map(score_map)
    avg_score = df["Score"].dropna().mean()
    val = round(float(avg_score), 2) if not np.isnan(avg_score) else None
    return {"indicator": "Average Budget Score", "value": val}


@app.get("/kpi/abstemious-rate")
def kpi_abstemious_rate():
    """Pourcentage de consommateurs abstemious"""
    if CONSUMERS.empty or DRINK_LVL_COL not in CONSUMERS or CONS_ID_COL not in CONSUMERS:
        return {"indicator": "% Abstemious", "value": None, "unit": "%"}

    cons = CONSUMERS.copy()
    cons[DRINK_LVL_COL] = cons[DRINK_LVL_COL].astype(str).str.strip().str.lower()

    total = cons[CONS_ID_COL].nunique()
    abst = cons.loc[cons[DRINK_LVL_COL] == "abstemious", CONS_ID_COL].nunique()
    pct_abst = (abst / total * 100) if total else 0
    return {"indicator": "% Abstemious", "value": round(float(pct_abst), 2), "unit": "%"}


@app.get("/kpi/all-consumers")
def kpi_all_consumers():
    """Regroupe tous les indicateurs consommateurs"""
    avg_age = pd.to_numeric(CONSUMERS[AGE_COL], errors="coerce").dropna().mean() if not CONSUMERS.empty else None

    df = CONSUMERS.copy()
    df[BUDGET_COL] = df[BUDGET_COL].astype(str).str.strip().str.lower()
    score_map = {"low": 1, "medium": 2, "high": 3}
    df["Score"] = df[BUDGET_COL].map(score_map)
    avg_score = df["Score"].dropna().mean() if not df.empty else None

    total = df[CONS_ID_COL].nunique() if CONS_ID_COL in df else 0
    abst = df.loc[df[DRINK_LVL_COL] == "abstemious", CONS_ID_COL].nunique() if DRINK_LVL_COL in df else 0
    pct_abst = (abst / total * 100) if total else 0

    return {
        "Average Age": round(float(avg_age), 2) if avg_age else None,
        "Average Budget Score": round(float(avg_score), 2) if avg_score else None,
        "% Abstemious": f"{pct_abst:.2f}%" if total else None,
    }
    
    # ================== PREDICTIF + KPIs complémentaires (API) ==================
# Ajoute ceci en bas de app/main.py (ou dans une section dédiée)

import textwrap
from fastapi import HTTPException

try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, mean_absolute_error
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

def _pretty_name(s: str) -> str:
    s = str(s)
    for p in ["consumers.", "restaurants.", "restaurant_cuisines.", "consumer_preferences."]:
        s = s.replace(p, "")
    s = s.replace("_", " ")
    return s.title()

def _wrap_label(s: str, width: int = 20) -> str:
    return "\n".join(textwrap.wrap(str(s), width=width, break_long_words=False))

def _load_or_mock():
    """
    Utilise les dataframes globaux si disponibles (RESTAURANTS, RATINGS, etc.).
    Sinon génère un mock identique à ton notebook.
    Retourne: restaurants, ratings, consumers, rest_cuis
    """
    if (RESTAURANTS is not None and not RESTAURANTS.empty and
        RATINGS is not None and not RATINGS.empty):
        # privilégie les data courant
        restaurants = RESTAURANTS.copy()
        ratings     = RATINGS.copy()
        consumers   = CONSUMERS.copy()   if CONSUMERS is not None   else pd.DataFrame()
        rest_cuis   = CUISINES.copy()    if CUISINES is not None    else pd.DataFrame()
        return restaurants, ratings, consumers, rest_cuis, False

    # --- MOCK DEMO (fallback) ---
    np.random.seed(7)
    cities = ["Mexico City","Guadalajara","Monterrey","Puebla"]
    cuisines = ["International","American","Cafe","Mexican","Italian","Japanese","Barbecue","Burgers"]
    n_rest = 112
    restaurants = pd.DataFrame({
        "Restaurant_ID": np.arange(1, n_rest+1),
        "Name": [f"R{i:03d}" for i in range(1, n_rest+1)],
        "City": np.random.choice(cities, size=n_rest),
        "Latitude": np.random.uniform(19.0, 25.0, size=n_rest),
        "Longitude": np.random.uniform(-103.0, -98.0, size=n_rest),
        "Has_Parking": np.random.choice([0,1], size=n_rest, p=[0.58, 0.42]),
        "Franchise_ID": np.random.choice([1,2,3,4], size=n_rest),
        "Cuisine": np.random.choice(cuisines, size=n_rest),
        "Price": np.random.choice([1,2,3], size=n_rest)  # 1=cheap 3=expensive
    })
    rows = 3200
    ratings = pd.DataFrame({
        "Rating_ID": np.arange(rows),
        "Restaurant_ID": np.random.choice(restaurants["Restaurant_ID"], size=rows),
        "Consumer_ID": np.random.randint(1, 900, size=rows),
        "Overall_Rating": np.clip(np.round(np.random.normal(3.6, 0.8, size=rows), 1), 0, 5)
    })
    consumers = pd.DataFrame({
        "Consumer_ID": np.arange(1, 900),
        "Transportation_Method": np.random.choice(["Car","Public","None"], size=899, p=[0.55,0.35,0.10])
    })
    rest_cuis = restaurants[["Restaurant_ID","Cuisine"]].copy()

    return restaurants, ratings, consumers, rest_cuis, True


def _prepare_dataset_for_model():
    """
    Construit le dataset pour le modèle (merge, nettoyage, encodage),
    retourne:
      df_base (restaurants + mean rating + cuisine éventuelle),
      X, y, columns, info dict
    """
    restaurants, ratings, consumers, rest_cuis, mocked = _load_or_mock()

    if "Restaurant_ID" not in ratings.columns:
        raise HTTPException(status_code=400, detail="Colonne Restaurant_ID absente de ratings.")

    # Moyenne rating par resto
    r_mean = ratings.groupby("Restaurant_ID", as_index=False)["Overall_Rating"].mean()

    df = restaurants.merge(r_mean, on="Restaurant_ID", how="left")

    # Injecte Cuisine si absente mais dispo côté rest_cuis
    if "Cuisine" not in df.columns and not rest_cuis.empty and {"Restaurant_ID","Cuisine"}.issubset(rest_cuis.columns):
        df = df.merge(rest_cuis[["Restaurant_ID","Cuisine"]], on="Restaurant_ID", how="left")

    target = "Overall_Rating"
    if target not in df.columns:
        raise HTTPException(status_code=400, detail="Colonne Overall_Rating absente après merge.")

    df = df.dropna(subset=[target]).drop_duplicates()

    # Booleans -> int
    for c in df.columns:
        if df[c].dtype == "bool":
            df[c] = df[c].astype(int)

    # Retire colonnes non pertinentes (ids, noms, géo)
    bad_like = []
    bad_like += [c for c in df.columns if any(k in c.lower() for k in ["name","id","zip","postal","code"])]
    bad_like += [c for c in df.columns if c.lower() in ["latitude","longitude","lat","lon","lng"]]
    bad_like = [c for c in set(bad_like) if c in df.columns and c != target]

    df_clean = df.drop(columns=bad_like, errors="ignore")

    # Encodage 1-hot
    cat_cols = df_clean.select_dtypes(include=["object"]).columns.tolist()
    df_ml = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True)

    if target not in df_ml.columns:
        raise HTTPException(status_code=400, detail="Cible absente après encodage.")

    X = df_ml.drop(columns=[target])
    y = df_ml[target].astype(float)

    info = {
        "mock_used": mocked,
        "n_rows": int(df.shape[0]),
        "n_features": int(X.shape[1]),
        "target": target,
        "has_cuisine": "Cuisine" in df.columns
    }

    return df, X, y, restaurants, ratings, info


@app.get("/predictive/summary")
def predictive_summary(top_features: int = 8, random_state: int = 42):
    """
    Entraîne un RandomForestRegressor et renvoie:
      - r2, mae, n_obs
      - drivers: top importances (feature, importance)
      - top_cuisines: moyennes prédites par cuisine (si dispo)
      - messages: avertissements éventuels
      - mock_used: bool
    """
    if not SKLEARN_OK:
        raise HTTPException(status_code=500, detail="scikit-learn n'est pas installé. pip install scikit-learn")

    df, X, y, restaurants, ratings, info = _prepare_dataset_for_model()
    messages = []

    # Vérifications minimales
    if X.shape[1] == 0 or y.nunique() <= 1:
        messages.append("Données explicatives insuffisantes pour entraîner un modèle utile.")
        return {
            "r2": None, "mae": None, "n_obs": int(df.shape[0]),
            "drivers": [], "top_cuisines": [],
            "messages": messages, "mock_used": info["mock_used"]
        }

    # Split stratifié sur quantiles de y
    try:
        y_bins = pd.qcut(y, q=min(5, y.nunique()), duplicates="drop", labels=False)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y_bins
        )
    except Exception:
        # fallback sans stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )
        messages.append("Stratification impossible (peu de dispersion). Fallback sans stratify.")

    # Modèle
    model = RandomForestRegressor(
        n_estimators=400, max_depth=None, min_samples_leaf=2,
        random_state=random_state, n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_all = model.predict(X)

    r2  = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    n_obs = int(df.shape[0])

    # Importances
    importances = model.feature_importances_
    k = int(min(max(1, top_features), len(importances)))
    idx = np.argsort(importances)[-k:][::-1]
    drivers = [
        {"feature": _pretty_name(X.columns[i]), "raw_feature": str(X.columns[i]), "importance": float(importances[i])}
        for i in idx
    ]

    # Top cuisines (si dispo)
    top_cuisines = []
    if "Cuisine" in df.columns:
        tmp = pd.DataFrame({"Cuisine": df["Cuisine"].astype(str), "Pred": y_pred_all})
        g = (tmp.groupby("Cuisine", as_index=False)["Pred"]
                .mean().sort_values("Pred", ascending=False).head(8))
        top_cuisines = [{"cuisine": str(row["Cuisine"]), "predicted_rating": float(row["Pred"])} for _, row in g.iterrows()]
    else:
        messages.append("Cuisine non disponible → top cuisines indisponible.")

    if r2 < 0:
        messages.append("⚠️ R² test < 0 : le modèle fait pire qu'une moyenne naïve (données explicatives limitées).")

    return {
        "r2": round(r2, 4),
        "mae": round(mae, 4),
        "n_obs": n_obs,
        "drivers": drivers,
        "top_cuisines": top_cuisines,
        "messages": messages,
        "mock_used": info["mock_used"]
    }


@app.get("/kpi/loyalty-rate")
def kpi_loyalty_rate():
    """
    Pourcentage d'avis dont la note >= 80% de la note maximale observée.
    Renvoie value en % (entier) + label.
    """
    _, ratings, _, _, mock = _load_or_mock()
    col = "Overall_Rating"
    if col not in ratings.columns or ratings.empty:
        return {"value": None, "label": "Loyalty Rate %"}

    max_note = float(ratings[col].max())
    if max_note <= 0:
        return {"value": 0, "label": "Loyalty Rate %"}

    threshold = 0.8 * max_note
    total_ratings = int(len(ratings))
    good_ratings = int((ratings[col] >= threshold).sum())
    loyalty_rate = (good_ratings / total_ratings) if total_ratings else 0.0
    return {"value": int(round(loyalty_rate * 100)), "label": "Loyalty Rate %", "mock_used": mock}


@app.get("/kpi/restaurant-density")
def kpi_restaurant_density():
    """
    Densité = nb restaurants / nb villes distinctes
    """
    restaurants, _, _, _, mock = _load_or_mock()
    if "City" not in restaurants.columns or restaurants["City"].nunique() == 0:
        return {"value": None, "label": "Restaurant Density"}

    density = len(restaurants) / restaurants["City"].nunique()
    return {"value": round(float(density), 2), "label": "Restaurant Density", "mock_used": mock}
