import folium, numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from shapely.geometry import MultiPoint
import geopandas as gpd
import branca.colormap as cm
import pandas as pd, geopandas as gpd
from pyproj import Transformer

df = pd.read_excel("/Users/mcwon/Downloads/final_distance.xlsx")

# 경위도 → EPSG:5179(UTM-K)  ──  거리·클러스터링용
trs = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True)
df["x"], df["y"] = zip(*df.apply(lambda r: trs.transform(r["longitude"], r["latitude"]), axis=1))

gdf = gpd.GeoDataFrame(df,
                       geometry=gpd.points_from_xy(df.longitude, df.latitude),
                       crs="EPSG:4326")   # 시각화용 원본 좌표계(경위도)

# ─────────────────── 데이터·클러스터링 ───────────────────
dfc = df.dropna(subset=["x","y","latitude","longitude","real_distance(m)"]).copy()

# ← 읍면동 결측은 '정보없음' 대체
dfc["읍면동"] = dfc["읍면동"].fillna("정보없음")

k = 15
coords = dfc[["x","y"]].values
km = KMeans(n_clusters=k, random_state=0).fit(coords)
dfc["cluster"] = km.labels_

def score_by_distance(d):
    if d <= 100:
        return 5
    elif d <= 200:
        return 4
    elif d <= 400:
        return 3
    elif d <= 800:
        return 2
    else:
        return 1

dfc["score"] = dfc["real_distance(m)"].apply(score_by_distance)

dfc["centroid_dist"] = np.linalg.norm(coords - km.cluster_centers_[dfc["cluster"]], axis=1)
# 대표 회관 추출
reps = dfc.loc[dfc.groupby("cluster")["centroid_dist"].idxmin()].copy()
# 클러스터별 평균 점수 계산
reps["avg_cluster_score"] = dfc.groupby("cluster")["score"].mean().values


# ───────────────────── 지도 생성 ─────────────────────
m = folium.Map(location=[dfc.latitude.mean(), dfc.longitude.mean()], zoom_start=11, tiles="cartodbpositron")
color_list = ["#e6194B","#3cb44b","#ffe119","#4363d8","#f58231","#911eb4",
              "#46f0f0","#f032e6","#bcf60c","#fabebe","#008080","#e6beff",
              "#9A6324","#fffac8","#800000"]

cmap = cm.LinearColormap(["red","yellow","green"], vmin=dfc.score.min(), vmax=dfc.score.max(),
                         caption="버스 인프라 점수")
m.add_child(cmap)

# ───────────── 모든 점: 결측 제거 + str 캐스팅 팝업 ─────────────
for _, r in dfc.iterrows():
    c = color_list[int(r["cluster"])]
    popup_html = (
        "<b>시설명:</b> {}<br>"
        "<b>읍면동:</b> {}<br>"
        "<b>real_distance:</b> {:.1f} m<br>"
        "<b>클러스터:</b> {}"
    ).format(str(r["시설명"]),
             str(r["읍면동"]),
             float(r["real_distance(m)"]),
             int(r["cluster"]) + 1)

    folium.CircleMarker(
        [r["latitude"], r["longitude"]],
        radius=6,
        color=c,
        fill=True,
        fill_color=cmap(r["score"]),
        fill_opacity=0.8,
        popup=folium.Popup(popup_html, max_width=250),
        tooltip=r["시설명"]
    ).add_to(m)

# ───────────── 대표 마을회관 ★ ─────────────
for _, r in reps.iterrows():
    pop = (
        "<b>대표 마을회관 ⭐</b><br>"
        "시설명: {}<br>"
        "클러스터: {}<br>"
        "버스 인프라 점수: {:.3f}"
    ).format(r["시설명"], int(r["cluster"])+1, r["avg_cluster_score"])
    folium.Marker(
        [r["latitude"], r["longitude"]],
        icon=folium.Icon(color="black", icon="star"),
        popup=folium.Popup(pop, max_width=220)
    ).add_to(m)

# ───────────── 클러스터 경계선 ─────────────
for cid in range(k):
    pts = dfc[dfc.cluster==cid][["longitude","latitude"]].values
    if len(pts) < 3: continue
    hull = MultiPoint(pts).convex_hull
    folium.GeoJson(
        gpd.GeoSeries([hull]).__geo_interface__,
        name=f"클러스터 {cid+1}",
        style_function=lambda feat, col=color_list[cid]: {
            "fillColor": col,
            "color": col,
            "weight": 2,
            "fillOpacity": 0.15
        },
        highlight_function=None,
        tooltip=None,
        # 이 옵션이 핵심!  폴리곤 자체는 클릭 이벤트를 막지 않도록
        embed=False,
        overlay=True,
        control=True,
        show=True,
        zoom_on_click=False,
        interactive=False         
    ).add_to(m)


folium.LayerControl().add_to(m)
m.save("Interactive_Layer_Map_.html")
print(" Interactive_Layer_Map_.html 저장 완료")
