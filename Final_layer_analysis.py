import folium
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import MultiPoint, Polygon
from sklearn.cluster import KMeans
from pyproj import Transformer
import branca.colormap as cm

# 데이터 불러오기
file_path = "/Users/mcwon/Downloads/final_distance.xlsx"
df = pd.read_excel(file_path)

# 경위도 → UTM-K (EPSG:5179) 변환
trs = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True)
df["x"], df["y"] = zip(*df.apply(lambda r: trs.transform(r["longitude"], r["latitude"]), axis=1))

# GeoDataFrame 생성
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")

# 전처리
dfc = df.dropna(subset=["x", "y", "latitude", "longitude", "real_distance(m)"]).copy()
dfc["읍면동"] = dfc["읍면동"].fillna("정보없음")

# 점수 계산 함수
def score_by_distance(d):
    if d <= 50:
        return 5
    elif d <= 100:
        return 4
    elif d <= 200:
        return 3
    elif d <= 300:
        return 2
    elif d <= 400:
        return 1
    else:
        return 0

dfc["score"] = dfc["real_distance(m)"].apply(score_by_distance)

# 클러스터링
cluster_id = 0
cluster_labels = []
cluster_centers = []
representatives = []
cluster_polygons = []

for name, group in dfc.groupby("읍면동"):
    coords = group[["x", "y"]].values
    if len(group) < 2:
        group["cluster"] = cluster_id
        group["centroid_dist"] = 0
        reps = group.copy()
        reps["avg_cluster_score"] = group["score"].mean()
        representatives.append(reps)
        cluster_labels.extend([cluster_id] * len(group))
        cluster_centers.append(coords[0])
        cluster_id += 1
        continue

    km = KMeans(n_clusters=2, random_state=0).fit(coords)
    group = group.copy()
    group["cluster"] = km.labels_ + cluster_id
    group["centroid_dist"] = np.linalg.norm(coords - km.cluster_centers_[km.labels_], axis=1)

    for cid in range(2):
        sub = group[group["cluster"] == cluster_id + cid]
        if not sub.empty:
            rep = sub.loc[sub["centroid_dist"].idxmin()].copy()
            rep["avg_cluster_score"] = sub["score"].mean()
            representatives.append(rep)
            # convex hull 생성
            points = MultiPoint(sub[["longitude", "latitude"]].values.tolist())
            if points.convex_hull.geom_type == "Polygon":
                cluster_polygons.append((points.convex_hull, cluster_id + cid))

    cluster_labels.extend(group["cluster"])
    cluster_centers.extend(km.cluster_centers_)
    dfc.loc[group.index, "cluster"] = group["cluster"]
    dfc.loc[group.index, "centroid_dist"] = group["centroid_dist"]

    cluster_id += 2

reps = pd.DataFrame(representatives)

# 지도 시각화
m = folium.Map(location=[dfc.latitude.mean(), dfc.longitude.mean()], zoom_start=11, tiles="cartodbpositron")
color_list = ["#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
              "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff",
              "#9A6324", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1",
              "#000075", "#808080", "#ffffff", "#000000", "#a9a9a9", "#dda0dd",
              "#ff69b4", "#cd5c5c", "#20b2aa", "#ff6347", "#adff2f", "#6495ed"] * 5

cmap = cm.LinearColormap(["red", "yellow", "green"], vmin=dfc.score.min(), vmax=dfc.score.max(), caption="버스 인프라 점수")
m.add_child(cmap)

# 읍면동 폴리라인
# 읍면동 폴리라인 + 이름 표시
for name, group in dfc.groupby("읍면동"):
    points = group[["longitude", "latitude"]].values.tolist()
    if len(points) > 2:
        polygon = MultiPoint(points).convex_hull
        coords = [(lat, lon) for lon, lat in polygon.exterior.coords]
        
        # 폴리라인
        folium.PolyLine(locations=coords,
                        color="black", weight=2.7, popup=name).add_to(m)

        # 중심 위치 계산
        center_lat = np.mean([lat for lat, lon in coords])
        center_lon = np.mean([lon for lat, lon in coords])

        # 텍스트 마커 추가
        folium.Marker(
            location=(center_lat, center_lon),
            icon=folium.DivIcon(html=f"""
                <div style="font-size: 12pt; font-weight: bold; color: black;">
                    {name}
                </div>""")
        ).add_to(m)


# 클러스터 폴리곤
for poly, cid in cluster_polygons:
    folium.Polygon(locations=[(lat, lon) for lon, lat in poly.exterior.coords],
                   color=color_list[cid % len(color_list)],
                   fill=True, fill_opacity=0.15,
                   popup=f"클러스터 {cid}").add_to(m)

# 회관 점 표시
for _, row in dfc.iterrows():
    popup = folium.Popup(f"시설명: {row['시설명']}<br>읍면동: {row['읍면동']}<br>real_distance(m): {row['real_distance(m)']}", max_width=250)
    folium.CircleMarker(location=(row.latitude, row.longitude),
                        radius=5,
                        fill=True,
                        fill_opacity=0.8,
                        color=color_list[int(row.cluster) % len(color_list)],
                        popup=popup).add_to(m)

# 대표 회관 표시
for _, row in reps.iterrows():
    popup = folium.Popup(f"⭐ 시설명: {row['시설명']}<br>클러스터: {int(row['cluster'])}<br>버스 인프라 점수: {round(row['avg_cluster_score'], 3)}", max_width=250)
    folium.Marker(location=(row.latitude, row.longitude),
                  icon=folium.Icon(color="black", icon="star"),
                  popup=popup).add_to(m)

# 결과 저장
m.save("Cluster_Map.html")
