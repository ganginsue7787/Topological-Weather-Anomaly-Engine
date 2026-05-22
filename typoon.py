import xarray as xr
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree

# ----------------------------------------------------------------------
# 1. NetCDF 파일 열기 및 내부 변수 확인
# ----------------------------------------------------------------------
file_path = 'data_stream-oper_stepType-instant.nc'
ds = xr.open_dataset(file_path)

print("=== 데이터셋 구조 확인 ===")
print(ds)

# ERA5에서 다운로드한 변수명에 맞게 매핑 (일반적으로 u10, v10 또는 u, v)
# 파일 구조에 따라 'u10', 'v10'을 실제 변수명으로 변경해 주세요.
u_wind = ds['u10'] # 10m 동서풍 벡터 component
v_wind = ds['v10'] # 10m 남북풍 벡터 component

# 위도(latitude), 경도(longitude), 시간 차원 추출 (ERA5/cfgrib은 valid_time 사용)
lats = ds['latitude'].values
lons = ds['longitude'].values
time_dim = 'valid_time' if 'valid_time' in ds.dims else 'time'
times = ds[time_dim].values

# ----------------------------------------------------------------------
# 2. 물리적 격자 간격 (dx, dy) 계산 (단위: km)
# ----------------------------------------------------------------------
# ERA5의 해상도는 보통 0.25도입니다. 이를 실제 물리 거리(km)로 변환합니다.
dlat = np.abs(lats[1] - lats[0])
dlon = np.abs(lons[1] - lons[0])

dy = dlat * 111.0 # 위도 1도 간격 = 약 111km
# 중심 위도(약 33~35도)를 기준으로 경도 간격 km 환산
mean_lat = np.mean(lats)
dx = dlon * 111.0 * np.cos(np.radians(mean_lat))

print(f"\n격자 간격 계산 완료: dx = {dx:.3f}km, dy = {dy:.3f}km")

# ----------------------------------------------------------------------
# 3. 시간대별 진짜 와도장(\omega = \partial_x v - \partial_y u) 계산 및 TAS 추출
# ----------------------------------------------------------------------
alpha_omega = 3.0 # 논문 기반 이방성 가중치
l_max_series = []

print("\n=== 시간대별 토폴로지 이상 탐지(TAS) 연산 시작 ===")

for t_idx, t_val in enumerate(times):
    # 특정 시간대의 u, v 2차원 격자장 추출
    u_field = u_wind.isel({time_dim: t_idx}).values
    v_field = v_wind.isel({time_dim: t_idx}).values
    
    # 중앙 차분법을 이용한 공간 편미분 연산
    # numpy.gradient는 축 방향에 따른 변화량을 반환하므로 격자 간격(dx, dy)으로 나눕니다.
    dv_dx = np.gradient(v_field, axis=1) / dx
    du_dy = np.gradient(u_field, axis=0) / dy
    
    # 유체역학 정의에 따른 진짜 와도장(Vorticity Field) 생성
    vorticity = dv_dx - du_dy
    
    # ------------------------------------------------------------------
    # 4. 고밀도 점구름(Point Cloud) 생성 및 이방성 지속성 수명 계산
    # ------------------------------------------------------------------
    # 연산 속도와 무결성을 위해 격자점 중 유의미한 소용돌이 영역을 샘플링합니다.
    # 여기서는 전체 격자점 좌표(X, Y)와 와도값(\omega)을 리스트로 정렬합니다.
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # 킬로미터 단위 공간 좌표로 변환
    x_coords = (lon_grid - lons[0]) * 111.0 * np.cos(np.radians(mean_lat))
    y_coords = (lat_grid - lats[0]) * 111.0
    
    # 평탄화(Flatten)하여 점구름 행렬 구축
    X = x_coords.flatten()
    Y = y_coords.flatten()
    W = vorticity.flatten()
    
    # 데이터 크기 관리를 위해 일정 간격으로 샘플링 (예: 5점당 1점씩)
    # 데이터가 아주 작다면 샘플링 없이 전체([::1]) 사용 가능합니다.
    sample_rate = 5
    X, Y, W = X[::sample_rate], Y[::sample_rate], W[::sample_rate]
    N_points = len(X)
    
    # 메모리 효율적 연산을 위한 이방성 거리 행렬 계산
    # 수만 개의 점일 경우 pdist 방식보다 이웃 그래프 방식을 추천하지만, 
    # 잘라낸 영역(Area)이 작다면 아래 루프로 정밀 계산이 가능합니다.
    dist_matrix = np.zeros((N_points, N_points))
    for i in range(N_points):
        dx_sq = (X[i] - X)**2
        dy_sq = (Y[i] - Y)**2
        dw_sq = alpha_omega**2 * (W[i] - W)**2
        dist_matrix[i, :] = np.sqrt(dx_sq + dy_sq + dw_sq)
        
    # MST 기법을 이용한 H_0 대장 바코드 수명(L_max) 추출
    mst = minimum_spanning_tree(dist_matrix)
    l_max = mst.toarray().max()
    l_max_series.append(l_max)
    
    print(f"[{pd.to_datetime(t_val)}] 계산 완료 -> L_max: {l_max:.4f}")

# ----------------------------------------------------------------------
# 5. 통계적 표준화를 통한 최종 TAS(Z-score) 도출
# ----------------------------------------------------------------------
l_max_series = np.array(l_max_series)
mu_L = np.mean(l_max_series)
sigma_L = np.std(l_max_series) if np.std(l_max_series) != 0 else 1.0

tas_series = (l_max_series - mu_L) / sigma_L

# 결과를 데이터프레임으로 변환하여 확인
results_df = pd.DataFrame({
    'Time': pd.to_datetime(times),
    'L_max': l_max_series,
    'TAS': tas_series
})

print("\n=== 최종 시뮬레이션 결과 ===")
print(results_df)

