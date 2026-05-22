import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.sparse.csgraph import minimum_spanning_tree

# ======================================================================
# [모듈 1] NetCDF 데이터 로드 및 유체역학적 진짜 와도장(Vorticity) 복원
# ======================================================================
file_path = 'data_stream-oper_stepType-instant.nc'

print("[Module 1] NetCDF 데이터셋 로딩 중...")
ds = xr.open_dataset(file_path)

# ERA5 데이터셋의 표준 변수명 확인 (일반적으로 10m 풍속은 u10, v10 / 단일 레벨은 u, v)
# 업로드하신 파일 구조에 맞추어 변수명을 자동으로 매핑합니다.
u_var = 'u10' if 'u10' in ds.variables else 'u'
v_var = 'v10' if 'v10' in ds.variables else 'v'

if u_var not in ds.variables or v_var not in ds.variables:
    raise KeyError(f"데이터셋 내에서 풍속 벡터 변수(u, v)를 찾을 수 없습니다. 존재 변수: {list(ds.variables)}")

u_wind = ds[u_var]
v_wind = ds[v_var]

# 좌표 및 시간 차원 배열 추출 (ERA5/cfgrib은 valid_time 사용)
lats = ds['latitude'].values
lons = ds['longitude'].values
time_dim = 'valid_time' if 'valid_time' in ds.dims else 'time'
times = ds[time_dim].values

# 물리적 그리드 간격 계산 (도 단위를 km 거리 단위로 변환)
dlat = np.abs(lats[1] - lats[0])
dlon = np.abs(lons[1] - lons[0])

dy = dlat * 111.0 # 위도 1도 간격 = 약 111km
mean_lat = np.mean(lats)
dx = dlon * 111.0 * np.cos(np.radians(mean_lat)) # 해당 위도에서의 경도 1도 간격 km 환산

print(f"[OK] 격자 공간해상도 분석 완료: dx = {dx:.2f}km, dy = {dy:.2f}km")
print(f"[OK] 총 분석 시간대 해상도: {len(times)} slots")

# ======================================================================
# [모듈 2 & 3] 이방성 점구름 생성 및 MST 기반 토폴로지 대장 수명(L_max) 추출
# ======================================================================
# 논문 명제 4.1 및 Table 1에 기반한 소용돌이 비중 계수 설정
alpha_omega = 3.0  
l_max_series = []

print("\n[Module 2 & 3] 시간축 기준 이방성 지속성 동조론(H0) 연산 시작...")

# 메쉬그리드를 미리 생성하여 2차원 공간 좌표 기하 구조를 km 단위로 맵핑
lon_grid, lat_grid = np.meshgrid(lons, lats)
x_coords_raw = (lon_grid - lons[0]) * 111.0 * np.cos(np.radians(mean_lat))
y_coords_raw = (lat_grid - lats[0]) * 111.0

for t_idx, t_val in enumerate(times):
    # 각 시간대의 u, v 격자 필드 추출
    u_field = u_wind.isel({time_dim: t_idx}).values
    v_field = v_wind.isel({time_dim: t_idx}).values
    
    # 1. 중앙 차분법(Central Difference)을 적용한 진짜 와도장 계산 (\omega = \partial_x v - \partial_y u)
    dv_dx = np.gradient(v_field, axis=1) / dx
    du_dy = np.gradient(u_field, axis=0) / dy
    vorticity_field = dv_dx - du_dy
    
    # 2. 토폴로지 점구름(Point Cloud) 매니폴드 평탄화 및 정렬
    X = x_coords_raw.flatten()
    Y = y_coords_raw.flatten()
    W = vorticity_field.flatten()
    
    # 3. 고밀도 데이터 연산 최적화를 위한 공간 샘플링 (Grid 정형화 격자 간격 조절)
    # 수만 개의 격자점을 다룰 때 메모리 오버헤드를 방지하기 위해 4점당 1점씩 엉성하게 샘플링합니다.
    sample_stride = 4  
    X, Y, W = X[::sample_stride], Y[::sample_stride], W[::sample_stride]
    N_points = len(X)
    
    # 4. 논문 공식 (3.1) 이방성 거리 행렬 연산
    # 각 점들 사이의 공간적 거리와 물리학적 와도의 격차를 단일 매트릭스로 통합합니다.
    dist_matrix = np.zeros((N_points, N_points))
    for i in range(N_points):
        dx_sq = (X[i] - X)**2
        dy_sq = (Y[i] - Y)**2
        dw_sq = alpha_omega**2 * (W[i] - W)**2
        dist_matrix[i, :] = np.sqrt(dx_sq + dy_sq + dw_sq)
        
    # 5. 계산위상수학(TDA) 정리 적용: H_0 바코드의 소멸 메커니즘은 최소 신장 트리(MST)와 일치합니다.
    # 소용돌이 집중 구조가 배경과 결합하기 직전의 가장 지독한 지속성 수명(L_max)을 구합니다.
    mst = minimum_spanning_tree(dist_matrix)
    l_max = mst.toarray().max()
    l_max_series.append(l_max)
    
    current_time_str = pd.to_datetime(t_val).strftime('%Y-%m-%d %H:%M')
    print(f" > [{current_time_str}] 매니폴드 수치 해석 완료 -> L_max: {l_max:.4f}")

# ======================================================================
# [모듈 4] 배경 규칙성 가정(BRA) 기반 최종 TAS 이상 점수 통계 산출
# ======================================================================
print("\n[Module 4] 대기 평상시 변동 베이스라인 산출 및 최종 TAS 통계 표준화...")
l_max_series = np.array(l_max_series)

# 전체 관측 기간의 대기 요동 통계를 기반으로 평균(mu_L)과 표준편차(sigma_L) 계산 (Z-score 기법)
mu_L = np.mean(l_max_series)
sigma_L = np.std(l_max_series) if np.std(l_max_series) != 0 else 1.0

# 논문 식 (3.7) 기반 최종 토폴로지 이상 점수(TAS) 산출
tas_series = (l_max_series - mu_L) / sigma_L

# ----------------------------------------------------------------------
# [결과 저장 및 시각화] 시계열 데이터프레임 빌드 및 플로팅
# ----------------------------------------------------------------------
results_df = pd.DataFrame({
    'Time': pd.to_datetime(times),
    'L_max': l_max_series,
    'TAS': tas_series
})

# 시뮬레이션 데이터 마스터 덤프 생성 (CSV 저장)
results_df.to_csv('era5_vorticity_tas_engine_output.csv', index=False, encoding='utf-8-sig')
print("[출력 완료] 시뮬레이션 마스터 테이블 'era5_vorticity_tas_engine_output.csv' 저장 성공.")

# 한글 폰트 깨짐 방지 설정 및 시각화 플롯 빌드
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

results_df = results_df.sort_values('Time').reset_index(drop=True)


def plot_timeseries_with_gaps(ax, times, values, gap_days=3, **plot_kwargs):
    """데이터 공백(월·연도 경계 등) 구간은 직선으로 연결하지 않고 구간별로 그립니다."""
    times = pd.to_datetime(times).to_numpy()
    values = np.asarray(values)
    seg_t, seg_v = [], []
    label_used = False

    def flush_segment():
        nonlocal label_used
        if not seg_t:
            return
        kw = plot_kwargs.copy()
        if label_used:
            kw.pop('label', None)
        else:
            label_used = True
        ax.plot(seg_t, seg_v, **kw)

    for i in range(len(times)):
        if i > 0 and (times[i] - times[i - 1]) > np.timedelta64(gap_days, 'D'):
            flush_segment()
            seg_t, seg_v = [], []
        seg_t.append(times[i])
        seg_v.append(values[i])
    flush_segment()


years = sorted(results_df['Time'].dt.year.unique())
fig, axes = plt.subplots(len(years), 1, figsize=(14, 4 * len(years)), sharey=True, squeeze=False)

for ax, year in zip(axes.flatten(), years):
    subset = results_df[results_df['Time'].dt.year == year]
    plot_timeseries_with_gaps(
        ax, subset['Time'], subset['TAS'],
        color='crimson', marker='o', linewidth=2, markersize=4,
        label='TAS (위상 예보 점수)',
    )
    ax.axhline(y=4.0, color='darkorange', linestyle='--', linewidth=1.5, label='비상 경보 임계치 (TAS = 4.0)')
    ax.set_title(f'{year}년 분석 구간 (8~10월, 6시간 간격)', fontsize=12, pad=10)
    ax.set_ylabel('Topological Anomaly Score (TAS)', fontsize=11)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax.tick_params(axis='x', rotation=20)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', fontsize=10)

axes.flatten()[-1].set_xlabel('분석 시간대 (Time Slot)', fontsize=12, labelpad=10)
fig.suptitle(
    'ERA5 진짜 와도장($\\omega$) 기반 위상수학적 태풍 예보 엔진 시계열 변동',
    fontsize=14, y=1.01,
)
fig.tight_layout()
plt.savefig('topological_cyclone_engine_monitor.png', dpi=300, bbox_inches='tight')
plt.close()
print("[시각화 완료] 실시간 모니터링 그래프 'topological_cyclone_engine_monitor.png' 저장 성공.")
print("\n[완료] 모든 위상수학적 분석 파이프라인이 정상적으로 종료되었습니다!")
