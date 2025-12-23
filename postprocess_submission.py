"""
예측 결과 후처리
- 이상치 제거
- 경기별 평균 좌표 보정
- Smoothing
"""
import pandas as pd
import numpy as np

def postprocess_submission(input_csv, output_csv):
    """
    제출 파일 후처리
    
    Args:
        input_csv: 원본 제출 파일 경로
        output_csv: 후처리된 제출 파일 경로
    """
    print("📊 제출 파일 후처리 시작...")
    
    # 1. 제출 파일 로드
    df = pd.read_csv(input_csv)
    print(f"   원본 데이터: {df.shape}")
    
    # 2. game_id 추출 (game_episode에서)
    df['game_id'] = df['game_episode'].astype(str).str.split('_').str[0]
    
    # 3. 이상치 제거 (필드 범위 밖)
    before_clip = len(df)
    df['end_x'] = df['end_x'].clip(0, 105)
    df['end_y'] = df['end_y'].clip(0, 68)
    print(f"   ✅ 좌표 범위 클리핑 완료")
    
    # 4. 경기별 통계 기반 보정
    game_stats = df.groupby('game_id').agg({
        'end_x': ['mean', 'std'],
        'end_y': ['mean', 'std']
    }).reset_index()
    
    game_stats.columns = ['game_id', 'x_mean', 'x_std', 'y_mean', 'y_std']
    
    # 전체 평균
    global_x_mean = df['end_x'].mean()
    global_y_mean = df['end_y'].mean()
    
    print(f"   전체 평균 좌표: ({global_x_mean:.2f}, {global_y_mean:.2f})")
    
    # 5. 극단적 이상치 보정 (3σ 규칙)
    df = df.merge(game_stats, on='game_id', how='left')
    
    # 경기별 평균으로부터 3σ 이상 벗어난 값 보정
    outlier_x_mask = np.abs(df['end_x'] - df['x_mean']) > 3 * df['x_std']
    outlier_y_mask = np.abs(df['end_y'] - df['y_mean']) > 3 * df['y_std']
    
    n_outliers = (outlier_x_mask | outlier_y_mask).sum()
    
    if n_outliers > 0:
        # 이상치를 경기 평균으로 보정
        df.loc[outlier_x_mask, 'end_x'] = df.loc[outlier_x_mask, 'x_mean']
        df.loc[outlier_y_mask, 'end_y'] = df.loc[outlier_y_mask, 'y_mean']
        print(f"   ✅ 이상치 {n_outliers}개 보정 완료")
    
    # 6. Smoothing (Moving Average) - 선택적
    # 같은 경기 내에서 시퀀스가 연속적이라면 적용
    # 주의: game_episode가 순서대로 정렬되어 있어야 함
    
    # 7. 최종 제출 파일 생성
    final_df = df[['game_episode', 'end_x', 'end_y']].copy()
    
    # 8. 최종 검증
    print(f"\n📈 후처리 통계:")
    print(f"   X 범위: [{final_df['end_x'].min():.2f}, {final_df['end_x'].max():.2f}]")
    print(f"   Y 범위: [{final_df['end_y'].min():.2f}, {final_df['end_y'].max():.2f}]")
    print(f"   X 평균: {final_df['end_x'].mean():.2f}")
    print(f"   Y 평균: {final_df['end_y'].mean():.2f}")
    
    # 9. 저장
    final_df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"\n✅ 후처리 완료: {output_csv}")
    
    return final_df

def blend_submissions(csv_list, weights=None, output_csv='submission_blended.csv'):
    """
    여러 제출 파일 블렌딩
    
    Args:
        csv_list: 제출 파일 경로 리스트
        weights: 각 파일의 가중치 (None이면 동일 가중치)
        output_csv: 출력 파일 경로
    """
    print(f"\n🔀 {len(csv_list)}개 제출 파일 블렌딩 시작...")
    
    if weights is None:
        weights = [1.0 / len(csv_list)] * len(csv_list)
    
    # 정규화
    weights = np.array(weights) / np.sum(weights)
    
    # 제출 파일 로드
    dfs = []
    for i, csv_path in enumerate(csv_list):
        df = pd.read_csv(csv_path)
        print(f"   {i+1}. {csv_path} (가중치: {weights[i]:.3f})")
        dfs.append(df)
    
    # game_episode 기준으로 정렬 (순서 맞추기)
    for df in dfs:
        df.sort_values('game_episode', inplace=True)
        df.reset_index(drop=True, inplace=True)
    
    # 블렌딩
    result = dfs[0][['game_episode']].copy()
    
    # 가중 평균
    result['end_x'] = sum(df['end_x'] * w for df, w in zip(dfs, weights))
    result['end_y'] = sum(df['end_y'] * w for df, w in zip(dfs, weights))
    
    # 좌표 범위 클리핑
    result['end_x'] = result['end_x'].clip(0, 105)
    result['end_y'] = result['end_y'].clip(0, 68)
    
    # 저장
    result.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"\n✅ 블렌딩 완료: {output_csv}")
    print(f"   최종 예측 개수: {len(result)}")
    
    return result

if __name__ == '__main__':
    # 사용 예시 1: 단일 파일 후처리
    postprocess_submission(
        input_csv='./submission.csv',
        output_csv='./submission_postprocessed.csv'
    )
    
    # 사용 예시 2: 여러 제출 파일 블렌딩
    # blend_submissions(
    #     csv_list=[
    #         './submission.csv',
    #         './submission_tta.csv',
    #         './submission_ensemble.csv'
    #     ],
    #     weights=[0.3, 0.3, 0.4],  # 앙상블에 더 높은 가중치
    #     output_csv='./submission_final.csv'
    # )