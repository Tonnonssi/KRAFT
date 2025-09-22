import os
# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import torch.nn as nn

############# DLinear 모델 부분
class Decomposition(nn.Module):
    def __init__(self, kernel_size, device, trainable=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.device = device
        self.padding = self.kernel_size // 2

        self.layer = nn.Conv1d(in_channels = 1, 
                               out_channels = 1, 
                               kernel_size = self.kernel_size,
                               bias = False,
                               padding = self.padding,
                               padding_mode='replicate',
                               dtype = torch.float32,
                               device = self.device)
        
        if not trainable:
            # 이동 평균이므로 가중치가 모두 동일해야 함
            weight = torch.ones(1, 1, self.kernel_size, device = self.device) / self.kernel_size
            self.register_buffer('weight', weight)

            with torch.no_grad():
                self.layer.weight.copy_(weight)
            self.layer.weight.requires_grad = False

    def forward(self, x):       # x.size : (batch_size, channel_size, seq_len)
        trend = self.layer(x)
        remainder = x - trend
        return trend, remainder

class DLinear(nn.Module):
    def __init__(self, kernel_size, seq_len, pred_len, channel_size, device, decompose_trainable):
        super().__init__()
        self.kernel_size = kernel_size
        self.seq_len = seq_len      # history L timesteps
        self.pred_len = pred_len    # future T timesteps
        self.channel_size = channel_size
        self.device = device

        self.decomposition = Decomposition(self.kernel_size, self.device, trainable=decompose_trainable)

        self.trend_layer = nn.Linear(self.seq_len, self.pred_len, dtype=torch.float32, device=self.device)
        self.remainder_layer = nn.Linear(self.seq_len, self.pred_len, dtype=torch.float32, device=self.device)


    def forward(self, x):
        x = x.transpose(1, 2)
        x_pred = x[:, 0].unsqueeze(1)       # 예측하는 feature
        trend, remainder = self.decomposition(x_pred)       # 시계열 분해 (trend(X_t), remainder(X_s))

        if self.channel_size > 1:
            x_rest = x[:, 1:]  # 추가 정보
            trend = torch.cat([trend, x_rest], dim=1)           # X_t
            remainder = torch.cat([remainder, x_rest], dim=1)   # X_s

        trend_pred = self.trend_layer(trend)[:,0]               # H_t
        remainder_pred = self.remainder_layer(remainder)[:,0]    # H_s
        x_hat = trend_pred + remainder_pred                 # X_hat = H_t + H_s
        return trend_pred, remainder_pred, x_hat


class DLinearModel(nn.Module):
    def __init__(self, num_heads, kernel_size, seq_len, pred_len_list, 
                channel_size, device, decompose_trainable=False, metadata=True):
        super().__init__()
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.seq_len = seq_len      # history L timesteps
        self.pred_len_list = pred_len_list    # future T timesteps
        self.channel_size = channel_size
        self.device = device
        self.metadata = metadata

        if metadata:
            self.output_len = 32        # 일단은 그냥 지정...
        else:
            self.output_len = max(self.pred_len_list)

        self.model_list = nn.ModuleList()
        for i in range(num_heads):
            pred_len = self.pred_len_list[i]
            model = DLinear(kernel_size, seq_len, pred_len, channel_size, device, decompose_trainable)
            self.model_list.append(model)

    def forward(self, x):
        pred_list = []

        for i in range(self.num_heads):
            _, _, x_hat = self.model_list[i].forward(x)
            pred_list.append(x_hat)

        if self.metadata:
            result = self._create_metafeatures(pred_list)
        else:
            batch_size = x.size(0)
            sum_tensor = torch.zeros(batch_size, self.output_len, device=self.device, dtype=torch.float32)
            count_tensor = torch.zeros(batch_size, self.output_len, device=self.device, dtype=torch.float32)
            # 각 텐서를 순회하며 합계와 횟수를 누적
            for t in pred_list:
                current_len = t.size(1)
                # 현 텐서의 값을 합계 텐서의 앞부분에 더함
                sum_tensor[:, :current_len] += t
                # 현 텐서가 기여한 부분의 횟수를 1씩 증가
                count_tensor[:, :current_len] += 1

            # 0으로 나누는 것을 방지하기 위해 count_tensor의 0인 값을 1로 변경
            # (값이 없는 구간은 sum도 0이므로 결과에 영향 없음)
            count_tensor = torch.where(count_tensor == 0, 1.0, count_tensor)
            # 최종 평균 텐서 계산
            result = sum_tensor / count_tensor

        return result

    def _calculate_stats(self, seq_batch):
        batch_size, pred_len = seq_batch.size()

        if pred_len < 2:
            # 계산이 불가능한 경우, 적절한 크기의 0 텐서를 반환
            nan_result = torch.full((batch_size,), float('nan'), device=self.device, dtype=torch.float32)
            return {
                "mean": nan_result, "std_dev": nan_result, "total_return": nan_result,
                "total_return_rate": nan_result, "max_drawdown": nan_result,
                "final_pos_in_range": nan_result, "linreg_slope": nan_result,
                "pred_trend_quality_ratio": nan_result,
            }

        # 1. 기본 통계량 (dim=1은 pred_len 차원을 의미)
        mean_val = torch.mean(seq_batch, dim=1)
        std_dev = torch.std(seq_batch, dim=1)

        # 2. 수익 및 리스크 관련 지표
        first_val = seq_batch[:, 0]
        last_val = seq_batch[:, -1]
        
        total_return = last_val - first_val
        # 0으로 나누는 것을 방지
        total_return_rate = torch.where(first_val != 0, total_return / first_val, torch.tensor(0.0, device=self.device))

        # 최대 낙폭 (Max Drawdown) - 배치 전체에 대해 벡터화
        cumulative_max = torch.cummax(seq_batch, dim=1).values
        drawdowns = (cumulative_max - seq_batch) / cumulative_max
        drawdowns = torch.nan_to_num(drawdowns, nan=0.0) # 0/0으로 인한 nan 처리
        max_drawdown = -torch.max(drawdowns, dim=1).values

        # 고점/저점 대비 최종 위치 - 배치 전체에 대해 벡터화
        min_val, _ = torch.min(seq_batch, dim=1)
        max_val, _ = torch.max(seq_batch, dim=1)
        range_val = max_val - min_val
        final_pos_in_range = torch.where(range_val != 0, (last_val - min_val) / range_val, torch.tensor(0.5, device=self.device))
        
        # 3. 추세 품질 및 강도 지표 - 배치 전체에 대해 벡터화
        # 선형 회귀 기울기
        x_axis = torch.arange(pred_len, device=self.device, dtype=torch.float32)
        x_mean = torch.mean(x_axis)
        y_mean = torch.mean(seq_batch, dim=1, keepdim=True) # 브로드캐스팅을 위해 차원 유지
        
        cov_xy = torch.mean((x_axis - x_mean) * (seq_batch - y_mean), dim=1)
        var_x = torch.var(x_axis)
        linreg_slope = cov_xy / (var_x + 1e-8)

        # 예측 추세 품질 지수
        changes = torch.diff(seq_batch, dim=1)
        mean_of_changes = torch.mean(changes, dim=1)
        std_of_changes = torch.std(changes, dim=1)
        pred_trend_quality_ratio = mean_of_changes / (std_of_changes + 1e-8)

        return {
            "mean": mean_val, "std_dev": std_dev, "total_return": total_return,
            "total_return_rate": total_return_rate, "max_drawdown": max_drawdown,
            "final_pos_in_range": final_pos_in_range, "linreg_slope": linreg_slope,
            "pred_trend_quality_ratio": pred_trend_quality_ratio,
        }

    def _create_metafeatures(self, pred_list):
        """
        (batch_size, pred_len) 텐서들의 리스트를 입력받아,
        모든 정보를 요약한 (batch_size, feature_size) 최종 상태 텐서를 반환합니다.
        """
        if not pred_list:
            return torch.tensor([])

        batch_size = pred_list[0].shape[0]
        device = pred_list[0].device
        
        # 각 예측 길이(horizon)에 대해 통계량 계산
        all_stats_by_horizon = [self._calculate_stats(seq_batch) for seq_batch in pred_list]

        # 유효한 계산 결과만 필터링
        valid_stats_by_horizon = [s for s in all_stats_by_horizon if s is not None and not torch.isnan(s['mean'][0])]
        if not valid_stats_by_horizon:
            # 모든 시퀀스가 너무 짧아서 유효한 통계가 없는 경우
            # 예시: 8개 지표 * 4개 메타피처 = 32. 0으로 채워진 텐서 반환
            return torch.zeros(batch_size, 8 * 4, device=device)
            
        indicator_names = list(valid_stats_by_horizon[0].keys())

        # 가장 짧고 긴 예측 길이를 가진 결과의 인덱스를 찾음
        seq_lengths = [seq.shape[1] for i, seq in enumerate(pred_list) if all_stats_by_horizon[i] is not None]
        shortest_idx_in_valid = seq_lengths.index(min(seq_lengths))
        longest_idx_in_valid = seq_lengths.index(max(seq_lengths))
        
        final_feature_list = []

        # 각 지표에 대해 메타 피처 계산
        for name in indicator_names:
            # 여러 예측 길이에 대한 해당 지표 값들을 모아 텐서를 만듦 (batch_size, num_horizons)
            values_across_horizons = torch.stack([stats[name] for stats in valid_stats_by_horizon], dim=1)
            
            # 메타 피처 계산
            mean_feature = torch.mean(values_across_horizons, dim=1)
            std_feature = torch.std(values_across_horizons, dim=1)
            short_term_feature = values_across_horizons[:, shortest_idx_in_valid]
            long_term_feature = values_across_horizons[:, longest_idx_in_valid]
            
            final_feature_list.extend([mean_feature, std_feature, short_term_feature, long_term_feature])
            
        # 최종적으로 (batch_size, feature_size) 텐서를 만듦
        return torch.stack(final_feature_list, dim=1)
