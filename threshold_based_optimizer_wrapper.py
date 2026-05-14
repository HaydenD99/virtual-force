"""
基于信道质量阈值的AP选择包装器
实现UC_CF_MIMO_IMPROVEMENT_ANALYSIS.md中的方案2
"""

import numpy as np

class ThresholdBasedAPSelection:
    """
    基于信道质量阈值的AP选择策略
    
    与固定数量选择不同，这种方法：
    1. 根据每个用户的信道质量分布选择AP
    2. 不同用户可能连接不同数量的AP
    3. 自动适应信道条件
    """
    
    def __init__(self, threshold_percentile=70, min_serving=3, max_serving=8):
        """
        参数：
        - threshold_percentile: 信道质量百分位阈值（0-100）
        - min_serving: 最少连接AP数
        - max_serving: 最多连接AP数
        """
        self.threshold_percentile = threshold_percentile
        self.min_serving = min_serving
        self.max_serving = max_serving
    
    def compute_AP_selection_mask(self, betas: np.ndarray) -> np.ndarray:
        """
        基于信道质量阈值计算AP选择掩码
        
        Parameters:
        -----------
        betas : np.ndarray
            大尺度衰落系数矩阵 (K, L)
            K: 用户数
            L: AP数
        
        Returns:
        --------
        mask : np.ndarray
            AP选择掩码 (K, L)，True表示该AP服务该用户
        """
        K, L = betas.shape
        mask = np.zeros_like(betas, dtype=bool)
        
        for k in range(K):
            # 计算该用户的信道质量阈值
            threshold = np.percentile(betas[k, :], self.threshold_percentile)
            
            # 选择超过阈值的AP
            selected_indices = np.where(betas[k, :] >= threshold)[0]
            
            # 如果选择的AP太多，只保留信道质量最好的max_serving个
            if len(selected_indices) > self.max_serving:
                # 按信道质量排序，选择最好的max_serving个
                sorted_indices = selected_indices[np.argsort(betas[k, selected_indices])[-self.max_serving:]]
                selected_indices = sorted_indices
            
            # 如果选择的AP太少，强制选择信道质量最好的min_serving个
            if len(selected_indices) < self.min_serving:
                selected_indices = np.argsort(betas[k, :])[-self.min_serving:]
            
            # 设置掩码
            mask[k, selected_indices] = True
        
        return mask


def create_threshold_based_vf_optimizer(base_optimizer_class):
    """创建支持阈值选择的VF优化器"""
    
    class ThresholdBasedVFOptimizer(base_optimizer_class):
        def __init__(self, config):
            super().__init__(config)
            # 替换AP选择方法
            self.threshold_percentile = config.get('threshold_percentile', 70)
            self.min_serving_APs = config.get('min_serving_APs', 3)
            self.max_serving_APs = config.get('max_serving_APs', 8)
            self.ap_selector = ThresholdBasedAPSelection(
                self.threshold_percentile,
                self.min_serving_APs,
                self.max_serving_APs
            )
        
        def compute_AP_selection_mask(self, betas: np.ndarray) -> np.ndarray:
            """使用阈值选择"""
            return self.ap_selector.compute_AP_selection_mask(betas)
    
    return ThresholdBasedVFOptimizer


def create_threshold_based_ga_optimizer(base_optimizer_class):
    """创建支持阈值选择的GA优化器"""
    
    class ThresholdBasedGAOptimizer(base_optimizer_class):
        def __init__(self, config):
            super().__init__(config)
            self.threshold_percentile = config.get('threshold_percentile', 70)
            self.min_serving_APs = config.get('min_serving_APs', 3)
            self.max_serving_APs = config.get('max_serving_APs', 8)
            self.ap_selector = ThresholdBasedAPSelection(
                self.threshold_percentile,
                self.min_serving_APs,
                self.max_serving_APs
            )
        
        def compute_AP_selection_mask(self, betas: np.ndarray) -> np.ndarray:
            """使用阈值选择"""
            return self.ap_selector.compute_AP_selection_mask(betas)
    
    return ThresholdBasedGAOptimizer


def create_threshold_based_pso_optimizer(base_optimizer_class):
    """创建支持阈值选择的PSO优化器"""
    
    class ThresholdBasedPSOOptimizer(base_optimizer_class):
        def __init__(self, config):
            super().__init__(config)
            self.threshold_percentile = config.get('threshold_percentile', 70)
            self.min_serving_APs = config.get('min_serving_APs', 3)
            self.max_serving_APs = config.get('max_serving_APs', 8)
            self.ap_selector = ThresholdBasedAPSelection(
                self.threshold_percentile,
                self.min_serving_APs,
                self.max_serving_APs
            )
        
        def compute_AP_selection_mask(self, betas: np.ndarray) -> np.ndarray:
            """使用阈值选择"""
            return self.ap_selector.compute_AP_selection_mask(betas)
    
    return ThresholdBasedPSOOptimizer


def create_threshold_based_newssa_optimizer(base_optimizer_class):
    """创建支持阈值选择的NewSSA优化器"""
    
    class ThresholdBasedNewSSAOptimizer(base_optimizer_class):
        def __init__(self, config):
            super().__init__(config)
            self.threshold_percentile = config.get('threshold_percentile', 70)
            self.min_serving_APs = config.get('min_serving_APs', 3)
            self.max_serving_APs = config.get('max_serving_APs', 8)
            self.ap_selector = ThresholdBasedAPSelection(
                self.threshold_percentile,
                self.min_serving_APs,
                self.max_serving_APs
            )
        
        def compute_AP_selection_mask(self, betas: np.ndarray) -> np.ndarray:
            """使用阈值选择"""
            return self.ap_selector.compute_AP_selection_mask(betas)
    
    return ThresholdBasedNewSSAOptimizer


def analyze_ap_selection_distribution(betas, mask):
    """分析AP选择分布"""
    K, L = betas.shape
    serving_counts = mask.sum(axis=1)  # 每个用户连接的AP数
    
    stats = {
        'mean': np.mean(serving_counts),
        'std': np.std(serving_counts),
        'min': np.min(serving_counts),
        'max': np.max(serving_counts),
        'total_APs': L,
        'users': K,
        'avg_utilization': np.mean(serving_counts) / L * 100
    }
    
    return stats
