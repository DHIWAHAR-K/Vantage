"""
Unit tests for MoE layer components
"""

import pytest
import mlx.core as mx
import mlx.nn as nn

from src.models.moe_layer import MoELayer, SparseMoEConfig
from src.models.router import SparseRouter
from src.models.expert import ExpertNetwork, ExpertParallel


class TestExpertNetwork:
    """Test expert FFN networks"""
    
    def test_expert_forward(self):
        """Test expert forward pass"""
        expert = ExpertNetwork(
            hidden_size=512,
            intermediate_size=2048,
            dropout=0.1,
        )
        
        # Create input
        x = mx.random.normal((2, 10, 512))  # batch=2, seq=10
        
        # Forward pass
        output = expert(x)
        
        # Check shape
        assert output.shape == (2, 10, 512)
        
        # Check not all zeros
        assert mx.abs(output).sum() > 0
    
    def test_expert_parameters(self):
        """Test expert parameter count"""
        expert = ExpertNetwork(
            hidden_size=512,
            intermediate_size=2048,
        )
        
        params = dict(expert.parameters())
        
        # Should have gate_proj, up_proj, down_proj
        assert len(params) >= 3
        
        # Check shapes
        assert params['gate_proj.weight'].shape == (512, 2048)
        assert params['up_proj.weight'].shape == (512, 2048)
        assert params['down_proj.weight'].shape == (2048, 512)


class TestSparseRouter:
    """Test sparse routing network"""
    
    def test_router_forward(self):
        """Test router forward pass"""
        router = SparseRouter(
            hidden_size=512,
            num_experts=8,
            num_experts_per_token=2,
        )
        
        x = mx.random.normal((2, 10, 512))
        
        expert_indices, expert_weights, aux_losses = router(x, training=True)
        
        # Check shapes
        assert expert_indices.shape == (2, 10, 2)
        assert expert_weights.shape == (2, 10, 2)
        
        # Check indices in valid range
        assert expert_indices.min() >= 0
        assert expert_indices.max() < 8
        
        # Check weights sum to 1
        weight_sums = expert_weights.sum(axis=-1)
        assert mx.allclose(weight_sums, mx.ones_like(weight_sums), atol=1e-5)
        
        # Check aux losses present
        assert "router_importance_loss" in aux_losses
        assert "router_load_loss" in aux_losses
        assert "router_z_loss" in aux_losses
    
    def test_router_top_k(self):
        """Test top-K selection"""
        router = SparseRouter(
            hidden_size=512,
            num_experts=16,
            num_experts_per_token=4,  # Top-4
        )
        
        x = mx.random.normal((1, 5, 512))
        
        expert_indices, expert_weights, _ = router(x, training=False)
        
        # Should select exactly K experts
        assert expert_indices.shape[-1] == 4
        assert expert_weights.shape[-1] == 4
    
    def test_router_no_training(self):
        """Test router without training mode"""
        router = SparseRouter(
            hidden_size=512,
            num_experts=8,
            num_experts_per_token=2,
        )
        
        x = mx.random.normal((2, 10, 512))
        
        _, _, aux_losses = router(x, training=False)
        
        # No aux losses when not training
        assert len(aux_losses) == 0


class TestMoELayer:
    """Test complete MoE layer"""
    
    def test_moe_forward(self):
        """Test MoE layer forward pass"""
        moe = MoELayer(
            hidden_size=512,
            intermediate_size=2048,
            num_experts=8,
            num_experts_per_token=2,
        )
        
        x = mx.random.normal((2, 10, 512))
        
        output, aux_losses = moe(x, training=True)
        
        # Check output shape
        assert output.shape == (2, 10, 512)
        
        # Check aux losses
        assert "load_balance_loss" in aux_losses
        assert "router_z_loss" in aux_losses
        assert "moe_aux_loss" in aux_losses
    
    def test_moe_residual(self):
        """Test residual connection"""
        moe = MoELayer(
            hidden_size=512,
            intermediate_size=2048,
            num_experts=4,
            num_experts_per_token=2,
        )
        
        x = mx.random.normal((1, 5, 512))
        
        # Forward without training for simpler test
        output, _ = moe(x, training=False)
        
        # Output should be different from input (not identity)
        assert not mx.allclose(output, x, atol=1e-3)
        
        # But should preserve shape
        assert output.shape == x.shape
    
    def test_moe_expert_statistics(self):
        """Test expert utilization statistics"""
        moe = MoELayer(
            hidden_size=512,
            intermediate_size=2048,
            num_experts=8,
            num_experts_per_token=2,
        )
        
        x = mx.random.normal((4, 20, 512))  # More tokens for statistics
        
        stats = moe.get_expert_statistics(x)
        
        # Check statistics keys
        assert "tokens_per_expert" in stats
        assert "tokens_per_expert_pct" in stats
        assert "avg_weight_per_expert" in stats
        assert "expert_utilization" in stats
        
        # Check expert utilization is reasonable
        utilization = stats["expert_utilization"].item()
        assert 0 <= utilization <= 100


class TestSparseMoEConfig:
    """Test MoE configuration"""
    
    def test_config_creation(self):
        """Test config creation"""
        config = SparseMoEConfig(
            hidden_size=512,
            num_experts=16,
            num_experts_per_token=2,
        )
        
        assert config.hidden_size == 512
        assert config.num_experts == 16
        assert config.num_experts_per_token == 2
    
    def test_config_from_dict(self):
        """Test config from dictionary"""
        config_dict = {
            "hidden_size": 1024,
            "num_experts": 32,
            "num_experts_per_token": 4,
        }
        
        config = SparseMoEConfig.from_dict(config_dict)
        
        assert config.hidden_size == 1024
        assert config.num_experts == 32
        assert config.num_experts_per_token == 4
    
    def test_parameter_calculation(self):
        """Test parameter count calculations"""
        config = SparseMoEConfig(
            hidden_size=512,
            intermediate_size=2048,
            num_experts=8,
            num_experts_per_token=2,
        )
        
        total_params = config.compute_total_parameters()
        active_params = config.compute_active_parameters()
        
        # Active should be less than total
        assert active_params < total_params
        
        # Efficiency ratio should be reasonable
        ratio = config.compute_efficiency_ratio()
        assert 0 < ratio < 1


class TestExpertParallel:
    """Test parallel expert execution"""
    
    def test_parallel_execution(self):
        """Test parallel expert execution"""
        experts = ExpertParallel(
            num_experts=4,
            hidden_size=256,
            intermediate_size=1024,
        )
        
        x = mx.random.normal((2, 8, 256))
        
        # Create routing decisions
        expert_indices = mx.array([
            [[0, 1], [1, 2], [2, 3], [0, 3], [1, 2], [0, 1], [2, 3], [0, 2]],
            [[1, 3], [0, 2], [1, 2], [0, 1], [2, 3], [1, 2], [0, 3], [1, 2]]
        ])  # (batch=2, seq=8, top_k=2)
        
        expert_weights = mx.ones((2, 8, 2)) / 2  # Equal weights
        
        output = experts(x, expert_indices, expert_weights)
        
        # Check output shape
        assert output.shape == (2, 8, 256)
        
        # Check output is not zero
        assert mx.abs(output).sum() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
