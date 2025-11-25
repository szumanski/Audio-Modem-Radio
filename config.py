# config.py - CONFIGURAÇÃO AVANÇADA E OTIMIZADA
import os
from typing import Dict, Any


class ConfigManager:
    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """Carrega configurações com valores otimizados"""
        self._config = {
            'modem': {
                'dsss_spreading_codes': {
                    'default': [1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0],
                    'secure': [1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1]
                },
                'fec_enabled': True,
                'fec_type': 'reed_solomon',  # 'reed_solomon' ou 'convolutional'
                'sample_rate': 96000,
                'quality_threshold': 0.4,  # Aumentado para melhor qualidade
                'duplicate_replacement_threshold': 0.15,
                'assembly_timeout': 7200,  # Timeout aumentado
                'max_quality_samples': 2000,
                'adaptive_equalization': True,
                'noise_reduction': True
            },
            'compression': {
                'enabled': True,
                'aggressive_threshold': 1024,
                'lzma_enabled': True,
                'delta_compression': True
            },
            'performance': {
                'max_workers': 4,
                'buffer_size': 8192,
                'real_time_processing': True,
                'cache_enabled': True,
                'neural_inference': True,  # NOVO: ativar inferência neural
                'gpu_acceleration': True   # NOVO: ativar aceleração GPU
            },
            'ui': {
                'auto_save_logs': True,
                'refresh_interval': 1000,
                'theme': 'dark'
            },
            # NOVO: Configurações do modem neural
            'neural_modem': {
            'enabled': True,
            'default_symbol_rate': 8000,
            'use_compression': False,
            'adaptive_modulation': True
            },

            'intelligent_communication': {
                'enabled': True,
                'auto_mode_selection': True,
                'channel_analysis': True,
                'learning_enabled': False
            },
        }

    def get(self, key: str, default=None):
        keys = key.split('.')
        value = self._config
        for k in keys:
            value = value.get(k, {})
        return value if value != {} else default

    def set(self, key: str, value: Any):
        keys = key.split('.')
        config = self._config
        for k in keys[:-1]:
            config = config.setdefault(k, {})
        config[keys[-1]] = value

    def save_to_file(self, filename: str = "filebeep_config.json"):
        import json
        with open(filename, 'w') as f:
            json.dump(self._config, f, indent=2)

    def load_from_file(self, filename: str = "filebeep_config.json"):
        import json
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                self._config.update(json.load(f))


# Instância global para acesso rápido
CONFIG = ConfigManager()


def get_quality_threshold():
    return CONFIG.get('modem.quality_threshold', 0.3)


def set_quality_threshold(value):
    CONFIG.set('modem.quality_threshold', max(0.0, min(1.0, value)))