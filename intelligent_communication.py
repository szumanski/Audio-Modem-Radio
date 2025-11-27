# intelligent_communication.py - IA de Comunicação Leve
import numpy as np
import time
from typing import Dict, Any


class ChannelAnalyzer:
    """Analisador leve de condições de canal"""

    def analyze_conditions(self, audio_samples=None):
        """Analisa condições atuais do canal"""
        conditions = {
            'snr_db': self.estimate_snr(audio_samples) if audio_samples is not None else 25.0,
            'bandwidth_hz': 8000,
            'noise_level': 0.2,
            'timestamp': time.time()
        }
        return conditions

    def estimate_snr(self, samples):
        """Estima SNR de forma simplificada"""
        if samples is None or len(samples) < 1000:
            return 25.0

        try:
            power = np.mean(samples ** 2)
            noise_estimate = np.mean((samples - np.mean(samples)) ** 2)
            snr = 10 * np.log10(power / (noise_estimate + 1e-10))
            return max(10, min(40, snr))
        except:
            return 25.0


class ModeRecommender:
    """Recomendador inteligente de modos"""

    def __init__(self):
        self.mode_profiles = {
            'FSK1200': {'robustness': 0.9, 'speed': 0.3, 'min_snr': 8},
            'FSK9600': {'robustness': 0.7, 'speed': 0.7, 'min_snr': 12},
            'QPSK': {'robustness': 0.6, 'speed': 0.8, 'min_snr': 15},
            'NEURAL': {'robustness': 0.8, 'speed': 0.9, 'min_snr': 10},
            'FSK19200': {'robustness': 0.5, 'speed': 0.9, 'min_snr': 18}
        }

    def recommend_mode(self, conditions, priority='balanced'):
        """Recomenda melhor modo baseado nas condições"""
        valid_modes = []

        for mode, profile in self.mode_profiles.items():
            if conditions['snr_db'] >= profile['min_snr']:
                if priority == 'robustness':
                    score = profile['robustness']
                elif priority == 'speed':
                    score = profile['speed']
                else:  # balanced
                    score = (profile['robustness'] + profile['speed']) / 2

                valid_modes.append((mode, score))

        if not valid_modes:
            return 'FSK1200'  # Fallback

        # Escolher modo com maior score
        best_mode = max(valid_modes, key=lambda x: x[1])[0]
        return best_mode


# Sistema principal leve
channel_analyzer = ChannelAnalyzer()
mode_recommender = ModeRecommender()


def analyze_channel(audio_samples=None):
    """Interface simplificada para análise de canal"""
    return channel_analyzer.analyze_conditions(audio_samples)


def get_recommended_mode(conditions, priority='balanced'):
    """Interface simplificada para recomendação de modo"""
    return mode_recommender.recommend_mode(conditions, priority)


def intelligent_encode_setup(file_size, priority='balanced'):
    """Configuração inteligente para encoding"""
    conditions = analyze_channel()
    recommended_mode = get_recommended_mode(conditions, priority)

    # Configurações baseadas no modo recomendado
    configs = {
        'FSK1200': {'symbol_rate': 1200, 'compress': True},
        'FSK9600': {'symbol_rate': 9600, 'compress': True},
        'QPSK': {'symbol_rate': 9600, 'compress': True},
        'NEURAL': {'symbol_rate': 8000, 'compress': False},
        'FSK19200': {'symbol_rate': 19200, 'compress': True}
    }

    config = configs.get(recommended_mode, configs['FSK9600'])
    config['mode'] = recommended_mode

    return config