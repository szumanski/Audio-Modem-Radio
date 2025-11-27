# neural_modem.py - MODEM NEURAL CORRIGIDO
import numpy as np
import torch
import torch.nn as nn


class SimpleNeuralModem:
    """Modem neural simplificado para compatibilidade imediata"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🎯 Modem Neural inicializado em: {self.device}")

    def bytes_to_iq(self, data_bytes, seq_len=1024):
        """Converte bytes para formato I/Q simplificado"""
        data_np = np.frombuffer(data_bytes, dtype=np.uint8)
        data_np = data_np.astype(np.float32) / 255.0

        # Preencher/truncar para tamanho fixo
        if len(data_np) < seq_len:
            padding = np.zeros(seq_len - len(data_np), dtype=np.float32)
            data_np = np.concatenate([data_np, padding])
        else:
            data_np = data_np[:seq_len]

        # Criar sinal I/Q simples (fase e amplitude)
        t = np.linspace(0, 1, seq_len)
        i_signal = data_np * np.cos(2 * np.pi * 5 * t)  # Portadora 5Hz normalizada
        q_signal = data_np * np.sin(2 * np.pi * 5 * t)

        return i_signal + 1j * q_signal

    def iq_to_bytes(self, iq_signal):
        """Converte I/Q de volta para bytes"""
        # Extrair amplitude como dados
        amplitude = np.abs(iq_signal)
        amplitude = (amplitude * 255).astype(np.uint8)
        return bytes(amplitude)

    def neural_modulate(self, data_bytes, symbol_rate=8000):
        """Modulação neural simplificada"""
        try:
            # Converter para I/Q
            iq_signal = self.bytes_to_iq(data_bytes)

            # Criar forma de onda (combinação de senoides)
            duration = len(data_bytes) / symbol_rate
            t = np.linspace(0, duration, len(iq_signal))

            # Portadora principal + modulação neural
            carrier_freq = 8000  # Hz
            waveform = np.real(iq_signal) * np.sin(2 * np.pi * carrier_freq * t)
            waveform += np.imag(iq_signal) * np.cos(2 * np.pi * carrier_freq * t)

            # Normalizar
            max_val = np.max(np.abs(waveform))
            if max_val > 0:
                waveform = waveform / max_val * 0.8

            return waveform.astype(np.float32)

        except Exception as e:
            print(f"❌ Erro na modulação neural: {e}")
            # Fallback para senóide simples
            return self._fallback_modulate(data_bytes, symbol_rate)

    def neural_demodulate(self, audio_samples, symbol_rate=8000):
        """Demodulação neural simplificada"""
        try:
            if len(audio_samples) == 0:
                return b''

            # Detecção de envelope + processamento simples
            envelope = np.abs(audio_samples)

            # Suavizar
            from scipy import signal
            b, a = signal.butter(3, 0.1)
            smoothed = signal.filtfilt(b, a, envelope)

            # Converter para bytes
            if np.max(smoothed) > 0:
                normalized = (smoothed * 255 / np.max(smoothed)).astype(np.uint8)
            else:
                normalized = np.zeros_like(smoothed, dtype=np.uint8)

            return bytes(normalized[:min(len(audio_samples) // 10, len(normalized))])

        except Exception as e:
            print(f"❌ Erro na demodulação neural: {e}")
            return b''

    def _fallback_modulate(self, data_bytes, symbol_rate):
        """Modulação de fallback"""
        duration = max(len(data_bytes) / symbol_rate, 0.1)
        t = np.linspace(0, duration, int(96000 * duration))

        # Senóide simples com frequência baseada nos dados
        freq = 8000 + (hash(data_bytes) % 1000) / 1000 * 2000
        waveform = 0.8 * np.sin(2 * np.pi * freq * t)

        return waveform.astype(np.float32)


# Instância global
neural_modem = SimpleNeuralModem()


# Funções de interface
def neural_modulate(data_bytes: bytes, symbol_rate: int = 8000) -> np.ndarray:
    return neural_modem.neural_modulate(data_bytes, symbol_rate)


def neural_demodulate(audio_samples: np.ndarray, symbol_rate: int = 8000) -> bytes:
    return neural_modem.neural_demodulate(audio_samples, symbol_rate)