# modem.py - CORRIGIDO COM MODULAÇÃO DIFERENCIAL (ROBUSTO)
import numpy as np
import math
import wave
import io
import struct
from scipy import signal
from config import CONFIG
from hellschreiber import hellschreiber_modulate, hellschreiber_demodulate

SAMPLE_RATE = 96000


class AdvancedModem:
    def __init__(self):
        self.sample_rate = CONFIG.get('modem.sample_rate', 96000)

    def _adaptive_gain_control(self, data: np.ndarray) -> np.ndarray:
        max_val = np.max(np.abs(data))
        if max_val > 0:
            return data / max_val * 0.95
        return data


# --- FUNÇÕES DE MODULAÇÃO DIFERENCIAL (DBPSK / DQPSK) ---
# Resolve o problema de inversão de fase e rotação no áudio

def bpsk_modulate(data_bytes: bytes, baud=1200, carrier=3000.0, samp_rate=96000) -> np.ndarray:
    """Implementa DBPSK (Differential BPSK) - Robusto contra inversão de polaridade"""
    bits = [int(b) for byte in data_bytes for b in f'{byte:08b}']

    # Preâmbulo longo para acordar o AGC (0xAA = 101010...)
    preamble = [1, 0] * 40
    bits = preamble + bits

    samples_per_symbol = int(samp_rate / baud)
    t_symbol = np.arange(samples_per_symbol) / samp_rate

    # Codificação Diferencial
    # 0 = Mantém fase, 1 = Inverte fase (180 graus)
    encoded_phases = []
    current_phase = 0

    for bit in bits:
        if bit == 1:
            current_phase += np.pi  # Muda 180 graus
        # Se 0, mantém a fase
        encoded_phases.append(current_phase)

    # Gerar forma de onda
    out = []
    for phase in encoded_phases:
        # Gera o símbolo com a fase acumulada
        symbol = np.sin(2 * np.pi * carrier * t_symbol + phase)
        # Suavização nas bordas para reduzir estalos (Windowing)
        window = np.hanning(len(symbol))
        # Aplicar janela apenas levemente nas bordas para não perder energia
        envelope = np.ones_like(symbol)
        ramp = int(len(symbol) * 0.1)
        envelope[:ramp] = np.linspace(0, 1, ramp)
        envelope[-ramp:] = np.linspace(1, 0, ramp)

        out.extend(symbol * envelope)

    return np.array(out, dtype=np.float32)


def bpsk_demodulate(samples: np.ndarray, baud=1200, carrier=3000.0, samp_rate=96000) -> bytes:
    """Demodula DBPSK usando detecção incoerente (multiplica pelo símbolo anterior)"""
    samples_per_symbol = int(samp_rate / baud)

    # 1. Filtro Passa-Banda
    nyquist = samp_rate / 2
    low = (carrier - baud) / nyquist
    high = (carrier + baud) / nyquist
    b, a = signal.butter(4, [max(0.01, low), min(0.99, high)], btype='band')
    filtered = signal.filtfilt(b, a, samples)

    # 2. Downconversion (Traz para banda base complexa)
    t = np.arange(len(filtered)) / samp_rate
    # Oscilador local
    lo = np.exp(-1j * 2 * np.pi * carrier * t)
    baseband = filtered * lo

    # 3. Filtro Passa-Baixa (para remover a frequência dupla gerada na mixagem)
    lpf_cutoff = baud / nyquist
    b_lp, a_lp = signal.butter(4, lpf_cutoff, btype='low')
    baseband = signal.filtfilt(b_lp, a_lp, baseband)

    # 4. Amostragem no centro do símbolo
    # Pula o início instável
    start_idx = samples_per_symbol
    symbols = baseband[start_idx::samples_per_symbol]

    if len(symbols) < 2:
        return b''

    # 5. Detecção Diferencial: Multiplica Símbolo[N] pelo Conjugado de Símbolo[N-1]
    # Se a fase mudou 180, o resultado real é negativo. Se mudou 0, é positivo.
    diff_symbols = symbols[1:] * np.conj(symbols[:-1])

    bits = []
    for s in diff_symbols:
        # Parte real negativa indica mudança de fase (bit 1), positiva indica manutenção (bit 0)
        bits.append(1 if np.real(s) < 0 else 0)

    # 6. Procurar Preâmbulo e Alinhar Bytes
    # O encoder envia [1, 0] * 40 como preâmbulo. Procuramos o fim dele.
    # Mas como é diferencial, precisamos procurar o padrão de bits decodificados.

    bit_str = ''.join(map(str, bits))

    # Procurar o Magic Byte do frame (FBPC -> 46 42 50 43 hex)
    # FBPC em bits: 01000110 01000010 01010000 01000011
    # Vamos procurar os primeiros 16 bits do Magic para sincronizar
    magic_pattern = "0100011001000010"

    sync_idx = bit_str.find(magic_pattern)

    out = bytearray()
    if sync_idx != -1:
        # Começa a decodificar bytes a partir do sync
        valid_bits = bit_str[sync_idx:]
        for i in range(0, len(valid_bits) - 7, 8):
            byte_bits = valid_bits[i:i + 8]
            try:
                out.append(int(byte_bits, 2))
            except:
                pass
    else:
        # Fallback: tenta converter tudo, o parser busca o frame depois
        for i in range(0, len(bit_str) - 7, 8):
            out.append(int(bit_str[i:i + 8], 2))

    return bytes(out)


def qpsk_modulate(data_bytes: bytes, baud=1200, carrier=3000.0, samp_rate=96000) -> np.ndarray:
    """Implementa DQPSK (Differential QPSK)"""
    # Mapear bytes para pares de bits (dibits)
    bits = []
    for byte in data_bytes:
        for i in range(7, -1, -2):  # 7,5,3,1
            bits.append((byte >> i) & 1)  # Bit mais significativo
            bits.append((byte >> (i - 1)) & 1)  # Bit menos significativo

    # Preâmbulo
    preamble_bits = [0, 0] * 30 + [1, 1] * 10  # Padrão para sincronia
    bits = preamble_bits + bits

    samples_per_symbol = int(samp_rate / baud)
    t_symbol = np.arange(samples_per_symbol) / samp_rate

    # Codificação Diferencial QPSK
    # Dibit -> Mudança de Fase
    # 00 -> 0
    # 01 -> +90 (pi/2)
    # 11 -> +180 (pi)
    # 10 -> +270 (-pi/2) (Código Gray)
    phase_map = {
        (0, 0): 0,
        (0, 1): np.pi / 2,
        (1, 1): np.pi,
        (1, 0): -np.pi / 2
    }

    encoded_phases = []
    current_phase = 0

    for i in range(0, len(bits), 2):
        dibit = (bits[i], bits[i + 1] if i + 1 < len(bits) else 0)
        phase_change = phase_map.get(dibit, 0)
        current_phase += phase_change
        encoded_phases.append(current_phase)

    out = []
    for phase in encoded_phases:
        symbol = np.sin(2 * np.pi * carrier * t_symbol + phase)
        # Windowing suave
        envelope = np.ones_like(symbol)
        ramp = int(len(symbol) * 0.1)
        envelope[:ramp] = np.linspace(0, 1, ramp)
        envelope[-ramp:] = np.linspace(1, 0, ramp)
        out.extend(symbol * envelope)

    return np.array(out, dtype=np.float32)


def qpsk_demodulate(samples: np.ndarray, baud=1200, carrier=3000.0, samp_rate=96000) -> bytes:
    """Demodula DQPSK"""
    samples_per_symbol = int(samp_rate / baud)

    # 1. Filtragem e Downconversion
    nyquist = samp_rate / 2
    low = (carrier - baud * 1.5) / nyquist
    high = (carrier + baud * 1.5) / nyquist
    b, a = signal.butter(4, [max(0.01, low), min(0.99, high)], btype='band')
    filtered = signal.filtfilt(b, a, samples)

    t = np.arange(len(filtered)) / samp_rate
    baseband = filtered * np.exp(-1j * 2 * np.pi * carrier * t)

    b_lp, a_lp = signal.butter(4, baud / nyquist, btype='low')
    baseband = signal.filtfilt(b_lp, a_lp, baseband)

    # 2. Sincronização de Símbolo (Gardner ou Energia simples)
    # Simplificado: oversampling e busca de pico de energia
    # Para produção, precisaria de um polyphase clock recovery
    symbols = baseband[samples_per_symbol // 2::samples_per_symbol]

    if len(symbols) < 2: return b''

    # 3. Detecção Diferencial
    diff_symbols = symbols[1:] * np.conj(symbols[:-1])

    bits = []
    for s in diff_symbols:
        # Calcular ângulo da diferença
        angle = np.angle(s)

        # Quantizar para 4 fases (0, pi/2, pi, -pi/2)
        # Adicionar pi/4 para rotacionar as zonas de decisão
        decision_angle = angle

        # Mapeamento reverso DQPSK
        # 0 rad -> 00
        # pi/2 -> 01
        # pi -> 11
        # -pi/2 -> 10

        # Normalizar para 0..2pi
        if decision_angle < 0: decision_angle += 2 * np.pi

        if decision_angle < np.pi / 4 or decision_angle > 7 * np.pi / 4:
            bits.extend([0, 0])
        elif np.pi / 4 <= decision_angle < 3 * np.pi / 4:
            bits.extend([0, 1])
        elif 3 * np.pi / 4 <= decision_angle < 5 * np.pi / 4:
            bits.extend([1, 1])
        else:
            bits.extend([1, 0])

    # 4. Reconstrução de Bytes
    bit_str = ''.join(map(str, bits))

    # Busca por preâmbulo FBPC
    magic_pattern = "0100011001000010"  # FB
    sync_idx = bit_str.find(magic_pattern)

    out = bytearray()
    if sync_idx != -1:
        valid_bits = bit_str[sync_idx:]
        for i in range(0, len(valid_bits) - 7, 8):
            try:
                out.append(int(valid_bits[i:i + 8], 2))
            except:
                pass
    else:
        # Tentar reconstruir mesmo sem sync perfeito
        for i in range(0, len(bit_str) - 7, 8):
            try:
                out.append(int(bit_str[i:i + 8], 2))
            except:
                pass

    return bytes(out)


# --- FSK Mantido (FSK é robusto por natureza se a frequência estiver certa) ---
def fsk_modulate(data_bytes: bytes, baud=1200, mark_freq=1200.0, space_freq=2200.0, samp_rate=96000) -> np.ndarray:
    bit_dur = 1.0 / baud
    samples_per_bit = int(round(samp_rate * bit_dur))
    t = np.arange(samples_per_bit) / samp_rate

    # Gerar fases contínuas (CPFSK) para evitar estalos espectrais
    phase = 0
    out = []

    # Preâmbulo: 0xAA 0xAA (alternando frequencias para sincronizar)
    full_data = b'\xAA\xAA\xAA\xAA' + data_bytes

    for byte in full_data:
        for i in range(7, -1, -1):
            bit = (byte >> i) & 1
            freq = mark_freq if bit == 1 else space_freq

            # Gera amostras mantendo continuidade de fase
            chunk = np.sin(2 * np.pi * freq * t + phase)
            out.extend(chunk)

            # Atualiza fase final para o próximo bit
            phase += 2 * np.pi * freq * (samples_per_bit / samp_rate)
            phase %= 2 * np.pi

    return np.array(out, dtype=np.float32) * 0.9


def fsk_demodulate(samples: np.ndarray, baud=1200, mark_freq=1200.0, space_freq=2200.0, samp_rate=96000) -> bytes:
    # Usar filtro de quadratura ou Goertzel
    # Implementação simplificada baseada em comparação de energia de filtro
    samples_per_bit = int(samp_rate / baud)

    # Filtros Bandpass para Mark e Space
    nyq = samp_rate / 2

    def get_envelope(freq):
        b, a = signal.butter(3, [(freq - baud) / nyq, (freq + baud) / nyq], btype='band')
        filt = signal.filtfilt(b, a, samples)
        return np.abs(signal.hilbert(filt))

    mark_env = get_envelope(mark_freq)
    space_env = get_envelope(space_freq)

    # Decisão: quem tem mais energia?
    bits = (mark_env > space_env).astype(int)

    # Decimation (pegar 1 amostra por bit)
    # Achar centro do bit é dificil sem clock recovery, vamos fazer média
    decoded_bits = []
    for i in range(samples_per_bit // 2, len(bits), samples_per_bit):
        chunk = bits[i - samples_per_bit // 4: i + samples_per_bit // 4]
        if len(chunk) > 0:
            decoded_bits.append(1 if np.mean(chunk) > 0.5 else 0)

    # Reconstruir bytes
    bit_str = ''.join(map(str, decoded_bits))

    # Sync com FBPC
    magic = "0100011001000010"
    idx = bit_str.find(magic)

    out = bytearray()
    start = idx if idx != -1 else 0

    for i in range(start, len(bit_str) - 7, 8):
        try:
            out.append(int(bit_str[i:i + 8], 2))
        except:
            pass

    return bytes(out)


# Mapeamentos para compatibilidade com código existente
def psk8_modulate(d, b=1200, c=3000.0, s=96000): return qpsk_modulate(d, b, c, s)  # Fallback para QPSK


def psk8_demodulate(s, b=1200, c=3000.0, s_r=96000): return qpsk_demodulate(s, b, c, s_r)


def fsk_high_speed_modulate(d, baud=19200, s=96000):
    return fsk_modulate(d, baud, 8000, 16000, s)


def fsk_high_speed_demodulate(s, baud=19200, s_r=96000):
    return fsk_demodulate(s, baud, 8000, 16000, s_r)


# Outros stubs necessários
def wav_from_array(arr, sr=96000):
    bio = io.BytesIO()
    with wave.open(bio, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        arr_int = (arr * 32767).astype(np.int16)
        wf.writeframes(arr_int.tobytes())
    return bio.getvalue()


def ofdm_modulate_simple(d, baud, carrier, num_subcarriers, samp_rate=96000): return qpsk_modulate(d, baud, carrier,
                                                                                                   samp_rate)


def ofdm_demodulate_simple(s, baud, carrier, num_subcarriers, samp_rate=96000): return qpsk_demodulate(s, baud, carrier,
                                                                                                       samp_rate)


def apsk16_modulate(d, b, c, s=96000): return qpsk_modulate(d, b, c, s)


def dsss_modulate(d, b, c, s=96000): return bpsk_modulate(d, b, c, s)


def msk_modulate(d, b, c, s=96000): return fsk_modulate(d, b, c, c + b, s)


def ft8_modulate(d, b, c, s=96000): return fsk_modulate(d, 50, c, c + 50, s)


def ft8_demodulate(s, b, c, sr=96000): return fsk_demodulate(s, 50, c, c + 50, sr)


def psk31_modulate(d, b, c, s=96000): return bpsk_modulate(d, 31.25, c, s)


def psk31_demodulate(s, b, c, sr=96000): return bpsk_demodulate(s, 31.25, c, sr)


def feld_hell_modulate(d, b, c, s=96000): return hellschreiber_modulate(d.decode('utf-8', 'ignore'), b, c, s)


def feld_hell_demodulate(s, b, c, sr=96000): return hellschreiber_demodulate(s, b, c, sr).encode('utf-8')