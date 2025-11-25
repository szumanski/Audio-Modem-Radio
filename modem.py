# modem.py
import numpy as np
import math
import wave
import io
import struct
from typing import Tuple
import scipy.signal as signal
import numpy as np
import math
import wave
import io
import struct
from typing import Tuple, Optional
import scipy.signal as signal
from scipy.fft import fft, ifft
from config import CONFIG


SAMPLE_RATE = 96000


class AdvancedModem:
    def __init__(self):
        self.sample_rate = CONFIG.get('modem.sample_rate', 96000)
        self.adaptive_equalization = CONFIG.get('modem.adaptive_equalization', True)
        self.noise_reduction = CONFIG.get('modem.noise_reduction', True)

    def _adaptive_filter(self, data: np.ndarray, cutoff_freq: float) -> np.ndarray:
        if not self.noise_reduction:
            return data

        nyquist = self.sample_rate / 2
        normal_cutoff = cutoff_freq / nyquist
        b, a = signal.butter(6, normal_cutoff, btype='low')
        return signal.filtfilt(b, a, data)

    def _calculate_snr(self, signal_data: np.ndarray) -> float:
        if len(signal_data) < 1000:
            return 30.0

        b, a = signal.butter(3, 0.1, btype='high')
        noise_estimate = signal.filtfilt(b, a, signal_data)

        signal_power = np.mean(signal_data ** 2)
        noise_power = np.mean(noise_estimate ** 2)

        return 10 * np.log10(signal_power / (noise_power + 1e-10)) if noise_power > 0 else 50.0

    def _adaptive_gain_control(self, data: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(data ** 2))
        if rms > 0:
            target_rms = 0.3
            gain = target_rms / rms
            return data * min(gain, 5.0)
        return data


def fsk_modulate(data_bytes: bytes, baud=1200, mark_freq=1200.0,
                           space_freq=2200.0, samp_rate=96000) -> np.ndarray:
    modem = AdvancedModem()

    print(f"🎵 Modulando FSK Otimizada: {len(data_bytes)} bytes, baud={baud}")

    if baud >= 9600:
        mark_freq = 8000.0
        space_freq = 16000.0

    bit_dur = 1.0 / baud
    samples_per_bit = int(round(samp_rate * bit_dur))

    t = np.arange(samples_per_bit) / samp_rate
    mark_wave = np.sin(2 * np.pi * mark_freq * t)
    space_wave = np.sin(2 * np.pi * space_freq * t)
    sync_wave = np.sin(2 * np.pi * space_freq * t)

    bits = []
    for byte in data_bytes:
        bits.extend([int(b) for b in f'{byte:08b}'])

    total_samples = samples_per_bit * (len(bits) + 2)
    out = np.zeros(total_samples, dtype=np.float32)

    out[:samples_per_bit] = sync_wave
    ptr = samples_per_bit

    for bit in bits:
        out[ptr:ptr + samples_per_bit] = mark_wave if bit else space_wave
        ptr += samples_per_bit

    out[ptr:ptr + samples_per_bit] = sync_wave

    out = modem._adaptive_gain_control(out)

    m = np.max(np.abs(out))
    if m > 0:
        out = out / m * 0.8

    return out


def fsk_demodulate(samples: np.ndarray, baud=1200, mark_freq=1200.0,
                            space_freq=2200.0, samp_rate=96000) -> bytes:
    modem = AdvancedModem()

    print(f"🔍 Demodulando FSK Avançada: {len(samples)} amostras")

    if baud >= 9600:
        mark_freq = 8000.0
        space_freq = 16000.0

    snr = modem._calculate_snr(samples)
    cutoff = min(space_freq + 2000, samp_rate / 2 * 0.9)

    if snr < 20:
        cutoff = space_freq + 1000

    filtered = modem._adaptive_filter(samples, cutoff)

    samples_per_bit = int(round(samp_rate / baud))

    def optimized_goertzel(chunk, target_freq):
        w = 2.0 * math.pi * target_freq / samp_rate
        cos_w = math.cos(w)
        coeff = 2.0 * cos_w

        s_prev, s_prev2 = 0.0, 0.0
        for x in chunk:
            s = x + coeff * s_prev - s_prev2
            s_prev2, s_prev = s_prev, s
        return s_prev2 * s_prev2 + s_prev * s_prev - coeff * s_prev * s_prev2

    bits = []
    for i in range(0, len(filtered) - samples_per_bit + 1, samples_per_bit):
        chunk = filtered[i:i + samples_per_bit]
        p_mark = optimized_goertzel(chunk, mark_freq)
        p_space = optimized_goertzel(chunk, space_freq)
        bits.append(1 if p_mark > p_space else 0)

    out = bytearray()
    for i in range(0, len(bits) - 7, 8):
        byte = sum((bit << (7 - j)) for j, bit in enumerate(bits[i:i+8]))
        out.append(byte)

    return bytes(out)


def bpsk_modulate(data_bytes: bytes, baud=1200, carrier=3000.0, samp_rate=96000) -> np.ndarray:
    bit_dur = 1.0 / baud
    samples_per_bit = int(round(samp_rate * bit_dur))

    t = np.arange(samples_per_bit) / samp_rate
    carrier_wave = np.sin(2 * np.pi * carrier * t)

    bits = [int(b) for byte in data_bytes for b in f'{byte:08b}']

    out = np.concatenate([carrier_wave if bit else -carrier_wave for bit in bits])
    m = np.max(np.abs(out))
    if m > 0:
        out = out / m * 0.8
    return out.astype(np.float32)


def bpsk_demodulate(samples: np.ndarray, baud=1200, carrier=3000.0, samp_rate=96000) -> bytes:
    samples_per_bit = int(round(samp_rate / baud))

    t = np.arange(samples_per_bit) / samp_rate
    ref = np.sin(2 * np.pi * carrier * t)

    bits = []
    for i in range(0, len(samples) - samples_per_bit + 1, samples_per_bit):
        chunk = samples[i:i + samples_per_bit]
        corr = np.sum(chunk * ref)
        bits.append(1 if corr > 0 else 0)

    out = bytearray()
    for i in range(0, len(bits) - 7, 8):
        byte = sum((bit << (7 - j)) for j, bit in enumerate(bits[i:i+8]))
        out.append(byte)

    return bytes(out)


def qpsk_modulate(data_bytes: bytes, baud=1200, carrier=3000.0, samp_rate=96000) -> np.ndarray:
    """Modulação QPSK Corrigida para compatibilidade com demodulação"""
    print(f"📡 QPSK Modulação INICIADA: {len(data_bytes)} bytes, baud={baud}, carrier={carrier}")

    # Converter bytes para bits
    bits = ''.join(f'{byte:08b}' for byte in data_bytes)

    # Garantir número par de bits
    if len(bits) % 2 != 0:
        bits += '0'
        print("⚠️  Número ímpar de bits, adicionado bit de padding")

    symbols = [int(bits[i:i + 2], 2) for i in range(0, len(bits), 2)]

    samples_per_symbol = int(samp_rate / baud)
    t_symbol = np.arange(samples_per_symbol) / samp_rate

    print(f"📊 {len(symbols)} símbolos a serem modulados, {samples_per_symbol} amostras por símbolo")

    # CORREÇÃO: Fases para QPSK padrão (45°, 135°, 225°, 315°)
    phases = {
        0: np.pi / 4,  # 00 -> 45°
        1: 3 * np.pi / 4,  # 01 -> 135°
        2: 5 * np.pi / 4,  # 11 -> 225°
        3: 7 * np.pi / 4  # 10 -> 315°
    }

    # Criar forma de onda
    waveform = np.array([], dtype=np.float32)

    for symbol_idx, symbol in enumerate(symbols):
        phase = phases[symbol]
        # Gerar o símbolo como cosseno com a fase correspondente
        symbol_wave = np.cos(2 * np.pi * carrier * t_symbol + phase)
        waveform = np.concatenate([waveform, symbol_wave])

    # Normalizar
    max_val = np.max(np.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val * 0.8

    print(f"✅ QPSK modulou {len(data_bytes)} bytes em {len(waveform)} amostras")

    # Verificação dos dados de entrada
    print(f"🔍 Dados de entrada (16 primeiros bytes): {data_bytes[:16].hex()}")

    return waveform.astype(np.float32)


def qpsk_demodulate(samples: np.ndarray, baud=1200, carrier=3000.0, samp_rate=96000) -> bytes:
    """Demodulação QPSK CORRIGIDA - Versão Robustecida"""
    print(f"🎯 QPSK Demodulação INICIADA: {len(samples)} amostras, baud={baud}, carrier={carrier}")

    # Parâmetros básicos
    samples_per_symbol = int(samp_rate / baud)
    total_symbols = len(samples) // samples_per_symbol

    print(f"📊 Amostras por símbolo: {samples_per_symbol}, Símbolos totais: {total_symbols}")

    # CORREÇÃO: Gerar portadoras com fase CORRETA para demodulação
    t_full = np.arange(len(samples)) / samp_rate

    # Portadoras em quadratura - CORREÇÃO: usar as mesmas fases da modulação
    I_carrier = np.cos(2 * np.pi * carrier * t_full)
    Q_carrier = np.sin(2 * np.pi * carrier * t_full)

    # CORREÇÃO: Aplicar filtro passa-banda para isolar o sinal
    from scipy import signal
    lowcut = carrier - baud * 2
    highcut = carrier + baud * 2
    nyq = 0.5 * samp_rate
    low = max(0.01, lowcut / nyq)
    high = min(0.99, highcut / nyq)

    if low < high:
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_samples = signal.filtfilt(b, a, samples)
    else:
        filtered_samples = samples
        print("⚠️  Filtro passa-banda ignorado (frequências inválidas)")

    bits = ''
    symbols_detected = 0

    # CORREÇÃO: Processar cada símbolo com ponto de amostragem no meio do símbolo
    for i in range(total_symbols):
        start_idx = i * samples_per_symbol
        mid_idx = start_idx + samples_per_symbol // 2
        end_idx = start_idx + samples_per_symbol

        if end_idx > len(filtered_samples):
            break

        # CORREÇÃO: Extrair componente I e Q usando correlação
        symbol_chunk = filtered_samples[start_idx:end_idx]
        t_symbol = np.arange(len(symbol_chunk)) / samp_rate

        # Correlacionar com as portadoras
        I_component = np.sum(symbol_chunk * np.cos(2 * np.pi * carrier * t_symbol))
        Q_component = np.sum(symbol_chunk * np.sin(2 * np.pi * carrier * t_symbol))

        # CORREÇÃO CRÍTICA: Decisão de símbolo com limiares adaptativos
        # Calcular limiares baseados na energia do sinal
        energy = np.sqrt(I_component ** 2 + Q_component ** 2)

        if energy < 0.1:  # Threshold para ruído
            # Provavelmente ruído, escolher símbolo mais provável
            bits += '00'
        else:
            # Decodificação QPSK baseada em quadrante
            if I_component >= 0 and Q_component >= 0:
                bits += '00'  # 45°
            elif I_component < 0 and Q_component >= 0:
                bits += '01'  # 135°
            elif I_component < 0 and Q_component < 0:
                bits += '11'  # 225°
            else:  # I_component >= 0 and Q_component < 0
                bits += '10'  # 315°

        symbols_detected += 1

    print(f"🔍 Símbolos processados: {symbols_detected}, Bits gerados: {len(bits)}")

    # CORREÇÃO: Converter bits para bytes com tratamento robusto de erros
    bytes_out = bytearray()
    bit_errors = 0
    bytes_processed = 0

    for i in range(0, len(bits) - 7, 8):
        byte_bits = bits[i:i + 8]
        try:
            byte_val = int(byte_bits, 2)
            bytes_out.append(byte_val)
            bytes_processed += 1
        except ValueError:
            bit_errors += 1
            # Em caso de erro, inserir byte de preenchimento
            bytes_out.append(0x3F)  # '?' em ASCII

    if bit_errors > 0:
        print(f"⚠️  {bit_errors} erros de bit detectados, {bytes_processed} bytes processados")

    print(f"✅ QPSK demodulou {len(bytes_out)} bytes")

    # Verificação crítica dos dados demodulados
    if len(bytes_out) >= 4:
        expected_preamble = b'\xAA\xAA\xAA\xAA'
        actual_preamble = bytes_out[:4]

        print(f"🔍 Verificação do preamble:")
        print(f"   Esperado: {expected_preamble.hex()}")
        print(f"   Obtido:   {actual_preamble.hex()}")

        if actual_preamble == expected_preamble:
            print("🎯 SUCESSO: Preamble detectado corretamente!")
        else:
            print("❌ FALHA: Preamble não encontrado!")
            print("🔄 Tentando sincronização manual...")

            # CORREÇÃO: Tentar encontrar o preamble manualmente
            data_bytes = bytes(bytes_out)
            preamble_pos = data_bytes.find(expected_preamble)

            if preamble_pos != -1:
                print(f"🎯 Preamble encontrado na posição {preamble_pos}")
                # Recortar dados a partir do preamble
                bytes_out = bytearray(data_bytes[preamble_pos:])
                print(f"📏 Dados recortados: {len(bytes_out)} bytes")
            else:
                print("❌ Preamble não encontrado em nenhuma posição")

    return bytes(bytes_out)

def psk8_modulate(data_bytes: bytes, baud=1200, carrier=3000.0, samp_rate=96000) -> np.ndarray:
    bits = ''.join(f'{byte:08b}' for byte in data_bytes)
    symbols = [int(bits[i:i+3], 2) for i in range(0, len(bits), 3)]

    samples_per_symbol = int(samp_rate / baud)
    t = np.arange(samples_per_symbol) / samp_rate

    phase_shifts = [k * np.pi / 4 for k in range(8)]

    out = np.concatenate([np.cos(2 * np.pi * carrier * t + phase_shifts[sym]) for sym in symbols])

    out /= np.max(np.abs(out)) * 0.8
    return out.astype(np.float32)


def psk8_demodulate(samples: np.ndarray, baud=1200, carrier=3000.0, samp_rate=96000) -> bytes:
    samples_per_symbol = int(samp_rate / baud)

    bits = ''
    for i in range(0, len(samples), samples_per_symbol):
        chunk = samples[i:i+samples_per_symbol]
        phase = np.angle(np.sum(chunk * np.exp(-1j * 2 * np.pi * carrier * np.arange(len(chunk)) / samp_rate)))
        sym = round((phase % (2*np.pi)) / (np.pi/4)) % 8
        bits += f'{sym:03b}'

    bits = bits[: (len(bits) // 8) * 8]

    bytes_out = [int(bits[j:j+8], 2) for j in range(0, len(bits), 8)]
    return bytes(bytes_out)


def apsk16_modulate(data_bytes: bytes, baud=1200, carrier=3000.0, samp_rate=96000) -> np.ndarray:
    bits = ''.join(f'{byte:08b}' for byte in data_bytes)
    symbols = [int(bits[i:i+4], 2) for i in range(0, len(bits), 4)]

    samples_per_symbol = int(samp_rate / baud)
    t = np.arange(samples_per_symbol) / samp_rate

    const = [ (a + 1j*b) for a in [-3,-1,1,3] for b in [-3,-1,1,3] ]

    out = np.concatenate([np.real(const[sym] * np.exp(1j * 2 * np.pi * carrier * t)) for sym in symbols])

    out /= np.max(np.abs(out)) * 0.8
    return out.astype(np.float32)


def apsk16_demodulate(samples: np.ndarray, baud=1200, carrier=3000.0, samp_rate=96000) -> bytes:
    samples_per_symbol = int(samp_rate / baud)

    bits = ''
    const = [ (a + 1j*b) for a in [-3,-1,1,3] for b in [-3,-1,1,3] ]

    for i in range(0, len(samples), samples_per_symbol):
        chunk = samples[i:i+samples_per_symbol]
        sym_complex = np.mean(chunk * np.exp(-1j * 2 * np.pi * carrier * np.arange(len(chunk)) / samp_rate))
        sym = np.argmin([np.abs(sym_complex - c) for c in const])
        bits += f'{sym:04b}'

    bytes_out = [int(bits[j:j+8], 2) for j in range(0, len(bits), 8)]
    return bytes(bytes_out)


def dsss_modulate(data_bytes: bytes, baud=1200, carrier=3000.0, samp_rate=96000) -> np.ndarray:
    pn = [1,0,1,1,0,0,1] * (baud // 7)
    bits = [2*int(b)-1 for byte in data_bytes for b in f'{byte:08b}']

    spread = np.repeat(bits, len(pn)) * np.tile(pn, len(bits))

    t = np.arange(len(spread)) / samp_rate
    out = np.sin(2 * np.pi * carrier * t) * (2 * spread - 1)

    out /= np.max(np.abs(out)) * 0.8
    return out.astype(np.float32)


def dsss_demodulate(samples: np.ndarray, baud=1200, carrier=3000.0, samp_rate=96000) -> bytes:
    pn = [1,0,1,1,0,0,1] * (baud // 7)
    chip_rate = len(pn) * baud

    t = np.arange(len(samples)) / samp_rate
    carrier_sig = np.sin(2 * np.pi * carrier * t)

    demod = samples * carrier_sig

    bits = []
    chips_per_bit = len(pn)
    for i in range(0, len(demod), chips_per_bit):
        chunk = demod[i:i+chips_per_bit]
        corr = np.sum(chunk * pn)
        bits.append(1 if corr > 0 else 0)

    bytes_out = [int(''.join(map(str, bits[k:k+8])), 2) for k in range(0, len(bits), 8)]
    return bytes(bytes_out)


def msk_modulate(data_bytes: bytes, baud=1200, carrier=3000.0, samp_rate=96000) -> np.ndarray:
    bits = [2*int(b)-1 for byte in data_bytes for b in f'{byte:08b}']

    samples_per_bit = int(samp_rate / baud)
    t = np.arange(samples_per_bit) / samp_rate

    phase = 0
    out = []
    for bit in bits:
        phase_inc = bit * np.pi / (2 * samples_per_bit)
        phase_arr = phase + np.cumsum(np.ones(samples_per_bit) * phase_inc)
        out.append(np.sin(2 * np.pi * carrier * t + phase_arr))
        phase = phase_arr[-1] % (2*np.pi)

    out = np.concatenate(out)
    out /= np.max(np.abs(out)) * 0.8
    return out.astype(np.float32)


def msk_demodulate(samples: np.ndarray, baud=1200, carrier=3000.0, samp_rate=96000) -> bytes:
    samples_per_bit = int(samp_rate / baud)

    i = samples * np.cos(2 * np.pi * carrier * np.arange(len(samples)) / samp_rate)
    q = samples * np.sin(2 * np.pi * carrier * np.arange(len(samples)) / samp_rate)

    bits = []
    prev_i, prev_q = 0, 0
    for j in range(0, len(samples), samples_per_bit):
        curr_i = np.mean(i[j:j+samples_per_bit])
        curr_q = np.mean(q[j:j+samples_per_bit])
        diff = curr_i * prev_q - curr_q * prev_i
        bits.append(1 if diff > 0 else 0)
        prev_i, prev_q = curr_i, curr_q

    bytes_out = [int(''.join(map(str, bits[k:k+8])), 2) for k in range(0, len(bits), 8)]
    return bytes(bytes_out)


def fsk_high_speed_modulate(data_bytes: bytes, baud=19200, mark_freq=12000.0, space_freq=18000.0,
                              samp_rate=SAMPLE_RATE) -> np.ndarray:
    return fsk_modulate(data_bytes, baud=baud, mark_freq=mark_freq, space_freq=space_freq, samp_rate=samp_rate)


def fsk_high_speed_demodulate(samples: np.ndarray, baud=19200, mark_freq=12000.0, space_freq=18000.0,
                              samp_rate=SAMPLE_RATE) -> bytes:
    lowcut = 8000
    highcut = 22000
    samples = bandpass_filter(samples, lowcut, highcut, samp_rate)

    samples = np.asarray(samples, dtype=np.float32)
    samples_per_bit = int(round(samp_rate / baud))

    def goertzel(chunk, freq):
        s_prev = 0.0
        s_prev2 = 0.0
        normalized = freq / samp_rate
        coeff = 2.0 * math.cos(2 * math.pi * normalized)
        for x in chunk:
            s = x + coeff * s_prev - s_prev2
            s_prev2 = s_prev
            s_prev = s
        power = s_prev2 * s_prev2 + s_prev * s_prev - coeff * s_prev * s_prev2
        return power

    bits = []
    for i in range(0, len(samples) - samples_per_bit + 1, samples_per_bit):
        chunk = samples[i:i + samples_per_bit]
        p_mark = goertzel(chunk, mark_freq)
        p_space = goertzel(chunk, space_freq)
        bits.append(1 if p_mark > p_space else 0)

    out = []
    for i in range(0, len(bits) - 7, 8):
        byte_bits = bits[i:i + 8]
        b = 0
        for k, bit in enumerate(byte_bits):
            b |= (bit & 1) << (7 - k)
        out.append(b)

    return bytes(out)


def ofdm_modulate_simple(data_bytes: bytes, baud=4800, carrier=12000.0, num_subcarriers=8,
                         samp_rate=SAMPLE_RATE) -> np.ndarray:
    from scipy.fft import ifft

    bits = ''.join(format(b, '08b') for b in data_bytes)

    bits_per_symbol = num_subcarriers * 2
    symbols = []

    for i in range(0, len(bits), bits_per_symbol):
        symbol_bits = bits[i:i + bits_per_symbol]
        if len(symbol_bits) < bits_per_symbol:
            symbol_bits += '0' * (bits_per_symbol - len(symbol_bits))

        symbol = []
        for j in range(0, len(symbol_bits), 2):
            two_bits = symbol_bits[j:j + 2]
            if two_bits == '00':
                symbol.append(1 + 1j)
            elif two_bits == '01':
                symbol.append(-1 + 1j)
            elif two_bits == '10':
                symbol.append(1 - 1j)
            else:
                symbol.append(-1 - 1j)

        symbols.append(symbol)

    time_domain = []
    samples_per_symbol = int(round(samp_rate / baud))

    for symbol in symbols:
        ifft_input = np.zeros(num_subcarriers, dtype=complex)
        ifft_input[1:len(symbol) + 1] = symbol

        td_symbol = ifft(ifft_input)

        cp_length = samples_per_symbol // 4
        cyclic_prefix = td_symbol[-cp_length:]
        full_symbol = np.concatenate([cyclic_prefix, td_symbol])

        time_domain.extend(full_symbol.real)

    output = np.array(time_domain, dtype=np.float32)
    m = np.max(np.abs(output))
    if m > 0:
        output = output / m * 0.8
    return output


def ofdm_demodulate_simple(samples: np.ndarray, baud=4800, carrier=12000.0, num_subcarriers=8,
                           samp_rate=SAMPLE_RATE) -> bytes:
    from scipy.fft import fft

    samples_per_symbol = int(round(samp_rate / baud))
    cp_length = samples_per_symbol // 4
    useful_length = samples_per_symbol - cp_length

    bits = ''
    i = 0

    while i + samples_per_symbol <= len(samples):
        # Remove cyclic prefix
        symbol_with_cp = samples[i:i + samples_per_symbol]
        symbol = symbol_with_cp[cp_length:cp_length + useful_length]

        # FFT
        freq_domain = fft(symbol)

        # Extrai subportadoras (ignora DC e extremos)
        subcarriers = freq_domain[1:num_subcarriers + 1]

        # Demodula QPSK em cada subportadora
        for sc in subcarriers:
            I, Q = sc.real, sc.imag

            if I >= 0 and Q >= 0:
                bits += '00'
            elif I < 0 and Q >= 0:
                bits += '01'
            elif I < 0 and Q < 0:
                bits += '11'
            else:
                bits += '10'

        i += samples_per_symbol

    if len(bits) % 8 != 0:
        bits = bits[:-(len(bits) % 8)]

    out = bytearray()
    for i in range(0, len(bits), 8):
        out.append(int(bits[i:i + 8], 2))
    return bytes(out)


def ft8_modulate(data_bytes: bytes, baud=6.25, carrier=1500.0, samp_rate=SAMPLE_RATE) -> np.ndarray:
    return ofdm_modulate_simple(data_bytes, baud=baud, carrier=carrier, num_subcarriers=79)


def ft8_demodulate(samples: np.ndarray, baud=6.25, carrier=1500.0, samp_rate=SAMPLE_RATE) -> bytes:
    return ofdm_demodulate_simple(samples, baud=baud, carrier=carrier, num_subcarriers=79)


def ft4_modulate(data_bytes: bytes, baud=20.83, carrier=1500.0, samp_rate=SAMPLE_RATE) -> np.ndarray:
    return ft8_modulate(data_bytes, baud=baud, carrier=carrier)


def ft4_demodulate(samples: np.ndarray, baud=20.83, carrier=1500.0, samp_rate=SAMPLE_RATE) -> bytes:
    return ft8_demodulate(samples, baud=baud, carrier=carrier)


def js8_modulate(data_bytes: bytes, baud=15.625, carrier=1500.0, samp_rate=SAMPLE_RATE) -> np.ndarray:
    return ft8_modulate(data_bytes, baud=baud, carrier=carrier)


def js8_demodulate(samples: np.ndarray, baud=15.625, carrier=1500.0, samp_rate=SAMPLE_RATE) -> bytes:
    return ft8_demodulate(samples, baud=baud, carrier=carrier)


def psk31_modulate(data_bytes: bytes, baud=31.25, carrier=1000.0, samp_rate=SAMPLE_RATE) -> np.ndarray:
    return bpsk_modulate(data_bytes, baud=baud, carrier=carrier)


def psk31_demodulate(samples: np.ndarray, baud=31.25, carrier=1000.0, samp_rate=SAMPLE_RATE) -> bytes:
    return bpsk_demodulate(samples, baud=baud, carrier=carrier)


def rtty_modulate(data_bytes: bytes, baud=45, mark_freq=2295.0, space_freq=2125.0, samp_rate=SAMPLE_RATE) -> np.ndarray:
    return fsk_modulate(data_bytes, baud=baud, mark_freq=mark_freq, space_freq=space_freq, samp_rate=samp_rate)


def rtty_demodulate(samples: np.ndarray, baud=45, mark_freq=2295.0, space_freq=2125.0, samp_rate=SAMPLE_RATE) -> bytes:
    return fsk_demodulate(samples, baud=baud, mark_freq=mark_freq, space_freq=space_freq, samp_rate=samp_rate)


def mfsk8_modulate(data_bytes: bytes, baud=7.8125, carrier=1000.0, num_tones=8, samp_rate=SAMPLE_RATE) -> np.ndarray:
    return ofdm_modulate_simple(data_bytes, baud=baud, carrier=carrier, num_subcarriers=num_tones)


def mfsk8_demodulate(samples: np.ndarray, baud=7.8125, carrier=1000.0, num_tones=8, samp_rate=SAMPLE_RATE) -> bytes:
    return ofdm_demodulate_simple(samples, baud=baud, carrier=carrier, num_subcarriers=num_tones)


def afsk1200_modulate(data_bytes: bytes, baud=1200, mark_freq=1200.0, space_freq=2200.0, samp_rate=SAMPLE_RATE) -> np.ndarray:
    return fsk_modulate(data_bytes, baud=baud, mark_freq=mark_freq, space_freq=space_freq, samp_rate=samp_rate)


def afsk1200_demodulate(samples: np.ndarray, baud=1200, mark_freq=1200.0, space_freq=2200.0, samp_rate=SAMPLE_RATE) -> bytes:
    return fsk_demodulate(samples, baud=baud, mark_freq=mark_freq, space_freq=space_freq, samp_rate=samp_rate)


def pactor_modulate(data_bytes: bytes, baud=200, carrier=1500.0, samp_rate=SAMPLE_RATE) -> np.ndarray:
    return psk8_modulate(data_bytes, baud=baud, carrier=carrier)


def pactor_demodulate(samples: np.ndarray, baud=200, carrier=1500.0, samp_rate=SAMPLE_RATE) -> bytes:
    return psk8_demodulate(samples, baud=baud, carrier=carrier)


def dmr_modulate(data_bytes: bytes, baud=4800, carrier=3000.0, samp_rate=SAMPLE_RATE) -> np.ndarray:
    return apsk16_modulate(data_bytes, baud=baud, carrier=carrier)


def dmr_demodulate(samples: np.ndarray, baud=4800, carrier=3000.0, samp_rate=SAMPLE_RATE) -> bytes:
    return apsk16_demodulate(samples, baud=baud, carrier=carrier)


def olivia_modulate(data_bytes: bytes, baud=31.25, carrier=1000.0, num_tones=32, samp_rate=SAMPLE_RATE) -> np.ndarray:
    return mfsk8_modulate(data_bytes, baud=baud, carrier=carrier, num_tones=num_tones)


def olivia_demodulate(samples: np.ndarray, baud=31.25, carrier=1000.0, num_tones=32, samp_rate=SAMPLE_RATE) -> bytes:
    return mfsk8_demodulate(samples, baud=baud, carrier=carrier, num_tones=num_tones)


def feld_hell_modulate(data_bytes: bytes, baud=122.5, carrier=1000.0, samp_rate=SAMPLE_RATE) -> np.ndarray:
    """Fallback implementation for Feld-Hell modulation"""
    # Use simple BPSK as fallback
    return bpsk_modulate(data_bytes, baud=baud, carrier=carrier, samp_rate=samp_rate)

def feld_hell_demodulate(samples: np.ndarray, baud=122.5, carrier=1000.0, samp_rate=SAMPLE_RATE) -> bytes:
    """Fallback implementation for Feld-Hell demodulation"""
    # Use simple BPSK as fallback
    return bpsk_demodulate(samples, baud=baud, carrier=carrier, samp_rate=samp_rate)

def lora_modulate(data_bytes: bytes, spreading_factor=7, bandwidth=125000, samp_rate=SAMPLE_RATE) -> np.ndarray:
    t = np.arange(len(data_bytes) * 100) / samp_rate
    chirp = np.sin(2 * np.pi * (bandwidth / 2) * t ** 2)
    return chirp.astype(np.float32)


def lora_demodulate(samples: np.ndarray, spreading_factor=7, bandwidth=125000, samp_rate=SAMPLE_RATE) -> bytes:
    return b''


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Filtro passa-banda com tratamento de erro robusto"""
    try:
        # Verificar parâmetros
        if lowcut <= 0 or highcut <= 0:
            raise ValueError("Frequências de corte devem ser positivas")

        if lowcut >= highcut:
            raise ValueError("lowcut deve ser menor que highcut")

        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq

        # Verificar se as frequências normalizadas são válidas
        if low <= 0 or high <= 0 or low >= 1 or high >= 1:
            raise ValueError("Frequências normalizadas inválidas")

        b, a = signal.butter(order, [low, high], btype='band')
        return signal.lfilter(b, a, data)

    except Exception as e:
        print(f"❌ Erro no filtro passa-banda: {e}")
        print(f"   Parâmetros: lowcut={lowcut}, highcut={highcut}, fs={fs}")
        # Retornar dados originais em caso de erro
        return data


def test_qpsk_loopback():
    """Teste direto de modulação/demodulação QPSK"""
    print("\n" + "=" * 60)
    print("🎯 TESTE DE LOOPBACK QPSK")
    print("=" * 60)

    # Dados de teste exatamente como no frame real
    test_data = b'\xAA\xAA\xAA\xAA' + b'FBPC' + b'\x04' + b'test' + b'\x00\x00\x00\x00\x01\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00' + b'data'

    print(f"📤 Dados de teste: {len(test_data)} bytes")
    print(f"🔍 Estrutura completa: {test_data.hex()}")

    # Parâmetros de modulação
    baud = 9600
    carrier = 3000.0
    samp_rate = 96000

    # Modular
    print("\n📡 MODULAÇÃO...")
    modulated = qpsk_modulate(test_data, baud=baud, carrier=carrier, samp_rate=samp_rate)

    # Demodular
    print("\n📥 DEMODULAÇÃO...")
    demodulated = qpsk_demodulate(modulated, baud=baud, carrier=carrier, samp_rate=samp_rate)

    # Verificação detalhada
    print("\n🔍 VERIFICAÇÃO DETALHADA:")
    print(f"📊 Dados originais:  {len(test_data)} bytes")
    print(f"📊 Dados demodulados: {len(demodulated)} bytes")

    print(f"🔍 Originais (hex):  {test_data.hex()}")
    print(f"🔍 Demodulados (hex): {demodulated.hex()}")

    # Verificar byte a byte
    if test_data == demodulated:
        print("🎯 ✅ TESTE BEM-SUCEDIDO: Dados idênticos!")
        return True
    else:
        print("❌ 💥 TESTE FALHOU: Dados diferentes!")

        # Encontrar a primeira diferença
        min_len = min(len(test_data), len(demodulated))
        for i in range(min_len):
            if test_data[i] != demodulated[i]:
                print(f"   🔍 Primeira diferença no byte {i}:")
                print(f"      Original:  {test_data[i:min(i + 8, len(test_data))].hex()}")
                print(f"      Demodulado: {demodulated[i:min(i + 8, len(demodulated))].hex()}")
                break

        if len(test_data) != len(demodulated):
            print(f"   🔍 Tamanhos diferentes: original={len(test_data)}, demodulado={len(demodulated)}")

        return False

def test_modulation_demodulation_loop():
    """Testa o pipeline completo de modulação/demodulação"""
    print("\n🎯 TESTE DE PIPELINE COMPLETO")

    # Dados de teste
    test_data = b'\xAA\xAA\xAA\xAA' + b'FBPC' + b'\x04' + b'test' + b'\x00\x00\x00\x00\x01\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00' + b'data'

    print(f"📤 Dados de teste: {len(test_data)} bytes")
    print(f"🔍 Estrutura: {test_data.hex()}")

    # Modular
    modulated = qpsk_modulate(test_data, baud=9600, carrier=3000.0)
    print(f"📡 Modulado: {len(modulated)} amostras")

    # Demodular
    demodulated = qpsk_demodulate(modulated, baud=9600, carrier=3000.0)
    print(f"📥 Demodulado: {len(demodulated)} bytes")

    # Verificar
    if test_data == demodulated:
        print("✅ TESTE BEM-SUCEDIDO: Dados idênticos!")
    else:
        print("❌ TESTE FALHOU: Dados diferentes!")
        print(f"   Esperado: {test_data.hex()}")
        print(f"   Obtido: {demodulated.hex()}")

    return test_data == demodulated


def test_qpsk_fixed():
    """Teste da QPSK corrigida"""
    print("\n" + "=" * 60)
    print("🎯 TESTE QPSK CORRIGIDO")
    print("=" * 60)

    # Dados de teste simples
    test_data = b'\xAA\xAA\xAA\xAA' + b'TEST'

    print(f"📤 Dados de teste: {test_data.hex()}")

    # Modular
    modulated = qpsk_modulate(test_data, baud=9600, carrier=3000.0)
    print(f"📡 Sinal modulado: {len(modulated)} amostras")

    # Adicionar um pouco de ruído para simular condições reais
    noise = np.random.normal(0, 0.05, len(modulated))
    modulated_noisy = modulated + noise

    # Demodular
    demodulated = qpsk_demodulate(modulated_noisy, baud=9600, carrier=3000.0)
    print(f"📥 Dados demodulados: {demodulated.hex()}")

    # Verificar
    if test_data in demodulated:
        print("✅ SUCESSO: Dados recuperados corretamente!")
        # Encontrar onde começam os dados corretos
        pos = demodulated.find(test_data)
        print(f"📍 Dados encontrados na posição: {pos}")
        return True
    else:
        print("❌ FALHA: Dados não recuperados")
        return False


# Executar teste se o arquivo for executado diretamente
if __name__ == "__main__":
    test_qpsk_fixed()

# --- helpers to write WAV ---
def wav_from_array(arr: np.ndarray, samp_rate=SAMPLE_RATE) -> bytes:
    a = (arr * 32767).astype(np.int16)
    bio = io.BytesIO()
    wf = wave.open(bio, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(samp_rate)
    wf.writeframes(a.tobytes())
    wf.close()
    bio.seek(0)
    return bio.read()