# encoder.py - VERSÃO MELHORADA E COMPLETA
import os, time, math, binascii, struct, hashlib, functools
from typing import Tuple, List
from modem import (fsk_modulate, bpsk_modulate, qpsk_modulate,
                   psk8_modulate, fsk_high_speed_modulate, ofdm_modulate_simple,
                   wav_from_array, SAMPLE_RATE, apsk16_modulate, apsk16_demodulate, dsss_modulate, dsss_demodulate,
                   msk_modulate, msk_demodulate)
from utils.compression import compress_data, prepare_sstv_like, super_compress, delta_compress
from hellschreiber import hellschreiber_modulate
from fec import ReedSolomonFEC, ConvolutionalEncoder
import numpy as np
import pygame
import time
from PyQt5.QtCore import QTimer
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Sistema de cache para evitar re-processamento
_encoding_cache = {}
_encoding_cancelled = False


def get_file_signature(file_path: str, mode: str, compress: bool, symbol_rate: int, target_duration_min: int) -> str:
    """Gera assinatura única para o arquivo e configurações"""
    file_stat = os.stat(file_path)
    file_id = f"{file_path}_{file_stat.st_size}_{file_stat.st_mtime}_{mode}_{compress}_{symbol_rate}_{target_duration_min}"
    return hashlib.md5(file_id.encode()).hexdigest()


def cancel_encoding():
    global _encoding_cancelled
    _encoding_cancelled = True


def reset_encoding_cancel():
    global _encoding_cancelled
    _encoding_cancelled = False


def adaptive_compress(data: bytes, mode: str) -> bytes:
    """Compressão adaptativa baseada no modo e tipo de dados"""
    if len(data) < 1024:  # Pequenos arquivos não compensam compressão
        return data

    # Modos de alta velocidade usam compressão mais agressiva
    if mode in ["8PSK", "FSK19200", "OFDM4", "OFDM8", "APSK16"]:
        compressed = super_compress(data)
        return delta_compress(compressed)
    else:
        return compress_data(data)


def calculate_transmission_stats(file_size: int, mode: str, symbol_rate: int, compress: bool = True) -> dict:
    """Calcula estatísticas detalhadas da transmissão"""
    # Eficiência em bytes/segundo
    efficiency_map = {
        "FSK1200": 100, "FSK9600": 800, "BPSK": symbol_rate // 8,
        "QPSK": symbol_rate // 4, "8PSK": (symbol_rate * 3) // 8,
        "FSK19200": 1600, "OFDM4": symbol_rate // 2, "OFDM8": symbol_rate,
        "SSTV": 50, "APSK16": symbol_rate * 4 // 8,  # 4 bits/símbolo
        "DSSS": symbol_rate // 16,  # Taxa reduzida devido ao espalhamento
        "MSK": symbol_rate // 4, "HELLSCHREIBER": 15  # Muito lento para texto
    }

    bytes_per_sec = efficiency_map.get(mode, symbol_rate // 4)

    # Estimativa de compressão
    compression_ratio = 0.4 if compress and mode not in ["SSTV", "HELLSCHREIBER"] else 1.0
    effective_size = file_size * compression_ratio

    duration_sec = effective_size / bytes_per_sec if bytes_per_sec > 0 else float('inf')

    return {
        'original_size': file_size,
        'effective_size': int(effective_size),
        'compression_ratio': compression_ratio,
        'bytes_per_sec': bytes_per_sec,
        'duration_sec': duration_sec,
        'duration_min': duration_sec / 60,
        'bitrate_bps': bytes_per_sec * 8
    }


def _frame_data(fname: str, data: bytes, part_number: int = 0, total_parts: int = 1,
                file_size: int = 0, file_crc: int = 0, quality_indicator: float = 1.0) -> bytes:
    """Frame data with robust header including error detection and quality indicator"""
    fname_b = fname.encode('utf-8')[:255]
    part_crc = binascii.crc32(data) & 0xffffffff

    preamble = b'\xAA\xAA\xAA\xAA'  # 4 bytes preamble
    magic = b'FBPC'  # 4 bytes magic

    # Adicionar indicador de qualidade (0.0 a 1.0 como byte)
    quality_byte = int(quality_indicator * 255) & 0xFF

    header = (preamble + magic +
              bytes([len(fname_b)]) + fname_b +
              struct.pack('<I', part_number) +
              struct.pack('<I', total_parts) +
              struct.pack('<I', file_size) +
              struct.pack('<I', file_crc) +
              struct.pack('<I', len(data)) +
              struct.pack('<I', part_crc) +
              bytes([quality_byte]))  # Novo: byte de qualidade

    framed_data = header + data

    # DEBUG: Verificar estrutura do frame
    print(f"📦 Frame criado: {len(framed_data)} bytes total")
    print(f"   Preamble+Magic: {preamble.hex() + magic.hex()}")
    print(f"   Nome arquivo: '{fname}' ({len(fname_b)} bytes)")
    print(f"   Part {part_number + 1}/{total_parts}, Size: {len(data)}")
    print(f"   CRC: {part_crc:08X}")

    return framed_data


def split_file_for_transmission(file_path: str, mode: str, symbol_rate: int,
                                target_duration_sec: int = 60) -> List[tuple]:
    """Split file into optimally sized parts"""
    file_size = os.path.getsize(file_path)
    fname = os.path.basename(file_path)

    with open(file_path, 'rb') as f:
        file_data = f.read()

    file_crc = binascii.crc32(file_data) & 0xffffffff

    efficiency_map = {
        "FSK1200": 100, "FSK9600": 800, "BPSK": symbol_rate // 8,
        "QPSK": symbol_rate // 4, "8PSK": (symbol_rate * 3) // 8,
        "FSK19200": 1600, "OFDM4": symbol_rate // 2, "OFDM8": symbol_rate,
        "SSTV": 50, "APSK16": symbol_rate * 4 // 8,
        "DSSS": symbol_rate // 16, "MSK": symbol_rate // 4,
        "HELLSCHREIBER": 15
    }

    bytes_per_sec = efficiency_map.get(mode, symbol_rate // 4)
    part_size = int(bytes_per_sec * target_duration_sec * 0.9)  # 90% para dados

    if file_size <= part_size:
        return [(fname, file_data, 0, 1, file_size, file_crc)]

    parts = []
    total_parts = math.ceil(file_size / part_size)

    for i in range(total_parts):
        start = i * part_size
        end = min((i + 1) * part_size, file_size)
        part_data = file_data[start:end]
        parts.append((fname, part_data, i, total_parts, file_size, file_crc))

    return parts


def encode_hellschreiber(data_bytes: bytes, baud=122.5, carrier=1000.0, samp_rate=SAMPLE_RATE) -> np.ndarray:
    """Modulação Hellschreiber para dados de texto a partir de bytes"""
    try:
        # Converter bytes para texto
        text = data_bytes.decode('utf-8', errors='ignore')
        return hellschreiber_modulate(text, baud, carrier, samp_rate)
    except Exception as e:
        print(f"Erro na codificação Hellschreiber: {e}")
        # Fallback: modulação FSK simples
        return fsk_modulate(data_bytes, baud=300, mark_freq=800, space_freq=1200, samp_rate=samp_rate)


def encode_single_part(part_info, mode, compress, symbol_rate, progress_callback=None, is_cancelled=None):
    """Encode single file part with enhanced error handling"""
    fname, part_data, part_num, total_parts, file_size, file_crc = part_info

    if is_cancelled and is_cancelled():
        raise Exception("Encoding cancelled by user")

    # Compressão adaptativa
    data = adaptive_compress(part_data, mode) if compress else part_data

    framed = _frame_data(fname, data, part_num, total_parts, file_size, file_crc)

    # Modulation dispatch - CORRIGIDO: usar funções diretamente
    if mode == "FSK1200":
        arr = fsk_modulate(framed, baud=1200, mark_freq=1200.0, space_freq=2200.0)
    elif mode == "FSK9600":
        arr = fsk_modulate(framed, baud=9600)
    elif mode == "BPSK":
        arr = bpsk_modulate(framed, baud=symbol_rate, carrier=3000.0)
    elif mode == "QPSK":
        arr = qpsk_modulate(framed, baud=symbol_rate, carrier=3000.0)
    elif mode == "8PSK":
        arr = psk8_modulate(framed, baud=symbol_rate, carrier=12000.0)
    elif mode == "FSK19200":
        arr = fsk_high_speed_modulate(framed, baud=19200)
    elif mode == "OFDM4":
        arr = ofdm_modulate_simple(framed, baud=symbol_rate, carrier=12000.0, num_subcarriers=4)
    elif mode == "OFDM8":
        arr = ofdm_modulate_simple(framed, baud=symbol_rate, carrier=12000.0, num_subcarriers=8)
    elif mode == "APSK16":
        arr = apsk16_modulate(framed, baud=symbol_rate, carrier=12000.0)
    elif mode == "DSSS":
        arr = dsss_modulate(framed, baud=symbol_rate, carrier=3000.0)
    elif mode == "MSK":
        arr = msk_modulate(framed, baud=symbol_rate, carrier=6000.0)
    elif mode == "HELLSCHREIBER":
        arr = encode_hellschreiber(framed, baud=122.5, carrier=1000.0)
    else:
        raise ValueError(f"Modo desconhecido: {mode}")

    wavb = wav_from_array(arr, SAMPLE_RATE)

    outname = os.path.join(CACHE_DIR, f"{fname}.part{part_num + 1:03d}_of_{total_parts:03d}.{mode}.wav")

    with open(outname, 'wb') as wf:
        wf.write(wavb)

    if progress_callback:
        progress_callback(part_num + 1, total_parts)

    return outname


def encode_hellschreiber_text(text: str, output_path: str = None,
                             baud=122.5, carrier=1000.0, samp_rate=SAMPLE_RATE) -> str:
    """Codifica texto em áudio usando Hellschreiber"""
    modulated = hellschreiber_modulate(text, baud, carrier, samp_rate)
    wav_data = wav_from_array(modulated, samp_rate)

    if output_path is None:
        output_path = os.path.join(CACHE_DIR, f"hell_{int(time.time())}.wav")

    with open(output_path, 'wb') as f:
        f.write(wav_data)

    return output_path


def encode_file_with_fec(path: str, mode: str, compress: bool = True,
                        symbol_rate: int = 9600, fec_type: str = "reed_solomon",
                        **kwargs) -> str:
    """Codificação de arquivo com correção de erro integrada"""
    with open(path, 'rb') as f:
        data = f.read()

    # Aplicar FEC
    if fec_type == "reed_solomon":
        fec = ReedSolomonFEC()
        data_encoded = fec.encode(data)
    elif fec_type == "convolutional":
        fec = ConvolutionalEncoder()
        data_encoded = fec.encode(data)
    else:
        data_encoded = data

    # Criar arquivo temporário com dados codificados
    temp_path = os.path.join(CACHE_DIR, f"fec_temp_{int(time.time())}.bin")
    with open(temp_path, 'wb') as f:
        f.write(data_encoded)

    try:
        # Codificar normalmente
        result = encode_file(temp_path, mode, compress, symbol_rate, **kwargs)
        return result
    finally:
        # Limpar arquivo temporário
        try:
            os.remove(temp_path)
        except:
            pass


def encode_file_parts(file_parts: List[tuple], mode: str, compress: bool = True,
                     symbol_rate: int = 9600, progress_callback=None, is_cancelled=None) -> List[str]:
    """Encode multiple file parts with enhanced error handling"""
    encoded_files = []

    for i, part_info in enumerate(file_parts):
        if is_cancelled and is_cancelled():
            # Cleanup partial files
            for f in encoded_files:
                try:
                    os.remove(f)
                except:
                    pass
            raise Exception("Encoding cancelled by user")

        try:
            encoded_file = encode_single_part(part_info, mode, compress, symbol_rate, progress_callback, is_cancelled)
            encoded_files.append(encoded_file)
        except Exception as e:
            # Cleanup on error
            for f in encoded_files:
                try:
                    os.remove(f)
                except:
                    pass
            raise e

    return encoded_files


def encode_file(path: str, mode: str = "FSK9600", compress: bool = True,
               symbol_rate: int = 9600, split_large_files: bool = True,
               target_duration_min: int = 1, progress_callback=None,
               is_cancelled=None) -> str:
    global _encoding_cancelled
    _encoding_cancelled = False

    # Verificar cache
    cache_key = get_file_signature(path, mode, compress, symbol_rate, target_duration_min)
    if cache_key in _encoding_cache:
        cached_file = _encoding_cache[cache_key]
        if os.path.exists(cached_file):
            print(f"Usando arquivo em cache: {cached_file}")
            if progress_callback:
                progress_callback(1, 1)  # Progresso completo
            return cached_file

    fname = os.path.basename(path)
    file_size = os.path.getsize(path)

    # Calcular estatísticas
    stats = calculate_transmission_stats(file_size, mode, symbol_rate, compress)
    print(f"Estatísticas da transmissão: {stats}")

    # Caso especial: Hellschreiber para arquivos de texto
    if mode == "HELLSCHREIBER" and path.endswith('.txt'):
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        result = encode_hellschreiber_text(text)
        _encoding_cache[cache_key] = result
        return result

    if mode == "SSTV":
        payload = prepare_sstv_like(path)
        framed = _frame_data(fname, payload)
        arr = qpsk_modulate(framed, baud=symbol_rate, carrier=3000.0)
        wavb = wav_from_array(arr, SAMPLE_RATE)
        outname = os.path.join(CACHE_DIR, f"{fname}.sstv.sr{symbol_rate}.wav")
        with open(outname, 'wb') as wf:
            wf.write(wavb)
        _encoding_cache[cache_key] = outname
        return outname

    # Verificar se precisa dividir o arquivo
    estimated_duration = stats['duration_sec']

    if split_large_files and estimated_duration > (target_duration_min * 60):
        print(f"Arquivo grande ({file_size} bytes), duração estimada: {estimated_duration:.1f}s")
        print(f"Dividindo em partes de ~{target_duration_min} minuto(s)...")

        file_parts = split_file_for_transmission(path, mode, symbol_rate, target_duration_min * 60)
        encoded_files = encode_file_parts(file_parts, mode, compress, symbol_rate, progress_callback, is_cancelled)

        # Criar arquivo de playlist
        playlist_path = os.path.join(CACHE_DIR, f"{fname}.{mode}.playlist.txt")
        with open(playlist_path, 'w', encoding='utf-8') as pf:
            pf.write(f"File: {fname}\n")
            pf.write(f"Total parts: {len(encoded_files)}\n")
            pf.write(f"Mode: {mode}\n")
            pf.write(f"Symbol rate: {symbol_rate}\n")
            pf.write(f"Original size: {file_size}\n")
            pf.write(f"Compression: {compress}\n")
            pf.write("Parts:\n")
            for ef in encoded_files:
                pf.write(f"{os.path.basename(ef)}\n")

        print(f"Codificado {len(encoded_files)} partes. Playlist: {playlist_path}")
        _encoding_cache[cache_key] = playlist_path
        return playlist_path
    else:
        # Codificação de arquivo único
        with open(path, 'rb') as f:
            raw = f.read()

        data = adaptive_compress(raw, mode) if compress else raw
        framed = _frame_data(fname, data)

        # Modulation - CORRIGIDO: mesma lógica do encode_single_part
        if mode == "FSK1200":
            arr = fsk_modulate(framed, baud=1200, mark_freq=1200.0, space_freq=2200.0)
        elif mode == "FSK9600":
            arr = fsk_modulate(framed, baud=9600)
        elif mode == "BPSK":
            arr = bpsk_modulate(framed, baud=symbol_rate, carrier=3000.0)
        elif mode == "QPSK":
            arr = qpsk_modulate(framed, baud=symbol_rate, carrier=3000.0)
        elif mode == "8PSK":
            arr = psk8_modulate(framed, baud=symbol_rate, carrier=12000.0)
        elif mode == "FSK19200":
            arr = fsk_high_speed_modulate(framed, baud=19200)
        elif mode == "OFDM4":
            arr = ofdm_modulate_simple(framed, baud=symbol_rate, carrier=12000.0, num_subcarriers=4)
        elif mode == "OFDM8":
            arr = ofdm_modulate_simple(framed, baud=symbol_rate, carrier=12000.0, num_subcarriers=8)
        elif mode == "APSK16":
            arr = apsk16_modulate(framed, baud=symbol_rate, carrier=12000.0)
        elif mode == "DSSS":
            arr = dsss_modulate(framed, baud=symbol_rate, carrier=3000.0)
        elif mode == "MSK":
            arr = msk_modulate(framed, baud=symbol_rate, carrier=6000.0)
        elif mode == "HELLSCHREIBER":
            arr = encode_hellschreiber(framed, baud=122.5, carrier=1000.0)
        else:
            raise ValueError(f"Modo desconhecido: {mode}")

        wavb = wav_from_array(arr, SAMPLE_RATE)
        outname = os.path.join(CACHE_DIR, f"{fname}.{mode}.sr{symbol_rate}.wav")

        with open(outname, 'wb') as wf:
            wf.write(wavb)

        _encoding_cache[cache_key] = outname
        return outname


def get_encoding_stats(file_path: str, mode: str, compress: bool = True, symbol_rate: int = 9600) -> dict:
    """Obtém estatísticas de encoding sem realmente codificar"""
    file_size = os.path.getsize(file_path)
    return calculate_transmission_stats(file_size, mode, symbol_rate, compress)


def clear_encoding_cache():
    """Limpa o cache de encoding"""
    global _encoding_cache
    _encoding_cache.clear()


# Adicionar imports necessários no início do arquivo se ainda não estiverem
import numpy as np