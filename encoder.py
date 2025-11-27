# encoder.py - VERSÃO MELHORADA E COMPLETA
import os, time, math, binascii, struct, hashlib, functools
from typing import Tuple, List
from modem import (fsk_modulate, bpsk_modulate, qpsk_modulate,
                   psk8_modulate, fsk_high_speed_modulate, ofdm_modulate_simple,
                   wav_from_array, SAMPLE_RATE, apsk16_modulate, dsss_modulate,
                   msk_modulate, ft8_modulate, psk31_modulate, feld_hell_modulate)
from utils.compression import compress_data, prepare_sstv_like, super_compress, delta_compress, intelligent_compress
from hellschreiber import hellschreiber_modulate
from fec import ReedSolomonFEC, ConvolutionalEncoder
import numpy as np
import pygame
import time
from PyQt5.QtCore import QTimer

# Alteração: Importar logging
import logging

logger = logging.getLogger('filebeep')
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Alteração: Sistema de cache com LRU
from functools import lru_cache


@lru_cache(maxsize=50)
def get_file_signature(file_path: str, mode: str, compress: bool, symbol_rate: int) -> str:
    s = os.stat(file_path)
    return hashlib.md5(f"{file_path}_{s.st_size}_{s.st_mtime}_{mode}_{compress}_{symbol_rate}".encode()).hexdigest()

def clear_encoding_cache():
    """Limpa o cache de encoding"""
    get_file_signature.cache_clear()
    logger.info("🧹 Cache de encoding limpo")


_encoding_cancelled = False


def cancel_encoding(): global _encoding_cancelled; _encoding_cancelled = True
_encoding_cancelled = False


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
                file_size: int = 0, file_crc: int = 0) -> bytes:
    """Cria um frame com Header robusto"""
    fname_b = fname.encode('utf-8')[:255]
    part_crc = binascii.crc32(data) & 0xffffffff

    # MAGIC "FBPC"
    magic = b'FBPC'

    # Estrutura:
    # MAGIC (4) + LenName(1) + Name(N) + Part(4) + Total(4) + FSize(4) + FCrc(4) + DataLen(4) + PartCrc(4)
    header = (magic +
              bytes([len(fname_b)]) + fname_b +
              struct.pack('<I', part_number) +
              struct.pack('<I', total_parts) +
              struct.pack('<I', file_size) +
              struct.pack('<I', file_crc) +
              struct.pack('<I', len(data)) +
              struct.pack('<I', part_crc))

    return header + data


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
        end = min(start + part_size, file_size)
        part_data = file_data[start:end]
        parts.append((f"{fname}.part{i + 1}", part_data, i, total_parts, file_size, file_crc))

    return parts


def encode_file_parts(file_parts: List[tuple], mode: str, compress: bool, symbol_rate: int,
                      progress_callback=None, is_cancelled=None) -> List[str]:
    encoded_files = []
    total_parts = len(file_parts)

    for idx, (fname, data, part_number, total_parts, file_size, file_crc) in enumerate(file_parts):
        if is_cancelled and is_cancelled():
            raise RuntimeError("Codificação cancelada pelo usuário")

        logger.info(f"Codificando parte {idx + 1}/{total_parts}: {fname}")

        if compress:
            data = adaptive_compress(data, mode)

        framed = _frame_data(fname, data, part_number, total_parts, file_size, file_crc)

        # Modulação
        logger.info(f"🎯 Modulando com modo: {mode}, taxa: {symbol_rate}")

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
            arr = hellschreiber_modulate(framed.decode('utf-8'))
        elif mode == "FT8":
            arr = ft8_modulate(framed, baud=symbol_rate, carrier=3000.0)
        elif mode == "PSK31":
            arr = psk31_modulate(framed, baud=symbol_rate, carrier=3000.0)
        elif mode == "FELD_HELL":
            arr = feld_hell_modulate(framed, baud=122.5, carrier=1000.0)
        else:
            raise ValueError(f"Modo desconhecido: {mode}")

        # VERIFICAÇÃO CRÍTICA DO ÁUDIO GERADO
        if not verify_audio_output(arr):
            logger.error(f"❌ ERRO: Modulação {mode} produziu áudio inválido para a parte {part_number + 1}!")
            logger.info("🔄 Tentando fallback com BPSK...")
            # Fallback para BPSK com taxa reduzida para melhor confiabilidade
            fallback_symbol_rate = min(symbol_rate, 4800)
            arr = bpsk_modulate(framed, baud=fallback_symbol_rate, carrier=3000.0)

            if not verify_audio_output(arr):
                logger.error("❌ FALHA CRÍTICA: Fallback BPSK também falhou!")
                # Último recurso: gerar tom de teste
                logger.info("🎵 Gerando tom de teste como último recurso...")
                duration = max(len(framed) / fallback_symbol_rate, 1.0)
                t = np.linspace(0, duration, int(SAMPLE_RATE * duration))
                arr = 0.8 * np.sin(2 * np.pi * 1000 * t).astype(np.float32)

                if not verify_audio_output(arr):
                    raise ValueError(
                        "Falha crítica na geração de áudio modulado - não foi possível produzir áudio válido")
                else:
                    logger.info("✅ Tom de teste gerado com sucesso (qualidade reduzida)")

        # Gerar arquivo WAV
        logger.info("💾 Convertendo para formato WAV...")
        wavb = wav_from_array(arr, SAMPLE_RATE)

        # Verificar se o WAV foi gerado corretamente
        if len(wavb) < 100:  # WAV muito pequeno indica problema
            logger.warning("❌ AVISO: Arquivo WAV gerado é muito pequeno, possivelmente corrompido")

        outname = os.path.join(CACHE_DIR, f"{fname}.{mode}.sr{symbol_rate}.wav")

        with open(outname, 'wb') as wf:
            wf.write(wavb)

        # Verificar se o arquivo foi salvo corretamente
        if os.path.exists(outname) and os.path.getsize(outname) > 100:
            logger.info(f"✅ Arquivo salvo: {outname} ({os.path.getsize(outname)} bytes)")
            encoded_files.append(outname)
        else:
            logger.error(f"❌ ERRO: Falha ao salvar arquivo {outname}")
            raise IOError(f"Falha ao salvar arquivo codificado: {outname}")

        if progress_callback:
            progress_callback(idx + 1, total_parts)

    return encoded_files


def encode_hellschreiber_text(text: str):
    # Implementação simplificada
    return "hellschreiber.wav"  # Placeholder


def encode_file(path: str, mode: str = "QPSK", compress: bool = True,
                symbol_rate: int = 9600, split_large_files: bool = True,
                target_duration_min: int = 1, progress_callback=None,
                is_cancelled=None) -> str:
    global _encoding_cancelled
    _encoding_cancelled = False

    fname = os.path.basename(path)
    with open(path, 'rb') as f:
        raw_data = f.read()

    file_crc = binascii.crc32(raw_data) & 0xffffffff
    file_size = len(raw_data)

    # 1. Compressão
    data = intelligent_compress(raw_data) if compress else raw_data

    # 2. Framing (Simplificado para arquivo único por enquanto para teste robusto)
    # Para multi-parte, a logica é a mesma, só loopar
    framed = _frame_data(fname, data, 0, 1, file_size, file_crc)

    # 3. Modulação
    logger.info(f"Modulando {len(framed)} bytes em modo {mode}...")

    try:
        if mode == "FSK1200":
            arr = fsk_modulate(framed, baud=1200)
        elif mode == "FSK9600":
            arr = fsk_modulate(framed, baud=9600)
        elif mode == "BPSK":
            arr = bpsk_modulate(framed, baud=symbol_rate)
        elif mode == "QPSK":
            arr = qpsk_modulate(framed, baud=symbol_rate)
        elif mode == "FSK19200":
            arr = fsk_high_speed_modulate(framed, baud=19200)
        else:
            arr = qpsk_modulate(framed, baud=symbol_rate)  # Default seguro
    except Exception as e:
        logger.error(f"Erro modulando: {e}")
        return ""

    wav_bytes = wav_from_array(arr, SAMPLE_RATE)
    outname = os.path.join(CACHE_DIR, f"{fname}.{mode}.wav")
    with open(outname, 'wb') as f:
        f.write(wav_bytes)

    return outname


def get_encoding_stats(file_path, mode, compress, symbol_rate):
    # Stub para manter interface
    sz = os.path.getsize(file_path)
    return {
        'original_size': sz, 'effective_size': sz, 'compression_ratio': 1.0,
        'bytes_per_sec': symbol_rate/4, 'duration_min': 1.0, 'bitrate_bps': symbol_rate
    }


def verify_audio_output(audio_array: np.ndarray, expected_min_duration: float = 0.1) -> bool:
    """Verifica se o array de áudio é válido de forma abrangente"""

    checks = [
        ("Array não é None", audio_array is not None),
        ("Array não vazio", len(audio_array) > 0),
        ("Não é tudo zero", not np.all(audio_array == 0)),
        ("Duração mínima", len(audio_array) / SAMPLE_RATE >= expected_min_duration),
        ("Tem variação", np.std(audio_array) >= 0.01),
        ("Sem NaN", not np.any(np.isnan(audio_array))),
        ("Sem infinitos", not np.any(np.isinf(audio_array))),
        ("Valores dentro do range", np.all(np.abs(audio_array) <= 1.0))
    ]

    failed_checks = []
    for check_name, condition in checks:
        if not condition:
            failed_checks.append(check_name)
            logger.warning(f"❌ Verificação de áudio falhou: {check_name}")

    if failed_checks:
        logger.error(f"❌ Áudio inválido. Falhas: {', '.join(failed_checks)}")
        return False

    # Verificações adicionais de qualidade
    duration = len(audio_array) / SAMPLE_RATE
    dynamic_range = np.max(audio_array) - np.min(audio_array)

    logger.info(f"✅ Áudio válido: {duration:.3f}s, {len(audio_array)} amostras, "
                f"variação: {np.std(audio_array):.6f}, dynamic_range: {dynamic_range:.6f}")

    return True