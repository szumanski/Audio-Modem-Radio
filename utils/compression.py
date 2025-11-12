# utils/compression.py - VERSÃO MELHORADA
# utils/compression.py - SISTEMA DE COMPRESSÃO INTELIGENTE
import zlib
import os
import lzma
import math
from io import BytesIO
from typing import Dict, Any
from config import CONFIG


class IntelligentCompressor:
    def __init__(self):
        self.compression_stats: Dict[str, Any] = {}
        self.enabled = CONFIG.get('compression.enabled', True)

    def analyze_data_pattern(self, data: bytes) -> Dict[str, Any]:
        """Analisa padrões nos dados para escolher melhor algoritmo"""
        if len(data) < 100:
            return {'recommended': 'none', 'ratio': 1.0}

        # Analisar entropia
        byte_freq = {}
        for byte in data:
            byte_freq[byte] = byte_freq.get(byte, 0) + 1

        entropy = 0
        total = len(data)
        for count in byte_freq.values():
            p = count / total
            entropy -= p * math.log2(p)

        # Detectar padrões
        repeated_patterns = self._detect_repeated_patterns(data)
        is_text = self._is_likely_text(data)

        if entropy < 2.0 or repeated_patterns:
            return {'recommended': 'lzma', 'ratio': 0.3, 'entropy': entropy}
        elif is_text:
            return {'recommended': 'zlib', 'ratio': 0.5, 'entropy': entropy}
        else:
            return {'recommended': 'delta+lzma', 'ratio': 0.4, 'entropy': entropy}

    def _detect_repeated_patterns(self, data: bytes, min_pattern=4, max_pattern=32) -> bool:
        """Detecta padrões repetidos nos dados"""
        if len(data) < min_pattern * 10:
            return False

        for pattern_len in range(min_pattern, min(max_pattern, len(data) // 10)):
            patterns = {}
            for i in range(0, len(data) - pattern_len, pattern_len):
                pattern = data[i:i + pattern_len]
                patterns[pattern] = patterns.get(pattern, 0) + 1

            if any(count > 3 for count in patterns.values()):
                return True
        return False

    def _is_likely_text(self, data: bytes) -> bool:
        """Verifica se os dados são provavelmente texto"""
        if len(data) == 0:
            return False

        text_chars = 0
        for byte in data[:1000]:  # Analisar apenas primeiros 1000 bytes
            if 32 <= byte <= 126 or byte in [9, 10, 13]:
                text_chars += 1

        return text_chars / min(1000, len(data)) > 0.8


def intelligent_compress(data: bytes, mode: str = "auto") -> bytes:
    """Compressão inteligente baseada na análise dos dados"""
    compressor = IntelligentCompressor()

    if not CONFIG.get('compression.enabled', True) or len(data) < 200:
        return b'RAW' + data

    # Análise automática se solicitado
    if mode == "auto":
        analysis = compressor.analyze_data_pattern(data)
        mode = analysis['recommended']

    try:
        if mode == "lzma" and CONFIG.get('compression.lzma_enabled', True):
            compressed = lzma.compress(data, preset=9)
            return b'LZMA' + compressed

        elif mode == "delta+lzma" and CONFIG.get('compression.delta_compression', True):
            delta_compressed = delta_compress(data)
            lzma_compressed = lzma.compress(delta_compressed, preset=9)
            return b'DLZM' + lzma_compressed

        else:  # Fallback para zlib
            zlib_compressed = zlib.compress(data, level=9)
            return b'ZLIB' + zlib_compressed

    except Exception as e:
        print(f"⚠️ Erro na compressão inteligente, usando fallback: {e}")
        return b'RAW' + data


def intelligent_decompress(compressed_data: bytes) -> bytes:
    """Descompressão inteligente com fallback"""
    try:
        if compressed_data.startswith(b'LZMA'):
            return lzma.decompress(compressed_data[4:])
        elif compressed_data.startswith(b'DLZM'):
            lzma_decompressed = lzma.decompress(compressed_data[4:])
            return delta_decompress(lzma_decompressed)
        elif compressed_data.startswith(b'ZLIB'):
            return zlib.decompress(compressed_data[4:])
        elif compressed_data.startswith(b'RAW'):
            return compressed_data[4:]
        else:
            # Tentar descompressão automática
            try:
                return zlib.decompress(compressed_data)
            except:
                return compressed_data
    except Exception as e:
        print(f"⚠️ Erro na descompressão inteligente: {e}")
        return compressed_data


# Manter funções existentes com melhorias
def super_compress_enhanced(data: bytes) -> bytes:
    """Versão melhorada da super compressão"""
    analysis = IntelligentCompressor().analyze_data_pattern(data)

    if analysis['recommended'] == 'lzma':
        return b'LZMA' + lzma.compress(data, preset=9)
    elif analysis['recommended'] == 'delta+lzma':
        delta_data = delta_compress(data)
        return b'DLZM' + lzma.compress(delta_data, preset=9)
    else:
        return b'ZLIB' + zlib.compress(data, level=9)

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import lzma
    LZMA_AVAILABLE = True
except ImportError:
    LZMA_AVAILABLE = False


def compress_data(data: bytes, level=9) -> bytes:
    """Compressão de dados genérica com zlib no nível máximo"""
    if len(data) < 100:  # Dados muito pequenos não compensam
        return data
    return zlib.compress(data, level)


def decompress_data(b: bytes) -> bytes:
    """Descompressão zlib com tratamento de erro"""
    try:
        return zlib.decompress(b)
    except zlib.error:
        # Se falhar, retorna os dados originais
        return b


def prepare_sstv_like(path: str, jpeg_quality=30, max_size=(400, 300)) -> bytes:
    """Prepara payload estilo SSTV com melhor qualidade"""
    if not PIL_AVAILABLE:
        with open(path, "rb") as f:
            return zlib.compress(f.read(), level=6)

    try:
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
        if os.path.splitext(path)[1].lower() not in image_extensions:
            with open(path, "rb") as f:
                return zlib.compress(f.read(), level=6)

        img = Image.open(path)

        if img.mode != 'RGB':
            img = img.convert('RGB')

        img.thumbnail(max_size, Image.Resampling.LANCZOS)

        buf = BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        jpeg_data = buf.getvalue()

        return zlib.compress(jpeg_data, level=6)

    except Exception as e:
        print(f"Erro no processamento de imagem, usando compressão padrão: {e}")
        with open(path, "rb") as f:
            return zlib.compress(f.read(), level=6)


# --- COMPRESSÃO AVANÇADA MELHORADA ---

def super_compress(data: bytes) -> bytes:
    """Compressão mais agressiva combinando métodos"""
    if len(data) < 500:  # Dados pequenos não compensam
        return b'RAW' + data

    if not LZMA_AVAILABLE:
        zlib_compressed = zlib.compress(data, level=9)
        return b'ZLIB' + zlib_compressed

    try:
        # Tentar zlib primeiro
        zlib_compressed = zlib.compress(data, level=9)

        # Tentar LZMA para dados maiores
        if len(data) > 1000:
            lzma_compressed = lzma.compress(data, preset=9)

            # Usar o melhor resultado
            if len(lzma_compressed) < len(zlib_compressed) * 0.8:  # LZMA é significativamente melhor
                return b'LZMA' + lzma_compressed

        return b'ZLIB' + zlib_compressed

    except Exception as e:
        print(f"Erro na super compressão, usando dados brutos: {e}")
        return b'RAW' + data


def super_decompress(b: bytes) -> bytes:
    """Descompressão para super_compress"""
    if b.startswith(b'LZMA'):
        if not LZMA_AVAILABLE:
            raise ImportError("LZMA não disponível")
        return lzma.decompress(b[4:])
    elif b.startswith(b'ZLIB'):
        return zlib.decompress(b[4:])
    elif b.startswith(b'RAW'):
        return b[4:]
    else:
        return decompress_data(b)


def delta_compress(data: bytes) -> bytes:
    """Compressão por diferença - ideal para dados sequenciais"""
    if len(data) <= 1:
        return data

    compressed = bytearray()
    last_byte = data[0]
    compressed.append(last_byte)

    for byte in data[1:]:
        delta = (byte - last_byte) & 0xFF
        compressed.append(delta)
        last_byte = byte

    return bytes(compressed)


def delta_decompress(compressed: bytes) -> bytes:
    """Descompressão por diferença"""
    if not compressed:
        return b''

    decompressed = bytearray()
    current_byte = compressed[0]
    decompressed.append(current_byte)

    for delta in compressed[1:]:
        current_byte = (current_byte + delta) & 0xFF
        decompressed.append(current_byte)

    return bytes(decompressed)


def adaptive_compress(data: bytes, mode: str) -> bytes:
    """Compressão adaptativa baseada no modo de transmissão"""
    if len(data) < 200:  # Dados muito pequenos
        return data

    # Modos de alta velocidade usam compressão mais agressiva
    if mode in ["8PSK", "FSK19200", "OFDM4", "OFDM8"]:
        return super_compress(data)
    else:
        return compress_data(data)