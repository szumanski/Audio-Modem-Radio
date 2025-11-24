# main.py - CLASSE PRINCIPAL COM INTERFACE MELHORADA
import os
import sys
import tempfile
import time
import threading
from datetime import datetime

import pygame
from PyQt5 import QtWidgets, QtCore, QtGui
import sounddevice as sd
import numpy as np
import soundfile as sf
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QFont, QPalette, QColor, QLinearGradient, QPainter, QPen
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QTabWidget, QGroupBox, QLabel,
                             QPushButton, QComboBox, QSpinBox, QCheckBox,
                             QProgressBar, QTextEdit, QListWidget, QListWidgetItem,
                             QFileDialog, QMessageBox, QSplitter, QFrame)

from encoder import encode_file, cancel_encoding, get_encoding_stats
from decoder import decode_wav_file, decode_from_buffer, get_assembly_status, get_reception_stats

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import pyqtgraph as pg

    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False


# ===============================================
# CONFIGURAÇÃO PARA EXECUTÁVEL
# ===============================================
def setup_executable_paths():
    if getattr(sys, 'frozen', False):
        # Se estiver rodando como .exe (PyInstaller)
        base_path = sys._MEIPASS
    else:
        # Se estiver rodando no PyCharm/Terminal
        base_path = os.path.dirname(os.path.abspath(__file__))

    sys.path.insert(0, base_path)

    # IMPORTANTE: Comente ou remova a linha abaixo.
    # O PyInstaller configura o Qt automaticamente. Forçar isso causa erro no .exe.
    # os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.join(base_path, 'PyQt5', 'Qt5', 'plugins')

    return base_path


BASE_PATH = setup_executable_paths()

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ===============================================
# CONSTANTES E CONFIGURAÇÕES DE UI
# ===============================================
MODES = ["FSK1200", "FSK9600", "BPSK", "QPSK", "SSTV", "8PSK", "FSK19200",
         "OFDM4", "OFDM8", "APSK16", "DSSS", "MSK", "HELLSCHREIBER"]

DIGITAL_MODES = [
    "FSK1200", "FSK9600", "BPSK", "QPSK", "8PSK", "FSK19200", "OFDM4", "OFDM8", "APSK16", "DSSS", "MSK",
    "FT8", "FT4", "JT65", "JT9", "MSK144", "WSPR", "JS8", "PSK31", "PSK63", "BPSK31", "RTTY", "FSK", "MFSK8", "MFSK16",
    "AFSK1200", "AFSK2400", "AX25", "PACTOR", "ARDOP", "VARA", "WINLINK", "DMR", "DSTAR", "NXDN", "P25", "YSF", "TETRA",
    "OLIVIA", "THOR", "MT63", "FSQ", "ALE", "CLOVER", "CHIRP", "COFDM", "LRPT", "DVB_S2", "LORA"
]

ANALOG_MODES = ["SSTV", "HELLSCHREIBER", "FELD_HELL", "SLOW_HELL"]

COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'dark': '#1A1A2E',
    'darker': '#16213E',
    'light': '#F8F9FA',
    'success': '#28A745',
    'warning': '#FFC107',
    'danger': '#DC3545',
    'info': '#17A2B8'
}


# ===============================================
# COMPONENTES DE UI PERSONALIZADOS
# ===============================================
class HeaderWidget(QWidget):
    def __init__(self, title="FileBeep Advanced v2", subtitle="Sistema Avançado de Transmissão de Dados por Áudio"):
        super().__init__()
        self.title = title
        self.subtitle = subtitle
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 10, 0, 10)

        title_label = QLabel(self.title)
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet(f"color: {COLORS['primary']}; padding: 5px;")

        subtitle_label = QLabel(self.subtitle)
        subtitle_label.setAlignment(QtCore.Qt.AlignCenter)
        subtitle_font = QFont()
        subtitle_font.setPointSize(10)
        subtitle_label.setFont(subtitle_font)
        subtitle_label.setStyleSheet(f"color: {COLORS['light']}; padding: 2px;")

        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)

        self.setLayout(layout)
        self.setStyleSheet(
            f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {COLORS['dark']}, stop:1 {COLORS['darker']}); border-radius: 5px;")


class ModeDiagramWidget(QWidget):
    def __init__(self, mode_name="QPSK"):
        super().__init__()
        self.mode_name = mode_name
        self.setMinimumHeight(120)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()
        height = self.height()

        # Fundo
        painter.fillRect(0, 0, width, height, QColor(COLORS['darker']))

        # Desenho baseado no modo
        if "FSK" in self.mode_name:
            self.draw_fsk_diagram(painter, width, height)
        elif "PSK" in self.mode_name:
            self.draw_psk_diagram(painter, width, height)
        elif "QPSK" in self.mode_name:
            self.draw_qpsk_diagram(painter, width, height)
        elif "OFDM" in self.mode_name:
            self.draw_ofdm_diagram(painter, width, height)
        else:
            self.draw_generic_diagram(painter, width, height)

        # Nome do modo
        painter.setPen(QColor(COLORS['light']))
        painter.drawText(10, height - 10, self.mode_name)

    def draw_fsk_diagram(self, painter, width, height):
        center_y = height // 2
        painter.setPen(QPen(QColor(COLORS['accent']), 2))

        for i in range(0, width, 10):
            freq = 1 if (i // 20) % 2 == 0 else 2
            y_offset = 20 * (freq - 1)
            painter.drawLine(i, center_y - y_offset, i + 10, center_y - y_offset)

    def draw_psk_diagram(self, painter, width, height):
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) // 3

        painter.setPen(QPen(QColor(COLORS['primary']), 2))
        painter.drawEllipse(center_x - radius, center_y - radius, radius * 2, radius * 2)

        # Pontos da constelação
        painter.setBrush(QColor(COLORS['accent']))
        for angle in [0, 90, 180, 270]:
            rad = angle * 3.14159 / 180
            x = center_x + radius * np.cos(rad)
            y = center_y + radius * np.sin(rad)
            painter.drawEllipse(int(x - 3), int(y - 3), 6, 6)

    def draw_qpsk_diagram(self, painter, width, height):
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) // 3

        painter.setPen(QPen(QColor(COLORS['secondary']), 2))
        painter.drawEllipse(center_x - radius, center_y - radius, radius * 2, radius * 2)

        # Pontos QPSK
        painter.setBrush(QColor(COLORS['success']))
        points = [
            (center_x + radius * 0.7, center_y + radius * 0.7),
            (center_x - radius * 0.7, center_y + radius * 0.7),
            (center_x + radius * 0.7, center_y - radius * 0.7),
            (center_x - radius * 0.7, center_y - radius * 0.7)
        ]
        for x, y in points:
            painter.drawEllipse(int(x - 3), int(y - 3), 6, 6)

    def draw_ofdm_diagram(self, painter, width, height):
        center_y = height // 2
        subcarriers = 8

        painter.setPen(QPen(QColor(COLORS['info']), 1))
        for i in range(subcarriers):
            freq = i + 1
            y = center_y - (freq - subcarriers / 2) * 8

            # Onda senoidal para cada subportadora
            for x in range(0, width, 2):
                y_pos = y + 10 * np.sin(2 * np.pi * freq * x / width)
                painter.drawPoint(x, int(y_pos))

    def draw_generic_diagram(self, painter, width, height):
        center_y = height // 2

        painter.setPen(QPen(QColor(COLORS['warning']), 2))
        for x in range(0, width, 3):
            y = center_y + 20 * np.sin(2 * np.pi * x / width)
            painter.drawPoint(x, int(y))


class StatusWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 5, 10, 5)

        # Ícones de status
        self.cpu_label = QLabel("CPU: --%")
        self.memory_label = QLabel("RAM: --%")
        self.disk_label = QLabel("DISK: --MB")

        for label in [self.cpu_label, self.memory_label, self.disk_label]:
            label.setStyleSheet(
                f"background: {COLORS['dark']}; color: {COLORS['light']}; padding: 3px 8px; border-radius: 3px;")
            layout.addWidget(label)

        layout.addStretch()

        self.status_label = QLabel("Sistema Pronto")
        self.status_label.setStyleSheet(f"color: {COLORS['success']}; font-weight: bold;")
        layout.addWidget(self.status_label)

        self.setLayout(layout)
        self.setStyleSheet(f"background: {COLORS['darker']}; border-top: 1px solid {COLORS['primary']};")

    def update_metrics(self, metrics):
        self.cpu_label.setText(f"CPU: {metrics.get('cpu', 0):.1f}%")
        self.memory_label.setText(f"RAM: {metrics.get('memory', 0):.1f}%")
        self.disk_label.setText(f"DISK: {metrics.get('disk_io', 0):.1f}MB")


# ===============================================
# WORKERS E THREADS (mantidos da versão anterior)
# ===============================================
class WorkerRecord(QtCore.QThread):
    finished = QtCore.pyqtSignal(list)
    volume = QtCore.pyqtSignal(float)

    def __init__(self, duration=30, mode="FSK9600", sr=9600):
        super().__init__()
        self._running = True
        self.duration = duration
        self.mode = mode
        self.sr = sr
        self.fs = 48000

    def run(self):
        # Implementação mantida da versão anterior
        fs = self.fs
        buf = []

        def callback(indata, frames, time, status):
            if status:
                print(f"Audio input error: {status}")

            if not self._running:
                raise sd.CallbackStop()

            data = indata[:, 0].copy().astype(np.float32)
            buf.append(data)

            rms = np.sqrt(np.mean(data ** 2))
            self.volume.emit(min(1.0, rms * 15))

        try:
            print("Starting recording thread...")
            with sd.InputStream(samplerate=fs, channels=1, callback=callback,
                                blocksize=2048, dtype=np.float32):
                start_time = time.time()
                while self._running and (time.time() - start_time) < self.duration:
                    time.sleep(0.1)

            if buf and self._running:
                audio_data = np.concatenate(buf)
                print(f"Processando {len(audio_data)} amostras...")

                saved = decode_from_buffer(audio_data, self.mode, self.sr)
                self.finished.emit(saved)
            else:
                self.finished.emit([])

        except Exception as e:
            print(f"Recording error: {e}")
            self.finished.emit([])


class EncodeWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int, int)
    cancelled = QtCore.pyqtSignal()

    def __init__(self, file_path, mode, compress, symbol_rate, target_duration_min):
        super().__init__()
        self.file_path = file_path
        self.mode = mode
        self.compress = compress
        self.symbol_rate = symbol_rate
        self.target_duration_min = target_duration_min
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True
        cancel_encoding()

    def run(self):
        try:
            result = encode_file(
                self.file_path,
                mode=self.mode,
                compress=self.compress,
                symbol_rate=self.symbol_rate,
                split_large_files=True,
                target_duration_min=self.target_duration_min,
                progress_callback=self.progress.emit,
                is_cancelled=lambda: self._is_cancelled
            )

            if self._is_cancelled:
                self.cancelled.emit()
            else:
                self.finished.emit(result)

        except Exception as e:
            if not self._is_cancelled:
                self.error.emit(str(e))
            else:
                self.cancelled.emit()


class PerformanceMonitor(QtCore.QThread):
    update_metrics = QtCore.pyqtSignal(dict)

    def run(self):
        while True:
            metrics = self.get_system_metrics()
            self.update_metrics.emit(metrics)
            time.sleep(2)

    def get_system_metrics(self):
        if not PSUTIL_AVAILABLE:
            return {'cpu': 0, 'memory': 0, 'disk_io': 0}

        try:
            return {
                'cpu': psutil.cpu_percent(),
                'memory': psutil.virtual_memory().percent,
                'disk_io': psutil.disk_io_counters().write_bytes / 1024 / 1024 if psutil.disk_io_counters() else 0
            }
        except Exception:
            return {'cpu': 0, 'memory': 0, 'disk_io': 0}


# ===============================================
# JANELA PRINCIPAL MELHORADA
# ===============================================
class ModernMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FileBeep Advanced v2 - Sistema de Transmissão por Áudio")
        self.resize(1400, 900)
        self.setWindowIcon(self.create_icon())

        # Configurar estilo da aplicação
        self.setup_stylesheet()

        # Inicializar componentes
        self.audio_player = AudioPlayer()
        self.played_files = set()
        self.playing_files = set()

        self.create_ui()
        self.setup_connections()

        # Inicializar threads e timers
        self.record_thread = None
        self.encode_thread = None
        self.encode_worker = None

        self.setup_timers()

        self.log_manager = LogManager()
        self.log_message("🚀 Sistema FileBeep Advanced inicializado com sucesso!")

    def setup_stylesheet(self):
        """Configura o estilo visual da aplicação"""
        self.setStyleSheet(f"""
            QMainWindow {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {COLORS['dark']}, stop:1 {COLORS['darker']});
                color: {COLORS['light']};
            }}

            QTabWidget::pane {{
                border: 1px solid {COLORS['primary']};
                background: {COLORS['darker']};
            }}

            QTabBar::tab {{
                background: {COLORS['dark']};
                color: {COLORS['light']};
                padding: 8px 16px;
                margin-right: 2px;
                border-radius: 4px;
            }}

            QTabBar::tab:selected {{
                background: {COLORS['primary']};
                color: white;
            }}

            QGroupBox {{
                font-weight: bold;
                color: {COLORS['accent']};
                border: 1px solid {COLORS['primary']};
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }}

            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }}

            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {COLORS['primary']}, stop:1 {COLORS['secondary']});
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }}

            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {COLORS['secondary']}, stop:1 {COLORS['primary']});
            }}

            QPushButton:pressed {{
                background: {COLORS['accent']};
            }}

            QComboBox, QSpinBox, QCheckBox {{
                background: {COLORS['dark']};
                color: {COLORS['light']};
                border: 1px solid {COLORS['primary']};
                border-radius: 3px;
                padding: 5px;
            }}

            QProgressBar {{
                border: 1px solid {COLORS['primary']};
                border-radius: 3px;
                text-align: center;
                color: {COLORS['light']};
            }}

            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {COLORS['success']}, stop:1 {COLORS['accent']});
            }}

            QTextEdit, QListWidget {{
                background: {COLORS['dark']};
                color: {COLORS['light']};
                border: 1px solid {COLORS['primary']};
                border-radius: 3px;
            }}
        """)

    def create_icon(self):
        """Cria um ícone simples para a aplicação"""
        pixmap = QtGui.QPixmap(32, 32)
        pixmap.fill(QColor(COLORS['primary']))
        painter = QPainter(pixmap)
        painter.setPen(QColor(COLORS['light']))
        painter.drawText(8, 20, "FB")
        painter.end()
        return QtGui.QIcon(pixmap)

    def create_ui(self):
        """Cria a interface do usuário"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Cabeçalho
        header = HeaderWidget()
        main_layout.addWidget(header)

        # Área principal com abas
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Criar abas
        self.create_encode_tab()
        self.create_decode_tab()
        self.create_player_tab()
        self.create_analysis_tab()

        # Log
        self.create_log_area(main_layout)

        # Status bar
        self.status_widget = StatusWidget()
        main_layout.addWidget(self.status_widget)

    def create_encode_tab(self):
        """Cria a aba de codificação"""
        encode_tab = QWidget()
        layout = QHBoxLayout(encode_tab)

        # Painel esquerdo - Configurações
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Grupo de seleção de arquivo
        file_group = QGroupBox("📁 Seleção de Arquivo")
        file_layout = QVBoxLayout(file_group)

        self.file_path_label = QLabel("Nenhum arquivo selecionado")
        self.file_path_label.setStyleSheet(f"color: {COLORS['warning']}; font-style: italic;")
        file_layout.addWidget(self.file_path_label)

        file_buttons_layout = QHBoxLayout()
        self.select_file_btn = QPushButton("📂 Selecionar Arquivo")
        self.select_file_btn.clicked.connect(self.select_file)
        file_buttons_layout.addWidget(self.select_file_btn)

        file_layout.addLayout(file_buttons_layout)
        left_layout.addWidget(file_group)

        # Grupo de configurações
        config_group = QGroupBox("⚙️ Configurações de Transmissão")
        config_layout = QVBoxLayout(config_group)

        # Modo de modulação
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Modo:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(MODES)
        self.mode_combo.setCurrentText("QPSK")
        self.mode_combo.currentTextChanged.connect(self.update_mode_diagram)
        mode_layout.addWidget(self.mode_combo)
        config_layout.addLayout(mode_layout)

        # Taxa de símbolo
        rate_layout = QHBoxLayout()
        rate_layout.addWidget(QLabel("Taxa de Símbolo:"))
        self.symbol_rate_spin = QSpinBox()
        self.symbol_rate_spin.setRange(100, 19200)
        self.symbol_rate_spin.setValue(9600)
        rate_layout.addWidget(self.symbol_rate_spin)
        config_layout.addLayout(rate_layout)

        # Opções
        self.compress_check = QCheckBox("📦 Usar Compressão")
        self.compress_check.setChecked(True)
        config_layout.addWidget(self.compress_check)

        duration_layout = QHBoxLayout()
        duration_layout.addWidget(QLabel("Duração por Parte (min):"))
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(1, 60)
        self.duration_spin.setValue(1)
        duration_layout.addWidget(self.duration_spin)
        config_layout.addLayout(duration_layout)

        left_layout.addWidget(config_group)

        # Botões de ação
        action_layout = QVBoxLayout()
        self.encode_button = QPushButton("🚀 Iniciar Codificação")
        self.encode_button.clicked.connect(self.encode_file_button_clicked)
        action_layout.addWidget(self.encode_button)

        self.encode_progress = QProgressBar()
        action_layout.addWidget(self.encode_progress)

        self.cancel_encode_button = QPushButton("❌ Cancelar Codificação")
        self.cancel_encode_button.clicked.connect(self.cancel_encoding)
        self.cancel_encode_button.setEnabled(False)
        action_layout.addWidget(self.cancel_encode_button)

        left_layout.addLayout(action_layout)
        left_layout.addStretch()

        # Painel direito - Visualização
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Diagrama do modo
        diagram_group = QGroupBox("📊 Diagrama do Modo")
        diagram_layout = QVBoxLayout(diagram_group)
        self.mode_diagram = ModeDiagramWidget("QPSK")
        diagram_layout.addWidget(self.mode_diagram)
        right_layout.addWidget(diagram_group)

        # Estatísticas
        stats_group = QGroupBox("📈 Estatísticas")
        stats_layout = QVBoxLayout(stats_group)
        self.stats_text = QTextEdit()
        self.stats_text.setMaximumHeight(150)
        self.stats_text.setPlainText("Selecione um arquivo para ver as estatísticas...")
        stats_layout.addWidget(self.stats_text)
        right_layout.addWidget(stats_group)

        right_layout.addStretch()

        # Adicionar painéis ao layout principal
        layout.addWidget(left_panel, 1)
        layout.addWidget(right_panel, 1)

        self.tabs.addTab(encode_tab, "🔧 Codificação")

    def create_decode_tab(self):
        """Cria a aba de decodificação"""
        decode_tab = QWidget()
        layout = QVBoxLayout(decode_tab)

        # Configurações de decodificação
        config_group = QGroupBox("🎛️ Configurações de Recepção")
        config_layout = QHBoxLayout(config_group)

        config_layout.addWidget(QLabel("Modo:"))
        self.decode_mode_combo = QComboBox()
        self.decode_mode_combo.addItems(MODES)
        self.decode_mode_combo.setCurrentText("QPSK")
        config_layout.addWidget(self.decode_mode_combo)

        config_layout.addWidget(QLabel("Taxa de Símbolo:"))
        self.decode_sr_spin = QSpinBox()
        self.decode_sr_spin.setRange(100, 19200)
        self.decode_sr_spin.setValue(9600)
        config_layout.addWidget(self.decode_sr_spin)

        config_layout.addStretch()
        layout.addWidget(config_group)

        # Controles de gravação
        record_group = QGroupBox("🎤 Captura de Áudio")
        record_layout = QVBoxLayout(record_group)

        # Medidor de volume
        volume_layout = QHBoxLayout()
        volume_layout.addWidget(QLabel("Nível de Áudio:"))
        self.volume_meter = QProgressBar()
        self.volume_meter.setRange(0, 100)
        volume_layout.addWidget(self.volume_meter)
        record_layout.addLayout(volume_layout)

        # Botões de gravação
        record_buttons_layout = QHBoxLayout()
        self.record_button = QPushButton("🔴 Iniciar Gravação (30s)")
        self.record_button.clicked.connect(self.start_record)
        record_buttons_layout.addWidget(self.record_button)

        self.decode_file_btn = QPushButton("📁 Decodificar Arquivo WAV")
        self.decode_file_btn.clicked.connect(self.decode_wav_file)
        record_buttons_layout.addWidget(self.decode_file_btn)

        record_layout.addLayout(record_buttons_layout)
        layout.addWidget(record_group)

        # Status de montagem
        splitter = QSplitter(QtCore.Qt.Vertical)

        # Status de montagem
        assembly_group = QGroupBox("📦 Status de Montagem de Arquivos")
        assembly_layout = QVBoxLayout(assembly_group)
        self.assembly_status = QTextEdit()
        self.assembly_status.setMaximumHeight(150)
        assembly_layout.addWidget(self.assembly_status)
        splitter.addWidget(assembly_group)

        # Estatísticas de recepção
        stats_group = QGroupBox("📊 Estatísticas de Recepção")
        stats_layout = QVBoxLayout(stats_group)
        self.reception_stats = QTextEdit()
        self.reception_stats.setMaximumHeight(150)
        stats_layout.addWidget(self.reception_stats)
        splitter.addWidget(stats_group)

        layout.addWidget(splitter)

        self.tabs.addTab(decode_tab, "🔍 Decodificação")

    def create_player_tab(self):
        """Cria a aba do player"""
        player_tab = QWidget()
        layout = QVBoxLayout(player_tab)

        # Playlist
        playlist_group = QGroupBox("🎵 Playlist de Transmissão")
        playlist_layout = QVBoxLayout(playlist_group)
        self.playlist_widget = QListWidget()
        self.playlist_widget.itemDoubleClicked.connect(self.on_playlist_item_double_click)
        playlist_layout.addWidget(self.playlist_widget)
        layout.addWidget(playlist_group)

        # Controles do player
        controls_group = QGroupBox("🎛️ Controles de Reprodução")
        controls_layout = QVBoxLayout(controls_group)

        # Informações do arquivo atual
        self.now_playing_label = QLabel("Nenhum arquivo em reprodução")
        self.now_playing_label.setAlignment(QtCore.Qt.AlignCenter)
        self.now_playing_label.setStyleSheet(f"color: {COLORS['accent']}; font-weight: bold; padding: 5px;")
        controls_layout.addWidget(self.now_playing_label)

        # Barra de progresso
        self.playback_progress = QProgressBar()
        controls_layout.addWidget(self.playback_progress)

        # Botões de controle
        player_buttons_layout = QHBoxLayout()

        self.play_selected_btn = QPushButton("▶️ Reproduzir")
        self.play_selected_btn.clicked.connect(self.on_play_selected)
        player_buttons_layout.addWidget(self.play_selected_btn)

        self.pause_button = QPushButton("⏸️ Pausar")
        self.pause_button.clicked.connect(self.on_pause)
        player_buttons_layout.addWidget(self.pause_button)

        self.stop_button = QPushButton("⏹️ Parar")
        self.stop_button.clicked.connect(self.on_stop)
        player_buttons_layout.addWidget(self.stop_button)

        self.clear_playlist_btn = QPushButton("🗑️ Limpar Playlist")
        self.clear_playlist_btn.clicked.connect(self.on_clear_playlist)
        player_buttons_layout.addWidget(self.clear_playlist_btn)

        controls_layout.addLayout(player_buttons_layout)
        layout.addWidget(controls_group)

        self.tabs.addTab(player_tab, "🎧 Player")

    def create_analysis_tab(self):
        """Cria a aba de análise"""
        analysis_tab = QWidget()
        layout = QVBoxLayout(analysis_tab)

        # Adicione aqui componentes de análise visual
        analysis_group = QGroupBox("📊 Análise de Sinais")
        analysis_layout = QVBoxLayout(analysis_group)

        analysis_label = QLabel("Recursos de análise visual em desenvolvimento...")
        analysis_label.setAlignment(QtCore.Qt.AlignCenter)
        analysis_layout.addWidget(analysis_label)

        layout.addWidget(analysis_group)
        layout.addStretch()

        self.tabs.addTab(analysis_tab, "📈 Análise")

    def create_log_area(self, main_layout):
        """Cria a área de log"""
        log_group = QGroupBox("📋 Log do Sistema")
        log_layout = QVBoxLayout(log_group)

        log_buttons_layout = QHBoxLayout()
        self.clear_log_btn = QPushButton("🗑️ Limpar Log")
        self.clear_log_btn.clicked.connect(self.clear_log)
        log_buttons_layout.addWidget(self.clear_log_btn)

        self.save_log_btn = QPushButton("💾 Salvar Log")
        self.save_log_btn.clicked.connect(self.save_log)
        log_buttons_layout.addWidget(self.save_log_btn)

        log_buttons_layout.addStretch()
        log_layout.addLayout(log_buttons_layout)

        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)

        main_layout.addWidget(log_group)

    def setup_connections(self):
        """Configura as conexões de sinais"""
        self.player_update_timer = QTimer()
        self.player_update_timer.timeout.connect(self.update_player_ui)
        self.player_update_timer.start(500)

    def setup_timers(self):
        """Configura os timers da aplicação"""
        self.assembly_timer = QTimer()
        self.assembly_timer.timeout.connect(self.update_assembly_status)
        self.assembly_timer.start(5000)

        self.metrics_timer = QTimer()
        self.metrics_timer.timeout.connect(self.update_performance_metrics)
        self.metrics_timer.start(2000)

    # ===============================================
    # MÉTODOS DE UI E INTERAÇÃO
    # ===============================================
    def update_mode_diagram(self, mode):
        """Atualiza o diagrama do modo selecionado"""
        self.mode_diagram.mode_name = mode
        self.mode_diagram.update()

    def select_file(self):
        """Seleciona arquivo para codificação"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Selecionar Arquivo")
        if file_path:
            self.file_path_label.setText(file_path)
            self.update_file_stats(file_path)

    def update_file_stats(self, file_path):
        """Atualiza estatísticas do arquivo selecionado"""
        try:
            file_size = os.path.getsize(file_path)
            mode = self.mode_combo.currentText()
            symbol_rate = self.symbol_rate_spin.value()
            compress = self.compress_check.isChecked()

            stats = get_encoding_stats(file_path, mode, compress, symbol_rate)

            stats_text = f"""
📊 ESTATÍSTICAS DE TRANSMISSÃO:

📁 Arquivo: {os.path.basename(file_path)}
📏 Tamanho Original: {file_size / 1024:.1f} KB
🎯 Modo: {mode}
⚡ Taxa de Símbolo: {symbol_rate} baud
📦 Compressão: {'Ativada' if compress else 'Desativada'}

⏱️ Duração Estimada: {stats['duration_min']:.1f} minutos
📈 Taxa de Dados: {stats['bitrate_bps']:.0f} bps
📉 Tamanho Efetivo: {stats['effective_size'] / 1024:.1f} KB
🔍 Razão de Compressão: {stats['compression_ratio']:.2f}
"""
            self.stats_text.setPlainText(stats_text)

        except Exception as e:
            self.log_message(f"❌ Erro ao calcular estatísticas: {e}")

    def encode_file_button_clicked(self):
        """Inicia a codificação do arquivo"""
        file_path = self.file_path_label.text()
        if not file_path or file_path == "Nenhum arquivo selecionado":
            QMessageBox.warning(self, "Atenção", "Por favor, selecione um arquivo primeiro.")
            return

        mode = self.mode_combo.currentText()
        compress = self.compress_check.isChecked()
        symbol_rate = self.symbol_rate_spin.value()
        target_duration_min = self.duration_spin.value()

        self.encode_worker = EncodeWorker(file_path, mode, compress, symbol_rate, target_duration_min)
        self.encode_worker.finished.connect(self.on_encode_finished)
        self.encode_worker.error.connect(self.on_encode_error)
        self.encode_worker.progress.connect(self.on_encode_progress)
        self.encode_worker.cancelled.connect(self.on_encode_cancelled)

        self.encode_thread = QtCore.QThread()
        self.encode_worker.moveToThread(self.encode_thread)
        self.encode_thread.started.connect(self.encode_worker.run)
        self.encode_thread.start()

        self.encode_button.setEnabled(False)
        self.cancel_encode_button.setEnabled(True)
        self.log_message(f"🚀 Iniciando codificação de {os.path.basename(file_path)} no modo {mode}")

    def on_encode_progress(self, current, total):
        """Atualiza o progresso da codificação"""
        progress = int((current / total) * 100) if total > 0 else 0
        self.encode_progress.setValue(progress)

    def on_encode_finished(self, result):
        """Finalização da codificação"""
        self.encode_thread.quit()
        self.encode_thread.wait()

        self.encode_button.setEnabled(True)
        self.cancel_encode_button.setEnabled(False)
        self.encode_progress.setValue(0)

        self.log_message(f"✅ Codificação concluída: {result}")
        self.add_file_to_playlist(result)

        QMessageBox.information(self, "Sucesso", f"Arquivo codificado com sucesso!\n{result}")

    def on_encode_error(self, error):
        """Erro na codificação"""
        self.encode_thread.quit()
        self.encode_thread.wait()

        self.encode_button.setEnabled(True)
        self.cancel_encode_button.setEnabled(False)
        self.encode_progress.setValue(0)

        self.log_message(f"❌ Erro na codificação: {error}")
        QMessageBox.critical(self, "Erro", f"Falha na codificação:\n{error}")

    def on_encode_cancelled(self):
        """Cancelamento da codificação"""
        self.encode_thread.quit()
        self.encode_thread.wait()

        self.encode_button.setEnabled(True)
        self.cancel_encode_button.setEnabled(False)
        self.encode_progress.setValue(0)

        self.log_message("⏹️ Codificação cancelada pelo usuário")

    def start_record(self):
        """Inicia a gravação para decodificação"""
        mode = self.decode_mode_combo.currentText()
        sr = self.decode_sr_spin.value()

        self.record_thread = WorkerRecord(duration=30, mode=mode, sr=sr)
        self.record_thread.finished.connect(self.on_record_finished)
        self.record_thread.volume.connect(self.volume_meter.setValue)
        self.record_thread.start()

        self.record_button.setEnabled(False)
        self.log_message("🎤 Iniciando gravação de 30 segundos...")

    def on_record_finished(self, saved):
        """Finalização da gravação"""
        self.record_button.setEnabled(True)
        self.volume_meter.setValue(0)

        if saved:
            self.log_message(f"✅ {len(saved)} arquivo(s) decodificado(s) com sucesso!")
            for file in saved:
                self.log_message(f"📁 Salvo: {file}")
        else:
            self.log_message("❌ Nenhum arquivo foi decodificado da gravação")

    def decode_wav_file(self):
        """Decodifica um arquivo WAV"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Selecionar Arquivo WAV", "", "Arquivos WAV (*.wav)")
        if file_path:
            mode = self.decode_mode_combo.currentText()
            sr = self.decode_sr_spin.value()

            try:
                saved = decode_wav_file(file_path, mode, sr)
                if saved:
                    self.log_message(f"✅ {len(saved)} arquivo(s) decodificado(s) de {os.path.basename(file_path)}")
                else:
                    self.log_message(f"❌ Não foi possível decodificar {os.path.basename(file_path)}")
            except Exception as e:
                self.log_message(f"❌ Erro na decodificação: {e}")

    def update_assembly_status(self):
        """Atualiza o status de montagem de arquivos"""
        status = get_assembly_status()
        if status:
            status_text = "📦 ARQUIVOS SENDO MONTADOS:\n\n"
            for item in status:
                status_text += f"📁 {item['filename']}\n"
                status_text += f"   📊 Progresso: {item['received']}/{item['total']} partes ({item['progress']:.1f}%)\n"
                if item['missing_parts']:
                    status_text += f"   ⚠️  Partes faltando: {item['missing_parts'][:5]}{'...' if len(item['missing_parts']) > 5 else ''}\n"
                status_text += "\n"
            self.assembly_status.setPlainText(status_text)
        else:
            self.assembly_status.setPlainText("📭 Nenhum arquivo em montagem no momento")

    def update_performance_metrics(self):
        """Atualiza as métricas de desempenho"""
        metrics = PerformanceMonitor().get_system_metrics()
        self.status_widget.update_metrics(metrics)

        # Atualizar estatísticas de recepção
        stats = get_reception_stats()
        stats_text = f"""
📊 ESTATÍSTICAS DE RECEPÇÃO:

📁 Arquivos Recebidos: {stats['total_files']}
📏 Bytes Recebidos: {stats['total_bytes'] / 1024:.1f} KB
🎯 Taxa de Sucesso: {stats['success_rate']:.1f}%
⭐ Qualidade Média: {stats['average_quality']:.3f}
🕒 Última Recepção: {time.ctime(stats['last_reception']) if stats['last_reception'] else 'Nunca'}

📈 Duplicatas Rejeitadas: {stats['duplicates_rejected']}
🔀 Partes Reordenadas: {stats['parts_reordered']}
"""
        self.reception_stats.setPlainText(stats_text)

    # ===============================================
    # MÉTODOS DO PLAYER
    # ===============================================
    def update_player_ui(self):
        """Atualiza a interface do player"""
        progress = self.audio_player.get_progress()
        self.playback_progress.setValue(int(progress))

        # Atualizar cores dos itens na playlist
        for i in range(self.playlist_widget.count()):
            item = self.playlist_widget.item(i)
            file_path = item.data(QtCore.Qt.UserRole)

            if file_path in self.played_files:
                item.setBackground(QColor(COLORS['success']))
                item.setForeground(QColor(COLORS['light']))
            elif file_path in self.playing_files:
                item.setBackground(QColor(COLORS['warning']))
                item.setForeground(QColor(COLORS['dark']))
            else:
                item.setBackground(QColor(COLORS['dark']))
                item.setForeground(QColor(COLORS['light']))

        # Atualizar informação do arquivo atual
        if self.audio_player.is_playing and self.audio_player.current_file:
            current_file = os.path.basename(self.audio_player.current_file)
            progress_text = f"▶️ Reproduzindo: {current_file} ({progress:.1f}%)"
            self.now_playing_label.setText(progress_text)
            self.now_playing_label.setStyleSheet(
                f"color: {COLORS['success']}; font-weight: bold; background: {COLORS['dark']}; padding: 5px; border-radius: 3px;")
        elif self.audio_player.current_file and not self.audio_player.is_playing:
            current_file = os.path.basename(self.audio_player.current_file)
            self.now_playing_label.setText(f"⏸️ Pausado: {current_file}")
            self.now_playing_label.setStyleSheet(
                f"color: {COLORS['warning']}; font-style: italic; background: {COLORS['dark']}; padding: 5px; border-radius: 3px;")
        else:
            self.now_playing_label.setText("⏹️ Nenhum arquivo em reprodução")
            self.now_playing_label.setStyleSheet(
                f"color: {COLORS['light']}; font-style: italic; background: {COLORS['dark']}; padding: 5px; border-radius: 3px;")

        # Verificar se a reprodução terminou
        if self.audio_player.update_playback():
            if self.audio_player.current_file:
                self.played_files.add(self.audio_player.current_file)
                self.playing_files.discard(self.audio_player.current_file)

    def add_file_to_playlist(self, file_path):
        """Adiciona um arquivo à playlist"""
        if not os.path.exists(file_path):
            return

        # Verificar se já existe na playlist
        for i in range(self.playlist_widget.count()):
            item = self.playlist_widget.item(i)
            if item.data(QtCore.Qt.UserRole) == file_path:
                return

        # Adicionar novo item
        item = QListWidgetItem(os.path.basename(file_path))
        item.setData(QtCore.Qt.UserRole, file_path)
        self.playlist_widget.addItem(item)

        # Marcar como não reproduzido
        self.played_files.discard(file_path)
        self.playing_files.discard(file_path)

        self.log_message(f"📋 Arquivo adicionado à playlist: {os.path.basename(file_path)}")

    def on_playlist_item_double_click(self, item):
        """Quando o usuário clica duas vezes em um item da playlist"""
        file_path = item.data(QtCore.Qt.UserRole)
        self.play_audio_file(file_path)

    def on_play_selected(self):
        """Reproduz o arquivo selecionado na playlist"""
        current_item = self.playlist_widget.currentItem()
        if current_item:
            file_path = current_item.data(QtCore.Qt.UserRole)
            self.play_audio_file(file_path)
        else:
            QMessageBox.warning(self, "Atenção", "Por favor, selecione um arquivo da playlist primeiro.")

    def play_audio_file(self, file_path):
        """Reproduz um arquivo de áudio"""
        if self.audio_player.load_file(file_path):
            self.audio_player.play()
            self.playing_files.add(file_path)
            self.played_files.discard(file_path)
            self.log_message(f"🎵 Reproduzindo: {os.path.basename(file_path)}")
        else:
            QMessageBox.warning(self, "Erro", f"Não foi possível reproduzir o arquivo: {os.path.basename(file_path)}")

    def on_pause(self):
        """Pausa a reprodução"""
        self.audio_player.pause()
        self.log_message("⏸️ Reprodução pausada")

    def on_stop(self):
        """Para a reprodução"""
        self.audio_player.stop()
        if self.audio_player.current_file:
            self.playing_files.discard(self.audio_player.current_file)
        self.log_message("⏹️ Reprodução parada")

    def on_clear_playlist(self):
        """Limpa a playlist"""
        if self.playlist_widget.count() > 0:
            reply = QMessageBox.question(self, "Confirmar", "Tem certeza que deseja limpar toda a playlist?",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.audio_player.stop()
                self.playlist_widget.clear()
                self.played_files.clear()
                self.playing_files.clear()
                self.log_message("🗑️ Playlist limpa")

    # ===============================================
    # MÉTODOS DE LOG E UTILITÁRIOS
    # ===============================================
    def log_message(self, message):
        """Adiciona mensagem ao log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        self.log_manager.write_log("INFO", message)

    def clear_log(self):
        """Limpa o log"""
        self.log_text.clear()
        self.log_message("📋 Log limpo pelo usuário")

    def save_log(self):
        """Salva o log em arquivo"""
        filename, _ = QFileDialog.getSaveFileName(self, "Salvar Log", "filebeep_log.txt", "Arquivos de Texto (*.txt)")
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.toPlainText())
                self.log_message(f"💾 Log salvo em: {filename}")
                QMessageBox.information(self, "Log Salvo", f"Log salvo com sucesso em:\n{filename}")
            except Exception as e:
                self.log_message(f"❌ Erro ao salvar log: {e}")
                QMessageBox.critical(self, "Erro", f"Erro ao salvar log: {e}")

    def cancel_encoding(self):
        """Cancela a codificação em andamento"""
        if self.encode_worker:
            self.encode_worker.cancel()

    def closeEvent(self, event):
        """Evento de fechamento da aplicação"""
        # Parar reprodução de áudio
        self.audio_player.stop()
        pygame.mixer.quit()

        # Parar threads
        if self.record_thread and self.record_thread.isRunning():
            self.record_thread._running = False
            self.record_thread.wait(2000)

        if self.encode_worker:
            self.encode_worker.cancel()

        if self.encode_thread and self.encode_thread.isRunning():
            self.encode_thread.quit()
            self.encode_thread.wait(1000)

        # Parar timers
        self.assembly_timer.stop()
        self.metrics_timer.stop()
        self.player_update_timer.stop()

        self.log_message("👋 Aplicação encerrada")
        event.accept()


# ===============================================
# CLASSES AUXILIARES (mantidas da versão anterior)
# ===============================================
class AudioPlayer:
    def __init__(self):
        pygame.mixer.init()
        self.current_file = None
        self.is_playing = False
        self.playback_position = 0
        self.total_length = 0
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.update_playback)

    def load_file(self, file_path):
        try:
            if self.is_playing:
                self.stop()
            pygame.mixer.music.load(file_path)
            self.current_file = file_path
            self.is_playing = False
            self.playback_position = 0
            return True
        except Exception as e:
            print(f"Erro ao carregar áudio: {e}")
            return False

    def play(self):
        if self.current_file:
            pygame.mixer.music.play()
            self.is_playing = True
            self.playback_timer.start(100)

    def pause(self):
        if self.is_playing:
            pygame.mixer.music.pause()
            self.is_playing = False
            self.playback_timer.stop()

    def stop(self):
        pygame.mixer.music.stop()
        self.is_playing = False
        self.playback_timer.stop()
        self.playback_position = 0

    def update_playback(self):
        if pygame.mixer.music.get_busy():
            self.playback_position += 0.1
            return False
        else:
            self.is_playing = False
            self.playback_timer.stop()
            return True

    def get_progress(self):
        if self.total_length > 0:
            return (self.playback_position / self.total_length) * 100
        return 0


class LogManager:
    def __init__(self):
        self.log_file = f"filebeep_log_{int(time.time())}.txt"
        self.max_size = 10 * 1024 * 1024  # 10MB

    def write_log(self, level: str, message: str):
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] [{level}] {message}\n"

        # Rotação de logs
        if os.path.exists(self.log_file):
            if os.path.getsize(self.log_file) > self.max_size:
                self.rotate_log()

        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)

    def rotate_log(self):
        """Realiza rotação de arquivos de log"""
        try:
            if os.path.exists(self.log_file):
                backup_name = f"{self.log_file}.backup"
                if os.path.exists(backup_name):
                    os.remove(backup_name)
                os.rename(self.log_file, backup_name)
        except Exception as e:
            print(f"Erro na rotação de log: {e}")


# ===============================================
# INICIALIZAÇÃO DA APLICAÇÃO
# ===============================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("FileBeep Advanced")
    app.setApplicationVersion("2.0")

    # Configurar fonte padrão
    font = QFont()
    font.setFamily("Segoe UI")
    font.setPointSize(9)
    app.setFont(font)

    win = ModernMainWindow()
    win.show()

    sys.exit(app.exec_())