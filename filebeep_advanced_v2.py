# main.py - CLASSE PRINCIPAL COMPLETA E CORRIGIDA
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
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))

    sys.path.insert(0, base_path)
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.join(base_path, 'PyQt5', 'Qt5', 'plugins')
    return base_path


BASE_PATH = setup_executable_paths()

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

MODES = ["FSK1200", "FSK9600", "BPSK", "QPSK", "SSTV", "8PSK", "FSK19200",
         "OFDM4", "OFDM8", "APSK16", "DSSS", "MSK", "HELLSCHREIBER"]


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


# Adicione esta classe antes da classe ModernMainWindow
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
            # Obter duração do arquivo (aproximada)
            import wave
            with wave.open(file_path, 'rb') as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                self.total_length = frames / float(rate)
            return True
        except Exception as e:
            print(f"Erro ao carregar arquivo: {e}")
            return False

    def play(self):
        if self.current_file and not self.is_playing:
            pygame.mixer.music.play()
            self.is_playing = True
            self.playback_timer.start(100)  # Atualizar a cada 100ms

    def pause(self):
        if self.is_playing:
            pygame.mixer.music.pause()
            self.is_playing = False
            self.playback_timer.stop()

    def stop(self):
        pygame.mixer.music.stop()
        self.is_playing = False
        self.playback_position = 0
        self.playback_timer.stop()

    def update_playback(self):
        if self.is_playing:
            self.playback_position += 0.1
            if self.playback_position >= self.total_length:
                self.stop()
                return True  # Reprodução concluída
        return False

    def get_progress(self):
        if self.total_length > 0:
            return (self.playback_position / self.total_length) * 100
        return 0


class ModernMainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FileBeep Advanced v2 - Professional Data Transfer")
        self.resize(1000, 750)
        self.setup_ui()
        self.last_wav = None
        self.record_thread = None
        self.encode_worker = None
        self.encode_thread = None

        # Sistema de player de áudio
        pygame.mixer.init()
        self.audio_player = AudioPlayer()
        self.played_files = set()  # Arquivos que já foram reproduzidos completamente
        self.playing_files = set()  # Arquivos atualmente em reprodução

        self.assembly_timer = QtCore.QTimer()
        self.assembly_timer.timeout.connect(self.update_assembly_status)
        self.assembly_timer.start(2000)

        self.metrics_timer = QtCore.QTimer()
        self.metrics_timer.timeout.connect(self.update_real_time_metrics)
        self.metrics_timer.start(1000)

        # Timer para atualizar a interface do player
        self.player_update_timer = QtCore.QTimer()
        self.player_update_timer.timeout.connect(self.update_player_ui)
        self.player_update_timer.start(500)  # Atualizar a cada 500ms

        self.setup_advanced_features()

    def setup_advanced_features(self):
        """Configura recursos avançados"""
        # Sistema de temas
        self.themes = {
            'dark': self.load_dark_theme,
            'light': self.load_light_theme,
            'blue': self.load_blue_theme
        }

        # Monitor de desempenho
        self.performance_monitor = PerformanceMonitor()
        self.performance_monitor.update_metrics.connect(self.update_performance_metrics)
        self.performance_monitor.start()

        # Sistema de logs avançado
        self.log_manager = LogManager()

    def load_dark_theme(self):
        """Tema escuro moderno"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QTabWidget::pane {
                border: 1px solid #555;
                background-color: #2b2b2b;
            }
            QTabBar::tab {
                background-color: #3c3c3c;
                color: #ffffff;
                padding: 8px 16px;
                border: 1px solid #555;
            }
            QTabBar::tab:selected {
                background-color: #0078d4;
            }
            QGroupBox {
                color: #ffffff;
                border: 1px solid #555;
                margin-top: 1ex;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)

    def load_light_theme(self):
        """Tema claro"""
        self.setStyleSheet("")

    def load_blue_theme(self):
        """Tema azul"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #e6f3ff;
            }
            QTabBar::tab:selected {
                background-color: #0078d4;
                color: white;
            }
        """)

    def update_performance_metrics(self, metrics):
        """Atualiza métricas de desempenho"""
        if hasattr(self, 'cpu_usage'):
            self.cpu_usage.setValue(int(metrics['cpu']))
        if hasattr(self, 'memory_usage'):
            self.memory_usage.setValue(int(metrics['memory']))
        if hasattr(self, 'disk_io'):
            self.disk_io.setText(f"IO: {metrics['disk_io']:.1f} MB/s")

    def create_advanced_monitor_tab(self):
        """Aba de monitoramento avançado"""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        if PYQTGRAPH_AVAILABLE:
            # Gráfico em tempo real
            self.signal_plot = pg.PlotWidget(title="Sinal em Tempo Real")
            self.signal_plot.setBackground('#2b2b2b')
            layout.addWidget(self.signal_plot)
        else:
            layout.addWidget(QtWidgets.QLabel("PyQtGraph não disponível para gráficos"))

        # Métricas avançadas
        metrics_grid = QtWidgets.QGridLayout()

        self.cpu_usage = QtWidgets.QProgressBar()
        self.memory_usage = QtWidgets.QProgressBar()
        self.disk_io = QtWidgets.QLabel("IO: 0 MB/s")

        metrics_grid.addWidget(QtWidgets.QLabel("CPU:"), 0, 0)
        metrics_grid.addWidget(self.cpu_usage, 0, 1)
        metrics_grid.addWidget(QtWidgets.QLabel("Memória:"), 1, 0)
        metrics_grid.addWidget(self.memory_usage, 1, 1)
        metrics_grid.addWidget(self.disk_io, 2, 0, 1, 2)

        layout.addLayout(metrics_grid)
        return widget

    def setup_ui(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)

        self.tab_widget = QtWidgets.QTabWidget()

        self.transmit_tab = self.create_transmit_tab()
        self.receive_tab = self.create_receive_tab()
        self.monitor_tab = self.create_monitor_tab()
        self.settings_tab = self.create_settings_tab()

        self.tab_widget.addTab(self.transmit_tab, "📤 Transmitir")
        self.tab_widget.addTab(self.receive_tab, "📥 Receber")
        self.tab_widget.addTab(self.monitor_tab, "📊 Monitor")
        self.tab_widget.addTab(self.settings_tab, "⚙️ Configurações")

        main_layout.addWidget(self.tab_widget)

        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Sistema inicializado e pronto")

    def create_transmit_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        config_group = QtWidgets.QGroupBox("Configurações de Transmissão")
        config_layout = QtWidgets.QGridLayout(config_group)

        config_layout.addWidget(QtWidgets.QLabel("Modulação:"), 0, 0)
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(MODES)
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        config_layout.addWidget(self.mode_combo, 0, 1)

        config_layout.addWidget(QtWidgets.QLabel("Taxa de Símbolo:"), 1, 0)
        self.sr_spin = QtWidgets.QSpinBox()
        self.sr_spin.setRange(600, 38400)
        self.sr_spin.setValue(9600)
        config_layout.addWidget(self.sr_spin, 1, 1)

        self.compress_check = QtWidgets.QCheckBox("Compressão Ativa")
        self.compress_check.setChecked(True)
        config_layout.addWidget(self.compress_check, 2, 0, 1, 2)

        self.mode_info_label = QtWidgets.QLabel()
        self.mode_info_label.setStyleSheet("color: #666; font-size: 10pt;")
        config_layout.addWidget(self.mode_info_label, 3, 0, 1, 2)

        layout.addWidget(config_group)

        action_layout = QtWidgets.QHBoxLayout()

        self.enc_btn = QtWidgets.QPushButton("🔒 Codificar Arquivo Único")
        self.enc_btn.clicked.connect(self.on_encode)
        self.enc_btn.setStyleSheet("QPushButton { padding: 8px; font-weight: bold; }")
        action_layout.addWidget(self.enc_btn)

        self.enc_large_btn = QtWidgets.QPushButton("📦 Codificar Arquivo Grande (Multi-partes)")
        self.enc_large_btn.clicked.connect(self.on_encode_large)
        self.enc_large_btn.setStyleSheet("QPushButton { padding: 8px; font-weight: bold; }")
        action_layout.addWidget(self.enc_large_btn)

        layout.addLayout(action_layout)

        transmit_layout = QtWidgets.QHBoxLayout()

        self.play_btn = QtWidgets.QPushButton("▶️ Reproduzir Último Arquivo")
        self.play_btn.clicked.connect(self.on_play)
        transmit_layout.addWidget(self.play_btn)

        self.transmit_btn = QtWidgets.QPushButton("📡 Transmitir (Reproduzir)")
        self.transmit_btn.clicked.connect(self.on_transmit)
        self.transmit_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; padding: 8px; }")
        transmit_layout.addWidget(self.transmit_btn)

        layout.addLayout(transmit_layout)

        self.cancel_encode_btn = QtWidgets.QPushButton("❌ Cancelar Codificação")
        self.cancel_encode_btn.clicked.connect(self.on_cancel_encode)
        self.cancel_encode_btn.setVisible(False)
        self.cancel_encode_btn.setStyleSheet("QPushButton { background-color: #ff6b6b; color: white; padding: 8px; }")
        layout.addWidget(self.cancel_encode_btn)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.detailed_progress = QtWidgets.QLabel("")
        self.detailed_progress.setStyleSheet("QLabel { padding: 5px; color: #666; }")
        self.detailed_progress.setVisible(False)
        layout.addWidget(self.detailed_progress)

        stats_group = QtWidgets.QGroupBox("Estatísticas da Transmissão")
        stats_layout = QtWidgets.QGridLayout(stats_group)

        self.file_size_label = QtWidgets.QLabel("Tamanho do arquivo: -")
        self.estimated_time_label = QtWidgets.QLabel("Tempo estimado: -")
        self.efficiency_label = QtWidgets.QLabel("Eficiência: -")

        stats_layout.addWidget(self.file_size_label, 0, 0)
        stats_layout.addWidget(self.estimated_time_label, 0, 1)
        stats_layout.addWidget(self.efficiency_label, 1, 0)

        fec_layout = QtWidgets.QHBoxLayout()
        fec_layout.addWidget(QtWidgets.QLabel("Correção de Erro:"))

        self.fec_combo = QtWidgets.QComboBox()
        self.fec_combo.addItems(["Nenhum", "Reed-Solomon", "Convolucional"])
        fec_layout.addWidget(self.fec_combo)

        config_layout.addLayout(fec_layout, 4, 0, 1, 2)

        layout.addWidget(stats_group)

        # ===============================================
        # Player de Áudio Integrado
        # ===============================================
        player_group = QtWidgets.QGroupBox("🎵 Player de Transmissão")
        player_layout = QtWidgets.QGridLayout(player_group)

        # Lista de arquivos para reprodução
        self.playlist_widget = QtWidgets.QListWidget()
        self.playlist_widget.itemDoubleClicked.connect(self.on_playlist_item_double_click)
        player_layout.addWidget(self.playlist_widget, 0, 0, 1, 4)

        # Controles do player
        self.play_btn = QtWidgets.QPushButton("▶️ Reproduzir")
        self.play_btn.clicked.connect(self.on_play_selected)
        player_layout.addWidget(self.play_btn, 1, 0)

        self.pause_btn = QtWidgets.QPushButton("⏸️ Pausar")
        self.pause_btn.clicked.connect(self.on_pause)
        player_layout.addWidget(self.pause_btn, 1, 1)

        self.stop_btn = QtWidgets.QPushButton("⏹️ Parar")
        self.stop_btn.clicked.connect(self.on_stop)
        player_layout.addWidget(self.stop_btn, 1, 2)

        self.clear_playlist_btn = QtWidgets.QPushButton("🗑️ Limpar Lista")
        self.clear_playlist_btn.clicked.connect(self.on_clear_playlist)
        player_layout.addWidget(self.clear_playlist_btn, 1, 3)

        # Barra de progresso
        self.playback_progress = QtWidgets.QProgressBar()
        self.playback_progress.setRange(0, 100)
        player_layout.addWidget(self.playback_progress, 2, 0, 1, 4)

        # Informações do arquivo atual
        self.now_playing_label = QtWidgets.QLabel("Nenhum arquivo selecionado")
        self.now_playing_label.setStyleSheet("QLabel { color: #666; font-style: italic; }")
        player_layout.addWidget(self.now_playing_label, 3, 0, 1, 4)

        layout.addWidget(player_group)

        layout.addStretch()
        return widget

    def create_receive_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        config_group = QtWidgets.QGroupBox("Configurações de Recepção")
        config_layout = QtWidgets.QGridLayout(config_group)

        config_layout.addWidget(QtWidgets.QLabel("Modulação:"), 0, 0)
        self.recv_mode_combo = QtWidgets.QComboBox()
        self.recv_mode_combo.addItems(MODES)
        config_layout.addWidget(self.recv_mode_combo, 0, 1)

        config_layout.addWidget(QtWidgets.QLabel("Taxa de Símbolo:"), 1, 0)
        self.recv_sr_spin = QtWidgets.QSpinBox()
        self.recv_sr_spin.setRange(600, 38400)
        self.recv_sr_spin.setValue(9600)
        config_layout.addWidget(self.recv_sr_spin, 1, 1)

        layout.addWidget(config_group)

        control_layout = QtWidgets.QHBoxLayout()

        self.recv_btn = QtWidgets.QPushButton("🎤 Iniciar Recepção")
        self.recv_btn.setCheckable(True)
        self.recv_btn.clicked.connect(self.on_toggle_receive)
        self.recv_btn.setStyleSheet("QPushButton { padding: 8px; font-weight: bold; }")
        control_layout.addWidget(self.recv_btn)

        self.decode_file_btn = QtWidgets.QPushButton("📁 Decodificar de Arquivo WAV")
        self.decode_file_btn.clicked.connect(self.on_decode_file)
        control_layout.addWidget(self.decode_file_btn)

        layout.addLayout(control_layout)

        volume_group = QtWidgets.QGroupBox("Nível de Entrada")
        volume_layout = QtWidgets.QVBoxLayout(volume_group)

        self.volmeter = QtWidgets.QProgressBar()
        self.volmeter.setRange(0, 100)
        self.volmeter.setValue(0)
        self.volmeter.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """)
        volume_layout.addWidget(self.volmeter)

        layout.addWidget(volume_group)

        self.assembly_status = QtWidgets.QLabel("Nenhum arquivo multi-partes sendo montado")
        self.assembly_status.setStyleSheet("QLabel { padding: 8px; background-color: #f0f0f0; border-radius: 4px; }")
        layout.addWidget(self.assembly_status)

        files_group = QtWidgets.QGroupBox("Arquivos Recebidos")
        files_layout = QtWidgets.QVBoxLayout(files_group)

        self.received_files_list = QtWidgets.QListWidget()
        files_layout.addWidget(self.received_files_list)

        layout.addWidget(files_group)

        return widget

    def create_monitor_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        metrics_group = QtWidgets.QGroupBox("Métricas em Tempo Real")
        metrics_layout = QtWidgets.QGridLayout(metrics_group)

        self.bitrate_label = QtWidgets.QLabel("Taxa de bits: 0 bps")
        self.snr_label = QtWidgets.QLabel("SNR: -- dB")
        self.ber_label = QtWidgets.QLabel("BER: --")
        self.quality_label = QtWidgets.QLabel("Qualidade: --")

        metrics_layout.addWidget(self.bitrate_label, 0, 0)
        metrics_layout.addWidget(self.snr_label, 0, 1)
        metrics_layout.addWidget(self.ber_label, 1, 0)
        metrics_layout.addWidget(self.quality_label, 1, 1)

        layout.addWidget(metrics_group)

        log_group = QtWidgets.QGroupBox("Log de Atividades")
        log_layout = QtWidgets.QVBoxLayout(log_group)

        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setMaximumHeight(300)
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)

        log_controls = QtWidgets.QHBoxLayout()
        self.clear_log_btn = QtWidgets.QPushButton("Limpar Log")
        self.clear_log_btn.clicked.connect(self.log_text.clear)
        log_controls.addWidget(self.clear_log_btn)

        self.save_log_btn = QtWidgets.QPushButton("Salvar Log")
        self.save_log_btn.clicked.connect(self.save_log)
        log_controls.addWidget(self.save_log_btn)

        log_controls.addStretch()
        log_layout.addLayout(log_controls)

        layout.addWidget(log_group)

        return widget

    def create_settings_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        audio_group = QtWidgets.QGroupBox("Configurações de Áudio")
        audio_layout = QtWidgets.QFormLayout(audio_group)

        self.sample_rate_combo = QtWidgets.QComboBox()
        self.sample_rate_combo.addItems(["44100", "48000", "96000"])
        self.sample_rate_combo.setCurrentText("96000")
        audio_layout.addRow("Taxa de Amostragem:", self.sample_rate_combo)

        self.audio_device_combo = QtWidgets.QComboBox()
        self.populate_audio_devices()
        audio_layout.addRow("Dispositivo de Áudio:", self.audio_device_combo)

        layout.addWidget(audio_group)

        file_group = QtWidgets.QGroupBox("Configurações de Arquivo")
        file_layout = QtWidgets.QFormLayout(file_group)

        self.cache_dir_edit = QtWidgets.QLineEdit(CACHE_DIR)
        self.browse_cache_btn = QtWidgets.QPushButton("Procurar...")
        self.browse_cache_btn.clicked.connect(self.browse_cache_dir)

        cache_layout = QtWidgets.QHBoxLayout()
        cache_layout.addWidget(self.cache_dir_edit)
        cache_layout.addWidget(self.browse_cache_btn)
        file_layout.addRow("Diretório de Cache:", cache_layout)

        self.auto_clean_check = QtWidgets.QCheckBox("Limpar cache automaticamente")
        self.auto_clean_check.setChecked(True)
        file_layout.addRow(self.auto_clean_check)

        layout.addWidget(file_group)

        action_layout = QtWidgets.QHBoxLayout()

        self.clear_cache_btn = QtWidgets.QPushButton("🧹 Limpar Cache")
        self.clear_cache_btn.clicked.connect(self.clear_cache)
        action_layout.addWidget(self.clear_cache_btn)

        self.reset_settings_btn = QtWidgets.QPushButton("🔄 Restaurar Padrões")
        self.reset_settings_btn.clicked.connect(self.reset_settings)
        action_layout.addWidget(self.reset_settings_btn)

        layout.addLayout(action_layout)
        layout.addStretch()

        return widget

    def populate_audio_devices(self):
        try:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if device['max_output_channels'] > 0:
                    self.audio_device_combo.addItem(f"{device['name']} ({i})", i)
        except Exception as e:
            self.log_message(f"Erro ao listar dispositivos de áudio: {e}")

    def on_mode_changed(self, mode):
        mode_info = {
            "FSK1200": "Lenta e robusta - Ideal para condições ruins",
            "FSK9600": "Equilibrada - Bom para uso geral",
            "BPSK": "Robusta - Boa imunidade a ruído",
            "QPSK": "Eficiente - 2x velocidade do BPSK",
            "8PSK": "Rápida - 3x velocidade do BPSK",
            "FSK19200": "Alta velocidade - Para bons canais",
            "OFDM4": "Avançada - Resistente a interferências",
            "OFDM8": "Máxima velocidade - Para condições ideais",
            "SSTV": "Para imagens - Velocidade muito lenta"
        }
        self.mode_info_label.setText(mode_info.get(mode, ""))

    def update_real_time_metrics(self):
        if self.record_thread and self.record_thread.isRunning():
            self.bitrate_label.setText(f"Taxa de bits: {self.sr_spin.value()} bps")
            self.snr_label.setText("SNR: 25 dB")
            self.ber_label.setText("BER: 1e-5")
            self.quality_label.setText("Qualidade: Excelente")

    def log_message(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def set_status(self, message):
        self.status_bar.showMessage(message)
        self.log_message(message)

    def on_encode(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Selecionar arquivo para codificar")
        if not fname:
            return

        mode = self.mode_combo.currentText()
        comp = self.compress_check.isChecked()
        sr = self.sr_spin.value()

        try:
            stats = get_encoding_stats(fname, mode, comp, sr)
            self.file_size_label.setText(f"Tamanho do arquivo: {stats['original_size']:,} bytes")
            self.estimated_time_label.setText(f"Tempo estimado: {stats['duration_min']:.1f} min")
            self.efficiency_label.setText(f"Eficiência: {stats['bytes_per_sec']:.0f} bytes/seg")
        except Exception as e:
            self.log_message(f"Erro ao calcular estatísticas: {e}")

        self.set_status(f"Codificando {os.path.basename(fname)} com {mode}...")
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)

        def job():
            try:
                wav = encode_file(fname, mode=mode, compress=comp, symbol_rate=sr, split_large_files=False)
                self.last_wav = wav
                QtCore.QMetaObject.invokeMethod(self, "encode_finished",
                                                QtCore.Qt.QueuedConnection,
                                                QtCore.Q_ARG(str, wav))
            except Exception as e:
                QtCore.QMetaObject.invokeMethod(self, "encode_error",
                                                QtCore.Qt.QueuedConnection,
                                                QtCore.Q_ARG(str, str(e)))

        threading.Thread(target=job, daemon=True).start()

    def on_encode_large(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Selecionar arquivo grande para codificar em partes")
        if not fname:
            return

        mode = self.mode_combo.currentText()
        comp = self.compress_check.isChecked()
        sr = self.sr_spin.value()

        duration, ok = QtWidgets.QInputDialog.getInt(self, "Duração da Parte",
                                                     "Duração alvo por parte (minutos):",
                                                     value=1, min=1, max=10)
        if not ok:
            return

        try:
            stats = get_encoding_stats(fname, mode, comp, sr)
            total_parts = max(1, int(stats['duration_min'] / duration))
            self.file_size_label.setText(f"Tamanho: {stats['original_size']:,} bytes")
            self.estimated_time_label.setText(f"Partes estimadas: {total_parts}")
            self.efficiency_label.setText(f"Eficiência: {stats['bytes_per_sec']:.0f} bytes/seg")
        except Exception as e:
            self.log_message(f"Erro ao calcular estatísticas: {e}")

        self.set_status(f"Codificando arquivo grande {os.path.basename(fname)} com {mode}...")
        self.progress.setVisible(True)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.detailed_progress.setVisible(True)
        self.detailed_progress.setText("Iniciando...")

        self.cancel_encode_btn.setVisible(True)
        self.enc_large_btn.setEnabled(False)
        self.enc_btn.setEnabled(False)

        self.encode_worker = EncodeWorker(fname, mode, comp, sr, duration)
        self.encode_thread = QtCore.QThread()

        self.encode_worker.moveToThread(self.encode_thread)
        self.encode_worker.finished.connect(self.encode_large_finished)
        self.encode_worker.error.connect(self.encode_large_error)
        self.encode_worker.progress.connect(self.encode_progress_update)
        self.encode_worker.cancelled.connect(self.encode_cancelled)

        self.encode_thread.started.connect(self.encode_worker.run)
        self.encode_thread.start()

    def on_cancel_encode(self):
        if self.encode_worker:
            self.set_status("Cancelando codificação...")
            self.encode_worker.cancel()

    def encode_progress_update(self, current_part, total_parts):
        progress = int((current_part / total_parts) * 100) if total_parts > 0 else 0
        self.progress.setValue(progress)
        self.detailed_progress.setText(f"Codificando parte {current_part}/{total_parts}")

    @QtCore.pyqtSlot()
    def encode_cancelled(self):
        self.progress.setVisible(False)
        self.detailed_progress.setVisible(False)
        self.cancel_encode_btn.setVisible(False)
        self.enc_large_btn.setEnabled(True)
        self.enc_btn.setEnabled(True)

        if self.encode_thread and self.encode_thread.isRunning():
            self.encode_thread.quit()
            self.encode_thread.wait(1000)

        self.set_status("Codificação cancelada pelo usuário")
        QtWidgets.QMessageBox.information(self, "Codificação Cancelada",
                                          "A codificação do arquivo foi cancelada. Arquivos parciais foram removidos.")

    @QtCore.pyqtSlot(str)
    def encode_finished(self, wav):
        self.progress.setVisible(False)
        self.set_status(f"Codificado com sucesso: {os.path.basename(wav)}")
        self.last_wav = wav
        # Adicionar à playlist automaticamente
        self.add_file_to_playlist(wav)

    @QtCore.pyqtSlot(str)
    def encode_large_finished(self, result):
        self.progress.setVisible(False)
        self.detailed_progress.setVisible(False)
        self.cancel_encode_btn.setVisible(False)
        self.enc_large_btn.setEnabled(True)
        self.enc_btn.setEnabled(True)

        if self.encode_thread and self.encode_thread.isRunning():
            self.encode_thread.quit()
            self.encode_thread.wait(1000)

        if result.endswith('.playlist.txt'):
            # Adicionar todas as partes à playlist
            try:
                with open(result, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip() and not line.startswith(('File:', 'Total parts:', 'Mode:', 'Symbol rate:',
                                                                 'Original size:', 'Compression:', 'Parts:')):
                            part_file = line.strip()
                            part_path = os.path.join(CACHE_DIR, part_file)
                            if os.path.exists(part_path):
                                self.add_file_to_playlist(part_path)
            except Exception as e:
                print(f"Erro ao ler playlist: {e}")
        else:
            self.last_wav = result
            self.add_file_to_playlist(result)
            self.set_status(f"Codificado com sucesso: {os.path.basename(result)}")

    @QtCore.pyqtSlot(str)
    def encode_error(self, error):
        self.progress.setVisible(False)
        self.detailed_progress.setVisible(False)
        self.cancel_encode_btn.setVisible(False)
        self.enc_large_btn.setEnabled(True)
        self.enc_btn.setEnabled(True)
        self.set_status(f"Erro na codificação: {error}")
        QtWidgets.QMessageBox.critical(self, "Erro na Codificação", f"Erro: {error}")

    @QtCore.pyqtSlot(str)
    def encode_large_error(self, error):
        self.progress.setVisible(False)
        self.detailed_progress.setVisible(False)
        self.cancel_encode_btn.setVisible(False)
        self.enc_large_btn.setEnabled(True)
        self.enc_btn.setEnabled(True)
        self.set_status(f"Erro na codificação: {error}")
        QtWidgets.QMessageBox.critical(self, "Erro na Codificação", f"Erro: {error}")

    def on_play(self):
        if not self.last_wav or not os.path.exists(self.last_wav):
            QtWidgets.QMessageBox.warning(self, "Sem arquivo WAV", "Por favor, codifique um arquivo primeiro.")
            return

        self.set_status("Reproduzindo WAV codificado...")
        try:
            data, sr = sf.read(self.last_wav, always_2d=False)
            sd.play(data, sr)
            sd.wait()
            self.set_status("Reprodução concluída")
        except Exception as e:
            self.set_status(f"Erro na reprodução: {e}")

    def on_transmit(self):
        self.on_play()

    def on_toggle_receive(self):
        if self.recv_btn.isChecked():
            self.start_receive()
        else:
            self.stop_receive()

    def start_receive(self):
        self.set_status("Iniciando recepção...")
        self.recv_btn.setText("Parar Recepção")

        duration = 300
        mode = self.recv_mode_combo.currentText()
        sr = self.recv_sr_spin.value()

        self.record_thread = WorkerRecord(duration=duration, mode=mode, sr=sr)
        self.record_thread.volume.connect(self.update_volume)
        self.record_thread.finished.connect(self.on_receive_finished)
        self.record_thread.start()

        self.set_status(f"Recebendo no modo {mode}...")

    def stop_receive(self):
        self.set_status("Parando recepção...")
        if self.record_thread:
            self.record_thread._running = False
        self.recv_btn.setChecked(False)
        self.recv_btn.setText("Iniciar Recepção")

    def update_volume(self, level):
        self.volmeter.setValue(int(level * 100))

    def update_assembly_status(self):
        try:
            assemblies = get_assembly_status()
            if assemblies:
                status_text = "Montando arquivos:\n"
                status_parts = []

                for assembly in assemblies:
                    missing_parts = assembly.get('missing_parts', [])
                    missing_info = f" - Faltando: {[p + 1 for p in missing_parts]}" if missing_parts else " - Completo!"
                    progress = f"{assembly['filename']} ({assembly['received']}/{assembly['total']}) {missing_info}"
                    status_parts.append(progress)

                status_text += "\n".join(status_parts)
                self.assembly_status.setText(status_text)

                total_assemblies = len(assemblies)
                self.status_bar.showMessage(f"Montando {total_assemblies} arquivo(s) multi-partes")
            else:
                self.assembly_status.setText("Nenhum arquivo multi-partes sendo montado")

            stats = get_reception_stats()
            if stats['last_reception']:
                last_time = time.strftime("%H:%M:%S", time.localtime(stats['last_reception']))
                quality_info = f" | Qualidade média: {stats.get('average_quality', 0):.1f}%" if 'average_quality' in stats else ""
                self.status_bar.showMessage(
                    f"Arquivos recebidos: {stats['total_files']} | Última recepção: {last_time}{quality_info}")

        except Exception as e:
            self.assembly_status.setText(f"Erro no status de montagem: {e}")

    def on_receive_finished(self, saved_files):
        self.recv_btn.setChecked(False)
        self.recv_btn.setText("Iniciar Recepção")

        self.update_assembly_status()

        if saved_files:
            self.set_status(f"Recebido {len(saved_files)} arquivo(s)")

            self.received_files_list.clear()
            for file_path in saved_files:
                self.received_files_list.addItem(os.path.basename(file_path))

            assemblies = get_assembly_status()
            if assemblies:
                assembly_info = "\n\nArquivos ainda sendo montados:\n"
                for assembly in assemblies:
                    assembly_info += f"- {assembly['filename']} ({assembly['received']}/{assembly['total']} partes)\n"
                saved_msg = "Arquivos recebidos:\n" + "\n".join(
                    os.path.basename(f) for f in saved_files) + assembly_info
            else:
                saved_msg = "Arquivos recebidos:\n" + "\n".join(os.path.basename(f) for f in saved_files)

            QtWidgets.QMessageBox.information(self, "Recepção Concluída", saved_msg)
        else:
            self.set_status("Recepção finalizada - nenhum arquivo decodificado")

            assemblies = get_assembly_status()
            if assemblies:
                assembly_info = "Arquivos ainda sendo montados:\n"
                for assembly in assemblies:
                    assembly_info += f"- {assembly['filename']} ({assembly['received']}/{assembly['total']} partes)\n"
                QtWidgets.QMessageBox.information(self, "Recepção Concluída",
                                                  "Nenhum arquivo completo decodificado.\n\n" + assembly_info)
            else:
                QtWidgets.QMessageBox.information(self, "Recepção Concluída", "Nenhum arquivo foi decodificado.")

    def on_decode_file(self):
        wav, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Selecionar WAV para decodificar",
                                                       filter="Arquivos WAV (*.wav)")
        if not wav:
            return

        mode = self.recv_mode_combo.currentText()
        sr = self.recv_sr_spin.value()

        self.set_status("Decodificando arquivo WAV...")
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)

        def job():
            try:
                saved = decode_wav_file(wav, mode=mode, symbol_rate=sr)
                QtCore.QMetaObject.invokeMethod(self, "decode_finished",
                                                QtCore.Qt.QueuedConnection,
                                                QtCore.Q_ARG(list, saved))
            except Exception as e:
                QtCore.QMetaObject.invokeMethod(self, "decode_error",
                                                QtCore.Qt.QueuedConnection,
                                                QtCore.Q_ARG(str, str(e)))

        threading.Thread(target=job, daemon=True).start()

    @QtCore.pyqtSlot(list)
    def decode_finished(self, saved):
        self.progress.setVisible(False)

        self.update_assembly_status()

        if saved:
            self.set_status(f"Decodificado {len(saved)} arquivo(s)")

            for file_path in saved:
                self.received_files_list.addItem(os.path.basename(file_path))

            msg = "Arquivos decodificados:\n" + "\n".join(os.path.basename(f) for f in saved)

            assemblies = get_assembly_status()
            if assemblies:
                assembly_info = "\n\nArquivos ainda sendo montados:\n"
                for assembly in assemblies:
                    assembly_info += f"- {assembly['filename']} ({assembly['received']}/{assembly['total']} partes)\n"
                msg += assembly_info

            QtWidgets.QMessageBox.information(self, "Decodificação Concluída", msg)
        else:
            self.set_status("Nenhum arquivo decodificado do WAV")

            assemblies = get_assembly_status()
            if assemblies:
                assembly_info = "Arquivos ainda sendo montados:\n"
                for assembly in assemblies:
                    assembly_info += f"- {assembly['filename']} ({assembly['received']}/{assembly['total']} partes)\n"
                QtWidgets.QMessageBox.information(self, "Decodificação Concluída",
                                                  "Nenhum arquivo completo decodificado.\n\n" + assembly_info)
            else:
                QtWidgets.QMessageBox.information(self, "Decodificação Concluída", "Nenhum arquivo foi decodificado.")

    @QtCore.pyqtSlot(str)
    def decode_error(self, error):
        self.progress.setVisible(False)
        self.set_status(f"Erro na decodificação: {error}")
        QtWidgets.QMessageBox.critical(self, "Erro na Decodificação", f"Erro: {error}")

    def browse_cache_dir(self):
        global CACHE_DIR
        new_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Selecionar Diretório de Cache", CACHE_DIR)
        if new_dir:
            CACHE_DIR = new_dir
            self.cache_dir_edit.setText(CACHE_DIR)
            os.makedirs(CACHE_DIR, exist_ok=True)
            self.log_message(f"Diretório de cache alterado para: {CACHE_DIR}")

    def clear_cache(self):
        try:
            count = 0
            for filename in os.listdir(CACHE_DIR):
                file_path = os.path.join(CACHE_DIR, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    count += 1
            self.log_message(f"Cache limpo com sucesso - {count} arquivos removidos")
            QtWidgets.QMessageBox.information(self, "Cache Limpo", f"{count} arquivos foram removidos do cache.")
        except Exception as e:
            self.log_message(f"Erro ao limpar cache: {e}")
            QtWidgets.QMessageBox.critical(self, "Erro", f"Erro ao limpar cache: {e}")

    def reset_settings(self):
        self.sr_spin.setValue(9600)
        self.recv_sr_spin.setValue(9600)
        self.mode_combo.setCurrentText("FSK9600")
        self.recv_mode_combo.setCurrentText("FSK9600")
        self.compress_check.setChecked(True)
        self.sample_rate_combo.setCurrentText("96000")
        self.log_message("Configurações restauradas para os padrões")
        QtWidgets.QMessageBox.information(self, "Configurações Restauradas",
                                          "Todas as configurações foram restauradas para os valores padrão.")

    def save_log(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Salvar Log", "filebeep_log.txt",
                                                            "Arquivos de Texto (*.txt)")
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.toPlainText())
                self.log_message(f"Log salvo em: {filename}")
                QtWidgets.QMessageBox.information(self, "Log Salvo", f"Log salvo com sucesso em:\n{filename}")
            except Exception as e:
                self.log_message(f"Erro ao salvar log: {e}")
                QtWidgets.QMessageBox.critical(self, "Erro", f"Erro ao salvar log: {e}")

    # ===============================================
    # Métodos do Player de Áudio
    # ===============================================

    def update_player_ui(self):
        """Atualiza a interface do player"""
        # Atualizar barra de progresso
        progress = self.audio_player.get_progress()
        self.playback_progress.setValue(int(progress))

        # Atualizar cores dos itens na playlist
        for i in range(self.playlist_widget.count()):
            item = self.playlist_widget.item(i)
            file_path = item.data(QtCore.Qt.UserRole)

            if file_path in self.played_files:
                # Verde - reprodução concluída
                item.setBackground(QtGui.QColor(200, 255, 200))
                item.setForeground(QtGui.QColor(0, 100, 0))
            elif file_path in self.playing_files:
                # Amarelo - em reprodução
                item.setBackground(QtGui.QColor(255, 255, 200))
                item.setForeground(QtGui.QColor(150, 150, 0))
            else:
                # Vermelho - ainda não reproduzido
                item.setBackground(QtGui.QColor(255, 200, 200))
                item.setForeground(QtGui.QColor(100, 0, 0))

        # Atualizar informação do arquivo atual
        if self.audio_player.is_playing and self.audio_player.current_file:
            current_file = os.path.basename(self.audio_player.current_file)
            progress_text = f"Reproduzindo: {current_file} ({progress:.1f}%)"
            self.now_playing_label.setText(progress_text)
            self.now_playing_label.setStyleSheet("QLabel { color: #006600; font-weight: bold; }")
        elif self.audio_player.current_file and not self.audio_player.is_playing:
            current_file = os.path.basename(self.audio_player.current_file)
            self.now_playing_label.setText(f"Pausado: {current_file}")
            self.now_playing_label.setStyleSheet("QLabel { color: #666600; font-style: italic; }")
        else:
            self.now_playing_label.setText("Nenhum arquivo selecionado")
            self.now_playing_label.setStyleSheet("QLabel { color: #666; font-style: italic; }")

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
        item = QtWidgets.QListWidgetItem(os.path.basename(file_path))
        item.setData(QtCore.Qt.UserRole, file_path)
        self.playlist_widget.addItem(item)

        # Marcar como não reproduzido (vermelho)
        self.played_files.discard(file_path)
        self.playing_files.discard(file_path)

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

    def play_audio_file(self, file_path):
        """Reproduz um arquivo de áudio"""
        if self.audio_player.load_file(file_path):
            self.audio_player.play()
            self.playing_files.add(file_path)
            self.played_files.discard(file_path)
            self.log_message(f"Reproduzindo: {os.path.basename(file_path)}")
        else:
            QtWidgets.QMessageBox.warning(self, "Erro",
                                          f"Não foi possível reproduzir o arquivo: {os.path.basename(file_path)}")

    def on_pause(self):
        """Pausa a reprodução"""
        self.audio_player.pause()
        self.log_message("Reprodução pausada")

    def on_stop(self):
        """Para a reprodução"""
        self.audio_player.stop()
        if self.audio_player.current_file:
            self.playing_files.discard(self.audio_player.current_file)
        self.log_message("Reprodução parada")

    def on_clear_playlist(self):
        """Limpa a playlist"""
        self.audio_player.stop()
        self.playlist_widget.clear()
        self.played_files.clear()
        self.playing_files.clear()
        self.log_message("Playlist limpa")

    def closeEvent(self, event):
        # Parar reprodução de áudio
        self.audio_player.stop()
        pygame.mixer.quit()

        if self.record_thread and self.record_thread.isRunning():
            self.record_thread._running = False
            self.record_thread.wait(2000)

        if self.encode_worker:
            self.encode_worker.cancel()

        if self.encode_thread and self.encode_thread.isRunning():
            self.encode_thread.quit()
            self.encode_thread.wait(1000)

        self.assembly_timer.stop()
        self.metrics_timer.stop()
        self.player_update_timer.stop()

        self.log_message("Aplicação encerrada")
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("FileBeep Advanced")
    app.setStyle('Fusion')

    win = ModernMainWindow()
    win.show()

    sys.exit(app.exec_())