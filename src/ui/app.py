"""
Smart Sampler PyGame UI
Designed for 480x320, tested on 1080p desktop (window stays 480x320).

Structure:
  events.py  — custom pygame event constants
  widgets.py — Button, Toggle, WaveformWidget, draw_header, draw_status
  workers.py — WorkersMixin  (background thread methods)
  handlers.py — HandlersMixin (event handling + browser helpers)
  screens.py — ScreensMixin  (all _draw_* methods)
  app.py     — SamplerApp    (__init__, widget construction, state transitions, run loop)
"""
import os
import sys
import shutil
import threading
import subprocess

import pygame

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from .theme import (
    SCREEN_W, SCREEN_H, FPS,
    BLACK, DARK, DARK_MID, MID, GREY, WHITE,
    ACCENT, ACCENT_DIM, RED, RED_DIM, GREEN, YELLOW, ORANGE,
    HEADER_H, PAD, font,
)
from ..config import (
    TEST_MODE, TEST_AUDIO_PATH, SAMPLES_DIR,
    RAW_FILENAME, CLEAN_FILENAME, SAMPLE_RATE,
    SFIZZ_BINARY,
)
from ..main import SmartSampler
from .events import (
    EV_RECORD_DONE, EV_PROCESS_DONE, EV_PROCESS_ERR,
    EV_MIDI_READY, EV_MIDI_ERR,
)
from .widgets import Button, Toggle, WaveformWidget, draw_header, draw_status
from .workers import WorkersMixin
from .handlers import HandlersMixin
from .screens import ScreensMixin


# ─────────────────────────────────────────────────────────────────────────────
#  Kept here only so external code that does `from .app import Button` still works
# ─────────────────────────────────────────────────────────────────────────────

class _DeprecatedButton(Button):
    """Backward-compat alias — use widgets.Button directly."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
#  Main application
# ─────────────────────────────────────────────────────────────────────────────

class SamplerApp(WorkersMixin, HandlersMixin, ScreensMixin):
    """
    State machine UI.
    States: home | test_pick | recording | processing | review | label |
            browser | browser_files | midi_play
    """

    def __init__(self):
        pygame.init()
        pygame.mixer.init(frequency=SAMPLE_RATE, channels=1)

        info = pygame.display.Info()
        if info.current_w <=480 and info.current_h <= 320:
            self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
            
        pygame.display.set_caption("Smart Sampler")
        self.clock  = pygame.time.Clock()

        self.sampler     = SmartSampler()
        self._state      = 'home'
        self._results    = None
        self._status     = ""
        self._status_col = YELLOW
        self._use_dtln   = False

        # Live recording state
        self._recording        = False
        self._stop_record_flag = threading.Event()
        self._live_chunk       = None

        # Test-file picker state
        self._test_files: list[str] = []
        self._test_scroll = 0

        # Sample browser — folder level
        self._browser_folders: list[str] = []
        self._browser_folder_scroll = 0
        self._browser_sel_folder: str | None = None

        # Sample browser — file level
        self._browser_files: list[tuple[str, str]] = []
        self._browser_scroll = 0
        self._browser_sel    = None
        self._browser_audio  = None

        # MIDI play state
        self._sfizz_proc: subprocess.Popen | None = None
        self._midi_sfz_path: str = ""
        self._midi_status: str   = ""
        self._midi_vel_fixed: bool = False

        self._build_widgets()

    # ──────────────────────────────────────────────────────────────────────
    #  Widget construction
    # ──────────────────────────────────────────────────────────────────────

    def _build_widgets(self):
        # ── HOME ──────────────────────────────────────────────────────────
        bw, bh = 215, 58
        self.h_btn_record  = Button((PAD,                  HEADER_H + 18, bw, bh),
                                    "Record", ACCENT, BLACK, bold=True)
        self.h_btn_test    = Button((SCREEN_W - bw - PAD,  HEADER_H + 18, bw, bh),
                                    "Test File", MID, WHITE)
        self.h_btn_browse  = Button((PAD,                  HEADER_H + 88, bw, bh),
                                    "Browse Samples", MID, WHITE)
        self.h_dtln_toggle = Toggle(SCREEN_W - bw - PAD + 8, HEADER_H + 98,
                                    "Use DTLN denoising")
        self.h_btn_quit    = Button((SCREEN_W - 78, 7, 68, 32),
                                    "Quit", RED_DIM, WHITE, fsize=14)

        # ── TEST FILE PICKER ───────────────────────────────────────────────
        self.tp_btn_back = Button((SCREEN_W - 88, 7, 78, 32),
                                  "Back", MID, WHITE, fsize=14)
        self.tp_btn_run  = Button((SCREEN_W//2 - 90, SCREEN_H - 50, 180, 38),
                                  "Process", ACCENT, BLACK, bold=True)

        # ── RECORDING ─────────────────────────────────────────────────────
        self.rec_waveform = WaveformWidget(
            (PAD, HEADER_H + 10, SCREEN_W - PAD*2, 80))
        self.rec_btn_stop = Button((SCREEN_W//2 - 75, SCREEN_H - 65, 150, 48),
                                   "Stop", RED, WHITE, bold=True)
        self.rec_vu_rect  = pygame.Rect(PAD, HEADER_H + 100, SCREEN_W - PAD*2, 14)

        # ── PROCESSING ────────────────────────────────────────────────────
        self._spin_angle = 0

        # ── REVIEW ────────────────────────────────────────────────────────
        self.rev_waveform    = WaveformWidget(
            (PAD, HEADER_H + 8, SCREEN_W - PAD*2, 72))
        bw2 = 195
        self.rev_btn_label   = Button((PAD,                 SCREEN_H - 50, bw2, 40),
                                      "Label & Save", ACCENT, BLACK, bold=True)
        self.rev_btn_discard = Button((SCREEN_W - bw2 - PAD, SCREEN_H - 50, bw2, 40),
                                      "Discard", RED, WHITE)

        # ── LABEL SELECT ──────────────────────────────────────────────────
        self.lbl_buttons: list[tuple[Button, str]] = []
        self.lbl_btn_back = Button((PAD, SCREEN_H - 46, 90, 36),
                                   "Back", MID, WHITE, fsize=14)

        # ── BROWSER — folder list ──────────────────────────────────────────
        self.br_btn_back = Button((SCREEN_W - 90, 7, 80, 32),
                                  "Back", MID, WHITE, fsize=14)
        self.br_btn_up   = Button((SCREEN_W - 44, HEADER_H + 4,  34, 34),
                                  "▲", DARK_MID, WHITE)
        self.br_btn_dn   = Button((SCREEN_W - 44, HEADER_H + 44, 34, 34),
                                  "▼", DARK_MID, WHITE)

        # ── BROWSER FILES — file list inside a folder ──────────────────────
        self.brf_btn_back = Button((SCREEN_W - 90, 7, 80, 32),
                                   "Back", MID, WHITE, fsize=14)
        self.brf_btn_up   = Button((SCREEN_W - 44, HEADER_H + 4,  34, 34),
                                   "▲", DARK_MID, WHITE)
        self.brf_btn_dn   = Button((SCREEN_W - 44, HEADER_H + 44, 34, 34),
                                   "▼", DARK_MID, WHITE)
        self.brf_waveform = WaveformWidget(
            (PAD, SCREEN_H - 92, SCREEN_W - PAD*2, 38))
        self.brf_btn_play = Button((PAD,          SCREEN_H - 60, 100, 36),
                                   "▶ Play", GREEN, BLACK, bold=True)
        self.brf_btn_midi = Button((PAD + 108,    SCREEN_H - 60, 120, 36),
                                   "MIDI", ACCENT, BLACK, bold=True)
        self.brf_vel_toggle = Toggle(PAD, SCREEN_H - 26,
                                     "Fix velocity")

        # ── MIDI PLAY ─────────────────────────────────────────────────────
        self.midi_waveform = WaveformWidget(
            (PAD, HEADER_H + 10, SCREEN_W - PAD*2, 80))
        self.midi_btn_stop = Button((SCREEN_W//2 - 75, SCREEN_H - 55, 150, 44),
                                    "Stop", RED, WHITE, bold=True)

    # ──────────────────────────────────────────────────────────────────────
    #  State transitions
    # ──────────────────────────────────────────────────────────────────────

    def _go(self, state: str):
        self._state  = state
        self._status = ""

    def _go_home(self):
        self._recording = False
        self._go('home')

    def _start_test_pick(self):
        test_dir = os.path.dirname(TEST_AUDIO_PATH)
        self._test_files = sorted(
            f for f in os.listdir(test_dir) if f.lower().endswith('.wav'))
        if not self._test_files:
            self._status     = "No WAV files found in test_samples/"
            self._status_col = RED
            return
        self._test_scroll = 0
        self._go('test_pick')

    def _run_test_file(self, filename: str):
        src = os.path.join(os.path.dirname(TEST_AUDIO_PATH), filename)
        shutil.copy2(src, RAW_FILENAME)
        self._go('processing')
        threading.Thread(target=self._process_worker,
                         args=(self._use_dtln,), daemon=True).start()

    def _start_recording(self):
        self._recording        = True
        self._stop_record_flag = threading.Event()
        self._go('recording')
        threading.Thread(target=self._record_worker, daemon=True).start()

    def _stop_recording(self):
        self._stop_record_flag.set()

    def _open_browser(self):
        """Scan SAMPLES_DIR for subfolders and show them."""
        self._browser_folders       = []
        self._browser_folder_scroll = 0
        self._browser_sel_folder    = None

        if os.path.exists(SAMPLES_DIR):
            self._browser_folders = sorted(
                f for f in os.listdir(SAMPLES_DIR)
                if os.path.isdir(os.path.join(SAMPLES_DIR, f))
            )
        self._go('browser')

    def _open_browser_folder(self, folder_name: str):
        """Open a folder and list its WAV files."""
        self._browser_sel_folder = folder_name
        self._browser_files      = []
        self._browser_scroll     = 0
        self._browser_sel        = None
        self._browser_audio      = None

        folder_path = os.path.join(SAMPLES_DIR, folder_name)
        for root, _, files in os.walk(folder_path):
            for f in sorted(files):
                if f.lower().endswith('.wav'):
                    full = os.path.join(root, f)
                    rel  = os.path.relpath(full, folder_path)
                    self._browser_files.append((rel, full))

        self._go('browser_files')

    def _go_midi_play(self):
        """Transition to midi_play — find SFZ and launch sfizz_jack."""
        if self._browser_sel is None:
            return
        _, wav_path = self._browser_files[self._browser_sel]

        sfz_dir = os.path.dirname(wav_path)
        sfz_candidates = [
            os.path.join(sfz_dir, f)
            for f in os.listdir(sfz_dir)
            if f.lower().endswith('.sfz') and '_fixedvel' not in f
        ]
        if not sfz_candidates:
            self._status     = "No SFZ file found for this sample"
            self._status_col = RED
            return
        
        # Rewrite SFZ with current velocity setting
        self._midi_vel_fixed = self.brf_vel_toggle.value
        base_sfz = sfz_candidates[0]
        audio_filename = os.path.basename(wav_path)

        # Always rewrite so velocity mode is guaranteed correct
        if self._midi_vel_fixed:
            target_sfz = base_sfz.replace('.sfz', '_fixedvel.sfz')
        else:
            target_sfz = base_sfz   # overwrite original with dynamic version

        # Detect pitch from the existing SFZ if _results not available
        detected_pitch = None
        if self._results:
            detected_pitch = self._results.get('detected_pitch')
        else:
            # Try to read pitch_keycenter from existing SFZ
            try:
                with open(base_sfz) as f:
                    for line in f:
                        if 'pitch_keycenter' in line:
                            midi_note = int(line.split('=')[1].strip())
                            import librosa
                            detected_pitch = librosa.midi_to_hz(midi_note)
                            break
            except Exception:
                pass

        self.sampler.sfz_generator.save(
            target_sfz, audio_filename,
            detected_pitch,
            self._browser_sel_folder or 'sample',
            fixed_velocity=self._midi_vel_fixed,
        )
        self._midi_sfz_path = target_sfz

        self._midi_status = "Starting MIDI engine…"
        self._go('midi_play')
        threading.Thread(
            target=self._midi_launch_worker,
            args=(self._midi_sfz_path,),
            daemon=True
        ).start()

    def _stop_midi_play(self):
        self._midi_stop_sfizz()
        self._go('browser_files')   # return to file list, not folder list

    # ──────────────────────────────────────────────────────────────────────
    #  Main loop
    # ──────────────────────────────────────────────────────────────────────

    def run(self):
        while True:
            for ev in pygame.event.get():
                self._handle_event(ev)
            self._draw()
            pygame.display.flip()
            self.clock.tick(FPS)


