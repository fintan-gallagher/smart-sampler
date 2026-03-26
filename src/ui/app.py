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
import json

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
    TEST_MODE, TEST_AUDIO_PATH, SAMPLES_DIR,IMPORT_DIR,
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

from .prefs import load_prefs, save_prefs

PREFS_PATH = os.path.expanduser('~/.smart_sampler_prefs.json')

def _load_prefs() -> dict:
    try:
        with open(PREFS_PATH) as f:
            return json.load(f)
    except Exception:
        return {}

def _save_prefs(data: dict):
    try:
        with open(PREFS_PATH, 'w') as f:
            json.dump(data, f)
    except Exception:
        pass


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
        self._browser_sel_folder_idx: int | None = None

        # Sample browser — file level
        self._browser_files: list[tuple[str, str]] = []
        self._browser_scroll = 0
        self._browser_sel    = None
        self._browser_audio  = None
        self._confirm_delete: dict | None = None

        # MIDI play state
        self._sfizz_proc: subprocess.Popen | None = None
        self._midi_sfz_path: str = ""
        self._midi_status: str   = ""
        self._midi_vel_fixed: bool = False

        # Persistent MIDI engine state
        # _midi_engine_active  — sfizz_jack is running and has confirmed ready
        # _midi_engine_loading — startup thread is in progress, not yet ready
        # _midi_live_sfz       — absolute path of the SFZ currently loaded
        # _midi_autoload       — when True, load selected sample on next EV_MIDI_READY
        # _midi_browser_mode   — True while the user is in the MIDI browse flow
        #                        (entered via home "MIDI Browse" button)
        self._midi_engine_active:  bool = False
        self._midi_engine_loading: bool = False
        self._midi_live_sfz:       str  = ""
        self._midi_autoload:       bool = False
        self._midi_browser_mode:   bool = False

        # DTLN Warning
        self._dtln_warning = False
        self._dtln_warn_no_show = False
        self._prefs = _load_prefs()

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
                                    "Imported Samples", MID, WHITE)
        self.h_btn_browse  = Button((PAD,                  HEADER_H + 88, bw, bh),
                                    "Browse Samples", MID, WHITE)
        self.h_dtln_toggle = Toggle(SCREEN_W - bw - PAD + 8, HEADER_H + 98,
                                    "Use DTLN denoising", w=70, h=36)
        self.h_btn_quit    = Button((SCREEN_W - 78, 7, 68, 32),
                                    "Quit", RED_DIM, WHITE, fsize=14)
        # Persistent MIDI engine toggle — sits below "Browse Samples" in the left column.
        # The label and bg colour are overwritten every frame in _draw_home() to reflect
        # the current engine state (off / loading / active).
        self.h_btn_midi    = Button((PAD, HEADER_H + 158, 215, 46),
                                    "MIDI Engine", DARK_MID, WHITE)
        
        # ── DTLN WARNING OVERLAY ──────────────────────────────────────────
        self.dtln_btn_ok     = Button((SCREEN_W//2 - 110, SCREEN_H//2 + 34, 100, 40),
                                      "Enable", ACCENT, BLACK, bold=True, fsize=13)
        self.dtln_btn_cancel = Button((SCREEN_W//2 + 10,  SCREEN_H//2 + 34, 100, 40),
                                      "Cancel", RED, WHITE, bold=True, fsize=13)
        self.dtln_warn_cb_rect = pygame.Rect(SCREEN_W//2 - 110, SCREEN_H//2 + 16, 18, 18)

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

        # ── PRE-RECORD ────────────────────────────────────────────────────
        self.pr_btn_back   = Button((SCREEN_W - 90, 7, 80, 32),
                                    "Back", MID, WHITE, fsize=14)
        self.pr_btn_record = Button((SCREEN_W//2 - 75, SCREEN_H - 65, 150, 48),
                                    "Record", RED, WHITE, bold=True)

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
        self.lbl_loop_toggle = Toggle(
            SCREEN_W - PAD - 70, SCREEN_H - 34,
            "Loop", value=False, w=70, h=36, label_left=True
        )
        self.lbl_buttons: list[tuple[Button, str]] = []

        # ── BROWSER — folder list ──────────────────────────────────────────
        self.br_btn_back = Button((SCREEN_W - 90, 7, 80, 32),
                                  "Back", MID, WHITE, fsize=14)
        self.br_btn_up   = Button((SCREEN_W - 44, HEADER_H + 4,  34, 56),
                                  "▲", DARK_MID, WHITE)
        self.br_btn_dn   = Button((SCREEN_W - 44, HEADER_H + 66, 34, 56),
                                  "▼", DARK_MID, WHITE)
        self.br_btn_open   = Button((PAD,          SCREEN_H - 50, 140, 38), 
                                  "Open",   ACCENT, BLACK, bold=True)
        self.br_btn_delete = Button((PAD + 148,    SCREEN_H - 50, 140, 38),
                                  "Delete", RED,    WHITE, bold=True)

        # ── BROWSER FILES — file list inside a folder ──────────────────────
        self.brf_btn_back = Button((SCREEN_W - 90, 7, 80, 32),
                                   "Back", MID, WHITE, fsize=14)
        self.brf_btn_up   = Button((SCREEN_W - 44, HEADER_H + 4,  34, 56),
                                   "▲", DARK_MID, WHITE)
        self.brf_btn_dn   = Button((SCREEN_W - 44, HEADER_H + 66, 34, 56),
                                   "▼", DARK_MID, WHITE)
        self.brf_waveform = WaveformWidget(
            (PAD, SCREEN_H - 78, SCREEN_W - PAD*2, 38))
        self.brf_btn_play = Button((PAD,          SCREEN_H - 36, 100, 34),
                                   "▶ Play", GREEN, BLACK, bold=True)
        self.brf_btn_midi = Button((PAD + 108,    SCREEN_H - 36, 100, 34),
                                   "MIDI", ACCENT, BLACK, bold=True)
        self.brf_vel_toggle = Toggle(PAD, SCREEN_H - 118,
                                     "Fix velocity", w=70, h=36)
        self.brf_btn_delete = Button((PAD + 236,   SCREEN_H - 36, 100, 34),
                                     "Delete", RED, WHITE, bold=True)
        
        self.cont_btn_yes = Button((SCREEN_W//2 - 110, SCREEN_H//2 + 20, 100, 40),
                                   "Yes", ACCENT, BLACK, bold=True, fsize=13)
        self.cont_btn_no  = Button((SCREEN_W//2 + 10,  SCREEN_H//2 + 20, 100, 40),
                                   "No", RED, WHITE, bold=True, fsize=13)

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
        if not os.path.isdir(IMPORT_DIR):
            self._status     = "No imported_samples folder on USB"
            self._status_col = RED
            return
        self._test_files = sorted(
            f for f in os.listdir(IMPORT_DIR) if f.lower().endswith('.wav'))
        if not self._test_files:
            self._status     = "No WAV files found in imported_samples/"
            self._status_col = RED
            return
        self._test_scroll = 0
        self._go('test_pick')

    def _run_test_file(self, filename: str):
        src = os.path.join(IMPORT_DIR, filename)
        shutil.copy2(src, RAW_FILENAME)
        self._go('processing')
        threading.Thread(target=self._process_worker,
                         args=(self._use_dtln,), daemon=True).start()

    def _start_recording(self):
        self._go('pre_record')

    def _begin_recording(self):
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
        self._browser_sel_folder_idx = None

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

    # ──────────────────────────────────────────────────────────────────────
    #  Persistent MIDI engine — helper methods
    # ──────────────────────────────────────────────────────────────────────

    def _send_sfizz_cmd(self, cmd: str) -> bool:
        if self._sfizz_proc is None or self._sfizz_proc.poll() is not None:
            # Process never started or has already exited — reset state.
            if self._midi_engine_active:
                self._midi_engine_active  = False
                self._midi_engine_loading = False
                self._midi_status = "Engine stopped unexpectedly — restart from home"
            return False
        try:
            self._sfizz_proc.stdin.write((cmd + '\n').encode())
            self._sfizz_proc.stdin.flush()
            return True
        except Exception:
            # stdin write failed (broken pipe etc.) — treat engine as dead.
            self._midi_engine_active  = False
            self._midi_engine_loading = False
            return False

    def _start_midi_engine(self):
        if self._midi_engine_active or self._midi_engine_loading:
            return   # already running or starting
        self._midi_engine_loading = True
        self._midi_live_sfz = ""
        self._midi_status = "Starting MIDI engine…"
        threading.Thread(
            target=self._midi_launch_worker,
            args=(None,),           # None = start without any SFZ file
            daemon=True,
        ).start()

    def _stop_midi_engine(self):       
        self._send_sfizz_cmd('quit')
        import time; time.sleep(0.2)   # give sfizz a moment to exit cleanly
        self._midi_stop_sfizz()        # existing hard-kill fallback + cleanup
        self._midi_engine_active  = False
        self._midi_engine_loading = False
        self._midi_live_sfz       = ""
        self._midi_status         = ""

    def _enter_midi_browse(self):
        if self._midi_engine_active or self._midi_engine_loading:
            self._stop_midi_engine()
        self._midi_browser_mode = True
        self._start_midi_engine()
        self._open_browser()

    def _exit_midi_browse(self):
        self._midi_browser_mode = False
        self._stop_midi_engine()
        self._go_home()

    def _load_sample_into_engine(self):
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

        self._midi_vel_fixed = self.brf_vel_toggle.value
        base_sfz       = sfz_candidates[0]
        audio_filename = os.path.basename(wav_path)

        # Fixed-velocity variant lives in a separate file so the original SFZ
        # (with dynamic velocity) is never overwritten by the toggle.
        target_sfz = (base_sfz.replace('.sfz', '_fixedvel.sfz')
                      if self._midi_vel_fixed else base_sfz)

        # Determine the pitch keycenter for the SFZ <region>.
        detected_pitch = None
        if self._results:
            detected_pitch = self._results.get('detected_pitch')
        else:
            try:
                with open(base_sfz) as f:
                    for line in f:
                        if 'pitch_keycenter' in line:
                            import librosa
                            detected_pitch = librosa.midi_to_hz(
                                int(line.split('=')[1].strip()))
                            break
            except Exception:
                pass

        self.sampler.sfz_generator.save(
            target_sfz, audio_filename,
            detected_pitch,
            self._browser_sel_folder or 'sample',
            fixed_velocity=self._midi_vel_fixed,
        )
       
        PREVIEW_SFZ  = '/tmp/ss_preview.sfz'
        sfz_preview  = self.sampler.sfz_generator.generate(
            wav_path,            # absolute path → written verbatim as sample=
            detected_pitch,
            self._browser_sel_folder or 'sample',
            audio_path=wav_path, # also used for loop_end frame count
            fixed_velocity=self._midi_vel_fixed,
        )
        with open(PREVIEW_SFZ, 'w') as _pf:
            _pf.write(sfz_preview)

        # Send the hot-swap command using the space-free preview path.
        if self._send_sfizz_cmd(f'load_instrument {PREVIEW_SFZ}'):
            self._midi_live_sfz = target_sfz   # display the real path in UI
            self._midi_status   = f"Loaded: {os.path.basename(target_sfz)}"
        else:
            self._midi_status = "Engine not responding — restart from home"

    # ──────────────────────────────────────────────────────────────────────
    #  MIDI play state transitions
    # ──────────────────────────────────────────────────────────────────────

    def _go_midi_play(self):
        if self._browser_sel is None:
            return

        if self._midi_engine_active:
            # Case A — instant hot-swap.
            self._load_sample_into_engine()

        elif self._midi_engine_loading:
            # Case B — engine is already starting; queue the load.
            self._midi_autoload = True
            self._status     = "Engine starting — sample will auto-load…"
            self._status_col = YELLOW

        else:
            # Case C — start engine then auto-load when ready.
            self._midi_autoload = True
            self._start_midi_engine()

    def _stop_midi_play(self):
        """Stop button handler for the legacy midi_play screen."""
        self._midi_stop_sfizz()
        self._midi_engine_active  = False
        self._midi_engine_loading = False
        self._midi_live_sfz       = ""
        self._go('browser_files')

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


