"""
Smart Sampler PyGame UI
Single-file UI — all screens in one place for easy iteration.
Designed for 480x320, tested on 1080p desktop (window stays 480x320).
"""
import os
import sys
import shutil
import threading
import subprocess

import numpy as np
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

# Custom pygame events
EV_RECORD_DONE  = pygame.USEREVENT + 1
EV_PROCESS_DONE = pygame.USEREVENT + 2
EV_PROCESS_ERR  = pygame.USEREVENT + 3
EV_MIDI_READY   = pygame.USEREVENT + 4
EV_MIDI_ERR     = pygame.USEREVENT + 5


# ─────────────────────────────────────────────────────────────────────────────
#  Reusable widgets
# ─────────────────────────────────────────────────────────────────────────────

class Button:
    def __init__(self, rect, label, bg=ACCENT, fg=BLACK,
                 fsize=17, bold=False, radius=7):
        self.rect    = pygame.Rect(rect)
        self.label   = label
        self.bg      = bg
        self.fg      = fg
        self._fsize  = fsize
        self._bold   = bold
        self.radius  = radius
        self.enabled = True
        self._down   = False

    def draw(self, surf):
        if not self.enabled:
            bg = tuple(c // 3 for c in self.bg)
            fg = GREY
        elif self._down:
            bg = tuple(max(0, c - 50) for c in self.bg)
            fg = self.fg
        else:
            bg = self.bg
            fg = self.fg
        pygame.draw.rect(surf, bg,    self.rect, border_radius=self.radius)
        pygame.draw.rect(surf, WHITE, self.rect, width=1, border_radius=self.radius)
        t = font(self._fsize, self._bold).render(self.label, True, fg)
        surf.blit(t, t.get_rect(center=self.rect.center))

    def handle(self, ev) -> bool:
        if not self.enabled:
            return False
        if ev.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(ev.pos):
            self._down = True
        elif ev.type == pygame.MOUSEBUTTONUP:
            if self._down and self.rect.collidepoint(ev.pos):
                self._down = False
                return True
            self._down = False
        return False


class Toggle:
    """Simple ON/OFF pill toggle."""
    def __init__(self, x, y, label, value=False):
        self.rect  = pygame.Rect(x, y, 54, 28)
        self.label = label
        self.value = value

    def draw(self, surf):
        track = GREEN if self.value else DARK_MID
        pygame.draw.rect(surf, track, self.rect, border_radius=14)
        pygame.draw.rect(surf, WHITE, self.rect, width=1, border_radius=14)
        kx = self.rect.right - 18 if self.value else self.rect.left + 4
        pygame.draw.circle(surf, WHITE, (kx + 10, self.rect.centery), 10)
        lbl = font(14).render(self.label, True, WHITE)
        surf.blit(lbl, (self.rect.right + 8,
                        self.rect.centery - lbl.get_height() // 2))

    def handle(self, ev) -> bool:
        if ev.type == pygame.MOUSEBUTTONUP and self.rect.collidepoint(ev.pos):
            self.value = not self.value
            return True
        return False


class WaveformWidget:
    """Draws a mono waveform from a numpy array."""
    def __init__(self, rect, color=ACCENT, bg=DARK_MID):
        self.rect  = pygame.Rect(rect)
        self.color = color
        self.bg    = bg

    def draw(self, surf, audio: np.ndarray | None):
        pygame.draw.rect(surf, self.bg, self.rect, border_radius=4)
        if audio is None or len(audio) == 0:
            return
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        w, h    = self.rect.width, self.rect.height
        mid_y   = self.rect.centery
        indices = np.linspace(0, len(audio) - 1, w).astype(int)
        chunks  = audio[indices]
        for i, s in enumerate(chunks):
            amp = int(np.clip(s, -1, 1) * (h // 2 - 2))
            x   = self.rect.left + i
            pygame.draw.line(surf, self.color,
                             (x, mid_y - amp), (x, mid_y + amp))


def draw_header(surf, title: str):
    pygame.draw.rect(surf, DARK, (0, 0, SCREEN_W, HEADER_H))
    pygame.draw.line(surf, ACCENT, (0, HEADER_H - 1), (SCREEN_W, HEADER_H - 1), 1)
    t = font(21, bold=True).render(title, True, ACCENT)
    surf.blit(t, (PAD * 2, HEADER_H // 2 - t.get_height() // 2))


def draw_status(surf, msg: str, color=YELLOW):
    if msg:
        t = font(13).render(msg, True, color)
        surf.blit(t, (PAD, SCREEN_H - t.get_height() - 4))


# ─────────────────────────────────────────────────────────────────────────────
#  Main application
# ─────────────────────────────────────────────────────────────────────────────

class SamplerApp:
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
            (PAD, SCREEN_H - 78, SCREEN_W - PAD*2, 38))
        self.brf_btn_play = Button((PAD,          SCREEN_H - 46, 100, 36),
                                   "▶ Play", GREEN, BLACK, bold=True)
        self.brf_btn_midi = Button((PAD + 108,    SCREEN_H - 46, 120, 36),
                                   "MIDI", ACCENT, BLACK, bold=True)

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
            if f.lower().endswith('.sfz')
        ]
        if not sfz_candidates:
            self._status     = "No SFZ file found for this sample"
            self._status_col = RED
            return

        self._midi_sfz_path = sfz_candidates[0]
        self._midi_status   = "Starting MIDI engine…"
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
    #  Background workers
    # ──────────────────────────────────────────────────────────────────────

    def _record_worker(self):
        import sounddevice as sd
        import soundfile as sf
        import librosa

        recorder  = self.sampler.recorder
        recording = []

        def callback(indata, frames, time, status):
            chunk = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
            recording.append(chunk)
            recent = np.concatenate(recording[-8:]) if len(recording) >= 8 \
                     else np.concatenate(recording)
            self._live_chunk = recent

        device_id = recorder.find_tascam_device()
        native_sr = recorder.TASCAM_SAMPLE_RATE

        try:
            with sd.InputStream(
                device=device_id,
                channels=1,
                samplerate=native_sr,
                callback=callback,
            ):
                self._stop_record_flag.wait()
        except Exception as e:
            pygame.event.post(pygame.event.Event(
                EV_PROCESS_ERR, {'msg': f"Recording error: {e}"}))
            return

        if not recording:
            pygame.event.post(pygame.event.Event(
                EV_PROCESS_ERR, {'msg': "No audio captured"}))
            return

        audio = np.concatenate(recording)
        if native_sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=native_sr, target_sr=SAMPLE_RATE)
        sf.write(RAW_FILENAME, audio, SAMPLE_RATE)

        pygame.event.post(pygame.event.Event(EV_RECORD_DONE))

    def _process_worker(self, use_dtln: bool):
        try:
            results = self.sampler.process(RAW_FILENAME, CLEAN_FILENAME, use_dtln)
            self._results = results
            pygame.event.post(pygame.event.Event(EV_PROCESS_DONE))
        except Exception as e:
            import traceback
            traceback.print_exc()
            pygame.event.post(pygame.event.Event(
                EV_PROCESS_ERR, {'msg': str(e)}))

    def _save_worker(self, label: str):
        results = self._results

        def gen_spectrogram(path):
            self.sampler.plotter.plot_comparison(
                results['raw_audio'], results['clean_audio'],
                results['detected_pitch'], results['detected_pitch'], path)

        def gen_sfz(path, audio_filename):
            self.sampler.sfz_generator.save(
                path, audio_filename, results['detected_pitch'], label)

        self.sampler.file_manager.save_with_label(
            CLEAN_FILENAME, RAW_FILENAME, label,
            results['detected_pitch'], gen_spectrogram, gen_sfz)

        self._status     = f"Saved under '{label}'"
        self._status_col = GREEN
        self._go_home()

    def _midi_launch_worker(self, sfz_path: str):
        import time
        try:
            uid         = os.getuid()
            runtime_dir = f'/run/user/{uid}'

            env = os.environ.copy()
            env['PIPEWIRE_LATENCY']         = '256/44100'
            env['PATH']                     = '/usr/bin:/usr/local/bin:/bin:' + env.get('PATH', '')
            env['XDG_RUNTIME_DIR']          = runtime_dir
            env['PIPEWIRE_RUNTIME_DIR']     = runtime_dir
            env['PIPEWIRE_REMOTE']          = f'{runtime_dir}/pipewire-0'
            env['DBUS_SESSION_BUS_ADDRESS'] = f'unix:path={runtime_dir}/bus'
            env['PULSE_SERVER']             = f'unix:{runtime_dir}/pulse/native'

            subprocess.run(['killall', '-9', 'sfizz_jack'], capture_output=True, env=env)
            time.sleep(2.0)

            self._sfizz_proc = subprocess.Popen(
                ['pw-jack', SFIZZ_BINARY, '--jack_autoconnect=true', sfz_path],
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            for attempt in range(50):
                time.sleep(0.1)
                if self._sfizz_proc.poll() is not None:
                    stdout_out = self._sfizz_proc.stdout.read().decode()
                    stderr_out = self._sfizz_proc.stderr.read().decode()
                    raise RuntimeError(
                        f"sfizz_jack exited.\nstdout={stdout_out[:500]}\n"
                        f"stderr={stderr_out[:500]}")
                if attempt >= 30:
                    break

            subprocess.run(
                ['pw-link', 'MPK mini 3:MPK mini 3 MIDI 1', 'sfizz:input'],
                capture_output=True, text=True, env=env, timeout=5)

            pygame.event.post(pygame.event.Event(EV_MIDI_READY))

        except Exception as e:
            import traceback
            traceback.print_exc()
            pygame.event.post(pygame.event.Event(EV_MIDI_ERR, {'msg': str(e)}))

    def _midi_stop_sfizz(self):
        if self._sfizz_proc is not None:
            try:
                if self._sfizz_proc.stdin:
                    self._sfizz_proc.stdin.close()
                self._sfizz_proc.wait(timeout=2)
            except Exception:
                try:
                    self._sfizz_proc.kill()
                except Exception:
                    pass
            self._sfizz_proc = None
        subprocess.run(['killall', 'sfizz_jack'], capture_output=True)

    # ──────────────────────────────────────────────────────────────────────
    #  Event handling — per state
    # ──────────────────────────────────────────────────────────────────────

    def _handle_event(self, ev):
        # ── background thread results ──────────────────────────────────
        if ev.type == EV_RECORD_DONE:
            self._go('processing')
            threading.Thread(target=self._process_worker,
                             args=(self._use_dtln,), daemon=True).start()
            return
        if ev.type == EV_PROCESS_DONE:
            self.rev_waveform_data = self._results.get('clean_audio')
            self._go('review')
            return
        if ev.type == EV_PROCESS_ERR:
            self._status     = ev.dict.get('msg', 'Unknown error')
            self._status_col = RED
            self._go('home')   # _go not _go_home so status isn't cleared
            return
        if ev.type == EV_MIDI_READY:
            self._midi_status = "Keyboard active — play away!"
            return
        if ev.type == EV_MIDI_ERR:
            self._midi_status = f"Error: {ev.dict.get('msg', 'Unknown')}"
            return

        # ── global quit ────────────────────────────────────────────────
        if ev.type == pygame.QUIT:
            self._midi_stop_sfizz()
            pygame.quit(); sys.exit()
        if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
            self._midi_stop_sfizz()
            pygame.quit(); sys.exit()

        s = self._state

        # ── HOME ──────────────────────────────────────────────────────
        if s == 'home':
            if self.h_btn_quit.handle(ev):
                self._midi_stop_sfizz()
                pygame.quit(); sys.exit()
            if self.h_btn_record.handle(ev):
                self._use_dtln = self.h_dtln_toggle.value
                self._start_recording()
            if self.h_btn_test.handle(ev):
                self._use_dtln = self.h_dtln_toggle.value
                self._start_test_pick()
            if self.h_btn_browse.handle(ev):
                self._open_browser()
            self.h_dtln_toggle.handle(ev)

        # ── TEST FILE PICKER ──────────────────────────────────────────
        elif s == 'test_pick':
            if self.tp_btn_back.handle(ev):
                self._go_home()
            if self.tp_btn_run.handle(ev):
                sel = getattr(self, '_tp_selected', None)
                if sel is not None:
                    self._run_test_file(sel)
                else:
                    self._status     = "Tap a file first"
                    self._status_col = YELLOW
            if ev.type == pygame.MOUSEBUTTONUP:
                ITEM_H = 38
                for i, fname in enumerate(
                        self._test_files[self._test_scroll:self._test_scroll + 5]):
                    row = pygame.Rect(PAD, HEADER_H + 4 + i * ITEM_H,
                                      SCREEN_W - PAD*2, ITEM_H - 3)
                    if row.collidepoint(ev.pos):
                        self._tp_selected = fname
                        break

        # ── RECORDING ─────────────────────────────────────────────────
        elif s == 'recording':
            if self.rec_btn_stop.handle(ev):
                self._stop_recording()
                self._status     = "Stopped — processing…"
                self._status_col = YELLOW

        # ── REVIEW ────────────────────────────────────────────────────
        elif s == 'review':
            if self.rev_btn_label.handle(ev):
                preds = self._results.get('predictions', [])
                self._build_label_buttons(preds)
                self._go('label')
            if self.rev_btn_discard.handle(ev):
                self._status     = "Sample discarded"
                self._status_col = ORANGE
                self._go_home()

        # ── LABEL SELECT ──────────────────────────────────────────────
        elif s == 'label':
            for btn, lbl in self.lbl_buttons:
                if btn.handle(ev):
                    threading.Thread(target=self._save_worker,
                                     args=(lbl,), daemon=True).start()
                    self._status     = "Saving…"
                    self._status_col = YELLOW
                    return
            if self.lbl_btn_back.handle(ev):
                self._go('review')

        # ── BROWSER — folder list ──────────────────────────────────────
        elif s == 'browser':
            if self.br_btn_back.handle(ev):
                self._go_home()
            if self.br_btn_up.handle(ev):
                self._browser_folder_scroll = max(0, self._browser_folder_scroll - 1)
            if self.br_btn_dn.handle(ev):
                max_s = max(0, len(self._browser_folders) - 5)
                self._browser_folder_scroll = min(max_s, self._browser_folder_scroll + 1)
            # Row tap → open folder
            if ev.type == pygame.MOUSEBUTTONUP:
                ITEM_H = 48
                for i in range(min(5, len(self._browser_folders) - self._browser_folder_scroll)):
                    row = pygame.Rect(PAD, HEADER_H + 4 + i * ITEM_H,
                                      SCREEN_W - 50, ITEM_H - 3)
                    if row.collidepoint(ev.pos):
                        self._open_browser_folder(
                            self._browser_folders[self._browser_folder_scroll + i])
                        break

        # ── BROWSER FILES — file list inside folder ────────────────────
        elif s == 'browser_files':
            if self.brf_btn_back.handle(ev):
                self._go('browser')          # back to folder list
            if self.brf_btn_up.handle(ev):
                self._browser_scroll = max(0, self._browser_scroll - 1)
            if self.brf_btn_dn.handle(ev):
                max_s = max(0, len(self._browser_files) - 5)
                self._browser_scroll = min(max_s, self._browser_scroll + 1)
            if self.brf_btn_play.handle(ev):
                self._browser_play()
            if self.brf_btn_midi.handle(ev):
                self._go_midi_play()
            # Row tap → select file
            if ev.type == pygame.MOUSEBUTTONUP:
                ITEM_H = 38
                for i in range(min(5, len(self._browser_files) - self._browser_scroll)):
                    row = pygame.Rect(PAD, HEADER_H + 4 + i * ITEM_H,
                                      SCREEN_W - 50, ITEM_H - 3)
                    if row.collidepoint(ev.pos):
                        self._browser_sel   = self._browser_scroll + i
                        self._browser_audio = None
                        self._load_browser_waveform()
                        break

        # ── MIDI PLAY ─────────────────────────────────────────────────
        elif s == 'midi_play':
            if self.midi_btn_stop.handle(ev):
                self._stop_midi_play()

    def _build_label_buttons(self, predictions):
        self.lbl_buttons.clear()
        BW, BH = 143, 54
        positions = [
            (PAD,            HEADER_H + 10),
            (PAD + BW + 5,   HEADER_H + 10),
            (PAD + (BW+5)*2, HEADER_H + 10),
            (PAD,            HEADER_H + 72),
            (PAD + BW + 5,   HEADER_H + 72),
            (PAD + (BW+5)*2, HEADER_H + 72),
        ]
        for (label, conf), pos in zip(predictions[:6], positions):
            short = label if len(label) <= 16 else label[:14] + "…"
            btn   = Button((*pos, BW, BH), short, DARK_MID, WHITE, fsize=13)
            self.lbl_buttons.append((btn, label))

    def _browser_play(self):
        if self._browser_sel is None:
            return
        _, path = self._browser_files[self._browser_sel]
        try:
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
        except Exception as e:
            self._status     = f"Playback error: {e}"
            self._status_col = RED

    def _load_browser_waveform(self):
        if self._browser_sel is None:
            return
        _, path = self._browser_files[self._browser_sel]

        def worker():
            try:
                import soundfile as sf
                audio, _ = sf.read(path, dtype='float32', always_2d=True)
                audio = audio.mean(axis=1)
                rms = np.sqrt(np.mean(audio ** 2))
                if rms > 0:
                    audio = audio * (0.3 / rms)
                audio = np.clip(audio, -1.0, 1.0)
                self._browser_audio = audio
            except Exception as e:
                self._status        = f"Waveform load error: {e}"
                self._status_col    = RED
                self._browser_audio = None

        threading.Thread(target=worker, daemon=True).start()

    # ──────────────────────────────────────────────────────────────────────
    #  Drawing — per state
    # ──────────────────────────────────────────────────────────────────────

    def _draw(self):
        self.screen.fill(BLACK)
        s = self._state

        if   s == 'home':           self._draw_home()
        elif s == 'test_pick':      self._draw_test_pick()
        elif s == 'recording':      self._draw_recording()
        elif s == 'processing':     self._draw_processing()
        elif s == 'review':         self._draw_review()
        elif s == 'label':          self._draw_label()
        elif s == 'browser':        self._draw_browser()
        elif s == 'browser_files':  self._draw_browser_files()
        elif s == 'midi_play':      self._draw_midi_play()

        draw_status(self.screen, self._status, self._status_col)

    # ── HOME ──────────────────────────────────────────────────────────────
    def _draw_home(self):
        draw_header(self.screen, "Smart Sampler")
        self.h_btn_record.draw(self.screen)
        self.h_btn_test.draw(self.screen)
        self.h_btn_browse.draw(self.screen)
        self.h_dtln_toggle.draw(self.screen)
        self.h_btn_quit.draw(self.screen)

        mode = "TEST MODE" if TEST_MODE else "LIVE MODE"
        col  = ORANGE if TEST_MODE else GREEN
        t    = font(12).render(mode, True, col)
        self.screen.blit(t, (PAD, HEADER_H + 4))

    # ── TEST FILE PICKER ──────────────────────────────────────────────────
    def _draw_test_pick(self):
        draw_header(self.screen, "Select Test File")
        self.tp_btn_back.draw(self.screen)

        ITEM_H   = 38
        VISIBLE  = 5
        selected = getattr(self, '_tp_selected', None)

        for i, fname in enumerate(
                self._test_files[self._test_scroll:self._test_scroll + VISIBLE]):
            y   = HEADER_H + 4 + i * ITEM_H
            sel = (fname == selected)
            bg  = ACCENT_DIM if sel else DARK_MID
            pygame.draw.rect(self.screen, bg,
                             (PAD, y, SCREEN_W - PAD*2, ITEM_H - 3),
                             border_radius=5)
            t = font(14, bold=sel).render(fname, True, WHITE)
            self.screen.blit(t, (PAD + 8, y + (ITEM_H - 3 - t.get_height()) // 2))

        if len(self._test_files) > VISIBLE:
            info = font(12).render(
                f"{self._test_scroll + 1}–"
                f"{min(self._test_scroll + VISIBLE, len(self._test_files))}"
                f" of {len(self._test_files)}", True, GREY)
            self.screen.blit(info, (SCREEN_W//2 - info.get_width()//2, SCREEN_H - 52))

        self.tp_btn_run.enabled = selected is not None
        self.tp_btn_run.draw(self.screen)

    # ── RECORDING ─────────────────────────────────────────────────────────
    def _draw_recording(self):
        draw_header(self.screen, "Recording…")
        self.rec_waveform.draw(self.screen, self._live_chunk)

        level = 0.0
        if self._live_chunk is not None and len(self._live_chunk):
            level = float(np.sqrt(np.mean(self._live_chunk ** 2))) * 4
            level = min(level, 1.0)
        bar_w = int(self.rec_vu_rect.width * level)
        pygame.draw.rect(self.screen, DARK_MID, self.rec_vu_rect, border_radius=4)
        if bar_w:
            col = GREEN if level < 0.7 else (YELLOW if level < 0.9 else RED)
            pygame.draw.rect(self.screen, col,
                             (*self.rec_vu_rect.topleft, bar_w, self.rec_vu_rect.height),
                             border_radius=4)
        lbl = font(12).render("Input level", True, GREY)
        self.screen.blit(lbl, (PAD, self.rec_vu_rect.bottom + 4))

        self.rec_btn_stop.draw(self.screen)

    # ── PROCESSING ────────────────────────────────────────────────────────
    def _draw_processing(self):
        import math
        draw_header(self.screen, "Processing…")
        self._spin_angle = (self._spin_angle + 5) % 360
        cx, cy = SCREEN_W // 2, SCREEN_H // 2 + 10
        for i in range(10):
            a   = math.radians(self._spin_angle + i * 36)
            r   = 32
            x   = int(cx + r * math.cos(a))
            y   = int(cy + r * math.sin(a))
            alp = max(30, 255 - i * 24)
            pygame.draw.circle(self.screen, (0, alp, min(alp, 160)), (x, y), 5)
        msg = font(16).render("Please wait…", True, WHITE)
        self.screen.blit(msg, msg.get_rect(center=(cx, cy + 55)))

    # ── REVIEW ────────────────────────────────────────────────────────────
    def _draw_review(self):
        draw_header(self.screen, "Review")
        audio = getattr(self, 'rev_waveform_data', None)
        self.rev_waveform.draw(self.screen, audio)

        if self._results:
            import librosa
            pitch = self._results.get('detected_pitch')
            preds = self._results.get('predictions', [])
            note  = librosa.hz_to_note(pitch) if pitch else "N/A"
            hz    = f"{pitch:.1f} Hz" if pitch else ""

            rows = [
                ("Pitch", f"{note}  {hz}"),
                ("1st",   preds[0][0] if len(preds) > 0 else "N/A"),
                ("2nd",   preds[1][0] if len(preds) > 1 else "N/A"),
                ("3rd",   preds[2][0] if len(preds) > 2 else "N/A"),
            ]
            y = HEADER_H + 86
            for key, val in rows:
                k = font(13).render(f"{key}:", True, GREY)
                v = font(13, bold=True).render(val, True, WHITE)
                self.screen.blit(k, (PAD, y))
                self.screen.blit(v, (80, y))
                y += 22

        self.rev_btn_label.draw(self.screen)
        self.rev_btn_discard.draw(self.screen)

    # ── LABEL SELECT ──────────────────────────────────────────────────────
    def _draw_label(self):
        draw_header(self.screen, "Choose Label")
        for btn, _ in self.lbl_buttons:
            btn.draw(self.screen)
        self.lbl_btn_back.draw(self.screen)
        hint = font(12).render("Tap a YAMNet prediction to save", True, GREY)
        self.screen.blit(hint, (PAD, SCREEN_H - 20))

    # ── BROWSER — folder list ──────────────────────────────────────────────
    def _draw_browser(self):
        draw_header(self.screen, "Sample Browser")
        self.br_btn_back.draw(self.screen)
        self.br_btn_up.draw(self.screen)
        self.br_btn_dn.draw(self.screen)

        ITEM_H  = 48
        VISIBLE = 5

        if not self._browser_folders:
            t = font(15).render("No sample folders found.", True, GREY)
            self.screen.blit(t, t.get_rect(center=(SCREEN_W//2, SCREEN_H//2)))
        else:
            for i in range(min(VISIBLE,
                               len(self._browser_folders) - self._browser_folder_scroll)):
                idx  = self._browser_folder_scroll + i
                name = self._browser_folders[idx]
                y    = HEADER_H + 4 + i * ITEM_H

                pygame.draw.rect(self.screen, DARK_MID,
                                 (PAD, y, SCREEN_W - 50, ITEM_H - 3),
                                 border_radius=5)

                # folder icon + name
                icon = font(15).render("📁", True, YELLOW)
                lbl  = font(14, bold=True).render(name, True, WHITE)
                self.screen.blit(icon, (PAD + 6,  y + (ITEM_H - 3 - icon.get_height()) // 2))
                self.screen.blit(lbl,  (PAD + 30, y + (ITEM_H - 3 - lbl.get_height())  // 2))

                # sample count
                folder_path = os.path.join(SAMPLES_DIR, name)
                count   = sum(1 for _, _, fs in os.walk(folder_path)
                              for f in fs if f.lower().endswith('.wav'))
                count_t = font(12).render(f"{count} samples", True, GREY)
                self.screen.blit(count_t,
                                 (SCREEN_W - 55 - count_t.get_width(),
                                  y + (ITEM_H - 3 - count_t.get_height()) // 2))

    # ── BROWSER FILES — file list inside folder ────────────────────────────
    def _draw_browser_files(self):
        draw_header(self.screen, f"📁 {self._browser_sel_folder or ''}")
        self.brf_btn_back.draw(self.screen)
        self.brf_btn_up.draw(self.screen)
        self.brf_btn_dn.draw(self.screen)

        ITEM_H  = 38
        VISIBLE = 5

        if not self._browser_files:
            t = font(15).render("No WAV files in this folder.", True, GREY)
            self.screen.blit(t, t.get_rect(center=(SCREEN_W//2, SCREEN_H//2 - 20)))
        else:
            for i in range(min(VISIBLE,
                               len(self._browser_files) - self._browser_scroll)):
                idx     = self._browser_scroll + i
                name, _ = self._browser_files[idx]
                y       = HEADER_H + 4 + i * ITEM_H
                sel     = (idx == self._browser_sel)
                bg      = ACCENT_DIM if sel else DARK_MID
                pygame.draw.rect(self.screen, bg,
                                 (PAD, y, SCREEN_W - 50, ITEM_H - 3),
                                 border_radius=5)
                t = font(13, bold=sel).render(name, True, WHITE)
                self.screen.blit(t, (PAD + 6, y + (ITEM_H - 3 - t.get_height()) // 2))

        # Waveform + buttons only when a file is selected
        if self._browser_sel is not None:
            self.brf_waveform.draw(self.screen, self._browser_audio)
            self.brf_btn_play.draw(self.screen)
            self.brf_btn_midi.draw(self.screen)

            # Playback progress bar
            if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
                pos_ms = pygame.mixer.music.get_pos()
                _, path = self._browser_files[self._browser_sel]
                try:
                    import soundfile as sf
                    info  = sf.info(path)
                    total = info.duration * 1000
                    ratio = min(pos_ms / total, 1.0) if total > 0 else 0
                    bar_x = self.brf_btn_midi.rect.right + 8
                    bar_w = SCREEN_W - bar_x - PAD
                    bar_y = self.brf_btn_play.rect.centery
                    pygame.draw.rect(self.screen, DARK_MID,
                                     (bar_x, bar_y - 5, bar_w, 10), border_radius=3)
                    pygame.draw.rect(self.screen, ACCENT,
                                     (bar_x, bar_y - 5, int(bar_w * ratio), 10),
                                     border_radius=3)
                except Exception:
                    pass

    # ── MIDI PLAY ─────────────────────────────────────────────────────────
    def _draw_midi_play(self):
        draw_header(self.screen, "MIDI Play")
        self.midi_waveform.draw(self.screen, self._browser_audio)

        sfz_name = os.path.basename(self._midi_sfz_path)
        name_t   = font(13).render(sfz_name, True, GREY)
        self.screen.blit(name_t, (PAD, HEADER_H + 96))

        col = (GREEN  if "active" in self._midi_status else
               RED    if "Error"  in self._midi_status else YELLOW)
        status_t = font(15, bold=True).render(self._midi_status, True, col)
        self.screen.blit(status_t, status_t.get_rect(
            center=(SCREEN_W // 2, HEADER_H + 120)))

        if "active" in self._midi_status:
            hint = font(13).render("Play notes on your MIDI keyboard", True, GREY)
            self.screen.blit(hint, hint.get_rect(
                center=(SCREEN_W // 2, HEADER_H + 142)))

        self.midi_btn_stop.draw(self.screen)

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