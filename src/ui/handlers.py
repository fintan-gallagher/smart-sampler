"""
HandlersMixin — pygame event handling and browser helper methods for SamplerApp.
"""
import threading

import numpy as np
import pygame

from .events import (
    EV_RECORD_DONE, EV_PROCESS_DONE, EV_PROCESS_ERR,
    EV_MIDI_READY, EV_MIDI_ERR,
)
from .theme import (
    SCREEN_W, HEADER_H, PAD,
    ACCENT_DIM, DARK_MID, WHITE,
    RED, YELLOW, ORANGE,
    font,
)
from .widgets import Button


class HandlersMixin:
    """Mixin providing event handling and browser helper methods for SamplerApp."""

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
            pygame.quit(); __import__('sys').exit()
        if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
            self._midi_stop_sfizz()
            pygame.quit(); __import__('sys').exit()

        s = self._state

        # ── HOME ──────────────────────────────────────────────────────
        if s == 'home':
            if self.h_btn_quit.handle(ev):
                self._midi_stop_sfizz()
                pygame.quit(); __import__('sys').exit()
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
                self._go('browser')
            if self.brf_btn_up.handle(ev):
                self._browser_scroll = max(0, self._browser_scroll - 1)
            if self.brf_btn_dn.handle(ev):
                max_s = max(0, len(self._browser_files) - 5)
                self._browser_scroll = min(max_s, self._browser_scroll + 1)
            if self.brf_btn_play.handle(ev):
                self._browser_play()
            if self.brf_btn_midi.handle(ev):
                self._go_midi_play()
            self.brf_vel_toggle.handle(ev)
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
