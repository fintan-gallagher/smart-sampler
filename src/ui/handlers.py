"""
HandlersMixin — pygame event handling and browser helper methods for SamplerApp.
"""
import os
import threading
import shutil

import numpy as np
import pygame

from src.config import SAMPLES_DIR

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
from .prefs import save_prefs


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
            # Engine is up and MIDI keyboard is connected.
            # Mark the engine as active so the home button turns green and
            # browser_files shows the MIDI indicator.
            self._midi_status         = "Keyboard active — play away!"
            self._midi_engine_active  = True
            self._midi_engine_loading = False

            # If the user clicked the MIDI button in browser_files while the
            # engine was still starting (or off), _midi_autoload was set.
            # Now that the engine is ready, load that sample immediately.
            if self._midi_autoload:
                self._midi_autoload = False
                self._load_sample_into_engine()
            return

        if ev.type == EV_MIDI_ERR:
            # Startup failed — reset all engine flags so the button goes back
            # to its dimmed 'off' state and the user can try again.
            self._midi_status         = f"Error: {ev.dict.get('msg', 'Unknown')}"
            self._midi_engine_active  = False
            self._midi_engine_loading = False
            self._midi_autoload       = False
            return

        # ── global quit ────────────────────────────────────────────────
        if ev.type == pygame.QUIT:
            self._midi_stop_sfizz()
            pygame.quit(); __import__('sys').exit()
        if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
            self._midi_stop_sfizz()
            pygame.quit(); __import__('sys').exit()

        s = self._state

        # ── DTLN WARNING OVERLAY ───────────────────────────────────────────
        if self._dtln_warning:
            if self.dtln_btn_cancel.handle(ev):
                self._dtln_warning = False
            elif self.dtln_btn_ok.handle(ev):
                self.h_dtln_toggle.value = True
                if self._dtln_warn_no_show:
                    self._prefs['dtln_warn_dismissed'] = True
                    save_prefs(self._prefs)
                self._dtln_warning = False
            elif (ev.type == pygame.MOUSEBUTTONUP
                  and self.dtln_warn_cb_rect.collidepoint(ev.pos)):
                self._dtln_warn_no_show = not self._dtln_warn_no_show
            return

        # ── DELETE CONFIRMATION OVERLAY ────────────────────────────────
        if self._confirm_delete is not None:
            if self.cont_btn_no.handle(ev):
                self._confirm_delete = None
            elif self.cont_btn_yes.handle(ev):
                action = self._confirm_delete.get('action')
                if action == 'delete_folder':
                    folder = self._confirm_delete['folder']
                    shutil.rmtree(os.path.join(SAMPLES_DIR, folder))
                    self._confirm_delete = None
                    self._open_browser()
                elif action == 'delete_file':
                    path = self._confirm_delete['path']
                    pygame.mixer.music.stop()
                    os.remove(path)
                    self._confirm_delete = None
                    self._open_browser_folder(self._browser_sel_folder)
            return   # block all other input while overlay is showing

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
            if self.h_btn_midi.handle(ev):
                # Navigate to the sample browser in MIDI mode.
                # The engine starts in the background while the user browses
                # folders.  Pressing Back from the folder list stops the engine
                # and returns here.  _enter_midi_browse() handles stopping any
                # stale engine instance before starting a fresh one.
                self._enter_midi_browse()
            if self.h_dtln_toggle.handle(ev):
                if self.h_dtln_toggle.value and not self._prefs.get('dtln_warn_dismissed'):
                    # undo the flip — overlay will confirm it
                    self.h_dtln_toggle.value = False
                    self._dtln_warn_no_show  = False
                    self._dtln_warning       = True

        # ── PRE-RECORD ────────────────────────────────────────────────────
        elif s == 'pre_record':
            if self.pr_btn_back.handle(ev):
                self._go_home()
            if self.pr_btn_record.handle(ev):
                self._begin_recording()

        # ── TEST FILE PICKER ─────────────────────────────────────────
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
                    self._go('processing')
                    self._status = "Saving..." 
                    self._status_col = YELLOW
                    threading.Thread(target=self._save_worker,
                                    args=(lbl,self.lbl_loop_toggle.value), 
                                    daemon=True).start()
                    return
            if self.lbl_btn_back.handle(ev):
                self._go('review')
            self.lbl_loop_toggle.handle(ev)

        # ── BROWSER — folder list ──────────────────────────────────────
        elif s == 'browser':
            if self.br_btn_back.handle(ev):
                if self._midi_browser_mode:
                    # Leaving MIDI mode: tear down the engine before going home.
                    # _exit_midi_browse() clears _midi_browser_mode, stops sfizz,
                    # and calls _go_home() — all in one step.
                    self._exit_midi_browse()
                else:
                    self._go_home()
            if self.br_btn_up.handle(ev):
                self._browser_folder_scroll = max(0, self._browser_folder_scroll - 1)
                self._browser_sel_folder_idx = None
            if self.br_btn_dn.handle(ev):
                max_s = max(0, len(self._browser_folders) - 5)
                self._browser_folder_scroll = min(max_s, self._browser_folder_scroll + 1)
                self._browser_sel_folder_idx = None  
            if self.br_btn_open.handle(ev):                                        
                if self._browser_sel_folder_idx is not None:                       
                    self._open_browser_folder(                                     
                        self._browser_folders[self._browser_sel_folder_idx])       
            if self.br_btn_delete.handle(ev):
                if self._browser_sel_folder_idx is not None:
                    folder = self._browser_folders[self._browser_sel_folder_idx]
                    self._confirm_delete = {
                        'action': 'delete_folder',
                        'folder': folder,
                        'name':   folder,
                    }
            if ev.type == pygame.MOUSEBUTTONUP:
                ITEM_H = 48
                # In MIDI mode the engine-status line above the folder list
                # shifts every row down by STATUS_H pixels.  The hit-rects
                # must use the same offset as the draw code or clicks land on
                # the wrong (invisible) position and require multiple taps.
                STATUS_H = 18 if self._midi_browser_mode else 0
                VISIBLE  = 4  if self._midi_browser_mode else 5
                for i in range(min(VISIBLE, len(self._browser_folders) - self._browser_folder_scroll)):
                    row = pygame.Rect(PAD, HEADER_H + 4 + STATUS_H + i * ITEM_H,
                                      SCREEN_W - 54, ITEM_H - 3)
                    if row.collidepoint(ev.pos):
                        self._browser_sel_folder_idx = self._browser_folder_scroll + i
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
            if self.brf_btn_delete.handle(ev):
                if self._browser_sel is not None:
                    name, path = self._browser_files[self._browser_sel]
                    self._confirm_delete = {
                        'action': 'delete_file',
                        'path':   path,
                        'name':   name,
                    }
            if self.brf_vel_toggle.handle(ev):
                # If the engine is already running, re-generate the SFZ with
                # the new velocity mode and hot-swap it in immediately so the
                # user hears the change without touching any other button.
                if self._midi_engine_active and self._browser_sel is not None:
                    self._load_sample_into_engine()
            if ev.type == pygame.MOUSEBUTTONUP:
                ITEM_H = 38
                for i in range(min(5, len(self._browser_files) - self._browser_scroll)):
                    row = pygame.Rect(PAD, HEADER_H + 4 + i * ITEM_H,
                                      SCREEN_W - 54, ITEM_H - 3)
                    if row.collidepoint(ev.pos):
                        self._browser_sel   = self._browser_scroll + i
                        self._browser_audio = None
                        self._load_browser_waveform()
                        if self._midi_browser_mode:
                            # MIDI mode: every tap immediately loads the sample.
                            # _go_midi_play() handles whichever engine state we're in:
                            #   active  → instant hot-swap via load_instrument stdin cmd
                            #   loading → sets _midi_autoload; fires on EV_MIDI_READY
                            #   off     → starts engine + sets _midi_autoload (edge case)
                            self._go_midi_play()
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
