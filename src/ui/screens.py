"""
ScreensMixin — all drawing / screen rendering methods for SamplerApp.
"""
import math
import os

import numpy as np
import pygame
from pygame.gfxdraw import box

from ..config import TEST_MODE, SAMPLES_DIR
from .theme import (
    SCREEN_W, SCREEN_H,
    BLACK, DARK, DARK_MID, GREY, WHITE,
    ACCENT, ACCENT_DIM, RED, GREEN, YELLOW, ORANGE,
    HEADER_H, PAD, font,
)
from .widgets import draw_header, draw_status


class ScreensMixin:
    """Mixin providing all screen drawing methods for SamplerApp."""

    def _draw(self):
        self.screen.fill(BLACK)
        s = self._state

        if   s == 'home':           self._draw_home()
        elif s == 'pre_record':     self._draw_pre_record()
        elif s == 'test_pick':      self._draw_test_pick()
        elif s == 'recording':      self._draw_recording()
        elif s == 'processing':     self._draw_processing()
        elif s == 'review':         self._draw_review()
        elif s == 'label':          self._draw_label()
        elif s == 'browser':        self._draw_browser()
        elif s == 'browser_files':  self._draw_browser_files()
        elif s == 'midi_play':      self._draw_midi_play()

        if self._confirm_delete:    self._draw_confirm_overlay()
        if self._dtln_warning:       self._draw_dtln_warning_overlay()

        draw_status(self.screen, self._status, self._status_col)

    # ── HOME ──────────────────────────────────────────────────────────────
    def _draw_home(self):
        draw_header(self.screen, "Smart Sampler")
        self.h_btn_record.draw(self.screen)
        self.h_btn_test.draw(self.screen)
        self.h_btn_browse.draw(self.screen)
        self.h_dtln_toggle.draw(self.screen)
        self.h_btn_quit.draw(self.screen)

        self.h_btn_midi.bg    = ACCENT
        self.h_btn_midi.fg    = BLACK
        self.h_btn_midi.label = "MIDI Browse"
        self.h_btn_midi.draw(self.screen)

        mode = "TEST MODE" if TEST_MODE else "LIVE MODE"
        col  = ORANGE if TEST_MODE else GREEN
        t    = font(12).render(mode, True, col)
        self.screen.blit(t, (PAD, HEADER_H + 4))

        # ── PRE-RECORD ────────────────────────────────────────────────────────
    def _draw_pre_record(self):
        draw_header(self.screen, "Record Sample")
        self.pr_btn_back.draw(self.screen)

        hint1 = font(15).render("Press Record when ready.", True, WHITE)
        hint2 = font(13).render("Recording will begin immediately.", True, GREY)
        self.screen.blit(hint1, hint1.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 - 30)))
        self.screen.blit(hint2, hint2.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2)))

        self.pr_btn_record.draw(self.screen)

    # ── TEST FILE PICKER ──────────────────────────────────────────────────
    def _draw_test_pick(self):
        draw_header(self.screen, "Imported Samples")
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
        self.lbl_loop_toggle.draw(self.screen) 
        hint = font(12).render("Tap a YAMNet prediction to save", True, GREY)
        self.screen.blit(hint, (PAD, SCREEN_H - 20))

    # ── BROWSER — folder list ──────────────────────────────────────────────
    def _draw_browser(self):
        if self._midi_browser_mode:
            draw_header(self.screen, "MIDI Browse")
            # Engine status: starting (yellow) / ready (green) / error (red).
            # This is the only feedback the user gets that the engine launched.
            if self._midi_engine_loading:
                st_col, st_txt = YELLOW, "Engine starting — choose a folder while you wait…"
            elif self._midi_engine_active:
                st_col, st_txt = GREEN,  "Engine ready  ●  open a folder and tap a sample"
            else:
                st_col, st_txt = RED,    "Engine error — try going back and pressing MIDI Browse again"
            st = font(11).render(st_txt, True, st_col)
            self.screen.blit(st, (PAD, HEADER_H + 4))
            STATUS_H = 18   # pixels reserved for the status line
            VISIBLE  = 4    # one fewer row to compensate
        else:
            draw_header(self.screen, "Sample Browser")
            STATUS_H = 0
            VISIBLE  = 5

        self.br_btn_back.draw(self.screen)
        self.br_btn_up.draw(self.screen)
        self.br_btn_dn.draw(self.screen)

        ITEM_H = 48

        if not self._browser_folders:
            t = font(15).render("No sample folders found.", True, GREY)
            self.screen.blit(t, t.get_rect(center=(SCREEN_W//2, SCREEN_H//2)))
        else:
            for i in range(min(VISIBLE,
                               len(self._browser_folders) - self._browser_folder_scroll)):
                idx  = self._browser_folder_scroll + i
                name = self._browser_folders[idx]
                # STATUS_H shifts every row down past the engine-status line.
                y    = HEADER_H + 4 + STATUS_H + i * ITEM_H
                sel  = (idx == self._browser_sel_folder_idx)
                bg   = ACCENT_DIM if sel else DARK_MID

                pygame.draw.rect(self.screen, bg,
                                 (PAD, y, SCREEN_W - 54, ITEM_H - 3),
                                 border_radius=5)

                icon = font(15).render("📁", True, YELLOW)
                lbl  = font(14, bold=True).render(name, True, WHITE)
                self.screen.blit(icon, (PAD + 6,  y + (ITEM_H - 3 - icon.get_height()) // 2))
                self.screen.blit(lbl,  (PAD + 30, y + (ITEM_H - 3 - lbl.get_height())  // 2))

                folder_path = os.path.join(SAMPLES_DIR, name)
                count   = sum(1 for _, _, fs in os.walk(folder_path)
                              for f in fs if f.lower().endswith('.wav'))
                count_t = font(12).render(f"{count} samples", True, GREY)
                self.screen.blit(count_t,
                                 (SCREEN_W - 55 - count_t.get_width(),
                                  y + (ITEM_H - 3 - count_t.get_height()) // 2))

        if self._browser_sel_folder_idx is not None:
            self.br_btn_open.draw(self.screen)
            # Delete is suppressed in MIDI mode — no reason to delete folders
            # mid-session, and keeping the button out reduces accident risk.
            if not self._midi_browser_mode:
                self.br_btn_delete.draw(self.screen)

    # ── BROWSER FILES — file list inside folder ────────────────────────────
    def _draw_browser_files(self):
        # In MIDI mode the header gets a green dot prefix so it's obvious the
        # engine is running and this isn't a regular browse session.
        if self._midi_browser_mode:
            draw_header(self.screen, f"MIDI ● {self._browser_sel_folder or ''}")
        else:
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
                # In MIDI mode the selected (= currently loaded) row is green
                # so the user can see at a glance which sample is playing.
                # In normal browse mode the selected row stays amber (ACCENT_DIM).
                if self._midi_browser_mode and sel:
                    bg = GREEN
                elif sel:
                    bg = ACCENT_DIM
                else:
                    bg = DARK_MID
                pygame.draw.rect(self.screen, bg,
                                 (PAD, y, SCREEN_W - 54, ITEM_H - 3),
                                 border_radius=5)
                # Use black text on green so contrast stays readable.
                fg = BLACK if (self._midi_browser_mode and sel) else WHITE
                t = font(13, bold=sel).render(name, True, fg)
                self.screen.blit(t, (PAD + 6, y + (ITEM_H - 3 - t.get_height()) // 2))

        if self._browser_sel is not None:
            # ── MIDI status indicator (above waveform) ───────────────────
            # In normal mode: shown only while the engine is active.
            # In MIDI mode: always shown so the user knows the engine state
            # (starting… / loaded filename / error).
            if self._midi_browser_mode:
                if self._midi_engine_loading:
                    ind_col, ind_txt = YELLOW, "● Engine starting — sample will load when ready"
                elif self._midi_engine_active:
                    sfz_name = os.path.basename(self._midi_live_sfz) if self._midi_live_sfz else "—"
                    if len(sfz_name) > 32:
                        sfz_name = sfz_name[:30] + "…"
                    ind_col, ind_txt = GREEN, f"● Loaded: {sfz_name}"
                else:
                    ind_col, ind_txt = RED, "● Engine not running"
                ind = font(11).render(ind_txt, True, ind_col)
                self.screen.blit(ind, (PAD, SCREEN_H - 92))
            elif self._midi_engine_active:
                # Normal mode legacy indicator.
                sfz_name = os.path.basename(self._midi_live_sfz) if self._midi_live_sfz else "no sample loaded"
                if len(sfz_name) > 32:
                    sfz_name = sfz_name[:30] + "…"
                ind = font(11).render(f"● MIDI: {sfz_name}", True, GREEN)
                self.screen.blit(ind, (PAD, SCREEN_H - 92))

            self.brf_waveform.draw(self.screen, self._browser_audio)
            self.brf_btn_play.draw(self.screen)

            if self._midi_browser_mode:
                # The MIDI button is replaced by a plain hint — tapping the
                # file row itself is the load action in MIDI mode.
                # The hint sits in the same horizontal slot the MIDI button
                # normally occupies so the layout feels balanced.
                hint = font(12).render("tap row to load", True, GREY)
                self.screen.blit(hint, hint.get_rect(
                    center=(PAD + 108 + 50, self.brf_btn_play.rect.centery)))
            else:
                # Normal mode: update MIDI button appearance to reflect engine state.
                # Green + "Load MIDI" when engine is active (hot-swap).
                # Amber + "MIDI" when engine is off (will start + load).
                if self._midi_engine_active:
                    self.brf_btn_midi.bg    = GREEN
                    self.brf_btn_midi.fg    = BLACK
                    self.brf_btn_midi.label = "Load MIDI"
                else:
                    self.brf_btn_midi.bg    = ACCENT
                    self.brf_btn_midi.fg    = BLACK
                    self.brf_btn_midi.label = "MIDI"
                self.brf_btn_midi.draw(self.screen)

            self.brf_btn_delete.draw(self.screen)
            self.brf_vel_toggle.draw(self.screen)

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

        # Read-only velocity indicator
        if self._midi_vel_fixed:
            vel_t = font(13).render("Velocity: Fixed (full)", True, GREEN)
        else:
            vel_t = font(13).render("Velocity: Dynamic", True, GREY)
        self.screen.blit(vel_t, (PAD, HEADER_H + 114))

        col = (GREEN  if "active" in self._midi_status else
               RED    if "Error"  in self._midi_status else YELLOW)
        status_t = font(15, bold=True).render(self._midi_status, True, col)
        self.screen.blit(status_t, status_t.get_rect(
            center=(SCREEN_W // 2, HEADER_H + 138)))

        if "active" in self._midi_status:
            hint = font(13).render("Play notes on your MIDI keyboard", True, GREY)
            self.screen.blit(hint, hint.get_rect(
                center=(SCREEN_W // 2, HEADER_H + 158)))

        self.midi_btn_stop.draw(self.screen)

        # ── DELETE CONFIRMATION OVERLAY ─────────────────────────────────────────────────────────
    def _draw_confirm_overlay(self):
        # Dim the background
        overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        # Dialog box
        box = pygame.Rect(PAD, SCREEN_H // 2 - 52, SCREEN_W - PAD * 2, 120)
        pygame.draw.rect(self.screen, DARK_MID, box, border_radius=8)
        pygame.draw.rect(self.screen, RED, box, width=2, border_radius=8)

        name = self._confirm_delete.get('name', '')
        line1 = font(14, bold=True).render("Are you sure you want to delete:", True, WHITE)
        line2 = font(13).render(f'"{name}"', True, YELLOW)
        self.screen.blit(line1, line1.get_rect(centerx=SCREEN_W // 2, top=box.top + 10))
        self.screen.blit(line2, line2.get_rect(centerx=SCREEN_W // 2, top=box.top + 30))

        self.cont_btn_yes.draw(self.screen)
        self.cont_btn_no.draw(self.screen)

    # ── DTLN WARNING OVERLAY ──────────────────────────────────────────────
    def _draw_dtln_warning_overlay(self):
        overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        box = pygame.Rect(PAD, SCREEN_H // 2 - 80, SCREEN_W - PAD * 2, 160)
        pygame.draw.rect(self.screen, DARK_MID, box, border_radius=8)
        pygame.draw.rect(self.screen, YELLOW, box, width=2, border_radius=8)

        lines = [
            ("DTLN Noise Removal", True,  14, YELLOW, box.top + 10),
            ("DTLN removes background noise",  False, 12, WHITE,  box.top + 32),
            ("and works best with vocal",      False, 12, WHITE,  box.top + 48),
            ("samples only. Non-vocal samples",False, 12, WHITE,  box.top + 64),
            ("may sound degraded.",            False, 12, WHITE,   box.top + 80),
        ]
        for text, bold, size, col, y in lines:
            t = font(size, bold).render(text, True, col)
            self.screen.blit(t, t.get_rect(centerx=SCREEN_W // 2, top=y))

        # Checkbox
        cb = self.dtln_warn_cb_rect
        pygame.draw.rect(self.screen, DARK_MID, cb, border_radius=3)
        pygame.draw.rect(self.screen, WHITE, cb, width=1, border_radius=3)
        if self._dtln_warn_no_show:
            pygame.draw.line(self.screen, GREEN,
                             (cb.left + 3, cb.centery), (cb.centerx - 1, cb.bottom - 4), 2)
            pygame.draw.line(self.screen, GREEN,
                             (cb.centerx - 1, cb.bottom - 4), (cb.right - 3, cb.top + 4), 2)
        lbl = font(12).render("Don't show this again", True, GREY)
        self.screen.blit(lbl, (cb.right + 6, cb.centery - lbl.get_height() // 2))

        self.dtln_btn_ok.draw(self.screen)
        self.dtln_btn_cancel.draw(self.screen)
