"""
Reusable pygame UI widgets and shared drawing helpers.
"""
import numpy as np
import pygame

from .theme import (
    SCREEN_W, SCREEN_H,
    BLACK, DARK, DARK_MID, MID, GREY, WHITE,
    ACCENT, ACCENT_DIM, RED, RED_DIM, GREEN, YELLOW, ORANGE,
    HEADER_H, PAD, font,
)


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
