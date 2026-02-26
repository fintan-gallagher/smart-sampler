"""Visual constants for Smart Sampler UI — designed for 480x320"""
import pygame

SCREEN_W = 480
SCREEN_H = 320
FPS      = 30

# Colours
BLACK      = (10,  10,  15)
DARK       = (25,  25,  35)
DARK_MID   = (40,  40,  55)
MID        = (65,  65,  85)
GREY       = (110, 110, 130)
WHITE      = (235, 235, 245)
ACCENT     = (0,   200, 160)
ACCENT_DIM = (0,   120, 95)
RED        = (210, 55,  55)
RED_DIM    = (130, 30,  30)
GREEN      = (55,  195, 75)
YELLOW     = (225, 180, 0)
ORANGE     = (225, 125, 0)

HEADER_H = 48
PAD      = 10

_fonts: dict = {}

def font(size: int, bold: bool = False) -> pygame.font.Font:
    key = (size, bold)
    if key not in _fonts:
        _fonts[key] = pygame.font.SysFont('dejavusans', size, bold=bold)
    return _fonts[key]