"""
Custom pygame event constants shared across all UI modules.
"""
import pygame

EV_RECORD_DONE  = pygame.USEREVENT + 1
EV_PROCESS_DONE = pygame.USEREVENT + 2
EV_PROCESS_ERR  = pygame.USEREVENT + 3
EV_MIDI_READY   = pygame.USEREVENT + 4
EV_MIDI_ERR     = pygame.USEREVENT + 5
