"""
WorkersMixin — all background-thread methods for SamplerApp.
"""
import os
import subprocess
import time

import numpy as np
import pygame

from ..config import RAW_FILENAME, CLEAN_FILENAME, SAMPLE_RATE, SFIZZ_BINARY
from .events import (
    EV_RECORD_DONE, EV_PROCESS_DONE, EV_PROCESS_ERR,
    EV_MIDI_READY, EV_MIDI_ERR,
)


class WorkersMixin:
    """Mixin providing all background worker methods for SamplerApp."""

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

    def _save_worker(self, label: str, loop: bool = True):
        results = self._results

        def gen_spectrogram(path):
            self.sampler.plotter.plot_comparison(
                results['raw_audio'], results['clean_audio'],
                results['detected_pitch'], results['detected_pitch'], path)

        def gen_sfz(path, audio_filename):
            self.sampler.sfz_generator.save(
                path, audio_filename, results['detected_pitch'], label,
                fixed_velocity=self.brf_vel_toggle.value, loop=loop)

        self.sampler.file_manager.save_with_label(
            CLEAN_FILENAME, RAW_FILENAME, label,
            results['detected_pitch'], gen_spectrogram, gen_sfz)

        self._status     = f"Saved under '{label}'"
        self._status_col = (0, 200, 80)   # GREEN — avoids circular theme import
        self._go_home()

    def _midi_launch_worker(self, sfz_path: str | None):
        """Start sfizz_jack as a persistent JACK client.

        sfz_path is now optional.  When None, sfizz_jack launches with no
        initial instrument loaded — subsequent 'load_instrument <path>'
        commands sent over stdin will hot-swap samples without restarting.
        When a path is provided (legacy midi_play flow) it is appended to the
        command as before.
        """
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

            # Kill any existing instance and wait until fully gone
            subprocess.run(['killall', '-9', 'sfizz_jack'], capture_output=True, env=env)
            for _ in range(20):
                result = subprocess.run(['pgrep', 'sfizz_jack'], capture_output=True)
                if result.returncode != 0:
                    break
                time.sleep(0.2)
            time.sleep(0.5)

            # Build the command.  The SFZ path is the only positional argument
            # and is optional per the sfizz_jack man page — omitting it lets
            # the engine start idle, ready to receive load_instrument commands.
            cmd = ['pw-jack', SFIZZ_BINARY, '--jack_autoconnect=true']
            if sfz_path:
                cmd.append(sfz_path)

            last_err = None
            for attempt in range(2):          # retry once on failure
                if attempt > 0:
                    time.sleep(2.0)

                self._sfizz_proc = subprocess.Popen(
                    cmd,
                    env=env,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

                exited = False
                for _ in range(30):
                    time.sleep(0.1)
                    if self._sfizz_proc.poll() is not None:
                        stdout_out = self._sfizz_proc.stdout.read().decode()
                        stderr_out = self._sfizz_proc.stderr.read().decode()
                        last_err = (f"sfizz_jack exited.\nstdout={stdout_out[:500]}\n"
                                    f"stderr={stderr_out[:500]}")
                        exited = True
                        break

                if not exited:
                    break  # process still running — success
            else:
                raise RuntimeError(last_err)  # both attempts failed

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
