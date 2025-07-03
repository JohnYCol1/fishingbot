import time
import queue
import threading
import wave
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
import pyautogui
import tkinter as tk
from tkinter import filedialog

@dataclass
class Settings:
    threshold_multiplier: float = 1.5
    similarity_threshold: float = 0.8
    failsafe_timer: float = 0.5
    device: int | None = None
    sound_path: str | None = None

settings = Settings()

cue_rate: int | None = None
cue_audio: np.ndarray | None = None

pyautogui.FAILSAFE = True


def choose_audio_device():
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            print(f"{idx}: {dev['name']}")
    while True:
        try:
            sel = int(input("Select audio input device index: "))
            if 0 <= sel < len(devices):
                settings.device = sel
                break
        except ValueError:
            pass
        print("Invalid selection. Try again.")


def load_sound_cue(path: str):
    """Load a WAV file to use as the sound cue."""
    global cue_rate, cue_audio
    with wave.open(path, 'rb') as wf:
        cue_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    cue_audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    if np.max(np.abs(cue_audio)) != 0:
        cue_audio /= np.max(np.abs(cue_audio))
    settings.sound_path = path


def listen_for_sound(stop_event: threading.Event):
    """Listen to the microphone and wait for the configured sound cue."""
    if cue_audio is None:
        print("No sound cue loaded.")
        return

    q: queue.Queue[np.ndarray] = queue.Queue()
    buffer = np.array([], dtype=np.float32)

    def callback(indata, frames, time_info, status):
        if status:
            print(status)
        q.put(indata[:, 0].copy())

    with sd.InputStream(device=settings.device, channels=1, samplerate=cue_rate, callback=callback):
        while not stop_event.is_set():
            data = q.get()
            buffer = np.concatenate((buffer, data))
            if len(buffer) >= len(cue_audio):
                segment = buffer[-len(cue_audio):]
                corr = np.dot(segment, cue_audio)
                norm = np.linalg.norm(segment) * np.linalg.norm(cue_audio)
                similarity = (corr / norm) * settings.threshold_multiplier
                if similarity >= settings.similarity_threshold:
                    return
                buffer = buffer[-len(cue_audio):]


def perform_actions():
    # Placeholder actions similar to AutoHotkey script
    pyautogui.press('space')
    pyautogui.moveRel(10, 0, duration=0.1)


class MacroBotGUI:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Macro Bot")

        tk.Label(self.root, text="Threshold Multiplier").pack()
        self.threshold_var = tk.DoubleVar(value=settings.threshold_multiplier)
        tk.Scale(self.root, from_=1.0, to=3.0, resolution=0.1,
                 orient="horizontal", variable=self.threshold_var).pack(fill="x")

        tk.Label(self.root, text="Similarity Threshold").pack()
        self.similarity_var = tk.DoubleVar(value=settings.similarity_threshold)
        tk.Scale(self.root, from_=0.1, to=1.0, resolution=0.01,
                 orient="horizontal", variable=self.similarity_var).pack(fill="x")

        tk.Label(self.root, text="Failsafe Timer").pack()
        self.failsafe_var = tk.DoubleVar(value=settings.failsafe_timer)
        tk.Scale(self.root, from_=0.1, to=2.0, resolution=0.1,
                 orient="horizontal", variable=self.failsafe_var).pack(fill="x")

        tk.Label(self.root, text="Audio Device").pack()
        devices = sd.query_devices()
        self.input_devices = [(i, d['name']) for i, d in enumerate(devices) if d['max_input_channels'] > 0]
        device_names = [name for _, name in self.input_devices]
        self.device_var = tk.StringVar(value=device_names[0] if device_names else "")
        tk.OptionMenu(self.root, self.device_var, *device_names).pack(fill="x")

        tk.Button(self.root, text="Select Sound Cue", command=self.select_sound).pack(fill="x")
        self.sound_label = tk.Label(self.root, text="No file selected")
        self.sound_label.pack(fill="x")

        self.start_button = tk.Button(self.root, text="Start", command=self.start)
        self.start_button.pack(fill="x")
        tk.Button(self.root, text="Stop", command=self.stop).pack(fill="x")

        self.stop_event = threading.Event()
        self.listener_thread: threading.Thread | None = None

    def select_sound(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if path:
            load_sound_cue(path)
            self.sound_label.config(text=path)

    def start(self) -> None:
        settings.threshold_multiplier = self.threshold_var.get()
        settings.similarity_threshold = self.similarity_var.get()
        settings.failsafe_timer = self.failsafe_var.get()
        for idx, name in self.input_devices:
            if name == self.device_var.get():
                settings.device = idx
                break
        if cue_audio is None:
            print("Please select a sound cue before starting.")
            return
        self.stop_event.clear()
        self.listener_thread = threading.Thread(target=self.run_listener)
        self.listener_thread.start()

    def run_listener(self) -> None:
        while not self.stop_event.is_set():
            listen_for_sound(self.stop_event)
            if self.stop_event.is_set():
                break
            time.sleep(settings.failsafe_timer)
            perform_actions()

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        self.root.mainloop()


def main():
    gui = MacroBotGUI()
    gui.run()


if __name__ == "__main__":
    main()
