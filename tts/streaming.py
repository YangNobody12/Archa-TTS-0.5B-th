import numpy as np
import sounddevice as sd
from threading import Thread
from queue import Queue
from scipy.signal import resample as scipy_resample
from transformers import TextStreamer

from .codec import SNACCodec
from .config import TTSConfig


def get_playback_samplerate(requested_sr: int = 24000) -> int:
    """ตรวจสอบว่า hardware รองรับ sample rate ที่ต้องการหรือไม่"""
    try:
        sd.check_output_settings(samplerate=requested_sr)
        return requested_sr
    except Exception:
        try:
            default_sr = int(sd.query_devices(kind="output")["default_samplerate"])
            print(f"⚠️ Hardware ไม่รองรับ {requested_sr}Hz → Resample เป็น {default_sr}Hz")
            return default_sr
        except Exception:
            return requested_sr


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """เปลี่ยน sample rate"""
    if orig_sr == target_sr or len(audio) == 0:
        return audio
    num_samples = int(len(audio) * target_sr / orig_sr)
    return scipy_resample(audio, num_samples).astype(np.float32)


class StreamingPlayer:
    """เล่นเสียงแบบ streaming ขณะ generate"""

    def __init__(self, codec: SNACCodec, config: TTSConfig, prefill_frames: int = 100):
        self.codec = codec
        self.snac_sr = config.snac_sr
        self.playback_sr = get_playback_samplerate(self.snac_sr)

        self.token_buffer: list[int] = []
        self.queue: Queue = Queue()
        self.all_audio: list[np.ndarray] = []
        self.prefill_done = False
        self.prefill_target_frames = prefill_frames
        self.stream_target_frames = 20
        self.context_frames = 5
        self.history_tokens: list[int] = []
        self.samples_per_frame: int | None = None

        self._thread = Thread(target=self._play_loop, daemon=True)
        self._thread.start()

    def _play_loop(self):
        try:
            stream = sd.OutputStream(samplerate=self.playback_sr, channels=1, dtype="float32")
            stream.start()
            while True:
                chunk = self.queue.get()
                if chunk is None:
                    break
                play_chunk = resample_audio(chunk, self.snac_sr, self.playback_sr)
                stream.write(play_chunk)
            stream.stop()
            stream.close()
        except Exception as e:
            print(f"❌ Playback error: {e}")

    def feed_tokens(self, new_tokens: list[int]):
        self.token_buffer.extend(new_tokens)
        target = self.prefill_target_frames if not self.prefill_done else self.stream_target_frames

        while len(self.token_buffer) >= target * 7:
            chunk_tokens = self.token_buffer[: target * 7]
            self.token_buffer = self.token_buffer[target * 7 :]
            tokens_to_decode = self.history_tokens + chunk_tokens
            audio = self.codec.decode_tokens(tokens_to_decode)
            if len(audio) == 0:
                continue

            if self.samples_per_frame is None:
                test_audio = self.codec.decode_tokens(chunk_tokens[:7])
                self.samples_per_frame = len(test_audio)

            if self.history_tokens:
                history_samples = (len(self.history_tokens) // 7) * self.samples_per_frame
                audio = audio[history_samples:]

            self.all_audio.append(audio)
            self.queue.put(audio)
            self.history_tokens = chunk_tokens[-(self.context_frames * 7) :]

            if not self.prefill_done:
                self.prefill_done = True
                print("▶️  เริ่มเล่นเสียงแล้ว!")

    def finish(self) -> np.ndarray | None:
        """Flush token ที่เหลือ แล้วรอ playback จบ"""
        valid_len = (len(self.token_buffer) // 7) * 7
        if valid_len > 0:
            chunk_tokens = self.token_buffer[:valid_len]
            tokens_to_decode = self.history_tokens + chunk_tokens
            audio = self.codec.decode_tokens(tokens_to_decode)
            if len(audio) > 0 and self.samples_per_frame is not None and self.history_tokens:
                history_samples = (len(self.history_tokens) // 7) * self.samples_per_frame
                audio = audio[history_samples:]
            if len(audio) > 0:
                self.all_audio.append(audio)
                self.queue.put(audio)

        self.queue.put(None)
        self._thread.join()
        return np.concatenate(self.all_audio) if self.all_audio else None


class TokenCollector(TextStreamer):
    """เก็บ audio tokens จาก generation (โหมด full)"""

    def __init__(self, tokenizer, audio_tokens_start: int, skip_prompt=True):
        super().__init__(tokenizer, skip_prompt=skip_prompt)
        self.audio_tokens_start = audio_tokens_start
        self.audio_tokens: list[int] = []

    def on_finalized_text(self, text: str, stream_end: bool = False):
        pass

    def put(self, value):
        if len(value.shape) > 1:
            value = value[0]
        for t in value.tolist():
            if t >= self.audio_tokens_start:
                self.audio_tokens.append(t)


class StreamingStreamer(TextStreamer):
    """ส่ง audio tokens ไปยัง StreamingPlayer แบบ real-time"""

    def __init__(self, tokenizer, player: StreamingPlayer, audio_tokens_start: int, skip_prompt=True):
        super().__init__(tokenizer, skip_prompt=skip_prompt)
        self.player = player
        self.audio_tokens_start = audio_tokens_start

    def on_finalized_text(self, text: str, stream_end: bool = False):
        pass

    def put(self, value):
        if len(value.shape) > 1:
            value = value[0]
        tokens = [t for t in value.tolist() if t >= self.audio_tokens_start]
        if tokens:
            self.player.feed_tokens(tokens)
