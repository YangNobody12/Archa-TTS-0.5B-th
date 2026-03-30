import time
import numpy as np
import torch
import soundfile as sf
import sounddevice as sd
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from .config import TTSConfig
from .codec import SNACCodec
from .streaming import (
    StreamingPlayer,
    TokenCollector,
    StreamingStreamer,
    resample_audio,
    get_playback_samplerate,
)


class TTSEngine:
    """
    Text-to-Speech engine ที่รวม Qwen2.5 + LoRA + SNAC

    ใช้งาน:
        engine = TTSEngine()
        engine.generate("สวัสดีครับ", mode="streaming")
    """

    def __init__(self, config: TTSConfig | None = None):
        self.config = config or TTSConfig()
        self._load_models()
        self.codec = SNACCodec(self.config)
        self.playback_sr = get_playback_samplerate(self.config.snac_sr)

    def _load_models(self):
        cfg = self.config
        print("กำลังโหลด Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_path, fix_mistral_regex=True)

        print("กำลังโหลด Base Model...")
        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model_path, torch_dtype=dtype, device_map=cfg.device
        )
        self.model.resize_token_embeddings(cfg.vocab_size)

        if cfg.lora_adapter_path:
            print(f"กำลังสวม LoRA Adapter จาก {cfg.lora_adapter_path}...")
            self.model = PeftModel.from_pretrained(self.model, cfg.lora_adapter_path)

    # ------------------------------------------------------------------
    def _build_prompt(self, text: str) -> torch.Tensor:
        cfg = self.config
        text_ids = self.tokenizer.encode(text, add_special_tokens=True)
        text_ids.append(cfg.end_of_text)
        prompt_ids = (
            [cfg.start_of_human]
            + text_ids
            + [cfg.end_of_human]
            + [cfg.start_of_ai]
            + [cfg.start_of_speech]
        )
        return torch.tensor([prompt_ids]).to(cfg.device)

    def _gen_kwargs(self, input_ids: torch.Tensor, **overrides) -> dict:
        cfg = self.config
        defaults = dict(
            input_ids=input_ids,
            max_new_tokens=max(16384, input_ids.shape[1] * 30),
            use_cache=True,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=cfg.end_of_speech,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        defaults.update(overrides)
        return defaults

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        text: str,
        output_path: str = "output_realtime.wav",
        mode: str = "streaming",
        play: bool = True,
        denoise: bool = True,
        **gen_overrides,
    ) -> np.ndarray | None:
        """
        สร้างเสียงจากข้อความ

        Args:
            text: ข้อความภาษาไทย
            output_path: path สำหรับบันทึกไฟล์ .wav (None = ไม่บันทึก)
            mode: "streaming" (เล่นสด) หรือ "full" (generate ทั้งหมดก่อน)
            play: เล่นเสียงผ่าน speaker หรือไม่
            denoise: ลดเสียงรบกวนหรือไม่
            **gen_overrides: override generation params เช่น temperature, top_p

        Returns:
            numpy waveform (24 kHz) หรือ None ถ้าไม่มี audio tokens
        """
        total_start = time.perf_counter()
        print(f"\n📝 ข้อความ: '{text}'")
        print(f"🔧 โหมด: {mode}")

        input_ids = self._build_prompt(text)
        kwargs = self._gen_kwargs(input_ids, **gen_overrides)

        if mode == "full":
            audio = self._generate_full(kwargs, play)
        else:
            audio = self._generate_streaming(kwargs)

        if audio is None:
            print("❌ ไม่ได้ audio tokens ที่ถูกต้อง")
            return None

        if denoise:
            print("🧹 กำลังตัดเสียงรบกวน...")
            audio = self.codec.denoise(audio, sr=self.config.snac_sr)

        if output_path:
            sf.write(output_path, audio, self.config.snac_sr)
            print(f"💾 บันทึกไว้ที่ {output_path} ({len(audio) / self.config.snac_sr:.1f}s)")

        elapsed = time.perf_counter() - total_start
        print(f"🚀 เวลารวม: {elapsed:.3f} วินาที")
        return audio

    # ------------------------------------------------------------------
    def _generate_full(self, kwargs: dict, play: bool) -> np.ndarray | None:
        print("🧠 กำลัง generate tokens...")
        collector = TokenCollector(self.tokenizer, self.config.audio_tokens_start)
        kwargs["streamer"] = collector

        with torch.no_grad():
            self.model.generate(**kwargs)

        n = len(collector.audio_tokens)
        print(f"✅ ได้ {n} audio tokens ({n // 7} frames)")
        audio = self.codec.decode_tokens(collector.audio_tokens)
        if len(audio) == 0:
            return None

        if play:
            print("▶️  กำลังเล่นเสียง...")
            play_audio = resample_audio(audio, self.config.snac_sr, self.playback_sr)
            sd.play(play_audio, self.playback_sr)
            sd.wait()

        return audio

    def _generate_streaming(self, kwargs: dict) -> np.ndarray | None:
        print("🧠 กำลัง generate + เล่นสด...")
        player = StreamingPlayer(self.codec, self.config)
        streamer = StreamingStreamer(
            self.tokenizer, player, self.config.audio_tokens_start
        )
        kwargs["streamer"] = streamer

        with torch.no_grad():
            self.model.generate(**kwargs)

        return player.finish()
