import numpy as np
import torch
import noisereduce as nr
from snac import SNAC

from .config import TTSConfig


class SNACCodec:
    """SNAC encoder/decoder สำหรับแปลง token ↔ audio"""

    def __init__(self, config: TTSConfig):
        self.config = config
        self.device = config.device
        self.model = SNAC.from_pretrained(config.snac_model_path).eval().to(self.device)

    def decode_tokens(self, token_list: list[int]) -> np.ndarray:
        """แปลง audio tokens กลับเป็น waveform"""
        valid_len = (len(token_list) // 7) * 7
        token_list = token_list[:valid_len]
        if valid_len < 7:
            return np.array([], dtype=np.float32)

        start = self.config.audio_tokens_start
        l1, l2, l3 = [], [], []

        for i in range(valid_len // 7):
            b = 7 * i
            codes = [t - start for t in token_list[b : b + 7]]
            l1.append(codes[0])
            l2.append(codes[1] - 4096)
            l3.append(codes[2] - 2 * 4096)
            l3.append(codes[3] - 3 * 4096)
            l2.append(codes[4] - 4 * 4096)
            l3.append(codes[5] - 5 * 4096)
            l3.append(codes[6] - 6 * 4096)

        snac_codes = [
            torch.tensor(l1, dtype=torch.long).unsqueeze(0).to(self.device),
            torch.tensor(l2, dtype=torch.long).unsqueeze(0).to(self.device),
            torch.tensor(l3, dtype=torch.long).unsqueeze(0).to(self.device),
        ]
        with torch.no_grad():
            audio = self.model.decode(snac_codes)
        return audio.squeeze().cpu().numpy()

    @staticmethod
    def denoise(audio: np.ndarray, sr: int = 24000, prop_decrease: float = 0.8) -> np.ndarray:
        """ลดเสียงรบกวนจาก waveform"""
        return nr.reduce_noise(y=audio, sr=sr, prop_decrease=prop_decrease).astype(np.float32)
