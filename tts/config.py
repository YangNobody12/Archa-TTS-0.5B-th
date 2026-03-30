import torch


class TTSConfig:
    """การตั้งค่าทั้งหมดสำหรับ TTS pipeline"""

    def __init__(
        self,
        base_model_path: str = "./Archa-TTS-0.5B-th",
        snac_model_path: str = "hubertsiuzdak/snac_24khz",
        lora_adapter_path: str | None = None,
        vocab_size: int = 180500,
        device: str | None = None,
    ):
        self.base_model_path = base_model_path
        self.snac_model_path = snac_model_path
        self.lora_adapter_path = lora_adapter_path
        self.vocab_size = vocab_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Token IDs
        self.tokeniser_length = 151665
        self.end_of_text = self.tokeniser_length + 2
        self.start_of_speech = self.tokeniser_length + 3
        self.end_of_speech = self.tokeniser_length + 4
        self.start_of_human = self.tokeniser_length + 5
        self.end_of_human = self.tokeniser_length + 6
        self.start_of_ai = self.tokeniser_length + 7
        self.audio_tokens_start = self.tokeniser_length + 10  # 151675

        # Audio
        self.snac_sr = 24000
