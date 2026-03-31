# ==========================================
# 🎵 Text-to-Audio (Qwen2.5 + LoRA + SNAC) - FIXED
# ==========================================
import torch
import soundfile as sf
import sounddevice as sd
import time
import numpy as np
import noisereduce as nr
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel
from snac import SNAC
from scipy.signal import resample
from threading import Thread
from queue import Queue, Empty

# ---------------------------------------------------------
# 1. ตั้งค่า — ตรงกับไฟล์เทรน
# ---------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
base_model_path = "Pakorn2112/Archa-TTS-0.5B-th"
# lora_adapter_path = "./myModel-v8-lora"
snac_model_path = "hubertsiuzdak/snac_24khz"

tokeniser_length = 151665
end_of_text = tokeniser_length + 2
start_of_speech = tokeniser_length + 3
end_of_speech = tokeniser_length + 4
start_of_human = tokeniser_length + 5
end_of_human = tokeniser_length + 6
start_of_ai = tokeniser_length + 7
audio_tokens_start = tokeniser_length + 10  # 151675

SNAC_SR = 24000

def get_supported_samplerate(requested_sr=SNAC_SR):
    """
    ตรวจสอบว่า hardware รองรับ sample rate ที่ต้องการหรือไม่
    หากไม่รองรับ จะคืนค่า 44100Hz หรือ default sample rate ของเครื่องแทน
    """
    try:
        sd.check_output_settings(samplerate=requested_sr)
        return requested_sr
    except Exception:
        fallback_sr = 44100
        try:
            sd.check_output_settings(samplerate=fallback_sr)
            print(f"⚠️ Warning: Hardware ไม่รองรับ {requested_sr}Hz. จะทำการ Resample เป็น {fallback_sr}Hz สำหรับการเล่นเสียง")
            return fallback_sr
        except Exception:
            try:
                default_sr = int(sd.query_devices(kind='output')['default_samplerate'])
                print(f"⚠️ Warning: Hardware ไม่รองรับ {requested_sr}Hz หรือ {fallback_sr}Hz. จะทำการ Resample เป็น {default_sr}Hz สำหรับการเล่นเสียง")
                return default_sr
            except Exception:
                print(f"⚠️ Warning: ไม่พบอุปกรณ์ Output. จะใช้ {requested_sr}Hz เป็นค่าพื้นฐาน")
                return requested_sr

# กำหนด Sample Rate สำหรับการเล่นเสียง (Playback)
PLAYBACK_SR = get_supported_samplerate(SNAC_SR)

def resample_audio(audio, orig_sr, target_sr):
    """
    เปลี่ยน Sample Rate ของข้อมูลเสียง (Resampling)
    """
    if orig_sr == target_sr or len(audio) == 0:
        return audio
    num_samples = int(len(audio) * target_sr / orig_sr)
    return resample(audio, num_samples).astype(np.float32)

# ---------------------------------------------------------
# 2. โหลดโมเดล
# ---------------------------------------------------------
print("กำลังโหลด Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

print("กำลังโหลด Base Model...")
torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
model = AutoModelForCausalLM.from_pretrained(
    base_model_path, torch_dtype=torch_dtype, device_map=device
)
model.resize_token_embeddings(180500)

# print(f"กำลังสวม LoRA Adapter จาก {lora_adapter_path}...")
# model = PeftModel.from_pretrained(model, lora_adapter_path)


print("กำลังโหลด SNAC Decoder...")
snac_model = SNAC.from_pretrained(snac_model_path).eval().to(device)

# ---------------------------------------------------------
# 3. SNAC decode
# ---------------------------------------------------------
def decode_tokens(token_list):
    valid_len = (len(token_list) // 7) * 7
    token_list = token_list[:valid_len]
    if valid_len < 7:
        return np.array([], dtype=np.float32)

    l1, l2, l3 = [], [], []
    for i in range(valid_len // 7):
        b = 7 * i
        codes = [t - audio_tokens_start for t in token_list[b:b+7]]
        l1.append(codes[0])
        l2.append(codes[1] - 4096)
        l3.append(codes[2] - 2*4096)
        l3.append(codes[3] - 3*4096)
        l2.append(codes[4] - 4*4096)
        l3.append(codes[5] - 5*4096)
        l3.append(codes[6] - 6*4096)

    snac_codes = [
        torch.tensor(l1, dtype=torch.long).unsqueeze(0).to(device),
        torch.tensor(l2, dtype=torch.long).unsqueeze(0).to(device),
        torch.tensor(l3, dtype=torch.long).unsqueeze(0).to(device),
    ]
    with torch.no_grad():
        audio = snac_model.decode(snac_codes)
    return audio.squeeze().cpu().numpy()

def denoise(audio, sr=SNAC_SR):
    return nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.8).astype(np.float32)

# ---------------------------------------------------------
# 4. Token collector
# ---------------------------------------------------------
class TokenCollector(TextStreamer):
    def __init__(self, tokenizer, skip_prompt=True):
        super().__init__(tokenizer, skip_prompt)
        self.audio_tokens = []

    def on_finalized_text(self, text: str, stream_end: bool = False):
        pass

    def put(self, value):
        if len(value.shape) > 1:
            value = value[0]
        for t in value.tolist():
            if t >= audio_tokens_start:
                self.audio_tokens.append(t)

# ---------------------------------------------------------
# 5. Streaming player
# ---------------------------------------------------------
class StreamingPlayer:
    def __init__(self, prefill_seconds=4.0):
        self.token_buffer = []
        self.queue = Queue()
        self.all_audio = []
        
        self.prefill_done = False
        self.prefill_target_frames = 100 
        self.stream_target_frames = 20   
        
        self.context_frames = 5          
        self.history_tokens = []         
        self.samples_per_frame = None    

        self.thread = Thread(target=self._play_loop, daemon=True)
        self.thread.start()

    def _play_loop(self):
        # ใช้ PLAYBACK_SR ที่ตรวจสอบแล้วว่า hardware รองรับ
        try:
            stream = sd.OutputStream(
                samplerate=PLAYBACK_SR,
                channels=1,
                dtype='float32',
            )
            stream.start()
            while True:
                chunk = self.queue.get()  
                if chunk is None:
                    break
                
                # ทำ Resampling ก่อนเล่นเสียง (ถ้าจำเป็น)
                play_chunk = resample_audio(chunk, SNAC_SR, PLAYBACK_SR)
                stream.write(play_chunk)
                
            stream.stop()
            stream.close()
        except Exception as e:
            print(f"❌ Playback error: {e}")

    def feed_tokens(self, new_tokens):
        self.token_buffer.extend(new_tokens)
        target_frames = self.prefill_target_frames if not self.prefill_done else self.stream_target_frames

        while len(self.token_buffer) >= target_frames * 7:
            chunk_tokens = self.token_buffer[:target_frames * 7]
            self.token_buffer = self.token_buffer[target_frames * 7:]

            tokens_to_decode = self.history_tokens + chunk_tokens
            audio = decode_tokens(tokens_to_decode)
            if len(audio) == 0:
                continue

            if self.samples_per_frame is None:
                test_audio = decode_tokens(chunk_tokens[:7])
                self.samples_per_frame = len(test_audio)

            if len(self.history_tokens) > 0:
                history_samples = (len(self.history_tokens) // 7) * self.samples_per_frame
                audio = audio[history_samples:]

            self.all_audio.append(audio)
            self.queue.put(audio)

            self.history_tokens = chunk_tokens[-(self.context_frames * 7):]

            if not self.prefill_done:
                self.prefill_done = True
                print("▶️  เริ่มเล่นเสียงแล้ว!")

    def finish(self):
        valid_len = (len(self.token_buffer) // 7) * 7
        if valid_len > 0:
            chunk_tokens = self.token_buffer[:valid_len]
            tokens_to_decode = self.history_tokens + chunk_tokens
            audio = decode_tokens(tokens_to_decode)

            if len(audio) > 0:
                if self.samples_per_frame is None:
                    self.samples_per_frame = len(audio)

                if len(self.history_tokens) > 0 and self.samples_per_frame is not None:
                    history_samples = (len(self.history_tokens) // 7) * self.samples_per_frame
                    if history_samples < len(audio):
                        audio = audio[history_samples:]
                    else:
                        audio = np.array([], dtype=np.float32)

                if len(audio) > 0:
                    self.all_audio.append(audio)
                    self.queue.put(audio)

        self.queue.put(None)  
        self.thread.join()    

        if self.all_audio:
            return np.concatenate(self.all_audio)
        return None


class StreamingStreamer(TextStreamer):
    def __init__(self, tokenizer, player, skip_prompt=True):
        super().__init__(tokenizer, skip_prompt)
        self.player = player

    def on_finalized_text(self, text: str, stream_end: bool = False):
        pass

    def put(self, value):
        if len(value.shape) > 1:
            value = value[0]
        tokens = [t for t in value.tolist() if t >= audio_tokens_start]
        if tokens:
            self.player.feed_tokens(tokens)


# ---------------------------------------------------------
# 6. ฟังก์ชันหลัก
# ---------------------------------------------------------
def generate_audio(text_prompt, output_filename="output_realtime.wav", mode="streaming"):
    total_start = time.perf_counter()
    print(f"\n📝 ข้อความ: '{text_prompt}'")
    print(f"🔧 โหมด: {mode}")

    text_ids = tokenizer.encode(text_prompt, add_special_tokens=True)
    text_ids.append(end_of_text)
    prompt_ids = (
        [start_of_human] + text_ids + [end_of_human]
        + [start_of_ai] + [start_of_speech]
    )
    input_ids = torch.tensor([prompt_ids]).to(device)

    estimated_tokens = len(text_prompt) * 30 
    max_tokens = max(16384, estimated_tokens)
    print(f"⚙️ ตั้งค่า max_new_tokens = {max_tokens}")

    gen_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=max_tokens, 
        use_cache=True,
        do_sample=True,          
        temperature=0.8,         
        top_p=0.9,               
        repetition_penalty=1.1,  
        eos_token_id=end_of_speech,
        pad_token_id=tokenizer.eos_token_id,
    )

    if mode == "full":
        print("🧠 กำลัง generate tokens...")
        collector = TokenCollector(tokenizer, skip_prompt=True)
        gen_kwargs["streamer"] = collector

        with torch.no_grad():
            model.generate(**gen_kwargs)

        n = len(collector.audio_tokens)
        print(f"✅ ได้ {n} audio tokens ({n//7} frames)")

        audio = decode_tokens(collector.audio_tokens)
        if len(audio) == 0:
            print("❌ ไม่ได้ audio tokens ที่ถูกต้อง")
            return

        print("🧹 กำลังตัดเสียงรบกวน...")
        audio = denoise(audio)
        
        # บันทึกไฟล์ที่ SNAC_SR (24000Hz) ตามเดิม
        sf.write(output_filename, audio, SNAC_SR)
        print(f"💾 บันทึกไว้ที่ {output_filename} ({len(audio)/SNAC_SR:.1f}s)")

        print("▶️  กำลังเล่นเสียง...")
        # ทำ Resampling สำหรับการเล่นเสียงผ่าน hardware ที่ไม่รองรับ 24000Hz
        play_audio = resample_audio(audio, SNAC_SR, PLAYBACK_SR)
        sd.play(play_audio, PLAYBACK_SR)
        sd.wait()

    else:
        print("🧠 กำลัง generate + เล่นสด...")
        player = StreamingPlayer(prefill_seconds=4.0) 
        streamer = StreamingStreamer(tokenizer, player, skip_prompt=True)
        gen_kwargs["streamer"] = streamer

        with torch.no_grad():
            model.generate(**gen_kwargs)

        audio = player.finish()
        if audio is not None:
            audio = denoise(audio)
            sf.write(output_filename, audio, SNAC_SR)
            print(f"💾 บันทึกไว้ที่ {output_filename} ({len(audio)/SNAC_SR:.1f}s)")

    total_end = time.perf_counter()
    print(f"🚀 เวลารวม: {total_end - total_start:.3f} วินาที")


if __name__ == "__main__":
    long_text = "สวัสดีครับใช้ผมฟรีและเร็วถึงจะเร็วไม่เท่าพี่ไทยเอ็นแอลพีก็ตาม"
    
    generate_audio(long_text, mode="streaming")