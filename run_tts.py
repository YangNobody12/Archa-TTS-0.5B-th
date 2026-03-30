"""ตัวอย่างการใช้งาน TTS Engine"""

from tts import TTSConfig, TTSEngine

# ตั้งค่า (ปรับ path ตามต้องการ)
config = TTSConfig(
    base_model_path="./Archa-TTS-0.5B-th",
    snac_model_path="hubertsiuzdak/snac_24khz",
    # lora_adapter_path="./myModel-v8-lora",  # uncomment ถ้าใช้ LoRA
)

engine = TTSEngine(config)

# ใช้งาน
text = "สวัสดีครับพี่น้องชาวไทยทุกท่าน วันนี้ผมจะมาเล่าเรื่องราวสุดฮาให้ฟังนะครับ คือเมื่อวานผมไปซื้อข้าวมันไก่ร้านประจำ"

# โหมด streaming (เล่นสดขณะ generate)
engine.generate(text, mode="streaming")

# โหมด full (generate ทั้งหมดก่อนเล่น)
# engine.generate(text, mode="full", output_path="output_full.wav")

# ไม่เล่นเสียง แค่บันทึกไฟล์
# audio = engine.generate(text, mode="full", play=False, output_path="silent.wav")

# ปรับ generation params
# engine.generate(text, temperature=0.6, top_p=0.85)
