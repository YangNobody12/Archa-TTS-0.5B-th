# Archa-TTS-0.5B-th

โมเดลแปลงข้อความภาษาไทยเป็นเสียงพูด (Text-to-Speech) พัฒนาบน Qwen2.5-0.5B ด้วยการ fine-tune แบบ LoRA ร่วมกับ [SNAC](https://github.com/hubertsiuzdak/snac) audio codec ที่ sample rate 24kHz

พัฒนาโดย ปรกณ์ อาชาคีรี — TU ACM SIGHPC Student Chapter | สนับสนุนโดย Thai SC

โมเดลจะสร้างลำดับ audio tokens จากข้อความภาษาไทย แล้วถอดรหัสเป็นคลื่นเสียงผ่าน SNAC neural audio codec

> ⚠️ โมเดลนี้รองรับเฉพาะภาษาไทยเท่านั้น และรองรับการสร้างเสียงแบบ Realtime

## รายละเอียดโมเดล

| รายการ | ค่า |
|---|---|
| โมเดลพื้นฐาน | Qwen/Qwen2.5-0.5B |
| สถาปัตยกรรม | Qwen2ForCausalLM |
| พารามิเตอร์ | ~0.5B |
| Audio codec | SNAC 24kHz |
| ความแม่นยำ | bfloat16 / float16 |
| ขนาด Vocab | 180,500 |
| Context สูงสุด | 32,768 tokens |

## สิ่งที่ต้องติดตั้ง

```bash
pip install torch transformers peft snac soundfile sounddevice noisereduce scipy numpy
```

- Python 3.8 ขึ้นไป
- แนะนำให้ใช้ GPU ที่รองรับ CUDA (รันบน CPU ได้แต่จะช้ามาก)
- ต้องมี LoRA adapter weights อยู่ในโฟลเดอร์ `Pakorn2112/Archa-TTS-0.5B-th`

## วิธีใช้งาน

### โหมด Streaming (เล่นเสียงแบบเรียลไทม์)

```python
from inference import generate_audio

generate_audio("สวัสดีครับ วันนี้อากาศดีมากเลยนะครับ", mode="streaming")
```

### โหมด Full (สร้างเสียงทั้งหมดก่อนเล่น)

```python
from inference import generate_audio

generate_audio("สวัสดีครับ วันนี้อากาศดีมากเลยนะครับ", mode="full")
```

### รันผ่าน CLI

```bash
python inference.py
```

จะรันข้อความตัวอย่างในโหมด streaming และบันทึกผลลัพธ์เป็นไฟล์ `output_realtime.wav`

## หลักการทำงาน

1. ข้อความภาษาไทยจะถูก tokenize และครอบด้วย control tokens พิเศษ (`start_of_human`, `start_of_speech` ฯลฯ)
2. โมเดล Qwen2.5 ที่ใส่ LoRA adapter จะสร้างลำดับ audio tokens แบบ autoregressive
3. Audio tokens จะถูกถอดรหัสทีละ 7 ตัว (1 SNAC frame) เป็น hierarchical codes 3 ระดับ
4. SNAC decoder จะสร้างคลื่นเสียง 24kHz จาก codes เหล่านี้
5. ลดเสียงรบกวน (noise reduction) เป็นขั้นตอนสุดท้าย

## โหมดการทำงาน

- `streaming` — ถอดรหัสและเล่นเสียงแบบเรียลไทม์ขณะที่กำลังสร้าง tokens โดยจะเก็บ buffer ไว้ก่อน (~100 frames) แล้วจึงเริ่มเล่น จากนั้นจะ stream ทีละ ~20 frames พร้อม context overlap เพื่อให้เสียงต่อเนื่องราบรื่น
- `full` — สร้าง tokens ทั้งหมดก่อน แล้วค่อยถอดรหัสและเล่นเสียงทีเดียว

## ผลลัพธ์

- เสียงจะถูกบันทึกเป็นไฟล์ WAV ที่ 24kHz (sample rate ดั้งเดิมของ SNAC)
- ระบบจะ resample อัตโนมัติหาก hardware ไม่รองรับ 24kHz

## สัญญาอนุญาต

กรุณาอ้างอิงสัญญาอนุญาตของโมเดล Qwen2.5 และ SNAC สำหรับเงื่อนไขการใช้งาน

## License

This project is licensed under the MIT License.

⚠️ Note:
- The model is trained on third-party datasets.
- Please ensure you comply with the original dataset licenses before using this model commercially.