# extract_text.py (增量处理 + 缓存 + 版本管理 + OCR 超优化)
import pdfplumber
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import os
import glob
import json
import hashlib
from openai import OpenAI
from datetime import datetime
import config

# ========== 配置 ==========



client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", "https://api.moonshot.cn/v1")
)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# 缓存目录
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)
LOG_FILE = os.path.join(CACHE_DIR, "processing_log.json")

# OCR 版本（每次改参数，版本号+1后会重新跑文本）
OCR_VERSION = "v6"  # 修复 MedianFilter
# =========================

def get_file_hash(pdf_path):
    """计算文件 MD5"""
    with open(pdf_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def load_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_log(log):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

def classify_document_with_ai(text_sample):
    """用 Kimi AI 自动分类文档"""
    prompt = f"""
请判断以下财务文档属于哪一类？输出 JSON 格式。

文本样本：
{text_sample[:3000]}

【可选类别】
1. 审计报告
2. 行业报告
3. 公司研究报告
4. 上市手册
5. 财报
6. 其他

【输出要求】
- 仅输出 JSON 格式
- 格式示例：
{{
  "type": "行业报告",
  "confidence": 0.93,
  "reason": "文本主要为行业市场分析和趋势预测"
}}
"""
    try:
        response = client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        # 尝试解析 JSON
        content = response.choices[0].message.content.strip()
        # 兼容模型多余文字
        start = content.find("{")
        end = content.rfind("}") + 1
        json_str = content[start:end]
        result = json.loads(json_str)
        return result
        # json.loads(response.choices[0].message.content)
        
    except Exception as e:
        print(f"分类失败: {e}")
        return {"type": "未知", "confidence": 0.0, "reason": "API 调用失败"}

def optimize_image_for_ocr(image):
    """OCR 超优化"""
    image = image.convert('L')
    image = image.filter(ImageFilter.MedianFilter(size=3))  # 修复：MedianFilter
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.5)
    image = image.filter(ImageFilter.SHARPEN)
    image = image.point(lambda x: 0 if x < 140 else 255, '1')
    return image

def extract_text_from_pdf(pdf_path, output_txt):
    text = f"# 文件: {os.path.basename(pdf_path)}\n\n"
    ocr_pages = 0
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text and len(page_text.strip()) > 50:
                text += f"\n--- Page {i+1} (Text) ---\n{page_text}"
            else:
                img = page.to_image(resolution=600).original
                img = optimize_image_for_ocr(img)
                ocr_text = pytesseract.image_to_string(
                    img, lang='chi_tra+chi_sim+eng', config='--oem 1 --psm 3'
                )
                text += f"\n--- Page {i+1} (OCR) ---\n{ocr_text}"
                ocr_pages += 1
                print(f"   [OCR] Page {i+1} 已识别（{OCR_VERSION}）")
    
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"提取完成: {output_txt} (OCR 页数: {ocr_pages})")
    
    sample = text[:3000]
    classification = classify_document_with_ai(sample)
    print(f"   [分类] {classification['type']} (置信度: {classification['confidence']:.2f})")
    
    return text, classification, ocr_pages

def batch_extract_docs():
    pdf_files = glob.glob("docs/*.pdf")
    log = load_log()
    all_text = ""
    classifications = {}
    total_ocr = 0
    processed = 0

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        file_hash = get_file_hash(pdf_path)
        txt_path = os.path.join("docs", os.path.splitext(filename)[0] + ".txt")
        
        # 检查是否需要处理
        cached = log.get(filename, {})
        if (cached.get("hash") == file_hash and 
            cached.get("ocr_version") == OCR_VERSION and 
            os.path.exists(txt_path)):
            print(f"跳过: {filename} (已处理，版本 {OCR_VERSION})")
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read()
            classification = cached["classification"]
        else:
            print(f"处理: {filename}")
            text, classification, ocr_count = extract_text_from_pdf(pdf_path, txt_path)
            total_ocr += ocr_count
            processed += 1
            
            # 更新日志
            log[filename] = {
                "hash": file_hash,
                "ocr_version": OCR_VERSION,
                "processed_at": datetime.now().isoformat(),
                "classification": classification,
                "ocr_pages": ocr_count
            }

        all_text += text + "\n\n" + "="*60 + "\n\n"
        classifications[filename] = classification

    # 保存
    with open("docs/all_extracted.txt", "w", encoding="utf-8") as f:
        f.write(all_text)
    with open("docs/classification.json", "w", encoding="utf-8") as f:
        json.dump(classifications, f, ensure_ascii=False, indent=2)
    save_log(log)

    print(f"\n批量处理完成！本次处理 {processed} 个文件，跳过 {len(pdf_files)-processed} 个")
    print(f"   OCR 总页数: {total_ocr}，版本: {OCR_VERSION}")
    print("   日志已保存：cache/processing_log.json")

if __name__ == "__main__":
    batch_extract_docs()