"""
Image Analysis Agent using OCR + Ollama Gemma3:12b
This agent receives HTTP requests with image URLs, downloads images,
extracts text using OCR, processes with Gemma3, and cleans up files.
"""

import os
import tempfile
import requests
import base64
import json
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
import uvicorn
import logging
from pathlib import Path
from PIL import Image
import pytesseract
import cv2
import numpy as np

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Pydantic models for request/response
class ImageAnalysisRequest(BaseModel):
    image_url: HttpUrl
    question: Optional[str] = "Hãy trích xuất và cải thiện text từ hình ảnh này."

class ImageAnalysisResponse(BaseModel):
    analysis: str
    status: str
    message: str

class ImageAnalysisAgent:
    def __init__(self, ollama_base_url: str = "http://localhost:11434"):
        """
        Initialize the Image Analysis Agent with OCR + Gemma3
        
        Args:
            ollama_base_url: Base URL for Ollama server
        """
        self.ollama_base_url = ollama_base_url
        self.model_name = "gemma3:12b"
        
        # Create directory for image storage in current working directory
        self.temp_dir = os.path.join(os.getcwd(), "temp_images")
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def _preprocess_image(self, image_path: str) -> str:
        """
        Preprocess image to improve OCR accuracy
        
        Args:
            image_path: Path to the original image
            
        Returns:
            Path to the preprocessed image
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply noise reduction
            denoised = cv2.medianBlur(gray, 3)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Morphological operations to remove noise
            kernel = np.ones((1,1), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            
            # Save preprocessed image
            preprocessed_path = image_path.replace('.', '_preprocessed.')
            cv2.imwrite(preprocessed_path, cleaned)
            
            return preprocessed_path
            
        except Exception as e:
            # If preprocessing fails, return original image
            return image_path
    
    def _extract_text_with_ocr(self, image_path: str) -> str:
        """
        Extract text from image using OCR with multiple techniques
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text
        """
        try:
            # Preprocess image for better OCR
            preprocessed_path = self._preprocess_image(image_path)
            
            # Configure Tesseract for better accuracy
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđĐ'
            
            # Try multiple OCR approaches
            text_results = []
            
            # Method 1: Direct OCR on preprocessed image
            img = Image.open(preprocessed_path)
            text1 = pytesseract.image_to_string(img, lang='vie+eng', config=custom_config)
            if text1.strip():
                text_results.append(text1.strip())
            
            # Method 2: OCR on original image
            img_original = Image.open(image_path)
            text2 = pytesseract.image_to_string(img_original, lang='vie+eng', config=custom_config)
            if text2.strip():
                text_results.append(text2.strip())
            
            # Method 3: Try different PSM modes
            for psm in [3, 6, 8, 13]:
                config = f'--oem 3 --psm {psm}'
                text_psm = pytesseract.image_to_string(img, lang='vie+eng', config=config)
                if text_psm.strip() and len(text_psm.strip()) > 10:
                    text_results.append(text_psm.strip())
            
            # Clean up preprocessed image
            if preprocessed_path != image_path and os.path.exists(preprocessed_path):
                os.remove(preprocessed_path)
            
            # Return the longest/best result
            if text_results:
                # Choose the result with most characters (usually better)
                best_text = max(text_results, key=len)
                return best_text
            else:
                return "Không thể trích xuất text từ hình ảnh này."
                
        except Exception as e:
            return f"Lỗi khi trích xuất text: {str(e)}"
    
    def _download_image(self, image_url: str) -> str:
        """
        Download image from URL and save to temporary directory
        
        Args:
            image_url: URL of the image to download
            
        Returns:
            Local path to the downloaded image
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(image_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Get file extension from URL or default to .jpg
            file_extension = Path(image_url).suffix.lower()
            if not file_extension or file_extension not in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']:
                file_extension = '.jpg'
            
            # Create unique filename
            import uuid
            filename = f"image_{uuid.uuid4()}{file_extension}"
            file_path = os.path.join(self.temp_dir, filename)
            
            # Save image
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            return file_path
            
        except Exception as e:
            raise Exception(f"Không thể tải hình ảnh: {str(e)}")
    
    def _analyze_image(self, image_path: str, question: str = None) -> str:
        """
        Extract text from image using OCR and improve with Gemma3
        
        Args:
            image_path: Path to the image file
            question: Question about the image processing
            
        Returns:
            Processed text result
        """
        extracted_text = ""
        try:
            if question is None:
                question = "Hãy trích xuất và cải thiện text từ hình ảnh này."
            
            # Check if file exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Không tìm thấy file: {image_path}")
            
            # Step 1: Extract text using OCR
            extracted_text = self._extract_text_with_ocr(image_path)
            
            if "Không thể trích xuất" in extracted_text or "Lỗi khi trích xuất" in extracted_text:
                return extracted_text
            
            # Step 2: Process with Gemma3 to improve and clean the text
            improvement_prompt = f"""
Text được trích xuất từ OCR: "{extracted_text}"

Nhiệm vụ: {question}

Hãy:
1. Sửa lỗi chính tả và ngữ pháp
2. Cải thiện định dạng và cấu trúc
3. Loại bỏ ký tự lạ hoặc không cần thiết
4. Đảm bảo text có ý nghĩa và dễ đọc

Chỉ trả về text đã được cải thiện, không cần giải thích.
"""
            
            # Call Gemma3 API (text-only, no images)
            payload = {
                "model": self.model_name,
                "prompt": improvement_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_k": 40,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                improved_text = result.get("response", "Không có phản hồi từ mô hình")
                
                if not improved_text or improved_text.strip() == "":
                    return f"Text gốc từ OCR: {extracted_text}"
                
                return improved_text
            else:
                # Return OCR text if Gemma3 fails
                return f"Text từ OCR (không thể cải thiện): {extracted_text}"
            
        except requests.exceptions.Timeout:
            return f"Text từ OCR (timeout): {extracted_text}" if extracted_text else "Lỗi timeout và không thể trích xuất text"
        except requests.exceptions.ConnectionError:
            return f"Text từ OCR (không kết nối được Gemma3): {extracted_text}" if extracted_text else "Lỗi kết nối và không thể trích xuất text"
        except Exception as e:
            return f"Lỗi khi xử lý: {str(e)}"
    
    def _cleanup_image(self, image_path: str) -> str:
        """
        Delete the downloaded image file
        
        Args:
            image_path: Path to the image file to delete
            
        Returns:
            Confirmation message
        """
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
                return f"Đã xóa file: {image_path}"
            else:
                return f"File không tồn tại: {image_path}"
        except Exception as e:
            return f"Lỗi khi xóa file: {str(e)}"
    
    def process_image_url(self, image_url: str, question: str = None) -> str:
        """
        Process image URL: download, analyze, and cleanup
        
        Args:
            image_url: URL of the image to analyze
            question: Optional question about the image
            
        Returns:
            Analysis result
        """
        if question is None:
            question = "Hãy trích xuất và cải thiện text từ hình ảnh này."
        
        image_path = None
        try:
            # Download image
            image_path = self._download_image(image_url)
            
            # Analyze image
            analysis_result = self._analyze_image(image_path, question)
            
            return analysis_result
            
        except Exception as e:
            raise Exception(f"Lỗi khi xử lý: {str(e)}")
        finally:
            # Always cleanup image regardless of success or failure
            if image_path and os.path.exists(image_path):
                try:
                    self._cleanup_image(image_path)
                except:
                    # If cleanup fails, try direct removal
                    try:
                        os.remove(image_path)
                    except:
                        pass
    
    def cleanup_temp_dir(self):
        """Clean up temporary directory"""
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            pass
    
    def force_cleanup_all_images(self):
        """Force cleanup all images in temp directory"""
        try:
            if os.path.exists(self.temp_dir):
                for filename in os.listdir(self.temp_dir):
                    file_path = os.path.join(self.temp_dir, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    except:
                        pass
        except Exception as e:
            pass

# Initialize FastAPI app
app = FastAPI(
    title="Image Text Extraction API",
    description="API for extracting and improving text from images using OCR + Gemma3",
    version="1.0.0"
)

# Initialize the agent
image_agent = ImageAnalysisAgent()

@app.post("/analyze_image", response_model=ImageAnalysisResponse)
async def analyze_image(request: ImageAnalysisRequest):
    """
    Analyze an image from URL
    """
    try:
        # Process the image
        analysis_result = image_agent.process_image_url(
            str(request.image_url), 
            request.question
        )
        
        return ImageAnalysisResponse(
            analysis=analysis_result,
            status="success",
            message="Trích xuất và cải thiện text thành công"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi xử lý yêu cầu: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Image Text Extraction Agent is running"}

@app.post("/cleanup")
async def force_cleanup():
    """Force cleanup all temporary images"""
    try:
        image_agent.force_cleanup_all_images()
        return {"status": "success", "message": "Đã dọn dẹp thư mục tạm thời"}
    except Exception as e:
        return {"status": "error", "message": f"Lỗi khi dọn dẹp: {str(e)}"}

@app.get("/temp_status")
async def temp_status():
    """Check temporary directory status"""
    try:
        temp_dir = image_agent.temp_dir
        if os.path.exists(temp_dir):
            files = os.listdir(temp_dir)
            return {
                "temp_dir": temp_dir,
                "exists": True,
                "file_count": len(files),
                "files": files[:10]  # Show first 10 files only
            }
        else:
            return {
                "temp_dir": temp_dir,
                "exists": False,
                "file_count": 0,
                "files": []
            }
    except Exception as e:
        return {"error": str(e)}

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    image_agent.cleanup_temp_dir()

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8005,
        log_level="info"
    )
