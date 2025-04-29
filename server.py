#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import uuid
import torch
import torchaudio
import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import logging
from logging.handlers import RotatingFileHandler
import argparse
import datetime
import json

# 添加当前目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 全局变量
FORWARD_ENABLED = False
FORWARD_IP = None
FORWARD_PORT = None
FORWARD_PATH = None

# 数据目录配置
DATA_DIR = os.path.join(current_dir, 'data')
STT_AUDIO_DIR = os.path.join(DATA_DIR, 'stt', 'audio')
STT_TRANSCRIPT_DIR = os.path.join(DATA_DIR, 'stt', 'transcripts')
TTS_INPUT_DIR = os.path.join(DATA_DIR, 'tts', 'input')
TTS_OUTPUT_DIR = os.path.join(DATA_DIR, 'tts', 'output')
METADATA_DIR = os.path.join(DATA_DIR, 'metadata')

# 创建必要的目录
for directory in [STT_AUDIO_DIR, STT_TRANSCRIPT_DIR, TTS_INPUT_DIR, TTS_OUTPUT_DIR, METADATA_DIR]:
    os.makedirs(directory, exist_ok=True)

# 导入STT和TTS模块
from server_asr.voice_recognition.voice_recognition import verify_voice
from server_asr.voice_to_text.speech_to_text import convert_speech_to_text, preprocess_audio, load_asr_model
import ChatTTS

# 创建FastAPI应用
app = FastAPI(
    title="语音处理服务",
    description="集成语音转文字(STT)和文字转语音(TTS)的服务"
)

# 配置日志
log_file = os.path.join(current_dir, 'logs', 'server.log')
os.makedirs(os.path.dirname(log_file), exist_ok=True)

handler = RotatingFileHandler(log_file, maxBytes=10000000, backupCount=5, encoding='utf-8')
handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
handler.setLevel(logging.INFO)

# 添加控制台日志处理器
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s'
))
console_handler.setLevel(logging.INFO)

# 配置日志记录器
logger = logging.getLogger("server")
logger.addHandler(handler)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)

# 解析命令行参数
parser = argparse.ArgumentParser(description='语音处理服务器')
parser.add_argument('--gpu', type=int, default=None, help='指定使用的GPU ID')
parser.add_argument('--host', type=str, default='0.0.0.0', help='服务器监听地址')
parser.add_argument('--port', type=int, default=8000, help='服务器监听端口')
parser.add_argument('--forward-ip', type=str, default='localhost', help='转发目标IP地址')
parser.add_argument('--forward-port', type=int, default=20252, help='转发目标端口号')
parser.add_argument('--forward-path', type=str, default='/api/v1/ai-terminal', help='转发URL路径')
args = parser.parse_args()

# 配置
UPLOAD_FOLDER = os.path.join(current_dir, 'uploads')
ALLOWED_EXTENSIONS = {'wav'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
GPU_ID = args.gpu

# 确保必要的目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("server_tts/speakers", exist_ok=True)
os.makedirs("server_tts/output", exist_ok=True)

# 转发配置
FORWARD_ENABLED = True if args.forward_ip else False
FORWARD_IP = args.forward_ip
FORWARD_PORT = args.forward_port
FORWARD_PATH = args.forward_path

# 初始化TTS
chat = ChatTTS.Chat()
chat.load(compile=False, custom_path="models/Test To TTS/ChatTTS")


# 请求模型
class TTSRequest(BaseModel):
    text: str
    speaker_id: Optional[str] = None
    temperature: float = 0.1
    top_p: float = 0.5
    top_k: int = 10
    prompt: str = '[oral_0][laugh_0][break_3]'


def allowed_file(filename: str) -> bool:
    """检查文件类型是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def forward_result(transcription: str) -> bool:
    """转发识别结果到目标地址"""
    if not FORWARD_ENABLED or not FORWARD_IP:
        return False
    
    path = FORWARD_PATH.lstrip('/') if FORWARD_PATH else "api/v1/ai-terminal"
    target_url = f"http://{FORWARD_IP}:{FORWARD_PORT}/{path}"
    
    try:
        headers = {'Content-Type': 'application/json'}
        response = requests.post(
            target_url,
            json={"data": transcription},
            headers=headers,
            timeout=5
        )
        
        if response.status_code in [200, 201, 202]:
            logger.info(f"已转发识别结果到URL: {target_url}")
            return True
        else:
            logger.warning(f"转发到URL失败，状态码: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"转发结果到URL失败: {e}")
        return False


def save_metadata(metadata: dict, prefix: str):
    """保存元数据到JSON文件"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}_{uuid.uuid4().hex[:8]}.json"
    filepath = os.path.join(METADATA_DIR, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    return filepath


def process_voice(audio_path: str, threshold: float = 0.7, preprocess: bool = True, domain: str = "telecom"):
    """处理语音文件，进行声纹识别和语音转文字"""
    start_time = time.time()
    result = {"status": "success", "audio_path": audio_path}
    
    try:
        # 声纹识别
        logger.info(f"开始声纹识别: {audio_path}")
        voiceprint_result = verify_voice(audio_path, threshold=threshold)
        result["voiceprint"] = voiceprint_result
        logger.info(f"声纹识别完成: {voiceprint_result['verified']}")
        
        # 语音转文字
        logger.info(f"开始语音转文字: {audio_path}")
        if preprocess:
            processed_path = preprocess_audio(audio_path)
            text = convert_speech_to_text(processed_path, domain=domain, gpu_id=GPU_ID)
            if os.path.exists(processed_path) and processed_path != audio_path:
                try:
                    os.remove(processed_path)
                except (OSError, IOError) as e:
                    logger.warning(f"清理预处理文件失败: {e}")
        else:
            text = convert_speech_to_text(audio_path, domain=domain, gpu_id=GPU_ID)
        
        if not text or text.strip() == "":
            logger.warning(f"语音转文字结果为空: {audio_path}")
            result["transcription"] = ""
            result["has_transcription"] = False
        else:
            # 保存转写结果到文件
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            transcript_filename = f"transcript_{timestamp}_{uuid.uuid4().hex[:8]}.txt"
            transcript_path = os.path.join(STT_TRANSCRIPT_DIR, transcript_filename)
            
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            result["transcription"] = text
            result["transcript_file"] = transcript_path
            result["has_transcription"] = True
            result["domain"] = domain
            logger.info(f"语音转文字完成: {text[:50]}...")
            
            if FORWARD_ENABLED:
                forward_success = forward_result(text)
                result["forwarded"] = forward_success
        
        # 保存处理元数据
        metadata = {
            "timestamp": datetime.datetime.now().isoformat(),
            "audio_file": audio_path,
            "processing_time": time.time() - start_time,
            "result": result
        }
        save_metadata(metadata, "stt_process")
        
    except Exception as e:
        logger.error(f"处理语音出错: {str(e)}", exc_info=True)
        result["status"] = "error"
        result["error"] = str(e)
        result["has_transcription"] = False
    
    result["processing_time"] = time.time() - start_time
    return result


def generate_speech(text: str, speaker_id: Optional[str] = None, temperature: float = 0.1,
                    top_p: float = 0.5, top_k: int = 10, prompt: str = '[oral_0][laugh_0][break_3]'):
    """生成语音并保存说话人特征"""
    # 保存输入文本
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    input_filename = f"input_{timestamp}_{uuid.uuid4().hex[:8]}.txt"
    input_path = os.path.join(TTS_INPUT_DIR, input_filename)
    
    with open(input_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    if speaker_id and os.path.exists(f"server_tts/speakers/{speaker_id}.pt"):
        spk_emb = torch.load(f"server_tts/speakers/{speaker_id}.pt")
        logger.info(f"使用已保存的说话人特征: {speaker_id}")
    else:
        spk_emb = chat.sample_random_speaker()
        if not speaker_id:
            speaker_id = f"speaker_{uuid.uuid4().hex[:8]}"
        torch.save(spk_emb, f"server_tts/speakers/{speaker_id}.pt")
        logger.info(f"生成并保存新的说话人特征: {speaker_id}")
    
    params_infer = ChatTTS.Chat.InferCodeParams(
        spk_emb=spk_emb,
        temperature=temperature,
        top_P=top_p,
        top_K=top_k,
    )
    
    params_refine_text = ChatTTS.Chat.RefineTextParams(
        prompt=prompt,
    )
    
    wavs = chat.infer(
        [text],
        params_refine_text=params_refine_text,
        params_infer_code=params_infer,
        use_decoder=True
    )
    
    output_filename = f"output_{timestamp}_{uuid.uuid4().hex[:8]}.wav"
    output_path = os.path.join(TTS_OUTPUT_DIR, output_filename)
    torchaudio.save(output_path, torch.from_numpy(wavs[0]).unsqueeze(0), 24000)
    logger.info(f"音频已保存为 {output_path}")
    
    # 保存处理元数据
    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "input_file": input_path,
        "output_file": output_path,
        "speaker_id": speaker_id,
        "parameters": {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "prompt": prompt
        }
    }
    save_metadata(metadata, "tts_process")
    
    return output_path


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return JSONResponse({
        "status": "ok",
        "time": datetime.datetime.now().isoformat()
    })


@app.post("/stt")
async def speech_to_text(
    file: UploadFile = File(...),
    domain: str = Form("telecom"),
    forward: bool = Form(True)
):
    """语音转文字接口"""
    global FORWARD_ENABLED
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="未选择文件")
    
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="不支持的文件类型，仅支持WAV格式")
    
    try:
        original_filename = os.path.basename(file.filename)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"audio_{timestamp}_{uuid.uuid4().hex[:8]}_{original_filename}"
        filepath = os.path.join(STT_AUDIO_DIR, filename)
        
        with open(filepath, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"文件已上传并保存为: {filepath}")
        
        # 保存原始转发设置
        original_forward_enabled = FORWARD_ENABLED
        
        # 根据请求参数决定是否进行转发
        if not forward:
            FORWARD_ENABLED = False
        
        result = process_voice(filepath, domain=domain)
        
        # 恢复原始转发设置
        FORWARD_ENABLED = original_forward_enabled
        
        if result.get("has_transcription", False) and result.get("status") == "success":
            logger.info("成功处理音频，返回结果")
            return result
        else:
            logger.info("音频未成功转录或处理失败，返回空结果")
            return {"status": "success", "transcription": "", "has_transcription": False}
    
    except Exception as e:
        logger.error(f"处理请求出错: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理请求出错: {str(e)}")


@app.get("/speakers")
async def list_speakers():
    """获取所有可用的说话人列表"""
    speakers = []
    for file in os.listdir("server_tts/speakers"):
        if file.endswith(".pt"):
            speakers.append(file.replace(".pt", ""))
    return {"speakers": speakers}


@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """文字转语音接口"""
    try:
        output_file = generate_speech(
            text=request.text,
            speaker_id=request.speaker_id,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            prompt=request.prompt
        )
        
        return FileResponse(
            path=output_file,
            media_type="audio/wav",
            filename=os.path.basename(output_file)
        )
    except Exception as e:
        logger.error(f"生成语音失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"生成语音失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    # 打印启动信息
    logger.info(f"服务器正在启动，监听 {args.host}:{args.port}")
    if GPU_ID is not None:
        logger.info(f"使用GPU {GPU_ID} 进行语音识别")
    else:
        if torch.cuda.is_available():
            logger.info("使用系统分配的默认GPU")
        else:
            logger.info("使用CPU模式运行")
    
    if FORWARD_ENABLED:
        logger.info(f"转发已启用，目标: http://{FORWARD_IP}:{FORWARD_PORT}/{FORWARD_PATH.lstrip('/')}")
    else:
        logger.info("转发功能未启用")
    
    # 预加载STT模型
    logger.info("正在预加载语音识别模型...")
    try:
        # 预加载语音识别模型
        _, _ = load_asr_model(gpu_id=GPU_ID)
        logger.info("语音识别模型加载完成")
    except Exception as e:
        logger.error(f"预加载语音识别模型失败: {str(e)}", exc_info=True)
    
    # 预热TTS模型
    logger.info("正在预热TTS模型...")
    try:
        # 生成一个非常短的示例文本来预热模型
        spk_emb = chat.sample_random_speaker()
        params_infer = ChatTTS.Chat.InferCodeParams(
            spk_emb=spk_emb,
            temperature=0.1,
            top_P=0.5,
            top_K=10,
        )
        params_refine_text = ChatTTS.Chat.RefineTextParams(
            prompt="[oral_0][laugh_0][break_3]",
        )
        _ = chat.infer(
            ["欢迎使用语音系统"],
            params_refine_text=params_refine_text,
            params_infer_code=params_infer,
            use_decoder=True
        )
        logger.info("TTS模型预热完成")
    except Exception as e:
        logger.error(f"TTS模型预热失败: {str(e)}", exc_info=True)
    
    uvicorn.run(app, host=args.host, port=args.port)