import os
import uuid
import torch
import ChatTTS
import torchaudio
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

# 初始化 FastAPI 应用
app = FastAPI(title="文字转语音 API", description="基于ChatTTS的文字转语音服务")

# 初始化 ChatTTS
chat = ChatTTS.Chat()
chat.load(compile=False, custom_path="models/Test To TTS/ChatTTS")  # 生产环境可以设为 True 以提高性能

# 确保存在输出目录
os.makedirs("server_tts/speakers", exist_ok=True)
os.makedirs("server_tts/output", exist_ok=True)

# 请求模型
class TTSRequest(BaseModel):
    text: str
    speaker_id: Optional[str] = None
    temperature: float = 0.1
    top_p: float = 0.5
    top_k: int = 10
    prompt: str = '[oral_0][laugh_0][break_3]'  # 清晰发音，无笑声，适中停顿


def generate_speech(text, speaker_id=None, temperature=0.1, top_p=0.5, top_k=10, prompt='[oral_0][laugh_0][break_3]'):
    """
    生成语音并保存说话人特征
    
    参数:
    - text: 要转换的文本
    - speaker_id: 说话人ID，如果为None则随机生成
    - temperature: 温度参数
    - top_p, top_k: 采样参数
    - prompt: 文本修饰提示
    
    返回:
    - 音频文件路径
    """
    # 确定说话人特征
    if speaker_id and os.path.exists(f"server_tts/speakers/{speaker_id}.pt"):
        # 加载已有说话人特征
        spk_emb = torch.load(f"server_tts/speakers/{speaker_id}.pt")
        print(f"使用已保存的说话人特征: {speaker_id}")
    else:
        # 随机生成新的说话人特征
        spk_emb = chat.sample_random_speaker()
        # 为新的说话人特征生成ID并保存
        if not speaker_id:
            speaker_id = f"speaker_{uuid.uuid4().hex[:8]}"
        torch.save(spk_emb, f"server_tts/speakers/{speaker_id}.pt")
        print(f"生成并保存新的说话人特征: {speaker_id}")
    
    # 生成参数配置
    params_infer = ChatTTS.Chat.InferCodeParams(
        spk_emb=spk_emb,  # 说话人特征
        temperature=temperature,
        top_P=top_p,
        top_K=top_k,
    )
    
    # 文本修饰配置
    params_refine_text = ChatTTS.Chat.RefineTextParams(
        prompt=prompt,
    )
    
    # 生成语音 (使用解码器)
    wavs = chat.infer(
        [text],
        params_refine_text=params_refine_text,
        params_infer_code=params_infer,
        use_decoder=True  # 使用解码器获得更好的质量
    )
    
    # 保存音频
    filename = f"server_tts/output/{speaker_id}_{uuid.uuid4().hex[:8]}.wav"
    torchaudio.save(filename, torch.from_numpy(wavs[0]).unsqueeze(0), 24000)
    print(f"音频已保存为 {filename}")
    
    return filename


@app.get("/speakers")
async def list_speakers():
    """获取所有可用的说话人列表"""
    speakers = []
    for file in os.listdir("server-tts/speakers"):
        if file.endswith(".pt"):
            speakers.append(file.replace(".pt", ""))
    return {"speakers": speakers}


@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """将文本转换为语音"""
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
        raise HTTPException(status_code=500, detail=f"生成语音失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 