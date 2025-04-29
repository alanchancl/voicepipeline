import torch
import ChatTTS
import torchaudio
import os
import time

chat = ChatTTS.Chat()
chat.load(compile=False, custom_path="models/Test To TTS/ChatTTS")  # Set to True for better performance

# 要转换的文本
texts = ["中国电信的宽带又快又好，服务又好，价格又便宜，真是太棒了！你考虑办一个吗，现在办理还有优惠哦！真的很便宜！"]

# 说话人特征目录
os.makedirs("speakers", exist_ok=True)

# 使用说话人特征或随机生成
def generate_speech(text_list, speaker_id=None):
    """
    生成语音并保存说话人特征
    
    参数:
    - text_list: 文本列表
    - speaker_id: 说话人ID，如果为None则随机生成
    
    返回:
    - wavs: 生成的音频
    - spk_emb: 使用的说话人特征
    """
    # 确定说话人特征
    if speaker_id and os.path.exists(f"speakers/{speaker_id}.pt"):
        # 加载已有说话人特征
        spk_emb = torch.load(f"speakers/{speaker_id}.pt")
        print(f"使用已保存的说话人特征: {speaker_id}")
    else:
        # 随机生成新的说话人特征
        spk_emb = chat.sample_random_speaker()
        # 为新的说话人特征生成ID并保存
        if not speaker_id:
            speaker_id = f"speaker_{int(time.time())}"
        torch.save(spk_emb, f"speakers/{speaker_id}.pt")
        print(f"生成并保存新的说话人特征: {speaker_id}")
    
    # 生成参数配置
    params_infer = ChatTTS.Chat.InferCodeParams(
        spk_emb=spk_emb,  # 说话人特征
        temperature=0.1,  # 低温度，更稳定清晰
        top_P=0.5,
        top_K=10,
    )
    
    # 文本修饰配置
    params_refine_text = ChatTTS.Chat.RefineTextParams(
        prompt='[oral_0][laugh_0][break_3]',  # 清晰发音，无笑声，适中停顿
    )
    
    # 生成语音 (使用解码器)
    wavs = chat.infer(
        text_list,
        params_refine_text=params_refine_text,
        params_infer_code=params_infer,
        use_decoder=True  # 使用解码器获得更好的质量
    )
    
    # 保存音频
    for i, wav in enumerate(wavs):
        filename = f"output_{speaker_id}_{i}.wav"
        torchaudio.save(filename, torch.from_numpy(wav).unsqueeze(0), 24000)
        print(f"音频已保存为 {filename}")
    
    return wavs, spk_emb


# 示例3: 生成一个新的说话人特征并将其保存为"最喜欢的"
wavs3, favorite_spk = generate_speech(texts, "favorite")

print("\n可用的说话人特征文件:")
for file in os.listdir("speakers"):
    if file.endswith(".pt"):
        print(f"- {file}")
