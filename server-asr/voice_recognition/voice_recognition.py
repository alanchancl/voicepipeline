#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import torch
import torchaudio
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import subprocess
import sys
import argparse
import logging

# 检查并安装必要的库
try:
    # 在实际使用时导入，避免未使用的导入警告
    pass
except ImportError:
    print("正在安装transformers库...")
    subprocess.check_call(["pip", "install", "transformers"])

# 声纹数据和模型目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.dirname(BASE_DIR)  # server-asr目录
PROJECT_ROOT = os.path.dirname(SERVER_DIR)  # 项目根目录

VOICE_SAMPLES_DIR = os.path.join(BASE_DIR, 'voice_samples')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_NAME = 'wav2vec2-large-960h'  # 更改为更精确的模型名称
LOCAL_MODEL_DIR = os.path.join(MODEL_DIR, MODEL_NAME)
CENTRALIZED_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "Automatic Speech Recognition", MODEL_NAME)  # 集中式模型目录
HF_MODEL_ID = 'facebook/wav2vec2-large-960h'  # 使用更好的预训练模型

# 创建必要的目录
if not os.path.exists(VOICE_SAMPLES_DIR):
    os.makedirs(VOICE_SAMPLES_DIR)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(LOCAL_MODEL_DIR):
    os.makedirs(LOCAL_MODEL_DIR)

# 特征缓存文件，避免重复提取特征
FEATURES_CACHE = os.path.join(BASE_DIR, 'features_cache.pkl')

# 全局模型缓存
_model_cache = None

# 使用预训练的声纹识别模型进行特征提取
def load_pretrained_model(gpu_id=None):
    """
    加载预训练的声纹识别模型
    
    Args:
        gpu_id: 指定GPU ID，如果不指定则使用默认设备
        
    Returns:
        加载的模型对象
    """
    # 实际使用时导入，避免未使用的导入警告
    from transformers import Wav2Vec2Model
    
    global _model_cache
    
    # 如果已缓存模型，直接返回
    if _model_cache is not None:
        print("使用已缓存的声纹识别模型")
        return _model_cache
        
    print("正在加载声纹识别模型...")
    
    # 设置当前GPU
    if gpu_id is not None and torch.cuda.is_available():
        if gpu_id >= 0 and gpu_id < torch.cuda.device_count():
            print(f"指定使用GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
            torch.cuda.set_device(gpu_id)
        else:
            print(f"警告: 指定的GPU {gpu_id} 不存在，将使用默认GPU")
            gpu_id = None
    
    # 首先尝试从集中式模型目录加载
    centralized_model_path_safetensors = os.path.join(CENTRALIZED_MODEL_DIR, 'model.safetensors')
    centralized_model_path_pytorch = os.path.join(CENTRALIZED_MODEL_DIR, 'pytorch_model.bin')
    centralized_config_path = os.path.join(CENTRALIZED_MODEL_DIR, 'config.json')
    
    # 检查集中式模型目录
    if ((os.path.exists(centralized_model_path_safetensors) or os.path.exists(centralized_model_path_pytorch)) 
            and os.path.exists(centralized_config_path)):
        print(f"从集中式模型目录加载: {CENTRALIZED_MODEL_DIR}")
        try:
            model = Wav2Vec2Model.from_pretrained(CENTRALIZED_MODEL_DIR)
            print(f"成功从集中式目录加载模型: {CENTRALIZED_MODEL_DIR}")
            # 后续处理...
        except Exception as e:
            print(f"从集中式目录加载失败: {e}，尝试从原始位置加载")
            model = None
    else:
        print("集中式模型目录不存在或为空，检查原始位置")
        model = None
    
    # 如果集中式目录加载失败，尝试从原始位置加载
    if model is None:
        # 模型文件路径
        local_model_path_pytorch = os.path.join(LOCAL_MODEL_DIR, 'pytorch_model.bin')
        local_model_path_safetensors = os.path.join(LOCAL_MODEL_DIR, 'model.safetensors')
        local_config_path = os.path.join(LOCAL_MODEL_DIR, 'config.json')
        
        # 尝试加载本地模型
        if ((os.path.exists(local_model_path_pytorch) or os.path.exists(local_model_path_safetensors)) 
                and os.path.exists(local_config_path)):
            print("发现本地模型文件")
            if os.path.exists(local_model_path_safetensors):
                print(f"找到safetensors格式模型: {local_model_path_safetensors}")
            elif os.path.exists(local_model_path_pytorch):
                print(f"找到pytorch格式模型: {local_model_path_pytorch}")
            
            try:
                # 从本地加载模型
                model = Wav2Vec2Model.from_pretrained(LOCAL_MODEL_DIR)
                print("成功从本地加载模型: {}".format(LOCAL_MODEL_DIR))
                
            except Exception as e:
                print("加载本地模型失败: {}".format(str(e)))
                print("将从Hugging Face重新下载模型...")
                model = None
        else:
            print("本地模型文件不存在，将从Hugging Face下载")
            model = None
    
    # 如果本地加载失败，从Hugging Face下载
    if model is None:
        try:
            print(f"正在从Hugging Face下载模型: {HF_MODEL_ID}")
            # 禁用进度条，使日志更清晰
            model = Wav2Vec2Model.from_pretrained(HF_MODEL_ID, cache_dir=MODEL_DIR)
            
            # 保存到本地目录
            print(f"正在将模型保存到本地: {LOCAL_MODEL_DIR}")
            model.save_pretrained(LOCAL_MODEL_DIR)
            print(f"模型已成功下载并保存到: {LOCAL_MODEL_DIR}")
            
            # 同时保存到集中式目录
            try:
                os.makedirs(os.path.dirname(CENTRALIZED_MODEL_DIR), exist_ok=True)
                model.save_pretrained(CENTRALIZED_MODEL_DIR)
                print(f"模型同时保存到集中式目录: {CENTRALIZED_MODEL_DIR}")
            except Exception as e:
                print(f"保存到集中式目录失败: {e}")
            
        except Exception as e:
            print(f"从Hugging Face下载模型失败: {str(e)}")
            raise RuntimeError(f"无法下载模型，请检查网络连接或手动下载: {HF_MODEL_ID}")
    
    # 设置为评估模式
    model.eval()
    
    # 如果有CUDA支持，则使用GPU
    if torch.cuda.is_available():
        model = model.cuda()  # 会使用当前设置的GPU
        print(f"已将模型移动到GPU: {torch.cuda.current_device()} ({torch.cuda.get_device_name(torch.cuda.current_device())})")
        # 添加内存优化
        torch.cuda.empty_cache()
    else:
        print("在CPU上运行模型")
    
    # 缓存模型
    _model_cache = model
    print("声纹识别模型已缓存到内存")
    
    return model

# 加载或初始化特征缓存
def load_features_cache():
    """
    加载特征缓存，如果不存在则创建新的
    """
    if os.path.exists(FEATURES_CACHE):
        with open(FEATURES_CACHE, 'rb') as f:
            return pickle.load(f)
    else:
        return {}

# 保存特征缓存
def save_features_cache(cache):
    """
    保存特征缓存
    """
    with open(FEATURES_CACHE, 'wb') as f:
        pickle.dump(cache, f)

# 提取音频特征
def extract_features(audio_path, model=None):
    """
    使用预训练模型从音频文件中提取声纹特征
    
    Args:
        audio_path: 音频文件路径
        model: 预训练模型，如果为None则加载默认模型
        
    Returns:
        特征向量
    """
    try:
        # 加载音频文件
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # 重采样到16kHz (wav2vec2要求)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            sample_rate = 16000
        
        # 确保音频是单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 如果没有提供模型，则加载默认模型
        if model is None:
            model = load_pretrained_model()
        
        # 准备模型输入
        # 需要 [batch_size, sequence_length] 格式
        if waveform.dim() == 2:  # [channels, time]
            waveform = waveform.squeeze(0)  # 移除通道维度，变成 [time]
        
        # 添加batch维度
        if waveform.dim() == 1:  # [time]
            waveform = waveform.unsqueeze(0)  # 变成 [1, time]
        
        # 确保张量在正确的设备上
        device = next(model.parameters()).device
        waveform = waveform.to(device)
        
        # 使用模型提取特征
        with torch.no_grad():
            outputs = model(waveform, output_hidden_states=True)
            
            # 获取最后一层的隐藏状态
            last_hidden = outputs.hidden_states[-1]  # [batch_size, sequence_length, hidden_size]
            
            # 计算平均值作为声纹特征
            features = last_hidden.mean(dim=1).cpu().numpy()  # [batch_size, hidden_size]
        
        return features.flatten()
        
    except Exception as e:
        print(f"特征提取失败: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# 获取指定文件夹中所有营业员的声纹特征
def get_all_employee_features():
    """
    从voice_samples目录中获取所有营业员的声纹特征
    
    Returns:
        员工ID到特征向量的字典映射
    """
    employee_features = {}
    features_cache = load_features_cache()
    model = load_pretrained_model()  # 使用缓存模型
    cache_updated = False
    
    # 遍历voice_samples目录下的所有子目录（每个子目录代表一个营业员）
    for employee_dir in glob.glob(os.path.join(VOICE_SAMPLES_DIR, '*')):
        if os.path.isdir(employee_dir):
            employee_id = os.path.basename(employee_dir)
            employee_features[employee_id] = []
            
            # 遍历该员工目录下的所有WAV文件
            for wav_file in glob.glob(os.path.join(employee_dir, '*.wav')):
                # 检查缓存中是否已有该文件的特征
                if wav_file in features_cache:
                    features = features_cache[wav_file]
                else:
                    # 提取特征并缓存
                    features = extract_features(wav_file, model)
                    features_cache[wav_file] = features
                    cache_updated = True
                
                employee_features[employee_id].append(features)
    
    # 如果缓存有更新，则保存
    if cache_updated:
        save_features_cache(features_cache)
    
    return employee_features

# 计算两个特征向量的相似度
def compute_similarity(features1, features2):
    """
    计算两个特征向量的余弦相似度
    
    Args:
        features1: 特征向量1
        features2: 特征向量2
        
    Returns:
        相似度分数 (0-1)
    """
    return float(cosine_similarity([features1], [features2])[0][0])

# 验证声纹
def verify_voice(audio_path, threshold=0.7, employee_id=None, gpu_id=None):
    """
    验证音频中的声音是否匹配已注册的员工声纹
    
    Args:
        audio_path: 待验证的音频文件路径
        threshold: 相似度阈值，高于此值视为匹配
        employee_id: 待验证的特定员工ID，如果为None则尝试匹配所有员工
        gpu_id: 指定使用的GPU ID，如果不指定则使用默认设备
        
    Returns:
        dict: 验证结果，包含验证状态、匹配的员工ID和相似度分数
    """
    try:
        # 加载预训练模型
        model = load_pretrained_model(gpu_id)
        
        # 加载并提取测试音频的特征向量
        audio_data = extract_features(audio_path, model)
        if audio_data is None:
            return {"verified": False, "employee_id": None, "similarity": 0, "error": "无法加载音频文件"}
            
        test_embedding = audio_data
        
        # 如果指定了员工ID，只验证该员工
        if employee_id:
            database_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_database")
            employee_file = os.path.join(database_dir, f"{employee_id}.pkl")
            
            if not os.path.exists(employee_file):
                return {"verified": False, "employee_id": employee_id, "similarity": 0, "error": "未找到该员工的声纹记录"}
                
            with open(employee_file, 'rb') as f:
                features_list = pickle.load(f)
                
            # 计算与已注册特征的相似度
            max_similarity = 0
            for stored_embedding in features_list:
                similarity = cosine_similarity([test_embedding], [stored_embedding])[0][0]
                max_similarity = max(max_similarity, similarity)
                
            # 根据阈值判断是否验证通过
            verified = bool(max_similarity >= threshold)  # 显式转换为Python布尔类型
            return {
                "verified": verified,
                "employee_id": employee_id,
                "similarity": float(max_similarity)
            }
        
        # 如果未指定员工ID，尝试匹配所有员工
        else:
            employee_features = get_all_employee_features()
            best_match = None
            best_similarity = 0
            
            for emp_id, features_list in employee_features.items():
                for stored_embedding in features_list:
                    similarity = cosine_similarity([test_embedding], [stored_embedding])[0][0]
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = emp_id
                        
            # 根据阈值判断是否验证通过
            verified = bool(best_similarity >= threshold)  # 显式转换为Python布尔类型
            return {
                "verified": verified,
                "employee_id": best_match,
                "similarity": float(best_similarity)
            }
            
    except Exception as e:
        logging.error(f"声纹验证时出错: {str(e)}")
        return {"verified": False, "employee_id": None, "similarity": 0, "error": str(e)}

# 添加声纹样本
def add_voice_sample(audio_path, employee_id, gpu_id=None):
    """
    添加员工声纹样本到数据库
    
    Args:
        audio_path: 音频文件路径
        employee_id: 员工ID
        gpu_id: 指定使用的GPU ID，如果不指定则使用默认设备
        
    Returns:
        bool: 是否成功
    """
    try:
        # 加载预训练模型
        model = load_pretrained_model(gpu_id)
        
        # 提取音频特征
        audio_data = extract_features(audio_path, model)
        if audio_data is None:
            logging.error(f"无法加载音频文件: {audio_path}")
            return False
            
        embedding = audio_data
        
        # 保存员工声纹特征
        database_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_database")
        os.makedirs(database_dir, exist_ok=True)
        
        employee_file = os.path.join(database_dir, f"{employee_id}.pkl")
        
        # 读取现有特征或创建新列表
        features_list = []
        if os.path.exists(employee_file):
            with open(employee_file, 'rb') as f:
                features_list = pickle.load(f)
                
        # 添加新的特征向量
        features_list.append(embedding)
        
        # 保存更新后的特征列表
        with open(employee_file, 'wb') as f:
            pickle.dump(features_list, f)
            
        return True
    except Exception as e:
        logging.error(f"添加声纹样本时出错: {str(e)}")
        return False

def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description="声纹识别工具")
    
    # 创建子命令解析器
    subparsers = parser.add_subparsers(dest="action", help="操作类型")
    
    # 下载模型子命令
    download_parser = subparsers.add_parser("download", help="下载预训练模型")
    
    # 列出声纹子命令
    list_parser = subparsers.add_parser("list", help="列出已注册的声纹")
    
    # 注册声纹子命令
    enroll_parser = subparsers.add_parser("enroll", help="注册新的声纹")
    enroll_parser.add_argument("--name", type=str, required=True, help="声纹名称")
    enroll_parser.add_argument("--audio", type=str, required=True, help="声音文件路径")
    
    # 验证身份子命令
    verify_parser = subparsers.add_parser("verify", help="验证声纹身份")
    verify_parser.add_argument("--audio", type=str, required=True, help="要验证的声音文件路径")
    verify_parser.add_argument("--name", type=str, help="指定要验证的声纹名称(可选)")
    verify_parser.add_argument("--threshold", type=float, default=0.75, help="匹配阈值(0-1之间,越高要求越严格)")
    
    # 对所有子命令添加GPU ID参数
    for subparser in [download_parser, enroll_parser, verify_parser]:
        subparser.add_argument("--gpu-id", type=int, help="指定使用的GPU ID")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    if args.action is None:
        parser.print_help()
        return
    
    # 处理下载模型请求
    if args.action == "download":
        print("开始下载预训练模型...")
        gpu_id = getattr(args, 'gpu_id', None)
        load_pretrained_model(gpu_id)
        print("模型下载完成")
        return
    
    # 处理列出声纹请求
    if args.action == "list":
        print("已注册的声纹列表:")
        embeddings_dict = get_all_employee_features()
        
        if not embeddings_dict:
            print("  没有找到已注册的声纹")
        else:
            for employee_id, features_list in embeddings_dict.items():
                print(f"  - 员工ID: {employee_id}, 样本数量: {len(features_list)}")
        return
    
    # 处理注册声纹请求
    if args.action == "enroll":
        name = args.name
        audio_path = args.audio
        
        if not os.path.exists(audio_path):
            print(f"错误: 音频文件不存在: {audio_path}")
            return
        
        print(f"正在处理声音样本: {audio_path}")
        
        try:
            gpu_id = getattr(args, 'gpu_id', None)
            add_voice_sample(audio_path, name, gpu_id)
            print(f"成功注册声纹: {name}")
        except Exception as e:
            print(f"注册声纹时出错: {str(e)}")
        return
    
    # 处理验证声纹请求
    if args.action == "verify":
        audio_path = args.audio
        name = args.name  # 可以是None
        threshold = args.threshold
        
        if not os.path.exists(audio_path):
            print(f"错误: 音频文件不存在: {audio_path}")
            return
        
        print(f"正在分析声音样本: {audio_path}")
        
        try:
            gpu_id = getattr(args, 'gpu_id', None)
            result = verify_voice(audio_path, threshold, name, gpu_id)
            if result['verified']:
                print(f"验证结果: 匹配 {result['employee_id']} (相似度: {result['similarity']:.4f})")
            else:
                print(f"验证结果: 不匹配 {result['employee_id']} (相似度: {result['similarity']:.4f})")
        except Exception as e:
            print(f"验证声纹时出错: {str(e)}")
        return

if __name__ == "__main__":
    main() 