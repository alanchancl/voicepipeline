#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import librosa
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import shutil
import argparse
import json
import jieba
import sys
import importlib
from custom_vocabulary import get_vocabulary_manager
# 模型路径和配置
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.dirname(MODULE_DIR)
PROJECT_ROOT = os.path.dirname(SERVER_DIR)  # 项目根目录

# 集中式模型目录和原始模型目录
CENTRALIZED_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "Voiceprint Recognition", "wav2vec2-large-xlsr-53-chinese-zh-cn")
MODEL_DIR = os.path.join(MODULE_DIR, 'models')
MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
LOCAL_MODEL_DIR = os.path.join(MODEL_DIR, "wav2vec2-large-xlsr-53-chinese-zh-cn")

# 使用集中式模型目录（如果存在），否则使用原始路径
ACTIVE_MODEL_DIR = CENTRALIZED_MODEL_DIR if os.path.exists(CENTRALIZED_MODEL_DIR) else LOCAL_MODEL_DIR

# 创建模型目录
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# 全局模型和处理器缓存
_model_cache = None
_processor_cache = None
_current_gpu_id = None  # 记录当前使用的GPU ID

# 初始化词库和jieba分词
def init_vocabulary():
    """初始化词库并加载到jieba分词中"""

    # 获取词库管理器
    vocab_manager = get_vocabulary_manager()
    # 获取所有术语并添加到jieba分词词典
    terms = vocab_manager.get_all_terms()
    for term in terms:
        jieba.add_word(term)
    print(f"已加载{len(terms)}个专业术语到分词系统")

# 在模块导入时初始化词库，但确保在所有函数定义完成后再调用
# 将init_vocabulary()的调用移到模块最后

# 加载语音识别模型
def load_asr_model(gpu_id=None):
    """
    加载语音识别模型，优先使用本地模型
    
    Args:
        gpu_id: 指定使用的GPU ID，None表示使用默认GPU
    
    Returns:
        (model, processor) 元组
    """
    global _model_cache, _processor_cache, _current_gpu_id

    # 如果已缓存模型且GPU设置未变，直接返回
    if (_model_cache is not None and _processor_cache is not None and
            _current_gpu_id == gpu_id):
        return _model_cache, _processor_cache

    # 如果缓存了模型但GPU设置变了，需要重新加载到指定GPU
    if _model_cache is not None and _current_gpu_id != gpu_id:
        print(f"GPU设置已更改: {_current_gpu_id} -> {gpu_id}，重新加载模型")
        _model_cache = None
        _processor_cache = None

    # 设置当前GPU
    if gpu_id is not None and torch.cuda.is_available():
        if gpu_id >= 0 and gpu_id < torch.cuda.device_count():
            print(f"指定使用GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
            torch.cuda.set_device(gpu_id)
        else:
            print(f"警告: 指定的GPU {gpu_id} 不存在，将使用默认GPU")
            gpu_id = None

    # 首先尝试从集中式模型目录加载
    if os.path.exists(CENTRALIZED_MODEL_DIR) and os.listdir(CENTRALIZED_MODEL_DIR):
        print(f"从集中式模型目录加载: {CENTRALIZED_MODEL_DIR}")
        try:
            processor = Wav2Vec2Processor.from_pretrained(CENTRALIZED_MODEL_DIR)
            model = Wav2Vec2ForCTC.from_pretrained(CENTRALIZED_MODEL_DIR)
            print("成功从集中式模型目录加载模型")
        except Exception as e:
            print(f"从集中式目录加载失败: {e}，尝试从原始位置加载")
            # 如果集中式目录加载失败，尝试从原始位置或下载
            model, processor = None, None
    else:
        # 检查原始模型目录是否存在
        if not os.path.exists(LOCAL_MODEL_DIR) or not os.listdir(LOCAL_MODEL_DIR):
            print(f"正在下载并保存语音识别模型到: {LOCAL_MODEL_DIR}")
            # 下载模型并保存到本地
            processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
            model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)

            # 保存模型和处理器到本地
            processor.save_pretrained(LOCAL_MODEL_DIR)
            model.save_pretrained(LOCAL_MODEL_DIR)
            print("语音识别模型已保存到本地")

            # 尝试同时保存到集中式目录
            try:
                os.makedirs(os.path.dirname(CENTRALIZED_MODEL_DIR), exist_ok=True)
                processor.save_pretrained(CENTRALIZED_MODEL_DIR)
                model.save_pretrained(CENTRALIZED_MODEL_DIR)
                print(f"模型同时保存到集中式目录: {CENTRALIZED_MODEL_DIR}")
            except Exception as e:
                print(f"保存到集中式目录失败: {e}")
        else:
            print(f"正在从原始位置加载模型: {LOCAL_MODEL_DIR}")
            # 从本地加载
            processor = Wav2Vec2Processor.from_pretrained(LOCAL_MODEL_DIR)
            model = Wav2Vec2ForCTC.from_pretrained(LOCAL_MODEL_DIR)

    # 如果有CUDA支持，则使用GPU
    if torch.cuda.is_available():
        model = model.cuda()  # 会使用当前设置的GPU
        print(f"已将模型加载到GPU: {torch.cuda.current_device()} ({torch.cuda.get_device_name()})")
        # 添加内存优化
        torch.cuda.empty_cache()
    else:
        print("CUDA不可用，使用CPU")

    # 更新全局缓存
    _model_cache = model
    _processor_cache = processor
    _current_gpu_id = gpu_id

    return model, processor

def calculate_pinyin_similarity(pinyin1, pinyin2):
    """
    计算两个拼音之间的相似度，考虑声母、韵母和声调
    
    Args:
        pinyin1: 第一个拼音字符串
        pinyin2: 第二个拼音字符串
        
    Returns:
        float: 相似度分数 (0-1)
    """
    # 如果完全相同，直接返回1.0
    if pinyin1 == pinyin2:
        return 1.0

    # 拼音长度差异过大，可能是完全不同的词
    len_diff = abs(len(pinyin1) - len(pinyin2))
    if len_diff > 3:
        return 0.0

    # 将拼音分解为声母、韵母和声调
    def parse_pinyin(pinyin):
        # 简单的声母列表
        initials = ['b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h',
                   'j', 'q', 'x', 'zh', 'ch', 'sh', 'r', 'z', 'c', 's', 'y', 'w']

        # 提取声调（最后一个数字）
        tone = None
        for char in reversed(pinyin):
            if char.isdigit():
                tone = int(char)
                pinyin = pinyin.replace(char, '')
                break

        # 提取声母
        initial = None
        for i in initials:
            if pinyin.startswith(i):
                initial = i
                pinyin = pinyin[len(i):]
                break

        # 剩下的部分是韵母
        final = pinyin

        return {'initial': initial, 'final': final, 'tone': tone}

    # 分解拼音
    p1 = parse_pinyin(pinyin1)
    p2 = parse_pinyin(pinyin2)

    # 计算各部分的相似性权重
    weights = {'initial': 0.3, 'final': 0.5, 'tone': 0.2}
    similarity = 0.0

    # 声母相似度
    if p1['initial'] == p2['initial']:
        similarity += weights['initial']

    # 韵母相似度 (可以进一步细化为韵头、韵腹、韵尾)
    if p1['final'] == p2['final']:
        similarity += weights['final']
    else:
        # 韵母有部分相同
        common_len = 0
        min_len = min(len(p1['final']), len(p2['final']))
        for i in range(min_len):
            if p1['final'][i] == p2['final'][i]:
                common_len += 1

        if min_len > 0:
            similarity += weights['final'] * (common_len / min_len) * 0.8

    # 声调相似度
    if p1['tone'] == p2['tone']:
        similarity += weights['tone']

    return similarity

def convert_speech_to_text(audio_path, domain=None, gpu_id=None):
    """
    将语音转换为文字
    
    Args:
        audio_path: 音频文件路径
        domain: 领域名称，用于应用特定领域的术语词库
        gpu_id: 指定使用的GPU ID，None表示使用默认GPU
        
    Returns:
        转换后的文本
    """
    try:
        # 加载模型，使用全局缓存，并指定GPU
        model, processor = load_asr_model(gpu_id)

        # 读取音频文件
        speech_array, sampling_rate = librosa.load(audio_path, sr=16000)

        # 处理音频输入
        inputs = processor(speech_array, sampling_rate=16000, return_tensors="pt", padding=True)

        # 如果有CUDA支持，则使用GPU
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")

        # 进行推理
        with torch.no_grad():
            logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

        # 获取最可能的标记
        predicted_ids = torch.argmax(logits, dim=-1)

        # 将标记转换为文本
        transcription = processor.batch_decode(predicted_ids)

        # 管理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 通用多音字纠正处理及领域词库匹配
        original_text, corrected_text, pinyin_list = apply_domain_correction(transcription[0], domain)

        print(f"语音识别结果 (领域: {domain if domain else '通用'}):")
        print(f"原始识别: {original_text}")
        print(f"改进结果: {corrected_text}")

        return corrected_text

    except Exception as e:
        print(f"语音转文字失败: {str(e)}")
        return f"语音转文字过程出错: {str(e)}"

def apply_domain_correction(text, domain=None):
    """
    应用领域词汇修正和多音字纠正
    
    Args:
        text: 原始识别文本
        domain: 领域名称
        
    Returns:
        tuple: (原始文本, 修正后的文本, 拼音列表)
    """
    if not text:
        return text, text, []

    try:
        from pypinyin import lazy_pinyin, Style
        has_pinyin = True
    except ImportError:
        has_pinyin = False
        print("警告: 未安装pypinyin模块，无法进行拼音处理")
        return text, text, []

    if not has_pinyin:
        return text, text, []

    # 分词
    words = list(jieba.cut(text))
    print(f"分词结果: {words}")

    # 获取拼音（含音调）
    pinyin_list = lazy_pinyin(text, style=Style.TONE3)
    print(f"原始拼音: {pinyin_list}")

    # 记录原始文本
    original_text = text
    corrected_text = text

    # 进行多音字纠正
    corrected_text, pinyin_list = correct_polyphone_errors(corrected_text, domain)

    # 如果指定了领域，应用领域词库进行纠正
    if domain:

        # 获取词库管理器
        vocab_manager = get_vocabulary_manager()

        # 记录要替换的词汇及其位置
        replacements = []

        # 获取领域词库中的所有术语
        domain_terms = vocab_manager.get_all_terms(vocab_type="domain", domain_name=domain)

        # 为每个分词结果计算与领域词汇的相似度
        for i, word in enumerate(words):
            if len(word) <= 1:  # 跳过单字词
                continue

            # 获取词的完整拼音
            word_pinyin = ''.join(lazy_pinyin(word, style=Style.TONE3))

            # 最佳匹配和相似度
            best_match = None
            max_similarity = 0.7  # 设置较高的阈值

            # 与领域词库中的所有词比较
            for term in domain_terms:
                if abs(len(term) - len(word)) > 2:  # 长度差异过大的跳过
                    continue

                # 计算术语的拼音
                term_pinyin = ''.join(lazy_pinyin(term, style=Style.TONE3))

                # 计算拼音相似度
                similarity = calculate_pinyin_similarity(word_pinyin, term_pinyin)
                print(f"词 '{word}' 与术语 '{term}' 的拼音相似度: {similarity}")

                # 找到更好的匹配
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = term

            # 如果找到较好的匹配，记录下来
            if best_match and best_match != word:
                print(f"✓ 领域词汇修正: '{word}' -> '{best_match}'")
                replacements.append((word, best_match))

        # 应用替换
        for old_word, new_word in replacements:
            corrected_text = corrected_text.replace(old_word, new_word)

        # 通过词库改进再次处理
        improved_text = improve_transcription(corrected_text, domain, pinyin_list)
        if improved_text != corrected_text:
            corrected_text = improved_text

    print(f"纠正后文本: {corrected_text}")

    return original_text, corrected_text, pinyin_list

def correct_polyphone_errors(text, domain=None):
    """
    纠正常见多音字错误，将多音字的拼音转换为正确的拼音
    
    Args:
        text: 原始转录文本
        domain: 领域名称，用于应用特定领域的规则
        
    Returns:
        tuple: (处理后的文本, 处理后的拼音列表)
    """
    if not text:
        return text, []

    # 1. 定义常见的多音字及其可能的读音
    # 格式: {字: {读音1: [适用上下文], 读音2: [适用上下文]}}
    POLYPHONE_DICT = {
        '长': {
            'chang2': ['长途', '长图', '长度', '长短', '长音', '长号', '长篇', '长江', '长城'],
            'zhang3': ['成长', '长大', '长高', '长辈', '年长', '长子', '茁长', '增长']
        },
        '行': {
            'xing2': ['行走', '行为', '行动', '行程', '行文', '品行', '行列'],
            'hang2': ['行业', '银行', '行情', '行市', '同行', '行号']
        },
        '数': {
            'shu4': ['数字', '数据', '数量', '数目', '数值', '计数'],
            'shuo4': ['数落', '数说']
        },
        '重': {
            'zhong4': ['重量', '重要', '重大', '重点', '沉重'],
            'chong2': ['重复', '重新', '重来', '重建', '重做']
        },
        '便': {
            'bian4': ['方便', '便利', '便当'],
            'pian2': ['便宜']
        },
        '差': {
            'cha4': ['差距', '差别', '差价', '差不多', '差错'],
            'chai1': ['出差', '差遣', '美差', '差事']
        },
        '靓': {
            'liang4': ['靓号', '靓丽', '靓女', '靓仔'],
            'jing4': ['标靓']
        }
    }

    # 处理多音字
    try:
        from pypinyin import lazy_pinyin, Style
        has_pinyin = True
    except ImportError:
        has_pinyin = False
        print("警告: 未安装pypinyin模块，无法进行多音字处理")
        return text, []

    if not has_pinyin:
        return text, []

    # 分词
    words = list(jieba.cut(text))
    print(f"分词结果: {words}")

    # 直接获取拼音
    pinyin_list = lazy_pinyin(text, style=Style.TONE3)
    print(f"原始拼音: {pinyin_list}")

    # 记录字符位置映射
    char_positions = {}
    pos = 0
    for i, word in enumerate(words):
        for j, char in enumerate(word):
            char_positions[(i, j)] = pos
            pos += 1

    # 遍历每个词，检查多音字
    for i, word in enumerate(words):
        # 检查词中是否含有多音字
        for j, char in enumerate(word):
            if char in POLYPHONE_DICT:
                char_pos = char_positions.get((i, j), 0)
                print(f"检测到多音字: '{char}' 在词 '{word}' 中，位置 {char_pos}")

                # 获取当前字符的默认拼音
                current_pinyin = pinyin_list[char_pos]
                print(f"多音字 '{char}' 当前拼音: {current_pinyin}")

                # 直接寻找词库中最匹配的词
                best_match = None
                best_reading = None

                # 先尝试完全匹配
                for reading, contexts in POLYPHONE_DICT[char].items():
                    for context in contexts:
                        # 判断当前词是否与上下文完全匹配
                        if word == context:
                            best_match = context
                            best_reading = reading
                            print(f"找到完全匹配词: '{context}'，读音 '{reading}'")
                            break
                    if best_match:
                        break

                # 如果没有完全匹配，尝试拼音匹配
                if not best_match:
                    # 获取当前词的拼音
                    word_pinyin = lazy_pinyin(word, style=Style.TONE3)

                    for reading, contexts in POLYPHONE_DICT[char].items():
                        for context in contexts:
                            # 检查多音字位置
                            if char in context:
                                # 获取上下文词的拼音
                                context_pinyin = lazy_pinyin(context,
                                                             style=Style.TONE3)

                                # 获取多音字在上下文中的位置
                                context_char_index = context.index(char)

                                # 检查多音字后面的字符拼音
                                if j + 1 < len(
                                        word) and context_char_index + 1 < len(
                                            context):
                                    # 如果多音字后面的字符拼音相同，这是一个强匹配
                                    if word_pinyin[j + 1] == context_pinyin[
                                            context_char_index + 1]:
                                        best_match = context
                                        best_reading = reading
                                        print(
                                            f"找到拼音匹配词: '{context}'，读音 '{reading}'"
                                        )
                                        print(
                                            f"  匹配位置: '{word[j+1]}({word_pinyin[j+1]})' 与 '{context[context_char_index+1]}({context_pinyin[context_char_index+1]})'"
                                        )
                                        break
                        if best_match:
                            break

                # 如果找到了匹配的词，修改多音字的拼音
                if best_reading and best_reading != current_pinyin:
                    print(
                        f"✓ 多音字 '{char}' 从 '{current_pinyin}' 改为 '{best_reading}'"
                    )
                    # 直接更新拼音列表
                    pinyin_list[char_pos] = best_reading
                else:
                    print(f"  未找到匹配词，保持多音字 '{char}' 当前拼音 '{current_pinyin}'")

    # 重新构建文本
    corrected_text = ''.join(words)

    print(f"多音字处理后拼音: {pinyin_list}")
    print(f"多音字处理后文本: {corrected_text}")

    return corrected_text, pinyin_list

def preprocess_audio(audio_path):
    """
    预处理音频文件以提高识别率
    
    Args:
        audio_path: 原始音频文件路径
        
    Returns:
        处理后的音频文件路径
    """
    try:
        # 加载音频
        y, sr = librosa.load(audio_path, sr=16000)

        # 标准化音量
        y = librosa.util.normalize(y)

        # 应用简单的降噪（去除静音部分的背景噪音）
        # 此处简化处理，实际项目中可以使用更复杂的降噪算法

        # 保存处理后的文件
        processed_path = os.path.join(os.path.dirname(audio_path),
                                     f"processed_{os.path.basename(audio_path)}")
        sf.write(processed_path, y, sr)

        return processed_path
    except Exception as e:
        print(f"音频预处理失败: {str(e)}")
        return audio_path  # 如果处理失败，返回原始文件路径

# 用于改进转录结果的函数
def improve_transcription(text, domain_name=None, pinyin=None):
    """
    使用自定义词库改进转录结果
    
    Args:
        text: 原始转录文本
        domain_name: 领域名称，用于应用特定领域的术语词库
        pinyin: 处理后的拼音列表，用于辅助判断多音字
        
    Returns:
        改进后的文本
    """
    if not text or not pinyin:
        return text

    # 获取词库管理器
    vocab_manager = get_vocabulary_manager()

    # 先用通用词库处理
    # 获取通用词库的拼音和长度映射
    common_pinyin_map = vocab_manager.get_domain_pinyin_map("common")
    common_term_by_length = vocab_manager.get_domain_length_map("common")

    print("使用通用词库和拼音辅助判断")
    improved_text = _apply_vocabulary_with_pinyin(text, pinyin, common_pinyin_map, common_term_by_length)

    # 如果指定了领域，再用领域词库处理
    if domain_name and domain_name != "common":
        # 获取领域词库的拼音和长度映射
        domain_pinyin_map = vocab_manager.get_domain_pinyin_map(domain_name)
        domain_term_by_length = vocab_manager.get_domain_length_map(domain_name)

        print(f"使用领域'{domain_name}'词库和拼音辅助判断")
        improved_text = _apply_vocabulary_with_pinyin(improved_text, pinyin, domain_pinyin_map, domain_term_by_length)

    print(f"改进前: '{text}'")
    print(f"改进后: '{improved_text}'")

    return improved_text


def _apply_vocabulary_with_pinyin(text, pinyin, term_pinyin_map,
                                  term_by_length):
    """
    使用拼音辅助应用词库进行文本修正
    
    Args:
        text: 需要修正的文本
        pinyin: 拼音列表
        term_pinyin_map: 拼音到术语的映射
        term_by_length: 按长度分组的术语列表
    
    Returns:
        修正后的文本
    """
    # 处理拼音列表
    try:
        # 将拼音列表转换为字符串形式，去除空格
        text_pinyin = ''.join([p.replace(' ', '') for p in pinyin])
        print(f"整个文本的拼音: {text_pinyin}")
    except Exception as e:
        print(f"处理拼音失败: {e}")
        text_pinyin = ''.join(pinyin)
        print(f"降级使用基本拼音: {text_pinyin}")

    # 存储替换信息
    replacements = []  # [(开始位置, 结束位置, 替换文本)]

    # 优先考虑2-4字的组合，首先尝试匹配更长的词
    prioritized_lengths = [4, 3, 2]  # 优先匹配的字长

    # 按照优先级遍历可能的词长
    for desired_len in prioritized_lengths:
        # 确保词库中有该长度的词
        if desired_len not in term_by_length:
            continue

        # 生成该长度可以匹配的拼音组合
        desired_terms = term_by_length[desired_len]

        # 根据文本长度判断可能的起始位置
        for start_pos in range(len(text) - desired_len + 1):
            end_pos = start_pos + desired_len

            # 检查当前位置是否已经被其他替换覆盖
            already_covered = False
            for r_start, r_end, _ in replacements:
                if (start_pos >= r_start and start_pos < r_end) or \
                   (end_pos > r_start and end_pos <= r_end) or \
                   (start_pos <= r_start and end_pos >= r_end):
                    already_covered = True
                    break

            if already_covered:
                continue

            # 获取当前位置的文本片段
            text_segment = text[start_pos:end_pos]

            # 从传入的pinyin参数中获取对应的拼音片段
            # 注意：pinyin列表的长度应该与text相同
            segment_pinyin = pinyin[start_pos:end_pos]
            segment_tone_pinyin = ''.join(
                [p.replace(' ', '') for p in segment_pinyin])
            print(
                f"检查片段: '{text_segment}' (位置 {start_pos}-{end_pos}), 拼音: '{segment_tone_pinyin}'"
            )

            # 寻找最佳匹配
            best_match = None
            best_similarity = 0.7  # 设置较高的阈值确保精确匹配

            # 遍历词库中指定长度的所有术语
            for term, term_pinyin in desired_terms:
                # 拼音完全匹配
                if term_pinyin == segment_tone_pinyin:
                    best_match = term
                    best_similarity = 1.0
                    break

                # 计算拼音相似度
                try:
                    # 提取声调
                    segment_tone = None
                    term_tone = None

                    for char in reversed(segment_tone_pinyin):
                        if char.isdigit():
                            segment_tone = int(char)
                            break

                    for char in reversed(term_pinyin):
                        if char.isdigit():
                            term_tone = int(char)
                            break

                    # 去除声调
                    segment_no_tone = ''.join(
                        [c for c in segment_tone_pinyin if not c.isdigit()])
                    term_no_tone = ''.join(
                        [c for c in term_pinyin if not c.isdigit()])

                    # 基础相似度计算
                    if segment_no_tone == term_no_tone:
                        # 无声调拼音相同，基础相似度为0.9
                        similarity = 0.9
                        # 声调也相同，完全匹配
                        if segment_tone == term_tone:
                            similarity = 1.0
                    else:
                        # 计算字符匹配率
                        matches = 0
                        total = max(len(segment_no_tone), len(term_no_tone))
                        for i in range(
                                min(len(segment_no_tone), len(term_no_tone))):
                            if segment_no_tone[i] == term_no_tone[i]:
                                matches += 1

                        # 基础相似度
                        similarity = matches / total * 0.8  # 最多80%相似度

                        # 声调相同加分
                        if segment_tone == term_tone:
                            similarity += 0.1  # 声调相同加10%相似度

                    # 单字词特殊处理
                    if desired_len == 1 and segment_no_tone == term_no_tone and segment_tone != term_tone:
                        similarity *= 0.9  # 单字词声调不同，降低10%相似度
                except Exception as e:
                    print(f"计算相似度出错: {e}")
                    similarity = 0.0

                # 更新最佳匹配
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = term
                    print(
                        f"找到相似匹配: '{term}' (标准拼音: '{term_pinyin}', 相似度: {similarity:.2f})"
                    )

            # 如果找到匹配且与原文不同，则添加替换
            if best_match and best_match != text_segment and best_similarity >= 0.7:
                print(
                    f"✓ 确认替换: '{text_segment}' -> '{best_match}' (位置 {start_pos}-{end_pos}, 相似度: {best_similarity:.2f})"
                )
                replacements.append((start_pos, end_pos, best_match))

    # 应用替换（从后向前，避免位置变化）
    replacements.sort(reverse=True)
    result_text = list(text)

    for start_pos, end_pos, replacement in replacements:
        result_text[start_pos:end_pos] = replacement

    return ''.join(result_text)

def main():
    """
    主函数，处理命令行参数并调用相应的功能
    """
    parser = argparse.ArgumentParser(description="语音转文字工具")
    parser.add_argument("--action", type=str, choices=["transcribe", "download_model", "list_terms"],
                       required=True, help="要执行的操作: transcribe(语音转文字), download_model(下载模型), list_terms(列出术语)")
    parser.add_argument("--audio", type=str, help="音频文件路径")
    parser.add_argument("--preprocess", action="store_true", help="是否对音频进行预处理")
    parser.add_argument("--output", type=str, help="输出文件路径，默认打印到控制台")
    parser.add_argument("--model_dir", type=str, default=LOCAL_MODEL_DIR, help="模型保存目录")
    parser.add_argument("--domain", type=str, help="领域名称，用于应用特定领域的专业术语")
    parser.add_argument("--vocab_type", type=str, choices=["common", "domain"], default="common",
                       help="词库类型: common(通用词库), domain(领域词库)")

    args = parser.parse_args()

    # 下载模型
    if args.action == "download_model":
        print(f"正在下载语音识别模型到目录: {args.model_dir}")
        # 强制重新下载模型
        if os.path.exists(LOCAL_MODEL_DIR):
            print(f"删除已存在的模型目录: {LOCAL_MODEL_DIR}")
            shutil.rmtree(LOCAL_MODEL_DIR)
        os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

        # 加载模型会自动下载
        model, processor = load_asr_model()
        print("语音识别模型下载完成")
        return

    # 列出术语
    elif args.action == "list_terms":
        vocab_manager = get_vocabulary_manager()
        terms = vocab_manager.get_all_terms(args.vocab_type, args.domain)

        print(f"术语列表 (共{len(terms)}个):")
        for term in terms:
            info = vocab_manager.get_term_info(term)
            if info:
                if info.get("definition"):
                    print(f"- {term}: {info['definition']}")
                else:
                    print(f"- {term}")
            else:
                print(f"- {term}")
        return

    # 语音转文字
    elif args.action == "transcribe":
        if not args.audio:
            print("错误: 语音转文字需要提供音频文件路径 (--audio)")
            return

        print(f"正在处理音频文件: {args.audio}")

        # 如果需要预处理
        if args.preprocess:
            print("正在进行音频预处理...")
            audio_path = preprocess_audio(args.audio)
            print(f"预处理后的音频文件: {audio_path}")
        else:
            audio_path = args.audio

        # 转换语音为文字
        print("正在进行语音识别...")
        text = convert_speech_to_text(audio_path, args.domain)

        # 输出结果
        result = {
            "audio_file": args.audio,
            "transcription": text,
            "domain": args.domain
        }

        # 如果指定了输出文件
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"识别结果已保存到: {args.output}")
        else:
            # 否则打印到控制台
            print("识别结果:")
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return


# 测试函数，用于开发和调试
def test_polyphone_correction(text, domain=None):
    """
    测试多音字纠正功能
    
    Args:
        text: 测试文本
        domain: 领域名称
    """
    print("=" * 50)
    print(f"测试文本: '{text}'")
    print(f"领域: {domain if domain else '无'}")
    print("=" * 50)

    # 处理多音字
    original_text, corrected_pinyin = correct_polyphone_errors(text, domain)
    print(f"原始文本: {original_text}")
    print(f"处理后拼音: {corrected_pinyin}")

    # 应用自定义词库改进转录结果
    try:
        # 先应用通用词库
        common_improved = improve_transcription(original_text, None, corrected_pinyin)
        print(f"通用词库改进: {common_improved}")

        # 如果指定了领域，再应用领域词库
        if domain:
            domain_improved = improve_transcription(common_improved, domain, corrected_pinyin)
            print(f"领域'{domain}'词库再次改进: {domain_improved}")
            final_result = domain_improved
        else:
            final_result = common_improved

        print(f"最终文本: {final_result}")
    except Exception as e:
        print(f"调用improve_transcription出错: {e}")

    print("=" * 50)

    # 返回处理结果
    if 'final_result' in locals():
        return original_text, corrected_pinyin, final_result
    return original_text, corrected_pinyin, None


if __name__ == "__main__":
    # 调用初始化词库
    try:
        init_vocabulary()
    except Exception as e:
        print(f"初始化词库失败: {e}")
        print("程序将继续，但某些词库功能可能不可用")

    # 如果有命令行参数，执行主函数
    if len(sys.argv) > 1:
        main()
    else:
        # 否则执行简单测试
        print("执行测试...")
        # 测试多音字修正和词库应用
        # print(convert_speech_to_text("server-asr/voice_to_text/noice2.wav", "telecom"))
        test_polyphone_correction("电型", "telecom")  # 无领域，仅使用通用词库

        # test_polyphone_correction("尼号，我想要进行果内长途花费查询", "telecom")  # 电信领域
