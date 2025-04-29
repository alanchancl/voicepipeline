#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import logging
import jieba  # noqa: F401
from pypinyin import lazy_pinyin  # noqa: F401

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("custom_vocabulary")

# 模块基础目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 词库文件路径
VOCAB_DIR = os.path.join(BASE_DIR, 'vocabulary')
COMMON_VOCAB_FILE = os.path.join(VOCAB_DIR, 'common_vocabulary.json')
USER_VOCAB_FILE = os.path.join(VOCAB_DIR, 'user_vocabulary.json')
DOMAIN_VOCAB_FILES = {}  # 领域特定词库文件

# 确保目录存在
if not os.path.exists(VOCAB_DIR):
    os.makedirs(VOCAB_DIR)

class VocabularyManager:
    """专业术语词库管理器"""
    
    def __init__(self):
        """初始化词库管理器"""
        self.vocabularies = {
            'common': {},    # 通用词库（包含默认词库和用户自定义词库）
            'domains': {}    # 领域特定词库
        }
        # 缓存术语的拼音信息
        self.term_pinyin_cache = {}  # {term: pinyin}
        # 缓存领域术语的拼音映射
        self.domain_pinyin_maps = {}  # {domain: {pinyin: term}}
        # 缓存领域术语的长度映射
        self.domain_length_maps = {}  # {domain: {length: [(term, pinyin)]}}
        
        self.load_vocabularies()
        self.build_pinyin_maps()
    
    def build_pinyin_maps(self):
        """构建所有领域的拼音映射"""
        # 清空现有缓存
        self.term_pinyin_cache = {}  # {term: pinyin}
        self.domain_pinyin_maps = {}  # {domain: {pinyin: term}}
        self.domain_length_maps = {}  # {domain: {length: [(term, pinyin)]}}
        
        # 创建通用拼音词库映射（用于所有领域）
        common_pinyin_map = {}
        common_length_map = {}
        
        # 构建通用词库的拼音映射作为基础
        for term, info in self.vocabularies['common'].items():
            if info and info.get('pronunciation'):
                pinyin = info['pronunciation'].replace(' ', '')
                self.term_pinyin_cache[term] = pinyin
                common_pinyin_map[pinyin] = term
                
                # 按长度分组
                term_len = len(term)
                if term_len not in common_length_map:
                    common_length_map[term_len] = []
                common_length_map[term_len].append((term, pinyin))
        
        # 将通用映射加入特殊域"common"中
        self.domain_pinyin_maps["common"] = common_pinyin_map
        self.domain_length_maps["common"] = common_length_map
        
        # 构建各领域词库的拼音映射（包含通用基础）
        for domain, terms in self.vocabularies['domains'].items():
            # 从通用映射复制基础数据
            domain_pinyin_map = common_pinyin_map.copy()
            domain_length_map = {k: v.copy() for k, v in common_length_map.items()}
            
            # 添加领域特有词汇（如有重复会覆盖通用词汇）
            for term, info in terms.items():
                if info and info.get('pronunciation'):
                    pinyin = info['pronunciation'].replace(' ', '')
                    self.term_pinyin_cache[term] = pinyin
                    domain_pinyin_map[pinyin] = term
                    
                    # 按长度分组
                    term_len = len(term)
                    if term_len not in domain_length_map:
                        domain_length_map[term_len] = []
                    domain_length_map[term_len].append((term, pinyin))
            
            self.domain_pinyin_maps[domain] = domain_pinyin_map
            self.domain_length_maps[domain] = domain_length_map
    
    def get_domain_pinyin_map(self, domain_name):
        """获取指定领域的拼音映射"""
        return self.domain_pinyin_maps.get(domain_name, {})
    
    def get_domain_length_map(self, domain_name):
        """获取指定领域的长度映射"""
        return self.domain_length_maps.get(domain_name, {})
    
    def get_term_pinyin(self, term):
        """获取术语的拼音"""
        return self.term_pinyin_cache.get(term)
    
    def load_vocabularies(self):
        """加载所有词库"""
        # 初始化通用词库
        self.vocabularies['common'] = {}
        
        # 加载默认词库到通用词库
        if os.path.exists(COMMON_VOCAB_FILE):
            try:
                with open(COMMON_VOCAB_FILE, 'r', encoding='utf-8') as f:
                    default_vocab = json.load(f)
                    self.vocabularies['common'].update(default_vocab)
                logger.info(f"加载默认词库到通用词库: {len(default_vocab)}个术语")
            except Exception as e:
                logger.error(f"加载默认词库失败: {str(e)}")
        
        # 加载用户词库并合并到通用词库（用户词库优先级更高）
        if os.path.exists(USER_VOCAB_FILE):
            try:
                with open(USER_VOCAB_FILE, 'r', encoding='utf-8') as f:
                    user_vocab = json.load(f)
                    self.vocabularies['common'].update(user_vocab)  # 用户词库会覆盖默认词库中的同名术语
                logger.info(f"加载用户词库到通用词库: {len(user_vocab)}个术语")
            except Exception as e:
                logger.error(f"加载用户词库失败: {str(e)}")
        
        logger.info(f"通用词库共有{len(self.vocabularies['common'])}个术语")
        
        # 加载领域词库目录下的所有JSON文件
        domain_files = [f for f in os.listdir(VOCAB_DIR) if f.startswith('domain_') and f.endswith('.json')]
        for file in domain_files:
            domain_name = file[7:-5]  # 去掉"domain_"前缀和".json"后缀
            file_path = os.path.join(VOCAB_DIR, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    domain_vocab = json.load(f)
                    self.vocabularies['domains'][domain_name] = domain_vocab
                    DOMAIN_VOCAB_FILES[domain_name] = file_path
                logger.info(f"加载领域词库 {domain_name}: {len(domain_vocab)}个术语")
            except Exception as e:
                logger.error(f"加载领域词库 {domain_name} 失败: {str(e)}")
    
    def get_all_terms(self, vocab_type='all', domain_name=None):
        """获取所有术语"""
        terms = []
        
        if vocab_type == 'all' or vocab_type == 'common':
            terms.extend(list(self.vocabularies['common'].keys()))
        
        if vocab_type == 'all' or vocab_type == 'domain':
            if domain_name:
                if domain_name in self.vocabularies['domains']:
                    terms.extend(list(self.vocabularies['domains'][domain_name].keys()))
            else:
                for domain_vocab in self.vocabularies['domains'].values():
                    terms.extend(list(domain_vocab.keys()))
        
        return terms
    
    def get_term_info(self, term):
        """获取术语详细信息"""
        # 按照优先级顺序查找: common > domain
        if term in self.vocabularies['common']:
            info = self.vocabularies['common'][term].copy()
            info['source'] = 'common'
            return info
        
        for domain_name, domain_vocab in self.vocabularies['domains'].items():
            if term in domain_vocab:
                info = domain_vocab[term].copy()
                info['source'] = f"domain:{domain_name}"
                return info
        
        return None

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

def _apply_vocabulary_with_pinyin(text, pinyin, term_pinyin_map, term_by_length):
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
            segment_tone_pinyin = ''.join([p.replace(' ', '') for p in segment_pinyin])
            print(f"检查片段: '{text_segment}' (位置 {start_pos}-{end_pos}), 拼音: '{segment_tone_pinyin}'")
            
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
                    segment_no_tone = ''.join([c for c in segment_tone_pinyin if not c.isdigit()])
                    term_no_tone = ''.join([c for c in term_pinyin if not c.isdigit()])
                    
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
                        for i in range(min(len(segment_no_tone), len(term_no_tone))):
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
                    print(f"找到相似匹配: '{term}' (标准拼音: '{term_pinyin}', 相似度: {similarity:.2f})")
            
            # 如果找到匹配且与原文不同，则添加替换
            if best_match and best_match != text_segment and best_similarity >= 0.7:
                print(f"✓ 确认替换: '{text_segment}' -> '{best_match}' (位置 {start_pos}-{end_pos}, 相似度: {best_similarity:.2f})")
                replacements.append((start_pos, end_pos, best_match))
    
    # 应用替换（从后向前，避免位置变化）
    replacements.sort(reverse=True)
    result_text = list(text)
    
    for start_pos, end_pos, replacement in replacements:
        result_text[start_pos:end_pos] = replacement
    
    return ''.join(result_text)

# 创建默认词库示例

# 单例模式的词库管理器
_vocabulary_manager = None

def get_vocabulary_manager():
    """获取词库管理器实例"""
    global _vocabulary_manager
    if _vocabulary_manager is None:
        _vocabulary_manager = VocabularyManager()
    return _vocabulary_manager

if __name__ == "__main__":
    # 测试代码
    print("专业术语词库管理模块")
    
    # 获取词库管理器
    manager = get_vocabulary_manager()
    
    # 打印所有术语
    all_terms = manager.get_all_terms()
    print(f"当前词库共有{len(all_terms)}个术语:")
    for term in all_terms:
        print(f"- {term}") 