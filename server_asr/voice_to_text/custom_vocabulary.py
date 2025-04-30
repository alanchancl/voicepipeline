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