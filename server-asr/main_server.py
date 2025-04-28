#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import wave
import flask
from flask import Flask, request, jsonify
import uuid
import datetime
import threading
import logging
from logging.handlers import RotatingFileHandler
import argparse
import torch
import requests  # 添加requests库用于发送HTTP请求

# 添加当前目录到系统路径，确保能够导入其他模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 导入声纹识别和语音转文字模块 - 修正导入路径
from voice_recognition.voice_recognition import verify_voice, add_voice_sample, get_all_employee_features
from voice_to_text.speech_to_text import convert_speech_to_text, preprocess_audio

# 解析命令行参数
parser = argparse.ArgumentParser(description='语音处理服务器')
parser.add_argument('--gpu', type=int, default=None, help='指定使用的GPU ID，默认使用系统分配')
parser.add_argument('--host', type=str, default='0.0.0.0', help='服务器监听地址')
parser.add_argument('--port', type=int, default=7860, help='服务器监听端口')
parser.add_argument('--debug', action='store_true', help='启用调试模式')
parser.add_argument('--forward-ip', type=str, default='localhost', help='转发目标IP地址')
parser.add_argument('--forward-port', type=int, default=20252, help='转发目标端口号')
parser.add_argument('--forward-path', type=str, default='/api/v1/ai-terminal', help='转发URL路径')
args = parser.parse_args()

# 创建Flask应用
app = Flask(__name__)

# 转发配置
FORWARD_ENABLED = True if args.forward_ip else False
FORWARD_IP = args.forward_ip
FORWARD_PORT = args.forward_port
FORWARD_PATH = args.forward_path

# 配置
UPLOAD_FOLDER = os.path.join(current_dir, 'uploads')
ALLOWED_EXTENSIONS = {'wav'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB 上传限制
GPU_ID = args.gpu  # 存储指定的GPU ID

# 确保上传目录存在
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 配置日志
log_file = os.path.join(current_dir, 'logs', 'server.log')
os.makedirs(os.path.dirname(log_file), exist_ok=True)

# 修改日志格式化程序，确保使用UTF-8编码
handler = RotatingFileHandler(log_file, maxBytes=10000000, backupCount=5, encoding='utf-8')
handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

# 添加控制台日志处理器，方便直接查看日志
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s'
))
console_handler.setLevel(logging.INFO)
app.logger.addHandler(console_handler)

# 设置默认编码，防止日志乱码
if sys.platform.startswith('win'):
    # Windows系统上设置控制台编码
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

# 打印GPU信息
if torch.cuda.is_available():
    app.logger.info("检测到 {} 个GPU".format(torch.cuda.device_count()))
    for i in range(torch.cuda.device_count()):
        app.logger.info("GPU {}: {}".format(i, torch.cuda.get_device_name(i)))
    if GPU_ID is not None:
        if GPU_ID >= 0 and GPU_ID < torch.cuda.device_count():
            app.logger.info("指定使用GPU {}: {}".format(GPU_ID, torch.cuda.get_device_name(GPU_ID)))
        else:
            app.logger.warning("指定的GPU ID {} 无效，将使用系统默认分配".format(GPU_ID))
            GPU_ID = None
else:
    app.logger.warning("未检测到GPU，将使用CPU模式运行")
    GPU_ID = None

# 预加载模型
def preload_models():
    """预加载所有需要的模型"""
    app.logger.info("开始预加载模型...")
    
    # 预加载声纹识别模型
    try:
        app.logger.info("预加载声纹识别模型...")
        from voice_recognition.voice_recognition import load_pretrained_model
        # 确保模型真正加载并缓存到全局变量
        voice_model = load_pretrained_model()
        app.logger.info("声纹识别模型加载完成")
    except Exception as e:
        app.logger.error("预加载声纹识别模型失败: {}".format(str(e)), exc_info=True)
    
    # 预加载语音识别模型
    try:
        app.logger.info("预加载语音识别模型...")
        from voice_to_text.speech_to_text import load_asr_model
        # 加载并保留模型和处理器的引用，传递GPU_ID
        model, processor = load_asr_model(gpu_id=GPU_ID)
        # 防止垃圾回收
        app.config["ASR_MODEL"] = model
        app.config["ASR_PROCESSOR"] = processor
        app.logger.info("语音识别模型加载完成")
    except Exception as e:
        app.logger.error("预加载语音识别模型失败: {}".format(str(e)), exc_info=True)
    
    # 预加载词库
    try:
        app.logger.info("预加载词库...")
        from voice_to_text.custom_vocabulary import get_vocabulary_manager
        vocab_manager = get_vocabulary_manager()
        # 确保词库被完全加载
        all_terms = vocab_manager.get_all_terms()
        app.logger.info("词库加载完成，共 {} 个术语".format(len(all_terms)))
    except Exception as e:
        app.logger.error("预加载词库失败: {}".format(str(e)), exc_info=True)
    
    app.logger.info("所有模型预加载完成")

# 在应用启动时预加载模型
preload_models()

# 允许的文件扩展名检查
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 转发文本结果函数
def forward_result(transcription):
    """
    转发识别结果到目标地址
    
    Args:
        transcription: 转写的文本结果
        
    Returns:
        布尔值，表示转发是否成功
    """
    if not FORWARD_ENABLED or not FORWARD_IP:
        return False
    
    # 构建转发URL
    path = FORWARD_PATH.lstrip('/') if FORWARD_PATH else "api/v1/ai-terminal"
    target_url = f"http://{FORWARD_IP}:{FORWARD_PORT}/{path}"
    
    try:
        # 将结果通过HTTP POST发送
        headers = {'Content-Type': 'application/json'}
        response = requests.post(
            target_url,
            json={"data": transcription},
            headers=headers,
            timeout=5
        )
        
        if response.status_code in [200, 201, 202]:
            app.logger.info(f"已转发识别结果到URL: {target_url}")
            return True
        else:
            app.logger.warning(f"转发到URL失败，状态码: {response.status_code}")
            return False
            
    except Exception as e:
        app.logger.error(f"转发结果到URL失败: {e}")
        return False

# 处理语音分析任务
def process_voice(audio_path, threshold=0.7, preprocess=True, domain="telecom"):
    """
    处理语音文件，进行声纹识别和语音转文字
    
    Args:
        audio_path: 音频文件路径
        threshold: 声纹识别阈值
        preprocess: 是否对音频进行预处理
        domain: 领域名称，默认为telecom
        
    Returns:
        处理结果字典
    """
    start_time = time.time()
    result = {"status": "success", "audio_path": audio_path}
    
    try:
        # 声纹识别
        app.logger.info("开始声纹识别: {}".format(audio_path))
        voiceprint_result = verify_voice(audio_path, threshold=threshold)
        result["voiceprint"] = voiceprint_result
        app.logger.info("声纹识别完成: {}".format(voiceprint_result['verified']))
        
        # 语音转文字
        app.logger.info("开始语音转文字: {}".format(audio_path))
        if preprocess:
            # 预处理音频以提高识别准确度
            processed_path = preprocess_audio(audio_path)
            text = convert_speech_to_text(processed_path, domain=domain, gpu_id=GPU_ID)
            # 清理预处理文件
            if os.path.exists(processed_path) and processed_path != audio_path:
                try:
                    os.remove(processed_path)
                except:
                    pass
        else:
            text = convert_speech_to_text(audio_path, domain=domain, gpu_id=GPU_ID)
        
        # 检查转录结果是否为空
        if not text or text.strip() == "":
            app.logger.warning("语音转文字结果为空: {}".format(audio_path))
            result["transcription"] = ""
            result["has_transcription"] = False
        else:
            result["transcription"] = text
            result["has_transcription"] = True
            result["domain"] = domain
            app.logger.info("语音转文字完成: {}...".format(text[:50]))
            
            # 转发识别结果到目标地址
            if FORWARD_ENABLED:
                forward_success = forward_result(text)
                result["forwarded"] = forward_success
        
    except Exception as e:
        app.logger.error("处理语音出错: {}".format(str(e)), exc_info=True)
        result["status"] = "error"
        result["error"] = str(e)
        result["has_transcription"] = False
    
    result["processing_time"] = time.time() - start_time
    return result

# API路由

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({"status": "ok", "time": datetime.datetime.now().isoformat()})

@app.route('/api/status', methods=['GET'])
def status_check():
    """状态检查接口 - 为VAD客户端提供"""
    return jsonify({
        "status": "ok", 
        "server_time": datetime.datetime.now().isoformat(),
        "models_loaded": True,
        "version": "1.0.0"
    })

@app.route('/api/upload', methods=['POST'])
def upload_audio():
    """
    上传音频文件接口 - 为VAD客户端提供
    
    请求参数:
    - file: 上传的WAV音频文件
    - domain: 可选，领域名称，默认telecom
    - forward: 可选，是否转发结果，默认true
    
    返回:
    - JSON结果，包含声纹识别和语音转文字结果
    """
    # 检查是否有文件上传
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "没有上传文件"}), 400
    
    file = request.files['file']
    
    # 检查文件名是否为空
    if file.filename == '':
        return jsonify({"status": "error", "message": "未选择文件"}), 400
    
    # 检查文件类型
    if not allowed_file(file.filename):
        return jsonify({"status": "error", "message": "不支持的文件类型，仅支持WAV格式"}), 400
    
    try:
        # 保留原始文件名，以保存时间信息
        original_filename = os.path.basename(file.filename)
        
        # 生成带有原始文件名的保存路径
        filepath = os.path.join(UPLOAD_FOLDER, original_filename)
        
        # 如果已存在同名文件，添加时间戳避免覆盖
        if os.path.exists(filepath):
            name, ext = os.path.splitext(original_filename)
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            original_filename = "{}_{}{}".format(name, timestamp, ext)
            filepath = os.path.join(UPLOAD_FOLDER, original_filename)
        
        file.save(filepath)
        app.logger.info("文件已上传并保存为: {}".format(filepath))
        
        # 获取领域参数和转发参数
        domain = request.form.get('domain', 'telecom')
        should_forward = request.form.get('forward', 'true').lower() == 'true'
        
        # 暂时保存原始转发设置
        original_forward_enabled = FORWARD_ENABLED
        
        # 根据请求参数决定是否进行转发
        if not should_forward:
            global FORWARD_ENABLED
            FORWARD_ENABLED = False
        
        # 使用默认参数处理语音
        result = process_voice(filepath, domain=domain)
        
        # 恢复原始转发设置
        FORWARD_ENABLED = original_forward_enabled
        
        # 只有当成功转录文字时才发送结果
        if result.get("has_transcription", False) and result.get("status") == "success":
            app.logger.info("成功处理音频，返回结果")
            return jsonify(result)
        else:
            app.logger.info("音频未成功转录或处理失败，返回空结果")
            return jsonify({"status": "success", "transcription": "", "has_transcription": False})
    
    except Exception as e:
        app.logger.error("处理请求出错: {}".format(str(e)), exc_info=True)
        return jsonify({"status": "error", "message": "处理请求出错: {}".format(str(e))}), 500

@app.route('/api/process_voice', methods=['POST'])
def api_process_voice():
    """
    处理上传的语音文件
    
    请求参数:
    - audio_file: 上传的WAV音频文件
    - threshold: 可选，声纹验证阈值，默认0.7
    - preprocess: 可选，是否预处理音频，默认true
    - domain: 可选，领域名称，默认telecom
    
    返回:
    - JSON结果，包含声纹识别和语音转文字结果
    """
    # 检查是否有文件上传
    if 'audio_file' not in request.files:
        return jsonify({"status": "error", "message": "没有上传文件"}), 400
    
    file = request.files['audio_file']
    
    # 检查文件名是否为空
    if file.filename == '':
        return jsonify({"status": "error", "message": "未选择文件"}), 400
    
    # 检查文件类型
    if not allowed_file(file.filename):
        return jsonify({"status": "error", "message": "不支持的文件类型，仅支持WAV格式"}), 400
    
    try:
        # 保留原始文件名，以保存时间信息
        original_filename = os.path.basename(file.filename)
        
        # 生成带有原始文件名的保存路径
        filepath = os.path.join(UPLOAD_FOLDER, original_filename)
        
        # 如果已存在同名文件，添加时间戳避免覆盖
        if os.path.exists(filepath):
            name, ext = os.path.splitext(original_filename)
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            original_filename = "{}_{}{}".format(name, timestamp, ext)
            filepath = os.path.join(UPLOAD_FOLDER, original_filename)
        
        file.save(filepath)
        app.logger.info("文件已上传并保存为: {}".format(filepath))
        
        # 获取参数
        threshold = float(request.form.get('threshold', 0.7))
        preprocess = request.form.get('preprocess', 'true').lower() == 'true'
        domain = request.form.get('domain', 'telecom')
        
        # 处理语音
        result = process_voice(filepath, threshold, preprocess, domain)
        
        # 根据参数决定是否只返回有效转录
        only_transcribed = request.form.get('only_transcribed', 'false').lower() == 'true'
        
        if only_transcribed and not result.get("has_transcription", False):
            app.logger.info("音频未成功转录，返回空结果")
            return jsonify({"status": "success", "transcription": "", "has_transcription": False})
        
        return jsonify(result)
    
    except Exception as e:
        app.logger.error("处理请求出错: {}".format(str(e)), exc_info=True)
        return jsonify({"status": "error", "message": "处理请求出错: {}".format(str(e))}), 500

@app.route('/api/employees', methods=['GET'])
def list_employees():
    """获取已注册员工列表"""
    try:
        employee_features = get_all_employee_features()
        employees = []
        
        for employee_id, features in employee_features.items():
            employees.append({
                "id": employee_id,
                "sample_count": len(features)
            })
        
        return jsonify({
            "status": "success",
            "count": len(employees),
            "employees": employees
        })
    
    except Exception as e:
        app.logger.error("获取员工列表出错: {}".format(str(e)), exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/enroll', methods=['POST'])
def enroll_employee():
    """
    注册新员工声纹
    
    请求参数:
    - audio_file: 上传的WAV音频文件
    - employee_id: 员工ID
    
    返回:
    - JSON结果，包含注册结果
    """
    # 检查是否有文件上传
    if 'audio_file' not in request.files:
        return jsonify({"status": "error", "message": "没有上传文件"}), 400
    
    # 检查员工ID
    employee_id = request.form.get('employee_id', '')
    if not employee_id:
        return jsonify({"status": "error", "message": "未提供员工ID"}), 400
    
    file = request.files['audio_file']
    
    # 检查文件名是否为空
    if file.filename == '':
        return jsonify({"status": "error", "message": "未选择文件"}), 400
    
    # 检查文件类型
    if not allowed_file(file.filename):
        return jsonify({"status": "error", "message": "不支持的文件类型，仅支持WAV格式"}), 400
    
    try:
        # 生成唯一文件名并保存
        filename = "{}.wav".format(uuid.uuid4().hex)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        app.logger.info("文件已上传并保存为: {}".format(filepath))
        
        # 添加声纹样本
        success = add_voice_sample(filepath, employee_id)
        
        if success:
            return jsonify({
                "status": "success", 
                "message": "成功注册员工{}的声纹样本".format(employee_id)
            })
        else:
            return jsonify({
                "status": "error", 
                "message": "声纹注册失败"
            }), 500
    
    except Exception as e:
        app.logger.error("注册声纹出错: {}".format(str(e)), exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

# 如果直接运行此脚本
if __name__ == "__main__":
    # 打印启动信息
    app.logger.info("服务器正在启动，监听 {}:{}".format(args.host, args.port))
    if GPU_ID is not None:
        app.logger.info("使用GPU {} 进行语音识别".format(GPU_ID))
    else:
        if torch.cuda.is_available():
            app.logger.info("使用系统分配的默认GPU")
        else:
            app.logger.info("使用CPU模式运行")
    
    # 打印转发配置
    if FORWARD_ENABLED:
        app.logger.info("转发已启用，目标: http://{}:{}/{}".format(
            FORWARD_IP, FORWARD_PORT, FORWARD_PATH.lstrip('/')))
    else:
        app.logger.info("转发功能未启用")
            
    # 启动Flask应用
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True) 