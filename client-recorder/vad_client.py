import numpy as np
import torch
import pyaudio
import wave
import threading
import time
import os
import socket
import argparse
import sys
import io
import json
import datetime
from dotenv import load_dotenv
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder

# 加载环境变量
load_dotenv()

# 创建必要的目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # 获取项目根目录

RECORDINGS_DIR = os.path.join(BASE_DIR, "recordings")
os.makedirs(RECORDINGS_DIR, exist_ok=True)

# 配置文件路径
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
HISTORY_FILE = os.path.join(BASE_DIR, "file_record.json")

# 模型相关配置
# 定义集中式模型路径和原始模型路径
CENTRALIZED_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "Voice Activity Detection", "silero_vad", "silero_vad.jit")


# 默认配置
DEFAULT_CONFIG = {
    "server": {
        "ip": "localhost",
        "port": 7860
    },
    "forward": {
        "ip": "localhost",
        "port": 20252,
        "path": "/api/v1/ai-terminal"
    },
    "vad": {
        "threshold": 0.45,
        "silence_duration": 1.2,
        "min_speech_duration": 0.3,
        "speech_pad_ms": 300
    },
    "audio": {
        "save_recordings": True,
        "input_device_index": None
    }
}

# 加载配置文件
def load_config():
    """从配置文件加载配置，如果不存在则创建默认配置"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"无法读取配置文件，使用默认配置: {e}")
    else:
        # 创建初始配置文件
        save_config(DEFAULT_CONFIG)
        print(f"已创建默认配置文件: {CONFIG_FILE}")
    
    return DEFAULT_CONFIG

# 保存配置到文件
def save_config(config):
    """保存配置到配置文件"""
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存配置文件失败: {e}")

# 加载配置
CONFIG = load_config()

# 从配置中获取默认值
DEFAULT_SERVER_IP = CONFIG["server"]["ip"]
DEFAULT_SERVER_PORT = CONFIG["server"]["port"]
DEFAULT_FORWARD_IP = CONFIG["forward"]["ip"]
DEFAULT_FORWARD_PORT = CONFIG["forward"]["port"]
DEFAULT_FORWARD_PATH = CONFIG["forward"]["path"]
DEFAULT_THRESHOLD = CONFIG["vad"]["threshold"]
DEFAULT_SILENCE_DURATION = CONFIG["vad"]["silence_duration"]
MIN_SPEECH_DURATION = CONFIG["vad"]["min_speech_duration"]
SPEECH_PAD_MS = CONFIG["vad"]["speech_pad_ms"]
SAVE_RECORDINGS = CONFIG["audio"]["save_recordings"]


class VoiceActivityDetector:
    """语音活动检测器，检测到语音后发送到服务器"""

    def __init__(self,
                 server_ip=DEFAULT_SERVER_IP,
                 server_port=DEFAULT_SERVER_PORT,
                 format=pyaudio.paInt16,
                 channels=1,
                 rate=16000,
                 chunk=512,  # 与模型要求匹配
                 threshold=DEFAULT_THRESHOLD,
                 silence_duration=DEFAULT_SILENCE_DURATION,
                 input_device_index=None,  # 添加输入设备索引参数
                 forward_ip=DEFAULT_FORWARD_IP,  # 转发目标IP
                 forward_port=DEFAULT_FORWARD_PORT,  # 转发目标端口
                 forward_path=DEFAULT_FORWARD_PATH):  # 转发URL路径
        """
        初始化语音活动检测器

        参数:
            server_ip: 服务器IP地址
            server_port: 服务器端口
            format: 音频格式 (默认: 16位PCM)
            channels: 声道数 (默认: 1，单声道)
            rate: 采样率 (默认: 16kHz)
            chunk: 单次采集帧数 (与模型要求匹配)
            threshold: VAD检测阈值
            silence_duration: 判定语音结束的静音持续时间(秒)
            input_device_index: 输入设备索引，None表示使用默认设备
            forward_ip: 转发目标IP地址，为空则不转发
            forward_port: 转发目标端口号
            forward_path: 转发URL路径 (默认: /api/v1/ai_terminal)
        """
        # 音频参数
        self.format = format
        self.channels = channels
        self.rate = rate
        self.chunk = chunk
        self.input_device_index = input_device_index
        
        # 服务器参数
        self.server_ip = server_ip
        self.server_port = server_port
        self.client_socket = None
        
        # 转发参数
        self.forward_ip = forward_ip
        self.forward_port = forward_port
        self.forward_path = forward_path  # 转发URL路径
        
        # VAD参数
        self.vad_threshold = threshold
        self.min_silence_frames = int(silence_duration * self.rate / self.chunk)
        self.min_speech_frames = int(MIN_SPEECH_DURATION * self.rate / self.chunk)
        self.pad_samples = int(SPEECH_PAD_MS * self.rate / 1000)
        
        # 状态控制
        self.is_recording = False
        self.is_speech = False
        self.speech_frames = []
        self.silence_frames = 0
        self.audio_buffer = []  # 前导缓冲区
        self.max_buffer_size = 10  # 保留前导音频的最大帧数
        
        # 连续性控制
        self.last_speech_end_time = 0
        self.speech_timeout = 0.3  # 语音片段间的最大允许间隔(秒)
        
        # 历史记录
        self.history = self._load_history()
        
        # 初始化PyAudio
        self.pyaudio_instance = pyaudio.PyAudio()
        
        # 设备选择
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")
        
        # 加载VAD模型
        try:
            self._load_silero_vad()
            print("Silero VAD模型加载成功")
        except Exception as e:
            print(f"加载Silero VAD失败: {e}")
            sys.exit(1)  # 如果VAD模型加载失败，退出程序
            
        # 连接服务器
        self._connect_to_server()
        
    @staticmethod
    def list_audio_devices():
        """列出所有可用的音频输入设备"""
        p = pyaudio.PyAudio()
        device_count = p.get_device_count()
        input_devices = []
        
        print("\n--- 可用的音频输入设备 ---")
        print("索引\t设备名称")
        print("-" * 50)
        
        for i in range(device_count):
            device_info = p.get_device_info_by_index(i)
            if device_info.get('maxInputChannels') > 0:  # 只列出输入设备
                input_devices.append((i, device_info))
                is_default = "*" if i == p.get_default_input_device_info().get('index') else " "
                print(f"{i}{is_default}\t{device_info.get('name')}")
        
        print("-" * 50)
        print("* 表示默认设备")
        
        p.terminate()
        return input_devices
        
    def _load_history(self):
        """加载历史记录"""
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"无法读取历史记录文件，创建新记录: {e}")
        return {"recordings": []}

    def _save_history(self):
        """保存历史记录"""
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
            
    def _connect_to_server(self):
        """连接到服务器"""
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            print(f"正在连接到服务器 {self.server_ip}:{self.server_port}...")
            self.client_socket.connect((self.server_ip, self.server_port))
            print("已连接到服务器")
        except Exception as e:
            print(f"连接服务器失败: {e}")
            sys.exit(1)  # 如果连接失败，退出程序
  
        # 如果设置了转发IP，显示相关信息
        if self.forward_ip:
            full_url = f"http://{self.forward_ip}:{self.forward_port}/{self.forward_path.lstrip('/')}"
            print(f"将使用HTTP转发到URL: {full_url}")
        
    def _load_silero_vad(self):
        """加载Silero VAD模型"""
        # 首先尝试从集中式模型目录加载
        if os.path.exists(CENTRALIZED_MODEL_PATH):
            print(f"从集中式模型目录加载模型: {CENTRALIZED_MODEL_PATH}")
            # 从本地文件加载模型
            self.vad_model = torch.jit.load(CENTRALIZED_MODEL_PATH, map_location=self.device)
        # 然后尝试从原始位置加载
        elif os.path.exists(ORIGINAL_MODEL_PATH):
            print(f"从原始位置加载模型: {ORIGINAL_MODEL_PATH}")
            # 从本地文件加载模型
            self.vad_model = torch.jit.load(ORIGINAL_MODEL_PATH, map_location=self.device)
        else:
            print("本地模型不存在，从 Torch Hub 下载模型...")
            # 从Torch Hub下载模型
            self.vad_model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False
            )
            
            # 尝试保存到集中式模型目录
            try:
                # 确保父目录存在
                os.makedirs(os.path.dirname(CENTRALIZED_MODEL_PATH), exist_ok=True)
                torch.jit.save(self.vad_model, CENTRALIZED_MODEL_PATH)
                print(f"模型已保存到集中式目录: {CENTRALIZED_MODEL_PATH}")
            except Exception as e:
                print(f"无法保存到集中式目录: {e}，将保存到原始位置")
                # 保存模型到原始位置
                torch.jit.save(self.vad_model, ORIGINAL_MODEL_PATH)
                print(f"模型已保存到: {ORIGINAL_MODEL_PATH}")
            
            # 保存工具函数引用
            self.get_speech_timestamps, self.save_audio, self.read_audio, _, _ = utils
        
        # 确保模型处于评估模式
        self.vad_model.eval()

    def _is_speech(self, audio_chunk):
        """判断音频数据是否包含语音"""
        try:
            # 确保数据长度合适
            if len(audio_chunk) != self.chunk:
                audio_chunk = np.resize(audio_chunk, self.chunk)
                
            # 转换为模型需要的格式
            audio_tensor = torch.FloatTensor(audio_chunk).to(self.device)
            # 归一化到[-1, 1]
            audio_tensor = audio_tensor / 32768.0
            
            # 进行VAD检测
            with torch.no_grad():
                speech_prob = self.vad_model(audio_tensor, self.rate)
                
            return speech_prob.item() > self.vad_threshold
        except Exception as e:
            print(f"VAD检测出错: {e}")
            return False

    def start_detection(self):
        """开始语音检测"""
        if self.is_recording:
            print("语音检测已经在运行")
            return
            
        self.is_recording = True
        
        # 创建并启动音频处理线程
        self.recording_thread = threading.Thread(target=self._audio_detection_loop)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        print("语音检测已启动")

    def stop_detection(self):
        """停止语音检测"""
        self.is_recording = False
        
        # 关闭与服务器的连接
        if self.client_socket:
            try:
                self.client_socket.close()
                print("已关闭与服务器的连接")
            except Exception as e:
                print(f"关闭连接出错: {e}")
                
        if hasattr(self, 'recording_thread'):
            self.recording_thread.join(timeout=2)
            
        # 保存历史记录
        self._save_history()
        
        print("语音检测已停止")

    def _save_audio_file(self, frames):
        """保存音频文件到本地"""
        if not SAVE_RECORDINGS:
            return None
            
        # 创建文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(RECORDINGS_DIR, f"recording_{timestamp}.wav")
        
        # 保存WAV文件
        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.pyaudio_instance.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(frames))
            
        print(f"录音已保存: {file_path}")
        return file_path

    def _forward_result(self, result_data):
        """转发结果到目标地址"""
        # 检查是否有转发目标
        if not self.forward_ip:
            return False
            
        # 构建转发URL
        path = self.forward_path.lstrip('/') if self.forward_path else "api/v1/ai_terminal"
        target_url = f"http://{self.forward_ip}:{self.forward_port}/{path}"
        
        try:
            # 将结果转为JSON并通过HTTP POST发送
            headers = {'Content-Type': 'application/json'}
            print(result_data['recognition_result']['transcription'])
            response = requests.post(
                target_url,
                json={"data": result_data['recognition_result']['transcription']},
                # data=json.dumps(result_data, ensure_ascii=False),
                headers=headers,
                timeout=5
            )
            
            if response.status_code in [200, 201, 202]:
                print(f"已转发识别结果到URL: {target_url}")
                return True
            else:
                print(f"转发到URL失败，状态码: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"转发结果到URL失败: {e}")
            return False
    
    def _forward_result_async(self, result_data):
        """异步转发结果到目标地址"""
        # 创建并启动转发线程
        forward_thread = threading.Thread(target=self._forward_result, args=(result_data,))
        forward_thread.daemon = True
        forward_thread.start()

    def _send_wav_to_server(self, frames):
        """将WAV格式音频发送到服务器（非阻塞方式）"""
        # 创建并启动发送线程
        send_thread = threading.Thread(target=self._process_and_send_wav, args=(frames,))
        send_thread.daemon = True
        send_thread.start()
        return True
        
    def _process_and_send_wav(self, frames):
        """处理并发送WAV格式音频到服务器（在单独的线程中运行）"""
        # 保存音频文件
        file_path = self._save_audio_file(frames)
        
        try:
            # 将帧数据转换为WAV格式
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.pyaudio_instance.get_sample_size(self.format))
                wf.setframerate(self.rate)
                wf.writeframes(b''.join(frames))
                
            # 获取WAV数据
            wav_data = wav_buffer.getvalue()
            
            print(f"发送语音片段到服务器，大小: {len(wav_data)/1024:.2f} KB")
            
            # 创建一个临时文件名，与本地保存格式一致
            timestamp1 = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            timestamp2 = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")[:-4]
            temp_filename = f"{timestamp1}_{timestamp2}.wav"
            
            try:
                # 创建一个临时文件对象
                temp_file = io.BytesIO(wav_data)
                
                # 准备要发送的表单数据
                multipart_data = MultipartEncoder(
                    fields={
                        'file': (temp_filename, temp_file, 'audio/wav'),
                        'domain': 'telecom'
                    }
                )
                
                # 设置请求头
                headers = {
                    'Content-Type': multipart_data.content_type
                }
                
                # 发送HTTP POST请求到/api/upload端点
                url = f"http://{self.server_ip}:{self.server_port}/api/upload"
                print(f"发送请求到: {url}")
                
                # 发送请求
                response = requests.post(
                    url, 
                    data=multipart_data,
                    headers=headers,
                    timeout=10
                )
                
                # 检查响应状态
                if response.status_code == 200:
                    try:
                        response_json = response.json()
                        # 直接输出完整的JSON响应
                        print("服务器响应:")
                        print(json.dumps(response_json, ensure_ascii=False, indent=2))
                        
                        # 添加时间戳和设备信息
                        timestamp = datetime.datetime.now().isoformat()
                        result_with_metadata = {
                            "timestamp": timestamp,
                            "recognition_result": response_json,
                            "audio_file": file_path
                        }
                        
                        # 转发结果到目标地址（异步方式）
                        if self.forward_ip:
                            self._forward_result_async(result_with_metadata)
                        
                        # 添加到历史记录
                        if file_path:
                            self.history["recordings"].append({
                                "timestamp": timestamp,
                                "file_path": file_path,
                                "response": response_json
                            })
                            # 保存更新后的历史记录
                            self._save_history()
                    except Exception as e:
                        print(f"解析服务器响应失败: {e}")
                        print(f"原始响应: {response.text}")
                else:
                    print(f"服务器返回错误状态码: {response.status_code}")
                    print(f"错误信息: {response.text}")
                
            except Exception as e:
                print(f"发送HTTP请求失败: {e}")
                
        except Exception as e:
            print(f"处理音频数据失败: {e}")

    def _audio_detection_loop(self):
        """音频检测主循环"""
        try:
            stream = self.pyaudio_instance.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                input_device_index=self.input_device_index,  # 使用指定的输入设备
                frames_per_buffer=self.chunk
            )
            
            if self.input_device_index is not None:
                device_info = self.pyaudio_instance.get_device_info_by_index(self.input_device_index)
                print(f"使用麦克风: {device_info.get('name', '未知设备')}")
            else:
                print("使用默认麦克风")
                
            print("正在监听麦克风...")
            
            # 状态指示
            dot_counter = 0
            
            while self.is_recording:
                # 读取音频数据
                data = stream.read(self.chunk, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                
                # 显示监听状态（每20帧显示一个点）
                dot_counter += 1
                if dot_counter % 20 == 0:
                    sys.stdout.write(".")
                    sys.stdout.flush()
                
                # 将当前帧加入缓冲区，保持最大长度
                self.audio_buffer.append(data)
                if len(self.audio_buffer) > self.max_buffer_size:
                    self.audio_buffer.pop(0)
                
                # 检测是否为语音
                is_current_speech = self._is_speech(audio_data)
                
                # 状态转换处理
                if is_current_speech:
                    # 当前帧是语音
                    if not self.is_speech:
                        # 语音开始
                        self.is_speech = True
                        
                        # 检查是否与上一段语音时间很接近（连续说话）
                        current_time = time.time()
                        if current_time - self.last_speech_end_time < self.speech_timeout:
                            print("检测到语音继续...")
                        else:
                            print("\n检测到新语音片段...")
                        
                        # 添加前导缓冲（包括当前帧以前的音频）
                        self.speech_frames = list(self.audio_buffer)
                    else:
                        # 继续语音
                        self.speech_frames.append(data)
                
                    # 重置静音帧计数
                    self.silence_frames = 0
                else:
                    # 当前帧是静音
                    if self.is_speech:
                        # 正在语音中，添加这一帧并计数
                        self.speech_frames.append(data)
                        self.silence_frames += 1
                        
                        # 检查静音帧是否达到静音判断阈值
                        if self.silence_frames >= self.min_silence_frames:
                            # 语音结束
                            self.is_speech = False
                            self.last_speech_end_time = time.time()
                            
                            # 检查语音片段长度是否足够
                            if len(self.speech_frames) >= self.min_speech_frames:
                                print(f"\n语音片段结束 - 长度: {len(self.speech_frames)} 帧")
                                
                                # 发送到服务器
                                self._send_wav_to_server(self.speech_frames)
                            else:
                                print(f"\n忽略过短的语音片段: {len(self.speech_frames)} 帧")
                            
                            # 重置语音帧
                            self.speech_frames = []
                            self.silence_frames = 0
            
            # 清理资源
            stream.stop_stream()
            stream.close()
        except Exception as e:
            print(f"无法打开音频流: {e}")
            self.is_recording = False
            return


def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='基于VAD的语音检测客户端')
    parser.add_argument('--ip', type=str, default=DEFAULT_SERVER_IP,
                        help=f'服务器IP地址 (默认: {DEFAULT_SERVER_IP})')
    parser.add_argument('--port', type=int, default=DEFAULT_SERVER_PORT,
                        help=f'服务器端口 (默认: {DEFAULT_SERVER_PORT})')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD,
                        help=f'VAD阈值 (默认: {DEFAULT_THRESHOLD})')
    parser.add_argument('--silence', type=float, default=DEFAULT_SILENCE_DURATION,
                        help=f'静音判断时长(秒) (默认: {DEFAULT_SILENCE_DURATION})')
    parser.add_argument('--no-save', action='store_true',
                        help='不保存录音文件')
    parser.add_argument('--list-devices', action='store_true',
                        help='列出所有可用的麦克风设备并退出')
    parser.add_argument('--device', type=int, default=None,
                        help='指定麦克风设备索引')
    parser.add_argument('--forward-ip', type=str, default=DEFAULT_FORWARD_IP,
                        help='转发结果到指定IP地址')
    parser.add_argument('--forward-port', type=int, default=DEFAULT_FORWARD_PORT,
                        help=f'转发目标端口号 (默认: {DEFAULT_FORWARD_PORT})')
    parser.add_argument('--forward-path', type=str, default=DEFAULT_FORWARD_PATH,
                        help=f'转发URL路径 (默认: {DEFAULT_FORWARD_PATH})')
    parser.add_argument('--reset-device', action='store_true',
                        help='重置并重新选择麦克风设备')
    parser.add_argument('--save-config', action='store_true',
                        help='保存当前命令行参数到配置文件')
    
    args = parser.parse_args()
    
    # 如果只需要列出设备
    if args.list_devices:
        VoiceActivityDetector.list_audio_devices()
        return
    
    # 获取当前配置
    config = load_config()
    
    # 处理命令行参数，更新配置
    if args.ip != DEFAULT_SERVER_IP:
        config["server"]["ip"] = args.ip
    
    if args.port != DEFAULT_SERVER_PORT:
        config["server"]["port"] = args.port
    
    if args.threshold != DEFAULT_THRESHOLD:
        config["vad"]["threshold"] = args.threshold
    
    if args.silence != DEFAULT_SILENCE_DURATION:
        config["vad"]["silence_duration"] = args.silence
    
    if args.forward_ip != DEFAULT_FORWARD_IP:
        config["forward"]["ip"] = args.forward_ip
    
    if args.forward_port != DEFAULT_FORWARD_PORT:
        config["forward"]["port"] = args.forward_port
    
    if args.forward_path != DEFAULT_FORWARD_PATH:
        config["forward"]["path"] = args.forward_path
    
    # 处理录音保存选项
    if args.no_save:
        config["audio"]["save_recordings"] = False
        global SAVE_RECORDINGS
        SAVE_RECORDINGS = False
    
    # 选择输入设备
    input_device_index = args.device
    
    # 如果用户没有指定设备且没有要求重置设备，尝试从配置读取
    if input_device_index is None and not args.reset_device:
        saved_device_index = config["audio"]["input_device_index"]
        if saved_device_index is not None:
            # 验证设备是否仍然可用
            available_devices = VoiceActivityDetector.list_audio_devices()
            valid_indices = [idx for idx, _ in available_devices]
            
            if saved_device_index in valid_indices:
                input_device_index = saved_device_index
                device_info = next((info for idx, info in available_devices if idx == saved_device_index), None)
                if device_info:
                    device_name = device_info.get('name', '未知设备')
                    print(f"\n使用配置的麦克风设备: [{saved_device_index}] {device_name}")
                else:
                    print(f"\n使用配置的麦克风设备 (索引: {saved_device_index})")
            else:
                print(f"\n配置的麦克风设备 (索引: {saved_device_index}) 已不可用，请重新选择")
    
    # 如果仍然没有设备索引或者用户要求重置，提示用户选择
    if input_device_index is None or args.reset_device:
        available_devices = VoiceActivityDetector.list_audio_devices()
        if available_devices:
            try:
                selection = input("\n请选择要使用的麦克风设备索引 (回车使用默认设备): ")
                if selection.strip():
                    input_device_index = int(selection)
                    # 检查索引是否有效
                    valid_indices = [idx for idx, _ in available_devices]
                    if input_device_index not in valid_indices:
                        print(f"警告: 设备索引 {input_device_index} 不是有效的输入设备，将使用默认设备")
                        input_device_index = None
                    else:
                        # 更新配置中的设备选择并立即保存
                        config["audio"]["input_device_index"] = input_device_index
                        save_config(config)
                        print(f"已更新麦克风设备设置: 设备索引 {input_device_index}")
            except ValueError:
                print("无效的输入，将使用默认设备")
                input_device_index = None
    
    # 保存其他配置变更
    if args.save_config or args.ip != DEFAULT_SERVER_IP or args.port != DEFAULT_SERVER_PORT or \
       args.threshold != DEFAULT_THRESHOLD or args.silence != DEFAULT_SILENCE_DURATION or \
       args.forward_ip != DEFAULT_FORWARD_IP or args.forward_port != DEFAULT_FORWARD_PORT or \
       args.forward_path != DEFAULT_FORWARD_PATH or args.no_save:
        save_config(config)
        print("已保存更新后的配置")
    
    # 创建VAD检测器
    detector = VoiceActivityDetector(
        server_ip=config["server"]["ip"],
        server_port=config["server"]["port"],
        threshold=config["vad"]["threshold"],
        silence_duration=config["vad"]["silence_duration"],
        input_device_index=input_device_index,
        forward_ip=config["forward"]["ip"],
        forward_port=config["forward"]["port"],
        forward_path=config["forward"]["path"]
    )
    
    # 开始检测
    detector.start_detection()
    
    # 设置中断处理
    def signal_handler(sig, frame):
        print("\n停止检测...")
        detector.stop_detection()
        sys.exit(0)
        
    import signal
    signal.signal(signal.SIGINT, signal_handler)
    
    # 保持程序运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        detector.stop_detection()
        print("程序已退出")


if __name__ == "__main__":
    main()