#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import requests
import json
import os
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import time
from datetime import datetime

class VoicePipelineClient:
    def __init__(self, root):
        self.root = root
        self.root.title("语音处理测试客户端")
        self.root.geometry("800x600")
        
        # 服务器配置
        self.server_url = "http://localhost:8000"
        
        # 录音配置
        self.recording = False
        self.audio_data = []
        self.sample_rate = 16000
        
        # 创建界面
        self.create_widgets()
        
    def create_widgets(self):
        # 创建选项卡
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=5)
        
        # 语音转文字选项卡
        stt_frame = ttk.Frame(notebook)
        notebook.add(stt_frame, text="语音转文字")
        
        # 文字转语音选项卡
        tts_frame = ttk.Frame(notebook)
        notebook.add(tts_frame, text="文字转语音")
        
        # 设置语音转文字界面
        self.setup_stt_tab(stt_frame)
        
        # 设置文字转语音界面
        self.setup_tts_tab(tts_frame)
    
    def setup_stt_tab(self, parent):
        # 录音控制区域
        control_frame = ttk.LabelFrame(parent, text="录音控制")
        control_frame.pack(fill='x', padx=5, pady=5)
        
        self.record_button = ttk.Button(control_frame, text="开始录音", command=self.toggle_recording)
        self.record_button.pack(side='left', padx=5, pady=5)
        
        self.upload_button = ttk.Button(control_frame, text="上传音频文件", command=self.upload_audio_file)
        self.upload_button.pack(side='left', padx=5, pady=5)
        
        # 转写结果显示区域
        result_frame = ttk.LabelFrame(parent, text="转写结果")
        result_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.stt_result = scrolledtext.ScrolledText(result_frame, wrap=tk.WORD, height=10)
        self.stt_result.pack(fill='both', expand=True, padx=5, pady=5)
    
    def setup_tts_tab(self, parent):
        # 文本输入区域
        input_frame = ttk.LabelFrame(parent, text="输入文本")
        input_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.tts_input = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, height=5)
        self.tts_input.pack(fill='both', expand=True, padx=5, pady=5)
        
        # 参数设置区域
        param_frame = ttk.LabelFrame(parent, text="参数设置")
        param_frame.pack(fill='x', padx=5, pady=5)
        
        # 说话人ID
        ttk.Label(param_frame, text="说话人ID:").grid(row=0, column=0, padx=5, pady=5)
        self.speaker_id = ttk.Entry(param_frame)
        self.speaker_id.grid(row=0, column=1, padx=5, pady=5)
        
        # 温度
        ttk.Label(param_frame, text="温度:").grid(row=0, column=2, padx=5, pady=5)
        self.temperature = ttk.Entry(param_frame)
        self.temperature.insert(0, "0.1")
        self.temperature.grid(row=0, column=3, padx=5, pady=5)
        
        # 控制按钮
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill='x', padx=5, pady=5)
        
        self.generate_button = ttk.Button(control_frame, text="生成语音", command=self.generate_speech)
        self.generate_button.pack(side='left', padx=5, pady=5)
        
        self.play_button = ttk.Button(control_frame, text="播放", command=self.play_audio)
        self.play_button.pack(side='left', padx=5, pady=5)
        
        # 状态显示
        self.status_label = ttk.Label(parent, text="就绪")
        self.status_label.pack(fill='x', padx=5, pady=5)
    
    def toggle_recording(self):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        self.recording = True
        self.record_button.configure(text="停止录音")
        self.audio_data = []
        
        def audio_callback(indata, frames, time, status):
            if status:
                print(status)
            self.audio_data.extend(indata[:, 0])
        
        self.stream = sd.InputStream(callback=audio_callback,
                                   channels=1,
                                   samplerate=self.sample_rate)
        self.stream.start()
    
    def stop_recording(self):
        self.recording = False
        self.record_button.configure(text="开始录音")
        self.stream.stop()
        self.stream.close()
        
        # 保存录音
        if self.audio_data:
            self.save_and_upload_recording()
    
    def save_and_upload_recording(self):
        # 保存录音文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.wav"
        sf.write(filename, np.array(self.audio_data), self.sample_rate)
        
        # 上传文件
        self.upload_audio_file(filename)
    
    def upload_audio_file(self, filename=None):
        if not filename:
            filename = filedialog.askopenfilename(
                title="选择音频文件",
                filetypes=[("WAV files", "*.wav")]
            )
            if not filename:
                return
        
        try:
            self.status_label.configure(text="正在上传并处理音频...")
            self.root.update()
            
            with open(filename, 'rb') as f:
                files = {'file': (os.path.basename(filename), f, 'audio/wav')}
                response = requests.post(
                    f"{self.server_url}/stt",
                    files=files,
                    data={'domain': 'telecom', 'forward': 'false'}
                )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("has_transcription"):
                    self.stt_result.delete(1.0, tk.END)
                    self.stt_result.insert(tk.END, result["transcription"])
                    self.status_label.configure(text="处理完成")
                else:
                    self.status_label.configure(text="未能识别出文字")
            else:
                self.status_label.configure(text=f"处理失败: {response.status_code}")
        
        except Exception as e:
            self.status_label.configure(text=f"错误: {str(e)}")
            messagebox.showerror("错误", str(e))
    
    def generate_speech(self):
        text = self.tts_input.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("警告", "请输入要转换的文本")
            return
        
        try:
            self.status_label.configure(text="正在生成语音...")
            self.root.update()
            
            # 准备请求数据
            data = {
                "text": text,
                "speaker_id": self.speaker_id.get() or None,
                "temperature": float(self.temperature.get()),
                "top_p": 0.5,
                "top_k": 10,
                "prompt": "[oral_0][laugh_0][break_3]"
            }
            
            # 发送请求
            response = requests.post(
                f"{self.server_url}/tts",
                json=data
            )
            
            if response.status_code == 200:
                # 保存音频文件
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"generated_{timestamp}.wav"
                with open(filename, 'wb') as f:
                    f.write(response.content)
                
                self.current_audio_file = filename
                self.status_label.configure(text="语音生成完成")
            else:
                self.status_label.configure(text=f"生成失败: {response.status_code}")
        
        except Exception as e:
            self.status_label.configure(text=f"错误: {str(e)}")
            messagebox.showerror("错误", str(e))
    
    def play_audio(self):
        if hasattr(self, 'current_audio_file') and os.path.exists(self.current_audio_file):
            try:
                data, samplerate = sf.read(self.current_audio_file)
                sd.play(data, samplerate)
                sd.wait()
            except Exception as e:
                messagebox.showerror("错误", f"播放失败: {str(e)}")
        else:
            messagebox.showwarning("警告", "没有可播放的音频文件")

if __name__ == "__main__":
    root = tk.Tk()
    app = VoicePipelineClient(root)
    root.mainloop() 