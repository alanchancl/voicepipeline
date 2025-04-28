#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import wave
import pyaudio
import time
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import threading
import datetime
import numpy as np
import uuid
import random
from voice_recognition import add_voice_sample, verify_voice, VOICE_SAMPLES_DIR

# 录音参数
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
MAX_RECORD_SECONDS = 30  # 最大录制时长
MIN_RECORD_SECONDS = 3   # 最小录制时长

class EmployeeVoiceRecorder:
    def __init__(self, master):
        self.master = master
        master.title("营业员声纹采集工具")
        master.geometry("600x800")  # 增加窗口高度以容纳麦克风选择
        master.minsize(600, 800)    # 增加最小高度
        
        # 设置样式
        self.style = ttk.Style()
        self.style.configure("TFrame", background="#f0f0f0")
        self.style.configure("TButton", font=("微软雅黑", 10))
        self.style.configure("TLabel", font=("微软雅黑", 10), background="#f0f0f0")
        self.style.configure("Header.TLabel", font=("微软雅黑", 16, "bold"), background="#f0f0f0")
        self.style.configure("SubHeader.TLabel", font=("微软雅黑", 12), background="#f0f0f0")
        self.style.configure("Prompt.TLabel", font=("微软雅黑", 12, "italic"), background="#e6f3ff", wraplength=550, padding=10)
        
        # 朗读提示文本列表
        self.prompts = [
            "请说：今天天气真不错，我很高兴为您服务。",
            "请说：欢迎光临，有什么可以帮您的吗？",
            "请说：我是XX号营业员，请问您需要办理什么业务？",
            "请说：如果您有任何问题，随时可以咨询我。",
            "请说：感谢您的耐心等待，马上为您处理。",
            "请说：您好，请出示您的有效证件。",
            "请说：您的业务已经办理完成，祝您生活愉快。",
            "请说：请确认信息是否正确，如有问题请告诉我。",
            "请说：非常抱歉让您久等了，请问还有其他需要吗？",
            "请说：这是您的回执单，请妥善保管。"
        ]
        self.current_prompt = random.choice(self.prompts)
        
        # 创建主框架
        self.main_frame = ttk.Frame(master, padding="20", style="TFrame")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题
        self.title_label = ttk.Label(self.main_frame, text="营业员声纹采集系统", style="Header.TLabel")
        self.title_label.pack(pady=(0, 20))
        
        # 员工信息区域
        self.employee_frame = ttk.LabelFrame(self.main_frame, text="员工信息", padding="10")
        self.employee_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 员工ID
        self.employee_id_frame = ttk.Frame(self.employee_frame)
        self.employee_id_frame.pack(fill=tk.X, pady=5)
        
        self.employee_id_label = ttk.Label(self.employee_id_frame, text="员工ID:", width=10)
        self.employee_id_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.employee_id_var = tk.StringVar()
        self.employee_id_entry = ttk.Entry(self.employee_id_frame, textvariable=self.employee_id_var)
        self.employee_id_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # 员工姓名
        self.employee_name_frame = ttk.Frame(self.employee_frame)
        self.employee_name_frame.pack(fill=tk.X, pady=5)
        
        self.employee_name_label = ttk.Label(self.employee_name_frame, text="员工姓名:", width=10)
        self.employee_name_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.employee_name_var = tk.StringVar()
        self.employee_name_entry = ttk.Entry(self.employee_name_frame, textvariable=self.employee_name_var)
        self.employee_name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # 添加麦克风选择区域
        self.mic_frame = ttk.LabelFrame(self.main_frame, text="麦克风设置", padding="10")
        self.mic_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 获取所有音频输入设备
        self.audio = pyaudio.PyAudio()
        self.mic_devices = []
        self.mic_names = []
        for i in range(self.audio.get_device_count()):
            dev_info = self.audio.get_device_info_by_index(i)
            if dev_info.get('maxInputChannels') > 0:  # 是输入设备
                self.mic_devices.append(i)
                name = dev_info.get('name', f'设备 {i}')
                self.mic_names.append(name)
                print(f"找到输入设备 {i}: {name}")
        
        # 麦克风选择下拉菜单
        self.mic_selection_frame = ttk.Frame(self.mic_frame)
        self.mic_selection_frame.pack(fill=tk.X, pady=5)
        
        self.mic_label = ttk.Label(self.mic_selection_frame, text="选择麦克风:", width=15)
        self.mic_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.selected_mic_var = tk.StringVar()
        self.mic_combo = ttk.Combobox(self.mic_selection_frame, textvariable=self.selected_mic_var)
        self.mic_combo['values'] = self.mic_names
        self.mic_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        if self.mic_names:  # 默认选择第一个设备
            self.mic_combo.current(0)
        
        self.refresh_mic_button = ttk.Button(self.mic_selection_frame, text="刷新设备", command=self.refresh_mic_devices)
        self.refresh_mic_button.pack(side=tk.RIGHT, padx=(5, 0))
        
        # 录音区域
        self.record_frame = ttk.LabelFrame(self.main_frame, text="录音控制", padding="10")
        self.record_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 朗读提示区域
        self.prompt_frame = ttk.Frame(self.record_frame)
        self.prompt_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.prompt_label = ttk.Label(self.prompt_frame, text=self.current_prompt, style="Prompt.TLabel")
        self.prompt_label.pack(fill=tk.X, pady=5)
        
        self.change_prompt_button = ttk.Button(self.prompt_frame, text="换一个提示", command=self.change_prompt)
        self.change_prompt_button.pack(pady=5)
        
        # 录音状态
        self.status_var = tk.StringVar(value="准备就绪")
        self.status_label = ttk.Label(self.record_frame, textvariable=self.status_var, font=("微软雅黑", 12))
        self.status_label.pack(pady=(0, 10))
        
        # 录音时长
        self.duration_var = tk.StringVar(value="00:00")
        self.duration_label = ttk.Label(self.record_frame, textvariable=self.duration_var, font=("微软雅黑", 24))
        self.duration_label.pack(pady=(0, 10))
        
        # 录音按钮
        self.button_frame = ttk.Frame(self.record_frame)
        self.button_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.record_button = ttk.Button(self.button_frame, text="开始录音", command=self.toggle_recording)
        self.record_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        self.play_button = ttk.Button(self.button_frame, text="播放录音", command=self.play_recording, state=tk.DISABLED)
        self.play_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        self.save_button = ttk.Button(self.button_frame, text="保存录音", command=self.save_recording, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        # 已采集样本区域
        self.samples_frame = ttk.LabelFrame(self.main_frame, text="已采集的声纹样本", padding="10")
        self.samples_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 添加树形视图显示样本 - 限制高度为5行
        self.samples_tree = ttk.Treeview(self.samples_frame, columns=("id", "date", "duration", "status"), 
                                         show="headings", selectmode="browse", height=5)
        self.samples_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 添加样本列表的滚动条
        self.tree_scrollbar = ttk.Scrollbar(self.samples_frame, orient="vertical", command=self.samples_tree.yview)
        self.tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.samples_tree.configure(yscrollcommand=self.tree_scrollbar.set)
        
        # 设置列
        self.samples_tree.heading("id", text="采样ID")
        self.samples_tree.heading("date", text="采样日期")
        self.samples_tree.heading("duration", text="录音时长")
        self.samples_tree.heading("status", text="状态")
        
        self.samples_tree.column("id", width=100)
        self.samples_tree.column("date", width=150)
        self.samples_tree.column("duration", width=80)
        self.samples_tree.column("status", width=100)
        
        # 底部按钮区域 - 确保有足够的间距
        self.bottom_frame = ttk.Frame(self.main_frame)
        self.bottom_frame.pack(fill=tk.X, pady=(10, 20))
        
        self.test_button = ttk.Button(self.bottom_frame, text="测试识别", command=self.test_recognition)
        self.test_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        self.refresh_button = ttk.Button(self.bottom_frame, text="刷新列表", command=self.refresh_samples)
        self.refresh_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        self.delete_button = ttk.Button(self.bottom_frame, text="删除选中", command=self.delete_sample)
        self.delete_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        # 录音相关变量
        self.is_recording = False
        self.frames = []
        self.current_audio_file = None
        self.record_start_time = None
        self.record_duration = 0
        self.timer_id = None
        
        # 初始化样本列表
        self.refresh_samples()
        
    def refresh_mic_devices(self):
        """刷新麦克风设备列表"""
        # 重新创建PyAudio实例以刷新设备列表
        if hasattr(self, 'audio'):
            self.audio.terminate()
        self.audio = pyaudio.PyAudio()
        
        self.mic_devices = []
        self.mic_names = []
        for i in range(self.audio.get_device_count()):
            dev_info = self.audio.get_device_info_by_index(i)
            if dev_info.get('maxInputChannels') > 0:  # 是输入设备
                self.mic_devices.append(i)
                name = dev_info.get('name', f'设备 {i}')
                self.mic_names.append(name)
                print(f"找到输入设备 {i}: {name}")
        
        # 更新下拉菜单
        self.mic_combo['values'] = self.mic_names
        if self.mic_names:  # 默认选择第一个设备
            self.mic_combo.current(0)
        self.status_var.set("麦克风设备列表已刷新")
        
    def get_selected_mic_index(self):
        """获取当前选定的麦克风设备索引"""
        try:
            selection = self.mic_combo.current()
            if selection >= 0 and selection < len(self.mic_devices):
                return self.mic_devices[selection]
            return None  # 没有选择设备
        except:
            return None
    
    def toggle_recording(self):
        """开始或停止录音"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """开始录音"""
        if not self.employee_id_var.get().strip():
            messagebox.showwarning("警告", "请先输入员工ID")
            return
        
        # 检查是否选择了麦克风
        if not self.get_selected_mic_index():
            messagebox.showwarning("警告", "请先选择麦克风设备")
            return
        
        # 禁用更换提示按钮和麦克风选择
        self.change_prompt_button.config(state=tk.DISABLED)
        self.mic_combo.config(state=tk.DISABLED)
        self.refresh_mic_button.config(state=tk.DISABLED)
        
        # 禁用保存和播放按钮
        self.play_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
        
        # 改变按钮文本和状态
        self.record_button.config(text="停止录音")
        self.status_var.set("正在录音...")
        
        # 重置录音数据
        self.frames = []
        self.record_start_time = time.time()
        self.record_duration = 0
        self.is_recording = True
        
        # 开始录音线程
        self.recording_thread = threading.Thread(target=self.record_audio)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        # 开始计时器
        self.update_duration()
    
    def record_audio(self):
        """录音线程"""
        try:
            mic_index = self.get_selected_mic_index()
            print(f"使用麦克风设备: {mic_index} - {self.mic_names[self.mic_devices.index(mic_index)]}")
            
            stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=mic_index,  # 指定输入设备
                frames_per_buffer=CHUNK
            )
            
            # 录音循环
            while self.is_recording and time.time() - self.record_start_time < MAX_RECORD_SECONDS:
                data = stream.read(CHUNK)
                self.frames.append(data)
            
            # 如果超时自动停止
            if time.time() - self.record_start_time >= MAX_RECORD_SECONDS:
                self.master.after(0, self.stop_recording)
            
            # 关闭流
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            self.master.after(0, lambda: self.show_error(f"录音过程中出错: {str(e)}"))
            self.master.after(0, self.stop_recording)
    
    def stop_recording(self):
        """停止录音"""
        if not self.is_recording:
            return
            
        self.is_recording = False
        
        # 等待录音线程结束
        if hasattr(self, 'recording_thread') and self.recording_thread.is_alive():
            self.recording_thread.join(1.0)
        
        # 取消计时器
        if self.timer_id:
            self.master.after_cancel(self.timer_id)
            self.timer_id = None
        
        # 计算最终录音时长
        self.record_duration = time.time() - self.record_start_time
        
        # 启用更换提示按钮和麦克风选择
        self.change_prompt_button.config(state=tk.NORMAL)
        self.mic_combo.config(state="readonly")
        self.refresh_mic_button.config(state=tk.NORMAL)
        
        # 判断录音时长是否符合要求
        if self.record_duration < MIN_RECORD_SECONDS:
            self.status_var.set(f"录音时间太短（最少{MIN_RECORD_SECONDS}秒）")
            self.record_button.config(text="开始录音")
            return
        
        # 生成临时文件名
        now = datetime.datetime.now()
        filename = f"{now.strftime('%Y%m%d%H%M%S')}_{int(now.microsecond/1000)}.wav"
        temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
        
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        self.current_audio_file = os.path.join(temp_dir, filename)
        
        # 保存录音到临时文件
        try:
            wf = wave.open(self.current_audio_file, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(self.frames))
            wf.close()
            
            # 更新UI
            self.status_var.set(f"录音完成，时长 {self.format_duration(self.record_duration)}")
            self.record_button.config(text="开始录音")
            self.play_button.config(state=tk.NORMAL)
            self.save_button.config(state=tk.NORMAL)
            
        except Exception as e:
            self.show_error(f"保存录音文件出错: {str(e)}")
            self.status_var.set("录音保存失败")
            self.record_button.config(text="开始录音")
    
    def update_duration(self):
        """更新录音时长显示"""
        if self.is_recording:
            elapsed = time.time() - self.record_start_time
            self.duration_var.set(self.format_duration(elapsed))
            self.timer_id = self.master.after(100, self.update_duration)
    
    def format_duration(self, seconds):
        """格式化时间显示"""
        minutes = int(seconds) // 60
        seconds = int(seconds) % 60
        return f"{minutes:02d}:{seconds:02d}"
    
    def play_recording(self):
        """播放录音"""
        if not self.current_audio_file or not os.path.exists(self.current_audio_file):
            self.show_error("没有可播放的录音")
            return
        
        # 使用系统默认播放器播放
        try:
            import platform
            import subprocess
            
            system = platform.system()
            
            if system == 'Windows':
                os.startfile(self.current_audio_file)
            elif system == 'Darwin':  # macOS
                subprocess.call(['open', self.current_audio_file])
            else:  # Linux
                subprocess.call(['xdg-open', self.current_audio_file])
                
            self.status_var.set("正在播放录音...")
            
        except Exception as e:
            self.show_error(f"播放录音时出错: {str(e)}")
    
    def save_recording(self):
        """保存录音到声纹库"""
        if not self.current_audio_file or not os.path.exists(self.current_audio_file):
            self.show_error("没有可保存的录音")
            return
        
        employee_id = self.employee_id_var.get().strip()
        if not employee_id:
            self.show_error("请先输入员工ID")
            return
        
        try:
            # 添加到声纹库
            success = add_voice_sample(self.current_audio_file, employee_id)
            
            if success:
                self.status_var.set("声纹样本保存成功!")
                # 刷新样本列表
                self.refresh_samples()
                # 禁用保存按钮，防止重复保存
                self.save_button.config(state=tk.DISABLED)
                # 更换提示文本，为下一次录音做准备
                self.change_prompt()
            else:
                self.status_var.set("声纹样本保存失败")
        except Exception as e:
            self.show_error(f"保存声纹样本时出错: {str(e)}")
    
    def refresh_samples(self):
        """刷新已采集的样本列表"""
        # 清空当前列表
        for item in self.samples_tree.get_children():
            self.samples_tree.delete(item)
        
        employee_id = self.employee_id_var.get().strip()
        if not employee_id:
            return
        
        # 员工目录
        employee_dir = os.path.join(VOICE_SAMPLES_DIR, employee_id)
        if not os.path.exists(employee_dir):
            return
        
        # 获取该员工的所有样本
        try:
            for file in os.listdir(employee_dir):
                if file.endswith('.wav'):
                    # 获取文件信息
                    file_path = os.path.join(employee_dir, file)
                    create_time = datetime.datetime.fromtimestamp(os.path.getctime(file_path))
                    
                    # 获取音频时长
                    try:
                        with wave.open(file_path, 'rb') as wf:
                            frames = wf.getnframes()
                            rate = wf.getframerate()
                            duration = frames / float(rate)
                    except:
                        duration = 0
                    
                    # 测试该样本的质量
                    result = verify_voice(file_path)
                    status = "有效" if result['similarity'] > 0.5 else "质量低"
                    
                    # 添加到列表
                    self.samples_tree.insert("", tk.END, values=(
                        file,
                        create_time.strftime('%Y-%m-%d %H:%M:%S'),
                        self.format_duration(duration),
                        status
                    ))
        except Exception as e:
            self.show_error(f"刷新样本列表时出错: {str(e)}")
    
    def delete_sample(self):
        """删除选中的样本"""
        selected = self.samples_tree.selection()
        if not selected:
            self.show_error("请先选择要删除的样本")
            return
        
        employee_id = self.employee_id_var.get().strip()
        if not employee_id:
            return
        
        # 确认删除
        if not messagebox.askyesno("确认删除", "确定要删除选中的声纹样本吗？"):
            return
        
        # 删除文件
        for item in selected:
            values = self.samples_tree.item(item, 'values')
            file_name = values[0]
            file_path = os.path.join(VOICE_SAMPLES_DIR, employee_id, file_name)
            
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                self.show_error(f"删除样本时出错: {str(e)}")
        
        # 刷新列表
        self.refresh_samples()
        self.status_var.set("样本已删除")
    
    def test_recognition(self):
        """测试声纹识别"""
        if not self.current_audio_file or not os.path.exists(self.current_audio_file):
            self.show_error("请先录制一段声音用于测试")
            return
        
        try:
            # 执行声纹识别
            result = verify_voice(self.current_audio_file)
            
            # 显示结果
            message = f"识别结果:\n"
            message += f"验证结果: {'通过' if result['verified'] else '未通过'}\n"
            message += f"最佳匹配员工: {result['employee_id'] if result['employee_id'] else '无'}\n"
            message += f"相似度: {result['similarity']:.4f}\n"
            message += f"消息: {result['message']}"
            
            messagebox.showinfo("识别结果", message)
            
        except Exception as e:
            self.show_error(f"测试识别时出错: {str(e)}")
    
    def change_prompt(self):
        """随机更换朗读提示"""
        new_prompt = random.choice(self.prompts)
        while new_prompt == self.current_prompt and len(self.prompts) > 1:
            new_prompt = random.choice(self.prompts)
        self.current_prompt = new_prompt
        self.prompt_label.config(text=self.current_prompt)
    
    def show_error(self, message):
        """显示错误消息"""
        messagebox.showerror("错误", message)

def main():
    """主函数"""
    root = tk.Tk()
    app = EmployeeVoiceRecorder(root)
    root.mainloop()

if __name__ == "__main__":
    main() 