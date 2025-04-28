# 语音流式处理系统 (VIOCEPIPELINE)

这是一个完整的语音流式处理系统，整合了语音活动检测(VAD)、声纹验证和语音转文字功能。系统包含客户端和服务器两个主要组件，通过HTTP协议进行通信。

## 使用方法

### 服务端

服务端已在服务器上完成部署，直接运行以下指令启动：

```bash
# 启动语音处理服务器（默认端口号为7860）
/data/chenl/voicepipeline/server/run_server.sh
```

### 客户端

客户端需要安装依赖并运行：

```bash
# 安装依赖
pip install -r voicepipeline/client/requirements.txt

# 启动客户端 (默认连接172.27.33.64:7860)
python voicepipeline/client/vad_client.py 

# 指定服务器IP和端口
python voicepipeline/client/vad_client.py --ip 服务器IP地址 --port 端口号

# 将语音转录结果转发到指定地址 (可选)
python voicepipeline/client/vad_client.py --forward-ip 转发IP地址 --forward-port 转发端口号
```

### 环境配置

使用conda环境：

```bash
# 使用已创建的环境
source conda activate chenl

# 或直接使用环境中的Python解释器
/data/envs/chenl/bin/python voicepipeline/client/vad_client.py
```

## 系统架构

```
viocepipeline/
├── client/                  # 客户端组件
│   ├── recordings/          # 录音文件存储目录
│   ├── models/              # VAD模型目录(Silero VAD)
│   │   └── silero_vad.jit   # 预训练的VAD模型
│   ├── vad_client.py        # 主客户端程序
│   └── file_record.json     # 录音历史记录
├── server/                  # 服务器组件
│   ├── voice_recognition/   # 声纹识别模块
│   ├── voice_to_text/       # 语音转文字模块 
│   ├── uploads/             # 上传文件存储目录
│   ├── logs/                # 日志文件目录
│   └── main_server.py       # 主服务器程序
└── README.md                # 项目说明文件
```

## 功能概述

### 客户端功能
- **实时语音检测**：使用Silero VAD模型检测用户说话活动
- **自动录音**：检测到语音后自动开始录制，静音后自动结束
- **录音保存**：将录制的语音按时间戳命名保存为WAV文件
- **历史记录**：保存所有录音文件的记录和服务器响应
- **HTTP通信**：将录音以标准HTTP请求发送到服务器进行处理

### 服务器功能
- **语音处理**：接收并处理客户端上传的音频文件
- **声纹验证**：识别说话人身份，支持员工声纹注册与验证
- **语音转文字**：将语音内容转换为文本
- **专业词库**：支持电信等领域的专业术语识别
- **JSON响应**：以标准JSON格式返回处理结果

## 典型工作流程

1. 启动服务器端
2. 启动客户端，连接到服务器
3. 开始说话，客户端自动检测语音
4. 语音结束后，客户端将WAV文件发送到服务器
5. 服务器进行声纹验证和语音转文字
6. 客户端接收服务器响应并显示结果

# 语音处理流水线

本项目包含完整的语音处理流水线，包括语音检测、语音识别和声纹识别等组件。

## 项目结构

- client-recorder: 语音检测客户端，使用Silero VAD模型进行语音活动检测
- server-asr: 语音识别服务器，包含中文和英文语音识别功能
- server-tts: 语音合成服务器 (尚未实现)
- models: 集中式模型管理目录

## 集中式模型管理

为了更好地管理各个组件使用的AI模型，项目使用了集中式模型管理结构：

```
models/
├── Automatic Speech Recognition/       # 自动语音识别模型
│   └── wav2vec2-large-960h/            # 英文语音识别模型
├── Voiceprint Recognition/             # 声纹识别模型
│   └── wav2vec2-large-xlsr-53-chinese-zh-cn/ # 中文语音识别模型
├── Text To Speech/                     # 语音合成模型（预留）
└── Voice Activity Detection/           # 语音活动检测模型
    └── silero_vad/                     # Silero VAD模型
```

### 集中式模型目录的优势

1. 模型文件集中管理，便于维护和更新
2. 减少重复存储相同模型的空间占用
3. 便于模型版本控制和备份
4. 项目组件可以共享相同的模型文件

### 模型加载逻辑

各组件的模型加载逻辑已更新，遵循以下优先级：

1. 首先尝试从集中式模型目录加载
2. 如果集中式目录不存在或加载失败，尝试从组件本地目录加载
3. 如果本地目录也不存在，则从网络下载并同时保存到集中式目录和本地目录

这种方式确保了即使在集中式目录不可用的情况下，系统仍然可以正常工作。

## 如何使用

1. 启动语音检测客户端：
   ```
   cd client-recorder
   python vad_client.py
   ```

2. 启动语音识别服务器：
   ```
   cd server-asr
   python main_server.py
   ```

## 模型文件

- Silero VAD: 用于语音活动检测的轻量级模型
- wav2vec2-large-960h: 用于英文语音识别和声纹识别的模型
- wav2vec2-large-xlsr-53-chinese-zh-cn: 用于中文语音识别的多语言模型

## 目录结构变更说明

为了更好地组织和管理模型文件，项目的模型目录结构已经从原来的简短名称更改为更具描述性的名称：

| 原目录名称 | 新目录名称 | 说明 |
|---------|---------|------|
| models/asr/ | models/Automatic Speech Recognition/ | 用于英文语音识别的模型目录 |
| models/asr/ | models/Voiceprint Recognition/ | 用于中文语音识别和声纹识别的模型目录 |
| models/tts/ | models/Text To Speech/ | 用于语音合成的模型目录（预留） |
| models/vad/ | models/Voice Activity Detection/ | 用于语音活动检测的模型目录 |

代码中已经更新了相应的路径引用，确保系统可以正确找到并加载模型文件。


