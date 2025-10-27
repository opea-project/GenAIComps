# FireRedASR ASR Microservice

这是一个基于 OPEA 框架的 FireRedASR 语音识别微服务，提供与 OpenAI API 完全兼容的音频转录功能。

## 功能特性

- 🎯 **OpenAI API 完全兼容**: 100% 兼容 OpenAI 的音频转录 API ([参考文档](https://platform.openai.com/docs/api-reference/audio/createTranscription))
- 🔥 **高性能**: 基于 FireRedASR 模型，支持 AED 和 LLM 两种模式
- 🐳 **容器化**: 支持 Docker 和 Kubernetes 部署
- 📊 **监控**: 内置健康检查和性能监控
- 🌐 **负载均衡**: 支持 Nginx 反向代理
- 🚀 **可扩展**: 支持水平扩展和自动扩缩容
- 🎤 **语音识别**: 专业的语音识别服务，支持多种音频格式

## 快速开始

### 1. 环境要求

- Python 3.9+
- Docker 20.10+
- Docker Compose 2.0+
- Kubernetes 1.20+ (可选)

### 2. 准备模型文件

下载 FireRedASR 模型文件到本地目录：

#### 方法1: 使用外部模型目录（推荐）

```bash
# 创建模型目录
mkdir -p /path/to/your/models

# 下载模型文件 (请替换为实际的模型下载链接)
# mkdir -p /path/to/your/models/FireRedASR-LLM-L
# wget [model_url] -O /path/to/your/models/FireRedASR-LLM-L/model.pth.tar
# wget [encoder_url] -O /path/to/your/models/FireRedASR-LLM-L/asr_encoder.pth.tar
# wget [cmvn_url] -O /path/to/your/models/FireRedASR-LLM-L/cmvn.ark
# git clone [qwen2_repo] /path/to/your/models/FireRedASR-LLM-L/Qwen2-7B-Instruct
```

#### 方法2: 使用命名卷（Docker Compose）

```bash
# 创建模型目录并下载文件
mkdir -p ./models/FireRedASR-LLM-L
# 下载模型文件到 ./models/FireRedASR-LLM-L/ 目录

# 启动服务（模型文件会自动挂载）
cd deployment/docker_compose
docker-compose -f docker-compose.yaml up -d
```

#### 方法3: 使用 Kubernetes PersistentVolume

```bash
# 1. 准备模型文件
mkdir -p /path/to/your/models/FireRedASR-LLM-L
# 下载模型文件到该目录

# 2. 创建 PVC（生产环境）
kubectl apply -f deployment/kubernetes/pvc.yaml

# 或者使用本地开发 PVC（开发环境）
# kubectl apply -f deployment/kubernetes/pvc.yaml

# 3. 部署服务
kubectl apply -f deployment/kubernetes/deployment.yaml
```

**注意**: 模型文件通过 Docker/Kubernetes 卷挂载，而不是包含在镜像中。默认挂载路径为 `/app/pretrained_models`。

### 3. Docker 部署

#### 使用构建脚本（推荐）

项目提供了一个智能构建脚本 [`build_docker.sh`](build_docker.sh)，具有以下特性：

- 🔧 **智能重试机制**：构建失败时自动重试（最多3次）
- 🛠️ **多种构建模式**：支持标准 Dockerfile 和简化版 Dockerfile
- 📦 **依赖自动安装**：如果构建失败，可尝试手动安装依赖
- 🌐 **代理支持**：自动检测并使用 HTTP/HTTPS 代理
- 🎨 **彩色输出**：清晰的构建状态提示

**使用方法：**

```bash
# 基本构建（推荐）
./build_docker.sh

# 使用代理构建
export HTTP_PROXY=http://proxy:port
export HTTPS_PROXY=http://proxy:port
./build_docker.sh

# 使用 Docker BuildKit 构建（可选）
DOCKER_BUILDKIT=1 ./build_docker.sh
```

**脚本特性：**

1. **自动重试**：构建失败时会自动重试最多3次，每次间隔30秒
2. **备用方案**：如果标准构建失败，会尝试使用 [`Dockerfile.simple`](Dockerfile.simple)
3. **依赖处理**：如果仍然失败，会尝试在容器中手动安装依赖
4. **错误提示**：提供详细的故障排除建议

#### 使用 Docker Compose

```bash
# 构建并启动服务
cd deployment/docker_compose

# 方法1: 使用外部卷挂载 (推荐)
# docker-compose -f docker-compose.yaml up -d

# 方法2: 使用命名卷（需要预先准备模型文件）
docker-compose -f docker-compose.yaml up -d

# 方法3: 使用本地目录挂载
# 修改 docker-compose.yaml 中的 volumes 配置
# volumes:
#   - /path/to/your/models:/app/pretrained_models:ro

# 查看日志
docker-compose logs -f fireredasr-asr
```

#### 使用 Docker 命令

**方法1：使用构建脚本（推荐）**
```bash
# 使用智能构建脚本（推荐）
./build_docker.sh

# 构建脚本会自动处理重试、依赖安装等问题
```

**方法2：使用标准 Docker 命令**
```bash
# 构建镜像
docker build -t opea/fireredasr-asr:latest .

# 运行容器 (使用外部模型目录)
docker run -d \
  --name fireredasr-asr \
  -p 9099:9099 \
  -v /path/to/your/models:/app/pretrained_models:ro \
  opea/fireredasr-asr:latest

# 运行容器 (使用本地模型目录)
docker run -d \
  --name fireredasr-asr \
  -p 9099:9099 \
  -v $(pwd)/models:/app/pretrained_models:ro \
  opea/fireredasr-asr:latest

# 方法3: 使用命名卷
docker run -d \
  --name fireredasr-asr \
  -p 9099:9099 \
  -v fireredasr-models:/app/pretrained_models:ro \
  opea/fireredasr-asr:latest
```

### 4. Kubernetes 部署

```bash
# 部署到 Kubernetes
kubectl apply -f deployment/kubernetes/deployment.yaml

# 查看部署状态
kubectl get pods -l app=fireredasr-asr

# 查看服务
kubectl get service fireredasr-asr-service

# 查看模型卷状态
kubectl get pvc fireredasr-asr-models-pvc
```

**注意**: Kubernetes部署需要预先准备模型文件并挂载到PersistentVolumeClaim中。

## API 使用

### API 端点

本服务提供与 OpenAI 完全兼容的音频转录 API：

- **POST** `/v1/audio/transcriptions` - 创建音频转录
- **GET** `/health` - 健康检查

### OpenAI API 兼容性

本 API 完全兼容 OpenAI 的音频转录 API，支持以下参数：

| 参数 | 类型 | 必需 | 描述 |
|------|------|------|------|
| file | File | 是 | 要转录的音频文件 |
| model | string | 否 | 要使用的模型，默认为 "fireredasr" |
| language | string | 否 | 语言代码，默认为 "auto" |
| prompt | string | 否 | 可选的转录提示 |
| response_format | string | 否 | 响应格式，"json" 或 "text"，默认为 "json" |
| temperature | number | 否 | 采样温度，默认为 0 |
| timestamp_granularities | array | 否 | 时间戳粒度，目前不支持 |

### 基本用法

#### 健康检查

```bash
curl http://localhost:9099/health
```

#### 音频转录（Base64 编码）

```bash
curl -X POST http://localhost:9099/v1/audio/transcriptions \
  -H "Content-Type: application/json" \
  -d '{
    "file": "base64_encoded_audio_data",
    "model": "fireredasr",
    "language": "auto",
    "response_format": "json"
  }'
```

#### 音频转录（文件上传）

```bash
curl -X POST http://localhost:9099/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=fireredasr" \
  -F "language=auto" \
  -F "response_format=json"
```

### 响应格式

#### JSON 响应

```json
{
  "text": "这是转录的文本内容。"
}
```

#### Text 响应

```
这是转录的文本内容。
```

### Python 客户端示例

```python
import requests
import base64

# 方法1: 文件上传
def transcribe_with_file_upload(audio_path, model="fireredasr", language="auto"):
    with open(audio_path, "rb") as audio_file:
        files = {"file": (audio_path, audio_file, "audio/wav")}
        data = {
            "model": model,
            "language": language,
            "response_format": "json"
        }
        response = requests.post(
            "http://localhost:9099/v1/audio/transcriptions",
            files=files,
            data=data
        )
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API Error: {response.status_code} - {response.text}")

# 方法2: Base64 编码
def transcribe_with_base64(audio_path, model="fireredasr", language="auto"):
    # 读取音频文件并编码
    with open(audio_path, "rb") as audio_file:
        audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')
    
    # 发送请求
    response = requests.post(
        "http://localhost:9099/v1/audio/transcriptions",
        json={
            "file": audio_base64,
            "model": model,
            "language": language,
            "response_format": "json"
        }
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API Error: {response.status_code} - {response.text}")

# 使用示例
try:
    # 文件上传方式
    result1 = transcribe_with_file_upload("audio.wav")
    print(f"转录结果: {result1['text']}")
    
    # Base64 编码方式
    result2 = transcribe_with_base64("audio.wav")
    print(f"转录结果: {result2['text']}")
    
except Exception as e:
    print(f"错误: {e}")
```

### JavaScript/TypeScript 客户端示例

```javascript
// 方法1: 文件上传
async function transcribeWithFileUpload(audioFile, model = "fireredasr", language = "auto") {
    const formData = new FormData();
    formData.append("file", audioFile);
    formData.append("model", model);
    formData.append("language", language);
    formData.append("response_format", "json");

    const response = await fetch("http://localhost:9099/v1/audio/transcriptions", {
        method: "POST",
        body: formData
    });

    if (!response.ok) {
        throw new Error(`API Error: ${response.status} - ${response.statusText}`);
    }

    return await response.json();
}

// 方法2: Base64 编码
async function transcribeWithBase64(audioBase64, model = "fireredasr", language = "auto") {
    const response = await fetch("http://localhost:9099/v1/audio/transcriptions", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            file: audioBase64,
            model: model,
            language: language,
            response_format: "json"
        })
    });

    if (!response.ok) {
        throw new Error(`API Error: ${response.status} - ${response.statusText}`);
    }

    return await response.json();
}

// 使用示例
document.getElementById('audioFile').addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (file) {
        try {
            // 文件上传方式
            const result1 = await transcribeWithFileUpload(file);
            console.log("转录结果:", result1.text);
            
            // 或者转换为 Base64 后使用
            const reader = new FileReader();
            reader.onload = async (e) => {
                const base64 = e.target.result.split(',')[1]; // 移除 data:audio/wav;base64, 前缀
                const result2 = await transcribeWithBase64(base64);
                console.log("转录结果:", result2.text);
            };
            reader.readAsDataURL(file);
            
        } catch (error) {
            console.error("错误:", error);
        }
    }
});
```

### 测试 API

运行测试脚本验证 API 功能：

```bash
# 设置测试音频文件路径
export FIREREDASR_TEST_AUDIO=examples/wav/BAC009S0764W0121.wav

# 运行测试
python src/check_asr_server.py
```

测试脚本包含以下测试：
- 健康检查
- 文件上传转录
- Base64 编码转录
- API 兼容性测试
- 错误处理测试

## 配置选项

### 环境变量

| 变量名 | 默认值 | 描述 |
|--------|--------|------|
| `FIREREDASR_MODEL_DIR` | `/app/pretrained_models` | 模型目录路径 |
| `FIREREDASR_ASR_TYPE` | `llm` | ASR 类型 (`aed` 或 `llm`) |
| `FIREREDASR_USE_GPU` | `false` | 是否使用 GPU |
| `FIREREDASR_BATCH_SIZE` | `1` | 批处理大小 |
| `FIREREDASR_BEAM_SIZE` | `1` | Beam search 大小 |
| `FIREREDASR_TEMPERATURE` | `1.0` | 温度参数 (LLM 模式) |
| `FIREREDASR_REPETITION_PENALTY` | `1.0` | 重复惩罚 (LLM 模式) |
| `LOGFLAG` | `true` | 启用日志 |
| `ENABLE_MCP` | `false` | 启用 MCP |

### 模型参数

#### AED 模式参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `FIREREDASR_NBEST` | `1` | N-best 大小 |
| `FIREREDASR_SOFTMAX_SMOOTHING` | `1.0` | Softmax 平滑 |
| `FIREREDASR_AED_LENGTH_PENALTY` | `0.0` | 长度惩罚 |
| `FIREREDASR_EOS_PENALTY` | `1.0` | EOS 惩罚 |
| `FIREREDASR_DECODE_MAX_LEN` | `0` | 最大解码长度 |

#### LLM 模式参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `FIREREDASR_DECODE_MIN_LEN` | `0` | 最小解码长度 |
| `FIREREDASR_LLM_LENGTH_PENALTY` | `0.0` | LLM 长度惩罚 |

## 测试

### 运行测试脚本

```bash
# 设置测试音频文件路径
export FIREREDASR_TEST_AUDIO=examples/wav/BAC009S0764W0121.wav

# 运行测试
python src/check_asr_server.py
```

### 手动测试

```bash
# 健康检查
curl http://localhost:9099/health

# 测试音频转录
python src/check_asr_server.py
```

## 监控和日志

### 健康检查

```bash
# 检查服务状态
curl http://localhost:9099/health

# 检查 Docker 容器状态
docker ps | grep fireredasr-asr

# 检查 Kubernetes Pod 状态
kubectl get pods -l app=fireredasr-asr
```

### 日志查看

```bash
# Docker 日志
docker logs fireredasr-asr

# Docker Compose 日志
docker-compose logs -f fireredasr-asr

# Kubernetes 日志
kubectl logs -f deployment/fireredasr-asr-deployment
```

## 性能优化

### 1. GPU 加速

```bash
# 启用 GPU
export FIREREDASR_USE_GPU=true

# Docker 运行时添加 GPU 支持
docker run --gpus all fireredasr-asr:latest
```

### 2. 批处理

```bash
# 增加批处理大小
export FIREREDASR_BATCH_SIZE=4
```

### 3. 模型优化

```bash
# 调整 beam size
export FIREREDASR_BEAM_SIZE=3

# 调整温度参数 (LLM 模式)
export FIREREDASR_TEMPERATURE=0.8
```

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型文件是否完整
   - 确认模型目录路径正确
   - 检查磁盘空间是否足够

2. **内存不足**
   - 减少 `FIREREDASR_BATCH_SIZE`
   - 启用 GPU 支持
   - 增加容器内存限制

3. **API 响应慢**
   - 检查网络连接
   - 优化模型参数
   - 考虑使用 GPU

4. **Docker 构建失败**
   - **使用构建脚本**：运行 `./build_docker.sh` 自动处理构建问题
   - **检查网络**：确保网络连接正常，或设置代理
   - **使用 BuildKit**：尝试 `DOCKER_BUILDKIT=1 ./build_docker.sh`
   - **简化构建**：构建脚本会自动尝试使用 [`Dockerfile.simple`](Dockerfile.simple)
   - **手动安装依赖**：脚本会尝试在容器中手动安装依赖

### 调试模式

```bash
# 启用详细日志
export LOGFLAG=true

# 检查模型文件
ls -la pretrained_models/FireRedASR-LLM-L/

# 检查容器资源使用
docker stats fireredasr-asr
```

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

本项目基于 Apache 2.0 许可证开源。

## 相关链接

- [FireRedASR 原始项目](https://github.com/[FireRedASR-repo])
- [OPEA 框架](https://github.com/opea-project/GenAIComps)
- [OpenAI API 文档](https://platform.openai.com/docs/api-reference/audio/createTranscription)