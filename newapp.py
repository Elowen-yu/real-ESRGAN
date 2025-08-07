# backend/newapp.py - 优化版本
import sys
print("当前Python路径：", sys.executable)

from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import os, shutil, uuid, subprocess
from werkzeug.utils import secure_filename
import zipfile
import threading
import time
from queue import Queue
import torch

app = Flask(__name__)
# 允许所有来源的跨域请求
CORS(app, origins=["*"])

BASE = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE, "uploads")
RESULT_FOLDER = os.path.join(BASE, "results")
print(f"BASE目录: {BASE}")
print(f"上传目录: {UPLOAD_FOLDER}")
print(f"结果目录: {RESULT_FOLDER}")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
print("目录创建完成")

# 全局变量用于存储模型实例
upsampler = None
model_lock = threading.Lock()

def initialize_model():
    """初始化Real-ESRGAN模型"""
    global upsampler
    try:
        print("正在初始化Real-ESRGAN模型...")
        # 添加Real-ESRGAN路径到sys.path
        realesrgan_path = "D:\\ai-image\\Real-ESRGAN"
        if realesrgan_path not in sys.path:
            sys.path.insert(0, realesrgan_path)
        
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        
        # 创建模型
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        model_path = os.path.join(realesrgan_path, "weights", "RealESRGAN_x2plus.pth")
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            return False
            
        # 创建upsampler
        upsampler = RealESRGANer(
            scale=2,
            model_path=model_path,
            model=model,
            tile=600,
            tile_pad=10,
            pre_pad=0,
            half=True,  # 使用半精度加速
            gpu_id=0
        )
        print("Real-ESRGAN模型初始化完成")
        return True
    except Exception as e:
        print(f"模型初始化失败: {str(e)}")
        return False

def process_image_with_model(input_path, output_path):
    """使用已加载的模型处理单张图像"""
    global upsampler
    try:
        import cv2
        # 1. 读取图像（CPU操作）
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"无法读取图像: {input_path}")
            return False
            
        # 处理图像
        # 2. 使用GPU模型处理图像（GPU操作）
        output, _ = upsampler.enhance(img, outscale=2)
        
        # 保存结果
        # 3. 保存处理后的图像（CPU操作）
        cv2.imwrite(output_path, output)
        print(f"图像处理完成: {input_path} -> {output_path}")
        return True
    except Exception as e:
        print(f"图像处理失败 {input_path}: {str(e)}")
        return False

# 配置上传限制
app.config['MAX_CONTENT_LENGTH'] = 1000000 * 1024 * 1024  # 500MB限制
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

@app.route("/")
def index():
    return jsonify({"message": "Real-ESRGAN Web API is running", "status": "ok"})

#上传文件
@app.route("/upload", methods=["POST"])
def upload():
    import time
    start_time = time.time()
    try:
        print("开始处理上传请求...")
        session_id = request.form.get("session") or str(uuid.uuid4())
        print(f"Session ID: {session_id}")
        
        upload_dir = os.path.join(UPLOAD_FOLDER, session_id)
        print(f"上传目录: {upload_dir}")
        os.makedirs(upload_dir, exist_ok=True)

        uploaded_files = []
        files = request.files.getlist("files")
        print(f"接收到 {len(files)} 个文件")
        
        for f in files:
            if f.filename:
                filename = secure_filename(f.filename)
                print(f"处理文件: {filename}")
                # 检查文件扩展名
                if not any(filename.lower().endswith(ext) for ext in app.config['UPLOAD_EXTENSIONS']):
                    print(f"跳过不支持的文件类型: {filename}")
                    continue

                file_path = os.path.join(upload_dir, filename)
                print(f"保存文件到: {file_path}")
                f.save(file_path)
                uploaded_files.append(filename)
                print(f"文件保存成功: {filename}")

        print(f"上传完成，共处理 {len(uploaded_files)} 个文件")
        end_time = time.time()
        print(f"上传处理耗时: {end_time - start_time:.3f} 秒")
        return jsonify({"status": "ok", "session": session_id, "files": uploaded_files})
    except Exception as e:
        end_time = time.time()
        print(f"上传处理异常，耗时: {end_time - start_time:.3f} 秒")
        import traceback
        error_msg = f"上传失败: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return jsonify({"status": "error", "msg": str(e)}), 500

#处理图像
@app.route("/process", methods=["POST"])
def process():
    #图像处理耗时+2
    import time
    start_time = time.time()
    print(f"开始时间: {start_time:.3f}")
    try:
        print("开始处理图像请求...")
        data = request.json or {}
        session_id = data.get("session")
        print(f"Session ID: {session_id}")
        
        if not session_id:
            print("错误: 没有提供session ID")
            return jsonify({"status": "error", "msg": "no session"}), 400

        in_dir = os.path.join(UPLOAD_FOLDER, session_id)
        out_dir = os.path.join(RESULT_FOLDER, session_id)
        print(f"输入目录: {in_dir}")
        print(f"输出目录: {out_dir}")

        # 确保输出目录存在
        os.makedirs(out_dir, exist_ok=True)
        print("输出目录创建完成")

        # 检查输入目录是否存在文件
        if not os.path.exists(in_dir):
            print(f"错误: 输入目录不存在: {in_dir}")
            return jsonify({"status": "error", "msg": "Input directory not found"}), 400
            
        files_in_dir = os.listdir(in_dir)
        if not files_in_dir:
            print(f"错误: 输入目录为空: {in_dir}")
            return jsonify({"status": "error", "msg": "No files to process"}), 400
            
        print(f"输入目录中有 {len(files_in_dir)} 个文件: {files_in_dir}")

        # 使用常驻模型处理图像
        global upsampler
        
        # 如果模型未初始化，先初始化
        model_load_start = time.time()
        if upsampler is None:
            with model_lock:
                if upsampler is None:
                    if not initialize_model():
                        return jsonify({"status": "error", "msg": "Failed to initialize model"}), 500
        model_load_time = time.time() - model_load_start
        print(f"模型加载/检查耗时: {model_load_time:.3f} 秒")
        
        # 处理所有图像
        success_count = 0
        total_count = len(files_in_dir)
        processing_start = time.time()
        
        for i, filename in enumerate(files_in_dir):
            if any(filename.lower().endswith(ext) for ext in app.config['UPLOAD_EXTENSIONS']):
                input_path = os.path.join(in_dir, filename)
                output_path = os.path.join(out_dir, filename)
                
                single_start = time.time()
                print(f"处理图像 {i+1}/{total_count}: {filename}")
                if process_image_with_model(input_path, output_path):
                    success_count += 1
                    single_time = time.time() - single_start
                    print(f"单张图片处理耗时: {single_time:.3f} 秒")
                else:
                    print(f"处理失败: {filename}")
        
        processing_time = time.time() - processing_start
        print(f"图像处理总耗时: {processing_time:.3f} 秒")
        print(f"平均每张图片耗时: {processing_time/total_count:.3f} 秒")
        
        if success_count == 0:
            return jsonify({"status": "error", "msg": "No images processed successfully"}), 500
        
        #图像处理耗时+2
        end_time = time.time()
        print(f"图像处理耗时: {end_time - start_time:.3f} 秒")
        return jsonify({"status": "done", "processed": success_count, "total": total_count})
    except Exception as e:
        #图像处理异常耗时+2
        end_time = time.time()
        print(f"图像处理异常，耗时: {end_time - start_time:.3f} 秒")
        import traceback
        error_msg = f"处理失败: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return jsonify({"status": "error", "msg": str(e)}), 500

#下载图像
@app.route("/download/<session_id>")
def download(session_id):
    import time
    start_time = time.time()
    try:
        print("开始处理下载请求...")
        print(f"Session ID: {session_id}")
        folder = os.path.join(RESULT_FOLDER, session_id)
        zipfile_path = os.path.join(RESULT_FOLDER, session_id + ".zip")
        print(f"结果目录: {folder}")
        print(f"压缩包路径: {zipfile_path}")

        # 检查结果目录是否存在
        if not os.path.exists(folder) or not os.listdir(folder):
            print(f"错误: 结果目录不存在或为空: {folder}")
            return jsonify({"status": "error", "msg": "Results folder not found or empty"}), 404

        # 删除已存在的旧压缩包（避免冲突）
        if os.path.exists(zipfile_path):
            print(f"删除旧压缩包: {zipfile_path}")
            os.remove(zipfile_path)

        # 使用Python的zipfile模块创建压缩包
        print(f"开始创建压缩包: {zipfile_path}")
        with zipfile.ZipFile(zipfile_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    # 计算相对路径，这样解压时不会包含完整路径
                    arcname = os.path.relpath(file_path, folder)
                    print(f"添加文件到压缩包: {arcname}")
                    zipf.write(file_path, arcname)

        print(f"压缩包创建完成: {zipfile_path}")
        
        # 检查压缩包是否生成
        if not os.path.exists(zipfile_path):
            print(f"错误: 压缩包未生成: {zipfile_path}")
            return jsonify({"status": "error", "msg": "Failed to create zip file"}), 500

        print("下载准备完成，开始发送文件")
        #下载耗时+2
        end_time = time.time()
        print(f"压缩下载耗时: {end_time - start_time:.3f} 秒")
        return send_file(zipfile_path, as_attachment=True)
    except Exception as e:
        import traceback
        error_msg = f"下载失败: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return jsonify({"status": "error", "msg": str(e)}), 500

if __name__ == "__main__":
    # 启动时初始化模型
    print("正在启动Real-ESRGAN Web服务...")
    if initialize_model():
        print("模型初始化成功，启动Web服务")
    else:
        print("警告: 模型初始化失败，将使用子进程模式")
    
    app.run(host="0.0.0.0", port=5000, threaded=True) 