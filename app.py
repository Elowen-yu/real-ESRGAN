# backend/app.py
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import os, shutil, uuid, subprocess
from werkzeug.utils import secure_filename

app = Flask(__name__)
# 允许所有来源的跨域请求
# 修复 CORS 配置，允许所有来源
#CORS(app, resources={r"/*": {"origins": "*"}})
CORS(app, origins=["*"])
#CORS(app, origins=["*"], allow_headers=["*"], methods=["GET", "POST", "OPTIONS"])

BASE = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE, "uploads")
RESULT_FOLDER = os.path.join(BASE, "results")
print(f"BASE目录: {BASE}")
print(f"上传目录: {UPLOAD_FOLDER}")
print(f"结果目录: {RESULT_FOLDER}")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
print("目录创建完成")

# 配置上传限制
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 100MB限制
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

@app.route("/")
def index():
    return jsonify({"message": "Real-ESRGAN Web API is running", "status": "ok"})

#上传文件
@app.route("/upload", methods=["POST"])
def upload():
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
        return jsonify({"status": "ok", "session": session_id, "files": uploaded_files})
    except Exception as e:
        import traceback
        error_msg = f"上传失败: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return jsonify({"status": "error", "msg": str(e)}), 500

#处理图像
@app.route("/process", methods=["POST"])
def process():
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

        cmd = [
            "python", "inference_realesrgan.py",
            "-n", "RealESRGAN_x2plus",
            "-i", in_dir,
            "-o", out_dir,
            # "--face_enhance",
            "--tile", "600"
        ]
        print(f"执行命令: {' '.join(cmd)}")
        print(f"工作目录: /workspace/Real-ESRGAN")
        
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd="/workspace/Real-ESRGAN")
        
        print(f"命令执行完成，返回码: {proc.returncode}")
        if proc.stdout:
            print(f"标准输出: {proc.stdout.decode()}")
        if proc.stderr:
            print(f"错误输出: {proc.stderr.decode()}")

        #命令示例：cd /workspace/Real-ESRGAN && \
        #python inference_realesrgan.py \
        #-n RealESRGAN_x2plus \
        #-i /app/uploads/abc123 \
        #-o /app/results/abc123 \
        #--tile 600
        
        if proc.returncode != 0:
            error_msg = proc.stderr.decode() if proc.stderr else "Unknown error"
            print(f"处理失败: {error_msg}")
            return jsonify({"status": "error", "msg": error_msg}), 500

        print("图像处理完成")
        return jsonify({"status": "done"})
    except Exception as e:
        import traceback
        error_msg = f"处理失败: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return jsonify({"status": "error", "msg": str(e)}), 500

#下载图像
@app.route("/download/<session_id>")
def download(session_id):
    try:
        folder = os.path.join(RESULT_FOLDER, session_id)
        zipfile = os.path.join(RESULT_FOLDER, session_id + ".zip")

        # 检查结果目录是否存在
        if not os.path.exists(folder) or not os.listdir(folder):
            return jsonify({"status": "error", "msg": "Results folder not found or empty"}), 404

        # 删除已存在的旧压缩包（避免冲突）
        if os.path.exists(zipfile):
            os.remove(zipfile)

        # 调用系统 zip 命令强制压缩（-r 递归，-q 静默模式，-9 最高压缩率）
        subprocess.run(
            ["zip", "-r", "-q", "-9", zipfile, "."],
            cwd=folder,  # 在目标目录内执行
            check=True,  # 检查命令是否成功
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        #命令示例：cd /app/results/abc123 && zip -r -q -9 /app/results/abc123.zip .

        # 检查压缩包是否生成
        if not os.path.exists(zipfile):
            return jsonify({"status": "error", "msg": "Failed to create zip file"}), 500

        return send_file(zipfile, as_attachment=True)
    except subprocess.CalledProcessError as e:
        return jsonify({"status": "error", "msg": f"Compression failed: {e.stderr.decode()}"}), 500
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)