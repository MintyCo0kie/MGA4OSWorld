Ubuntu VM Flask Server 使用教程
下面是一份面向 使用者（宿主机 / Agent 调用者） 的 Ubuntu VM Flask Server 使用教程。
默认你的：
server.py
pyxcursor.py
在 VM 同一目录下，并且 VM 里运行这个 server，宿主机通过端口转发访问它。
1. 这是什么？能做什么？
这是一个运行在 Ubuntu 虚拟机内部 的 HTTP 服务。
你在 宿主机（或 agent） 通过 HTTP 调用它，就能在 虚拟机里执行操作。
目前支持：
✅ 健康检查：确认服务存活、DISPLAY、会话类型
 ✅ 运行一次命令 (subprocess.run)：适合短命令
 ✅ 启动长命令 (subprocess.Popen)：后台任务，支持 poll / wait / terminate
 ✅ 执行 Python 代码 (run_python)
 ✅ 上传文件到 VM (upload_file)
 ✅ 下载 VM 任意可读文件 (download_file)
 ✅ 截屏 (screenshot)
2. 目录结构建议
在 VM 里建立一个目录，例如：
/home/opengui/Server/

  server.py
  pyxcursor.py

  screenshots/        # 自动创建
  uploads/            # 上传文件会保存到这里
说明：
screenshots/ 自动创建
uploads/ 自动创建
download 不再限制目录，可以读取任意文件
3. VM 端安装依赖
在 Ubuntu VM 中执行：
pip install flask pyautogui pillow numpy
然后安装截图依赖：
sudo apt-get update
sudo apt-get install -y scrot python3-tk python3-dev x11-utils
说明：
Linux 上 pyautogui.screenshot() 通常依赖 scrot。
4. 启动 server（VM 内）
4.1 最简单启动
进入 server 目录：
cd /home/opengui/Server
python3 server.py
默认监听：
0.0.0.0:5000
4.2 确保 GUI / 截图可用（推荐）
如果你使用 GUI / VNC / X11：
export DISPLAY=:0
python3 server.py
检查：
echo $DISPLAY
echo $XDG_SESSION_TYPE
5. 宿主机如何访问
假设你的 QEMU 端口转发：
-netdev user,id=net0,hostfwd=tcp::5000-:5000
那么：
宿主机 127.0.0.1:5000
↓
VM 5000
所以所有请求都用：
http://127.0.0.1:5000
6. API 使用说明
以下所有示例都在 宿主机执行。
6.1 health_check
检查 server 是否正常运行。
请求
curl http://127.0.0.1:5000/health_check
返回示例：
{
  "ok": true,
  "platform": "Linux",
  "time_ms": 1739160000000,
  "display": ":0",
  "session_type": "x11"
}
重点关注：
display
session_type
6.2 subprocess.run（短命令）
适合：
ls
cat
ps
curl
bash脚本
示例：
curl -X POST http://127.0.0.1:5000/subprocess/run \
  -H "Content-Type: application/json" \
  -d '{
    "cmd": ["bash","-lc","echo hello | wc -c"],
    "shell": false
  }'
返回：
{
 "ok": true,
 "returncode": 0,
 "stdout": "6\n",
 "stderr": ""
}
推荐方式：
["bash","-lc","..."]
而不是：
shell=true
6.3 subprocess.Popen（长任务）
适合：
长脚本
sleep
后台程序
QEMU
服务进程
start
curl -X POST http://127.0.0.1:5000/subprocess/popen/start \
  -H "Content-Type: application/json" \
  -d '{
    "cmd": ["bash","-lc","sleep 100"]
  }'
返回：
job_id
pid
poll
curl http://127.0.0.1:5000/subprocess/popen/<job_id>/poll
wait
curl -X POST http://127.0.0.1:5000/subprocess/popen/<job_id>/wait
terminate
温和结束：
curl -X POST http://127.0.0.1:5000/subprocess/popen/<job_id>/terminate \
  -H "Content-Type: application/json" \
  -d '{"sig":"TERM"}'
强制杀死：
{"sig":"KILL"}
6.4 run_python
执行 Python 代码：
curl -X POST http://127.0.0.1:5000/run_python \
  -H "Content-Type: application/json" \
  -d '{
    "code": "print(1+1)"
  }'
返回：
stdout
stderr
returncode
6.5 upload_file（宿主机 → VM）
上传文件。
上传文件会保存到：
uploads/
示例：
curl -X POST http://127.0.0.1:5000/setup/upload \
  -F "file_path=/tmp/demo.txt" \
  -F "file_data=@./local.txt"
实际路径：
uploads/tmp/demo.txt
6.6 download_file（VM → 宿主机）
新增功能：
允许下载 VM 上任意可读文件。
只要 Flask server 进程对该文件有读取权限，就可以下载。
请求
GET /setup/download?file_path=...
示例：
curl "http://127.0.0.1:5000/setup/download?file_path=/tmp/test.txt" -o ./test.txt
支持路径
支持：
绝对路径
相对路径
~
环境变量
例如：
/tmp/test.txt
/home/opengui/file.json
./outputs/log.txt
~/demo.txt
$HOME/test.txt
保存到宿主机指定目录
宿主机决定保存路径：
curl "http://127.0.0.1:5000/setup/download?file_path=/tmp/test.txt" \
-o /home/user/Desktop/test.txt
返回结果
成功：
返回文件本体（不是 JSON）。
失败示例：
404 File not found
403 File not readable
400 Not regular file
6.7 screenshot
获取 VM 屏幕截图。
curl -o shot.png http://127.0.0.1:5000/screenshot
server 会尝试：
截图
叠加鼠标光标
如果光标抓取失败，会 fallback。
7. 常见问题排查
screenshot 报错
先检查：
curl http://127.0.0.1:5000/health_check
如果：
display = null
说明 server 没有 DISPLAY。
解决：
export DISPLAY=:0
subprocess 管道不工作
推荐写法：
["bash","-lc","command | command"]
不要：
shell=false + "ls | grep"
popen 任务丢失
server 重启后：
JOBS 会丢失
解决：
ps
pgrep
手动查进程。
8. 安全注意事项
⚠️ 这个 server 允许：
执行命令
执行 Python
上传文件
下载文件
尤其现在 download 可以读取 VM 任意可读文件。
因此：
不要：
暴露公网
给不可信用户访问
在生产机器上使用
建议只在：
本地 VM
实验环境
agent research
使用。
9. 快速自检流程
建议每次部署跑一遍：
health
curl http://127.0.0.1:5000/health_check
command
curl -X POST http://127.0.0.1:5000/subprocess/run \
  -H "Content-Type: application/json" \
  -d '{"cmd":["bash","-lc","echo ok"]}'
screenshot
curl -o shot.png http://127.0.0.1:5000/screenshot
upload
echo "hello" > local.txt

curl -X POST http://127.0.0.1:5000/setup/upload \
  -F "file_path=/tmp/demo.txt" \
  -F "file_data=@./local.txt"
download
VM 内：
echo hello > /tmp/test.txt
宿主机：
curl "http://127.0.0.1:5000/setup/download?file_path=/tmp/test.txt" -o ./test.txt