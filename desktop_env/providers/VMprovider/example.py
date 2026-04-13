"""VMprovider 使用示例."""

import logging
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from VMprovider import VM, VMConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# 默认配置
DEFAULT_DISK_PATH = PROJECT_ROOT / "workspace/vm/base/base.qcow2"
DEFAULT_LOGS_DIR = PROJECT_ROOT / "workspace/logs"


def get_default_config(**kwargs) -> VMConfig:
    """获取默认配置，可覆盖指定参数."""
    defaults = {
        "disk_path": str(DEFAULT_DISK_PATH),
        "logs_dir": str(DEFAULT_LOGS_DIR),
        "mount_tmp": True,
        "snapshot": True,
    }
    defaults.update(kwargs)
    return VMConfig(**defaults)


def basic_usage():
    """基本使用：启动 VM，执行命令，截图，停止."""
    print("=" * 50)
    print("示例 1: 基本使用")
    print("=" * 50)

    config = get_default_config()

    # 使用 context manager 自动管理生命周期
    with VM(config) as vm:
        # 检查健康状态
        health = vm.health_check()
        print(f"Health check: {health}")

        # 执行 bash 命令
        result = vm.run_bash("echo 'Hello from VM!' && whoami")
        print(f"Bash result: {result}")

        # 执行 Python 代码
        result = vm.run_python("import sys; print(f'Python {sys.version}')")
        print(f"Python result: {result}")

        # 截图
        screenshot_path = PROJECT_ROOT / "workspace/screenshot_example.png"
        vm.screenshot(str(screenshot_path))
        print(f"Screenshot saved to: {screenshot_path}")


def custom_config():
    """自定义配置：调整内存、CPU、暴露 VNC 端口."""
    print("=" * 50)
    print("示例 2: 自定义配置")
    print("=" * 50)

    config = get_default_config(
        memory=4096,  # 4GB 内存
        cpus=2,  # 2 核 CPU
        vnc_port=5901,  # 暴露 VNC 端口用于调试查看
        container_name="my-opengui-vm",  # 指定容器名称
    )

    vm = VM(config)

    try:
        vm.start(timeout=180)  # 最多等待 3 分钟启动
        print(f"VM started, container ID: {vm.container_id[:12]}")
        print(f"VNC available at: localhost:{config.vnc_port}")

        # 执行一些操作...
        health = vm.health_check()
        print(f"Health: {health}")

    finally:
        vm.stop()
        print("VM stopped")


def file_operations():
    """文件操作：上传文件到 VM."""
    print("=" * 50)
    print("示例 3: 文件操作")
    print("=" * 50)

    config = get_default_config()

    with VM(config) as vm:
        # 创建一个临时文件
        local_file = PROJECT_ROOT / "workspace/test_upload.txt"
        local_file.parent.mkdir(parents=True, exist_ok=True)
        local_file.write_text("This is a test file for upload.")

        # 上传到 VM
        result = vm.upload_file(str(local_file), "test_upload.txt")
        print(f"Upload result: {result}")

        # 验证文件存在
        result = vm.run_bash("cat ~/uploads/test_upload.txt")
        print(f"File content in VM: {result.get('stdout', '')}")

        # 清理本地文件
        local_file.unlink()


def long_running_process():
    """长时间运行的进程：使用 popen 接口."""
    print("=" * 50)
    print("示例 4: 长时间运行的进程")
    print("=" * 50)

    config = get_default_config()

    with VM(config) as vm:
        # 启动一个长时间运行的进程
        result = vm.client.popen_start(["sleep", "5"])
        job_id = result.get("job_id")
        print(f"Started job: {job_id}")

        # 轮询检查状态
        import time
        for i in range(10):
            poll_result = vm.client.popen_poll(job_id)
            print(f"Poll {i+1}: running={poll_result.get('running')}")
            if not poll_result.get("running"):
                break
            time.sleep(1)

        # 等待并获取结果
        wait_result = vm.client.popen_wait(job_id)
        print(f"Final result: returncode={wait_result.get('returncode')}")


def multiple_vms():
    """多个 VM：同时运行多个虚拟机（需要多个磁盘镜像）."""
    print("=" * 50)
    print("示例 5: 多个 VM")
    print("=" * 50)

    # 注意：同时运行多个 VM 需要：
    # 1. 不同的磁盘镜像（或使用 snapshot 模式）
    # 2. 不同的 VNC 端口
    # 3. 足够的系统资源

    config1 = get_default_config(container_name="vm-1", vnc_port=5901)
    config2 = get_default_config(container_name="vm-2", vnc_port=5902)

    vm1 = VM(config1)
    vm2 = VM(config2)

    try:
        print("Starting VM 1...")
        vm1.start()
        print(f"VM 1 IP: {vm1.client.host}")

        print("Starting VM 2...")
        vm2.start()
        print(f"VM 2 IP: {vm2.client.host}")

        # 在两个 VM 上执行不同的命令
        result1 = vm1.run_bash("hostname")
        result2 = vm2.run_bash("hostname")
        print(f"VM 1 hostname: {result1.get('stdout', '').strip()}")
        print(f"VM 2 hostname: {result2.get('stdout', '').strip()}")

    finally:
        vm1.stop()
        vm2.stop()
        print("Both VMs stopped")


def error_handling():
    """错误处理示例."""
    print("=" * 50)
    print("示例 6: 错误处理")
    print("=" * 50)

    # 尝试不设置 disk_path
    try:
        vm = VM(VMConfig())
        vm.start()
    except RuntimeError as e:
        print(f"Expected error (no disk_path): {e}")

    # 尝试在未启动时访问 client
    config = get_default_config()
    vm = VM(config)
    try:
        vm.client.health_check()
    except RuntimeError as e:
        print(f"Expected error (not started): {e}")

    # 启动并处理命令错误
    with VM(config) as vm:
        # 执行一个会失败的命令
        result = vm.run_bash("exit 1")
        print(f"Failed command: returncode={result.get('returncode')}")

        # 执行不存在的命令
        result = vm.run_bash("nonexistent_command_xyz")
        print(f"Unknown command: returncode={result.get('returncode')}")
        print(f"stderr: {result.get('stderr', '')[:100]}")


if __name__ == "__main__":
    examples = {
        "1": ("基本使用", basic_usage),
        "2": ("自定义配置", custom_config),
        "3": ("文件操作", file_operations),
        "4": ("长时间运行的进程", long_running_process),
        "5": ("多个 VM", multiple_vms),
        "6": ("错误处理", error_handling),
    }

    if len(sys.argv) > 1:
        choice = sys.argv[1]
        if choice in examples:
            name, func = examples[choice]
            print(f"\n运行示例: {name}\n")
            func()
        elif choice == "all":
            for key, (name, func) in examples.items():
                print(f"\n运行示例 {key}: {name}\n")
                try:
                    func()
                except Exception as e:
                    print(f"Error: {e}")
                print()
        else:
            print(f"未知示例: {choice}")
            print("可用示例: 1-6 或 'all'")
    else:
        print("VMprovider 使用示例")
        print("-" * 30)
        print(f"磁盘镜像: {DEFAULT_DISK_PATH}")
        print(f"日志目录: {DEFAULT_LOGS_DIR}")
        print()
        for key, (name, _) in examples.items():
            print(f"  {key}. {name}")
        print()
        print("用法: python example.py <示例编号>")
        print("      python example.py all  # 运行所有示例")
