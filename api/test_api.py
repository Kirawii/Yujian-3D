"""
测试 SOKE API
"""
import requests
import time
import sys

BASE_URL = "http://127.0.0.1:6006"  # 本地端口
# 公网访问: https://u895901-9072-0273df24.westc.seetacloud.com:8443


def test_health():
    """测试服务状态"""
    print("[TEST] Checking API health...")
    try:
        response = requests.get(f"{BASE_URL}/")
        data = response.json()
        print(f"  Name: {data['name']}")
        print(f"  Version: {data['version']}")
        print(f"  Model loaded: {data['model_loaded']}")
        return data['model_loaded']
    except Exception as e:
        print(f"  Error: {e}")
        return False


def test_generate(text="你好，世界"):
    """测试生成任务"""
    print(f"\n[TEST] Submitting generation task: '{text}'")
    try:
        response = requests.post(
            f"{BASE_URL}/generate",
            json={"text": text, "num_samples": 1}
        )
        data = response.json()
        print(f"  Task ID: {data['task_id']}")
        print(f"  Status: {data['status']}")
        return data['task_id']
    except Exception as e:
        print(f"  Error: {e}")
        return None


def test_status(task_id):
    """测试状态查询"""
    print(f"\n[TEST] Checking status for task: {task_id}")
    try:
        response = requests.get(f"{BASE_URL}/status/{task_id}")
        data = response.json()
        print(f"  Status: {data['status']}")
        print(f"  Text: {data['text']}")
        if data['output_video']:
            print(f"  Output: {data['output_video']}")
        return data['status']
    except Exception as e:
        print(f"  Error: {e}")
        return None


def test_list_tasks():
    """测试任务列表"""
    print("\n[TEST] Listing tasks...")
    try:
        response = requests.get(f"{BASE_URL}/tasks?limit=5")
        data = response.json()
        print(f"  Tasks count: {len(data['tasks'])}")
        for task in data['tasks'][:3]:
            print(f"    - {task['task_id']}: {task['status']} - '{task['text'][:20]}...'")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


def wait_for_completion(task_id, timeout=300):
    """等待任务完成"""
    print(f"\n[WAIT] Waiting for task {task_id} to complete (timeout: {timeout}s)...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        status = test_status(task_id)
        if status == "completed":
            print("  Task completed!")
            return True
        elif status == "failed":
            print("  Task failed!")
            return False

        print("  Waiting 5 seconds...")
        time.sleep(5)

    print("  Timeout!")
    return False


def download_video(task_id, output_path="test_output.mp4"):
    """下载视频"""
    print(f"\n[TEST] Downloading video for task: {task_id}")
    try:
        response = requests.get(f"{BASE_URL}/video/{task_id}?sample=0")
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(response.content)
            print(f"  Video saved to: {output_path}")
            return True
        else:
            print(f"  Error: {response.status_code}")
            return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def main():
    print("=" * 60)
    print("SOKE Sign Language API Test")
    print("=" * 60)

    # Test health
    if not test_health():
        print("\n[ERROR] API is not ready or model not loaded!")
        sys.exit(1)

    # Test list tasks
    test_list_tasks()

    # Test generate
    test_text = "很高兴认识你" if len(sys.argv) < 2 else sys.argv[1]
    task_id = test_generate(test_text)

    if not task_id:
        print("\n[ERROR] Failed to submit task!")
        sys.exit(1)

    # Wait for completion
    if wait_for_completion(task_id):
        download_video(task_id, f"output_{task_id}.mp4")
    else:
        print("\n[ERROR] Task did not complete successfully!")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
