import os
import shutil
import stat
import sys
import time

SRC = r"saved_models/sentiment/models--ProsusAI--finbert/snapshots/4556d13015211d73dccd3fdd39d39232506f3e43/tokenizer_config.json"
DST = r"C:/Users/piotr/Desktop/tokenizer_config_copy.json"


def try_unlock(path):
    try:
        # Remove read-only, system, hidden attributes if set
        os.chmod(path, stat.S_IWRITE)
    except Exception as e:
        print(f"[WARN] Could not change file attributes: {e}")


def try_copy(src, dst):
    try:
        shutil.copy2(src, dst)
        print(f"[OK] File copied to: {dst}")
        return True
    except PermissionError as e:
        print(f"[ERROR] PermissionError: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def main():
    print(f"[INFO] Attempting to unlock and copy: {SRC}")
    if not os.path.exists(SRC):
        print(f"[ERROR] Source file does not exist: {SRC}")
        sys.exit(1)

    try_unlock(SRC)
    time.sleep(0.5)
    success = try_copy(SRC, DST)
    if not success:
        print("[INFO] Retrying after 1s...")
        time.sleep(1)
        try_unlock(SRC)
        success = try_copy(SRC, DST)
        if not success:
            print(
                "[FATAL] Could not copy or unlock the file. Try closing all programs that may use the file and run as administrator."
            )
            sys.exit(2)
    print(
        "[DONE] If the copy is readable, update your model loading path to use the new file."
    )


if __name__ == "__main__":
    main()
