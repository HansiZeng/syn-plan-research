import argparse
import json
import os
import tempfile
import subprocess
import uuid


def is_jsonl(path):
    return path.endswith('.jsonl')

def is_json(path):
    return path.endswith('.json')

def download_from_s3(s3_uri, local_path):
    result = subprocess.run(["aws", "s3", "cp", s3_uri, local_path], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"⚠️ Warning: Failed to download from S3: {s3_uri}. Proceeding with empty cache.")
        return None
    return local_path


def upload_to_s3(local_path, s3_uri):
    subprocess.check_call(["aws", "s3", "cp", local_path, s3_uri])
    print(f"✅ Uploaded merged cache to {s3_uri}")


def load_cache(path):
    cache = {}
    if not os.path.exists(path):
        return cache

    try:
        with open(path, "r", encoding="utf-8") as f:
            if is_jsonl(path):
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        key, val = obj.get("key"), obj.get("value")
                        if key is not None:
                            cache[key] = val
                    except json.JSONDecodeError:
                        print(f"⚠️ Skipping invalid line in JSONL file: {line}")
            elif is_json(path):
                cache = json.load(f)
            else:
                print(f"⚠️ Unsupported format for file: {path}")
    except Exception as e:
        print(f"⚠️ Failed to load cache from {path}: {e}")
    return cache


def save_cache(data, path):
    tmp_path = f"{path}.{os.getpid()}.{uuid.uuid4().hex}.tmp.jsonl" if is_jsonl(path) else f"{path}.{os.getpid()}.{uuid.uuid4().hex}.tmp.json"

    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            if is_jsonl(path):
                for key, value in data.items():
                    json.dump({"key": key, "value": value}, f, ensure_ascii=False)
                    f.write("\n")
            elif is_json(path):
                json.dump(data, f, ensure_ascii=False, indent=2)
            else:
                raise ValueError(f"Unsupported format: {path}")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)  # atomic move
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"❌ Failed to save cache to {path}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_file", type=str, required=True, help="Path to local cache file (e.g., /workspace/cache/serper_search_cache.json[l])")
    parser.add_argument("--s3_uri", type=str, required=True, help="Target S3 URI (e.g., s3://shopqa-users/WebThinker/cache/serper_search_cache.json[l])")
    args = parser.parse_args()

    # Download S3 file to temporary location
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        s3_local_path = tmp.name

    s3_cache = {}
    if download_from_s3(args.s3_uri, s3_local_path):
        s3_cache = load_cache(s3_local_path)

    local_cache = load_cache(args.cache_file)

    # Merge: local has priority over s3
    merged_cache = {**s3_cache, **local_cache}

    save_cache(merged_cache, args.cache_file)
    print(f"✅ Local cache updated at {args.cache_file}")

if __name__ == "__main__":
    main()