import os

class FileTools:

    def __init__(self, base_dir="project_storage"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    # -----------------------------
    # Write TEXT file
    # -----------------------------
    def write(self, filepath, content):
        try:
            full_path = os.path.join(self.base_dir, filepath)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)

            return {"status": "success", "path": full_path}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # -----------------------------
    # Write BINARY file (FIX)
    # -----------------------------
    def write_binary(self, filepath, content_bytes):
        try:
            full_path = os.path.join(self.base_dir, filepath)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            with open(full_path, "wb") as f:
                f.write(content_bytes)

            return {"status": "success", "path": full_path}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # -----------------------------
    # Read file
    # -----------------------------
    def read(self, filepath):
        try:
            full_path = os.path.join(self.base_dir, filepath)
            with open(full_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # -----------------------------
    # Read BINARY
    # -----------------------------
    def read_binary(self, filepath):
        try:
            full_path = os.path.join(self.base_dir, filepath)
            with open(full_path, "rb") as f:
                return f.read()
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # -----------------------------
    # Delete file
    # -----------------------------
    def delete(self, filepath):
        try:
            full_path = os.path.join(self.base_dir, filepath)
            os.remove(full_path)
            return {"status": "success"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # -----------------------------
    # List directory
    # -----------------------------
    def list(self, directory=""):
        try:
            full_dir = os.path.join(self.base_dir, directory)
            return os.listdir(full_dir)
        except Exception as e:
            return {"status": "error", "error": str(e)}
