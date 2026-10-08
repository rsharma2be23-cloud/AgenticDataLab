# src/tools/job_tools.py
import time

class JobTools:

    def long_job(self, seconds=5):
        try:
            time.sleep(seconds)
            return {"status": "success", "message": f"Completed {seconds}s job"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
