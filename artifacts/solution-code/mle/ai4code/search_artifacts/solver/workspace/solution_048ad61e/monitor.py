import time
import os

def monitor():
    while True:
        with open("/workspace/solution_048ad61e/output.log", "r") as f:
            log = f.read()
            if "Saved submission.csv" in log:
                print("JOB FINISHED SUCCESSFULLY!")
                break
            if "Traceback (most recent call last):" in log:
                print("JOB FAILED WITH ERROR!")
                break
        time.sleep(60)

if __name__ == "__main__":
    monitor()
