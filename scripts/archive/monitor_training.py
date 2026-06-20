import time
import os

output_file = "training_output.log"

print("Monitoring training (updates every 30 seconds)...")
print("Press Ctrl+C to stop monitoring")

last_size = 0
start_time = time.time()

try:
    while True:
        if os.path.exists(output_file):
            current_size = os.path.getsize(output_file)
            elapsed = time.time() - start_time

            if current_size > last_size:
                print(f"[{elapsed/60:.1f}min] File size: {current_size} bytes (growing...)")
                last_size = current_size

                # Show last 20 lines
                try:
                    with open(output_file, 'r') as f:
                        lines = f.readlines()
                        for line in lines[-5:]:
                            print("  " + line.rstrip())
                except:
                    pass
            else:
                print(f"[{elapsed/60:.1f}min] File size unchanged ({current_size} bytes)")

        if elapsed > 900:  # 15 minutes
            print("\nTraining should be complete!")
            break

        time.sleep(30)

except KeyboardInterrupt:
    print("\nMonitoring stopped by user")
