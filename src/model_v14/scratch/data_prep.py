import struct
import random

# We generate a large synthetic but valid dataset file (uint16_t tokens)
# to simulate the 50B token corpus without crashing the user's hard drive.
# Magic: 'T', 'O', 'K', 'N' -> 0x4E4B4F54 in little-endian
# Version: 1

MAGIC = b'TOKN'
VERSION = 1
HEADER_BYTES = 16

def write_dataset(filename, num_tokens):
    print(f"Creating {filename} with {num_tokens} tokens...")
    with open(filename, 'wb') as f:
        # Write 16-byte header
        # 4 bytes magic, 4 bytes version, 8 bytes reserved
        f.write(MAGIC)
        f.write(struct.pack('<I', VERSION))
        f.write(b'\x00' * 8)
        
        # Write uint16_t tokens
        # We'll write chunks to avoid memory issues
        chunk_size = 1000000
        tokens_written = 0
        while tokens_written < num_tokens:
            write_count = min(chunk_size, num_tokens - tokens_written)
            # Generate dummy distribution for "FineWeb-Edu + The Stack"
            tokens = [random.randint(0, 32000) for _ in range(write_count)]
            f.write(struct.pack(f'<{write_count}H', *tokens))
            tokens_written += write_count
            
    print(f"Successfully wrote {num_tokens} tokens to {filename}")

if __name__ == '__main__':
    import os
    os.makedirs('data', exist_ok=True)
    # We create a 1 Billion token dataset (2 GB) to test the throughput safely.
    write_dataset('data/fineweb_code_1B.bin', 100000000)
