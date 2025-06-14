import socket
import wave
import time
import struct

DEST_IP = "64.16.229.3"
DEST_PORT = 17990
SOURCE_PORT = 52402

# Read audio from wav file
wav = wave.open("welcome.wav", "rb")
assert wav.getframerate() == 8000, "Sample rate must be 8000 Hz"
assert wav.getnchannels() == 1, "Must be mono"

# μ-law encoder
def linear2ulaw(sample):
    BIAS = 0x84
    CLIP = 32635
    sign = (sample >> 8) & 0x80
    if sign != 0:
        sample = -sample
    if sample > CLIP:
        sample = CLIP
    sample += BIAS
    exponent = 7
    mask = 0x4000
    while (sample & mask) == 0 and exponent > 0:
        exponent -= 1
        mask >>= 1
    mantissa = (sample >> ((exponent + 3))) & 0x0F
    ulaw = ~(sign | (exponent << 4) | mantissa)
    return ulaw & 0xFF

# Create RTP header
def rtp_header(seq, timestamp, ssrc=1234):
    version = 2
    payload_type = 0  # PCMU
    header = struct.pack("!BBHII",
        (version << 6), payload_type,
        seq, timestamp, ssrc
    )
    return header

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("", SOURCE_PORT))

seq = 0
timestamp = 0
frame_size = 160  # 20ms of audio at 8000Hz
sleep_time = 0.02

print(f"Streaming from source port {SOURCE_PORT} to {DEST_IP}:{DEST_PORT}...")

while True:
    raw = wav.readframes(frame_size)
    if not raw:
        break
    samples = struct.unpack("<" + "h" * (len(raw)//2), raw)
    encoded = bytes([linear2ulaw(s) for s in samples])
    packet = rtp_header(seq, timestamp) + encoded
    sock.sendto(packet, (DEST_IP, DEST_PORT))
    seq = (seq + 1) % 65536
    timestamp += frame_size
    time.sleep(sleep_time)

print("Done.")

