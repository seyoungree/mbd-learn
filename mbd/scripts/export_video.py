import os
import pickle
import ffmpeg
import matplotlib.pyplot as plt
import numpy as np

from brax.base import Transform

path = "../../results/hopper"
video_path = f"{path}/hopper_rollout.mp4"

# If the video file already exists, delete it
if os.path.exists(video_path):
    os.remove(video_path)

with open(f"{path}/rollouts.pkl", "rb") as f:
    rollouts = pickle.load(f)

states = rollouts[0]
positions = [float(state.x.pos[0, 0]) for state in states]

frame_rate = 20
width, height = 600, 400

output = ffmpeg.input('pipe:0', framerate=frame_rate, format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}')
ffmpeg_output = ffmpeg.output(output, video_path, vcodec='libx264', pix_fmt='yuv420p')

process = ffmpeg_output.run_async(pipe_stdin=True)

fig, ax = plt.subplots(figsize=(6, 4))
fig.set_dpi(100)

for i, x in enumerate(positions):
    ax.clear()

    ax.set_xlim(min(positions)-1, max(positions)+1)
    ax.set_ylim(-1, 2)
    ax.set_title(f"Hopper Step {i}")

    ax.plot(x, 0, 'ro', markersize=10)

    fig.canvas.draw()

    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape((height, width, 3))  # Ensure it matches the expected dimensions

    process.stdin.write(image.tobytes())

process.stdin.close()
process.wait()

print(f"Video saved to {video_path}")
