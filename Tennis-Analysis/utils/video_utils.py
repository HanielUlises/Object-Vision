import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()

        if ret:
            frames.append(frame)
        else: 
            break
    cap.release()
    return frames

def save_video(output_vid_frames, output_vid_path):
    fcc = cv2.VideoWriter_fourcc(* 'MJPG')
    out = cv2.VideoWriter(output_vid_path, fcc, 24, (output_vid_frames[0].shape[1], output_vid_frames[0].shape[1]))

    for frame in output_vid_frames:
        out.write(frame)

    out.release()