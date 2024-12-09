import os
import argparse
import cv2
import mediapipe as mp

def process_img(img, face_detection):
    H, W, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)
    if results.detections is not None:
        for detection in results.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box
            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height
            x1, y1, w, h = int(x1 * W), int(y1 * H), int(w * W), int(h * H)
            img[y1:y1 + h, x1:x1 + w] = cv2.blur(img[y1:y1 + h, x1:x1 + w], (30, 30))
    return img

parser = argparse.ArgumentParser(description="Face blurring using Mediapipe.")
parser.add_argument("--mode", choices=["image", "video", "webcam"], required=True, help="Mode of operation")
parser.add_argument("--filePath", default=None, help="File path for image or video (required for image/video mode)")
parser.add_argument("--cameraIndex", type=int, default=0, help="Camera index for webcam mode (default: 0)")
args = parser.parse_args()

output_dir = './output'
os.makedirs(output_dir, exist_ok=True)

mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    if args.mode == "image":
        if not args.filePath:
            print("Error: Please provide a valid image file path.")
            exit()
        img = cv2.imread(args.filePath)
        if img is None:
            print("Error: Unable to read the image file.")
            exit()
        img = process_img(img, face_detection)
        output_path = os.path.join(output_dir, 'output.png')
        cv2.imwrite(output_path, img)
        print(f"Processed image saved to {output_path}")

    elif args.mode == "video":
        if not args.filePath:
            print("Error: Please provide a valid video file path.")
            exit()
        cap = cv2.VideoCapture(args.filePath)
        if not cap.isOpened():
            print("Error: Unable to open the video file.")
            exit()
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        output_video_path = os.path.join(output_dir, 'output.mp4')
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = process_img(frame, face_detection)
            out.write(frame)
        cap.release()
        out.release()
        print(f"Processed video saved to {output_video_path}")

    elif args.mode == "webcam":
        cap = cv2.VideoCapture(args.cameraIndex)
        if not cap.isOpened():
            print(f"Error: Unable to access the camera at index {args.cameraIndex}.")
            exit()
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Warning: Empty frame received. Skipping...")
                continue
            frame = process_img(frame, face_detection)
            cv2.imshow("Webcam - Press 'q' to exit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

