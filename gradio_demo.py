import cv2
import tempfile
import os
from ultralytics import YOLO
import gradio as gr

# 加载预训练的YOLOv8模型
model = YOLO("yolov8n.pt")

def detect_video(video_path):
    # 创建临时文件用于保存处理后的视频
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', dir=os.getcwd())
    output_path = temp_file.name
    print(f'Temporary file created at: {output_path}')
    temp_file.close()  # 关闭文件

    # 打开输入视频文件
    cap = cv2.VideoCapture(video_path)

    # 获取视频帧宽度、高度和帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 定义视频编解码器和输出视频文件
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 将帧转换为RGB格式并进行检测
        results = model(frame)

        # 对每个检测结果进行渲染并转换为NumPy数组
        for result in results:
            for bbox in result.boxes:
                # 提取检测框坐标和标签信息
                x1, y1, x2, y2 = map(int, bbox.xyxy[0])
                label = result.names[int(bbox.cls)]
                confidence = float(bbox.conf)  # 转换为Python浮点数

                # 绘制检测框和标签
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 将处理后的帧写入输出视频文件
            out.write(frame)

    cap.release()
    out.release()

    return video_path, output_path

with gr.Blocks() as demo:
    gr.Markdown("# Video Detection with YOLOv8")
    with gr.Row():
        with gr.Column():
            input_video = gr.Video(label="Upload your video here")
        with gr.Column():
            original_video = gr.Video(label="Original Video")
            detected_video = gr.Video(label="Detected Video")
    detect_button = gr.Button("Detect")
    detect_button.click(detect_video, inputs=input_video, outputs=[original_video, detected_video])

# 启动Gradio应用
if __name__ == "__main__":
    demo.launch()
