from gradio import Blocks, Video, Dataframe, Button, State
import gradio as gr
import easydict
import os
import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import v2
from util.io import load_json
from torch.utils.data import Dataset, DataLoader
from moviepy.editor import VideoFileClip, ImageSequenceClip
from util.score import (
    filter_events_by_score,
    non_max_suppression_events,
)

from inference_on_mp4 import (
    get_args,
    get_model,
    VideoFrameSlidingDataset,
    run_inference,
)
import json
from openai import OpenAI

raw_results = None  # add global variable to store predictions


def reply_message(message, history):
    global raw_results  # use global raw_results
    system_message = "Volleyball Events: "
    if raw_results:
        system_message += json.dumps(raw_results)
    else:
        system_message += "No prediction results available."
    history.extend(
        [
            {"role": "user", "content": message},
            {"role": "system", "content": system_message},
        ]
    )
    stream = client.chat.completions.create(
        messages=history,
        model="google/gemini-2.0-flash-lite-preview-02-05:free",
        stream=True,
    )
    chunks = []
    for chunk in stream:
        chunks.append(chunk.choices[0].delta.content or "")
        yield "".join(chunks)

    return message


def process_video(video_file):
    dataset = VideoFrameSlidingDataset(
        video_file,
        mode="mp4",
        window_size=args.clip_len,
        stride=args.clip_len,
        channel_first=False,
        transform=transform,
    )
    fps = dataset.fps

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=os.cpu_count() // 2
    )

    pred_events = run_inference(
        model,
        dataloader,
        args.classes,
        args.pred_file,
        postprocess=True,
    )
    results = pred_events[0]["events"]
    global raw_results
    raw_results = results  # save the results globally

    if len(results) == 0:
        results_df = pd.DataFrame([{"message": "No events detected"}])
        btn_update = gr.update(visible=False)
        vid_update = gr.update(visible=False)
    else:
        results = sorted(results, key=lambda x: x["frame"])
        # Add timestamp (in seconds) to each event
        for event in results:
            event["timestamp"] = round(event["frame"] / fps, 2)
        results_df = pd.DataFrame(results)
        # Rename columns and include Timestamp
        results_df.columns = ["Event", "Frame", "Score", "Location", "Timestamp (sec)"]
        results_df["Score"] = results_df["Score"].apply(lambda x: round(x, 2))
        results_df["Location"] = results_df["Location"].apply(
            lambda coords: [round(v, 2) for v in coords]
        )
        btn_update = gr.update(visible=True)
        vid_update = gr.update(visible=True)
    # Return predictions, original video file, raw results and UI updates for render_button and preview.
    return (
        results_df,
        video_file,
        results,
        btn_update,
        vid_update,
        gr.update(visible=True),
    )


def render_results(video_file, results):
    print("Rendering video...")
    rendered_video = render_video(video_file, results)
    print("Video rendered:", rendered_video)
    return rendered_video


def render_video(
    video_file,
    events,
    freeze_frames=25,
    main_color=(242, 243, 244),
    out_width=1280,
    out_height=720,
):

    event_by_frame = {x["frame"]: x for x in events}
    output_video_file = video_file.replace(".mp4", "_output.mp4")
    clip = VideoFileClip(video_file)
    fps = clip.fps
    frames = list(clip.iter_frames())
    height, width = frames[0].shape[:2]
    new_frames = []

    for idx, frame in enumerate(frames):
        if idx not in event_by_frame:
            new_frames.append(frame)
        else:
            event = event_by_frame[idx]
            x, y = int(event["xy"][0] * width), int(event["xy"][1] * height)
            score = event["score"]
            label = event["label"]
            text = f"{label.upper()} {score:.0%}"

            for it in range(freeze_frames):
                dt = it / freeze_frames
                canvas = frame.copy()

                draw_filled_parallelogram(text, (x, y), canvas, dt)

                alpha = max(0.9, 0.65 + 0.25 * dt)
                canvas = cv2.addWeighted(frame, 1 - alpha, canvas, alpha, 0)

                radius = 10 + 40 * ((1 - dt) ** 8)
                thickness = 1 + dt * 2
                cv2.circle(
                    canvas,
                    (x, y),
                    int(radius),
                    main_color,
                    int(thickness),
                    lineType=cv2.LINE_AA,
                )
                new_frames.append(canvas)

    # resize if needed
    if out_width != width or out_height != height:
        new_frames = [cv2.resize(x, (out_width, out_height)) for x in new_frames]
    new_clip = ImageSequenceClip(new_frames, fps=fps)
    new_clip.write_videofile(output_video_file, codec="libx264")

    return output_video_file


def draw_filled_parallelogram(text, center_coordinates, image, dt):
    # Initial parameters
    radius = 10
    color = (255, 255, 255)  # White color in BGR
    circle_thickness = 1  # Solid circle
    delta_shadow = np.array([3, 3])
    color_shadow = (57, 61, 71)  # Black color for the shadow
    color_text = (0, 0, 0)  # Black color for the text

    # Text and padding
    font_scale = 0.7
    font_thickness = 1
    # font = cv2.FONT_HERSHEY_SIMPLEX
    font = cv2.FONT_HERSHEY_COMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

    left_padding = 7
    right_padding = 20
    top_padding = 12
    bottom_padding = 7

    # Adjust the length of the parallelogram based on the text size and padding
    parallelogram_length = text_size[0] + left_padding + right_padding
    parallelogram_height = text_size[1] + top_padding + bottom_padding

    # Define angle for the 60-degree line
    angle = np.pi / 3  # 60 degrees in radians
    positive_angle = 2 * np.pi / 3  # 120 degrees in radians

    # Calculate the 60-degree line points
    start_point = center_coordinates
    end_point = (
        int(center_coordinates[0] + 100 * np.cos(angle)),
        int(center_coordinates[1] - 100 * np.sin(angle)),
    )

    # Calculate the points of the parallelogram, with the first point being below the 60-degree line
    gap = 5
    point1 = (end_point[0] + gap, end_point[1] + gap)  # Below the 60-degree line
    point2 = (
        int(point1[0] + parallelogram_length),
        point1[1],
    )  # Move horizontally (parallel to horizontal line)
    point3 = (
        int(point2[0] + parallelogram_height * np.cos(positive_angle)),
        int(point2[1] + parallelogram_height * np.sin(positive_angle)),
    )  # Adjusted to 120 degrees
    point4 = (
        int(point1[0] + parallelogram_height * np.cos(positive_angle)),
        int(point1[1] + parallelogram_height * np.sin(positive_angle)),
    )  # Adjusted to 120 degrees

    # Draw the white dot with radius 10
    cv2.circle(
        image,
        center_coordinates,
        radius,
        color,
        circle_thickness,
        lineType=cv2.LINE_AA,
    )

    # Draw the 60-degree line
    end_point_ = np.add(
        start_point, np.subtract(end_point, center_coordinates) * min(1.0, dt * 4)
    ).astype(np.int32)

    start_point_shadow = np.add(start_point, delta_shadow * 0.5).astype(np.int32)
    end_point_shadow = np.add(end_point_, delta_shadow * 0.5).astype(np.int32)

    cv2.line(
        image,
        start_point_shadow,
        end_point_shadow,
        color_shadow,  # Black color for the shadow
        thickness=2,
        lineType=cv2.LINE_AA,
    )

    cv2.line(image, start_point, end_point_, color, thickness=2, lineType=cv2.LINE_AA)

    # Draw the horizontal line from the end of the first line

    if dt >= 1 / 4.0:
        horizontal_end_point = (
            int(end_point[0] + 60 * min(1.0, 4 * (dt - 0.25))),
            end_point[1],
        )
        horizontal_end_point_shadow = np.add(
            horizontal_end_point, delta_shadow * 0.5
        ).astype(np.int32)
        cv2.line(
            image,
            end_point_shadow,
            horizontal_end_point_shadow,
            color_shadow,  # Black color for the shadow
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        cv2.line(
            image,
            end_point,
            horizontal_end_point,
            color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    # Fill the parallelogram by using the fillPoly function
    delta = np.array([max(0, 1 - dt * 4) * 15, 0])
    pts = np.array([point1, point2, point3, point4], np.int32)
    pts = (pts + delta).astype(np.int32)
    pts_shadow = (pts + delta_shadow).astype(np.int32)
    pts = pts.reshape((-1, 1, 2))
    pts_shadow = pts_shadow.reshape((-1, 1, 2))

    # draw main parallelogram
    cv2.fillPoly(image, [pts_shadow], color_shadow, lineType=cv2.LINE_AA)
    cv2.fillPoly(image, [pts], color, lineType=cv2.LINE_AA)

    # Calculate text position (centered inside the parallelogram with adjusted padding)
    text_x = pts[0][0][0] + left_padding
    text_y = pts[0][0][1] + top_padding + text_size[1] // 2

    # Add the text inside the parallelogram
    cv2.putText(
        image,
        text,
        (text_x, text_y),
        font,
        font_scale,
        color_text,
        font_thickness,
        lineType=cv2.LINE_AA,
    )

    return image


if __name__ == "__main__":
    global model, transform, args, client, results_state

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPEN_ROUTER_API_KEY"),
    )

    print("Loading model")
    config_dir = "exp/best/stes_vnl15_base"
    config_path = os.path.join(config_dir, "config.json")
    checkpoint_path = os.path.join(config_dir, "checkpoint.pt")
    save_dir = os.path.join(config_dir, "tmp")
    os.makedirs(save_dir, exist_ok=True)
    args = get_args(
        **load_json(config_path), checkpoint_path=checkpoint_path, save_dir=save_dir
    )

    transform = v2.Compose(
        [
            v2.Resize(args.resize, antialias=False),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    model = get_model(args)
    args = easydict.EasyDict(args)
    args.pred_file = os.path.join(args.save_dir, "predictions.json")

    # Set up a Blocks interface with states and buttons
    with Blocks() as demo:
        gr.Markdown("# Spatio-Temporal Event Spotting in Volleyball Videos")
        gr.Markdown(
            "Upload a video clip and click 'Detect Events' to analyze events in your video."
        )
        with gr.Row():
            with gr.Column(scale=1):
                video_input = Video(label="Upload Video Clip", autoplay=True)
                process_button = Button("Detect Events")
                predictions_output = Dataframe(
                    label="Event Detection Results",
                    value=pd.DataFrame([{"Message": "..."}]),
                )
                video_state = State()  # stores the original video file
                results_state = State()  # stores raw results from inference

            with gr.Column(scale=1):
                render_button = Button("Render results", visible=False)
                rendered_video = Video(
                    label="Rendered Video Preview",
                    visible=False,
                    autoplay=True,
                    # height=240,
                )
                chat_container = gr.Column(visible=False)
                with chat_container:
                    gr.ChatInterface(
                        fn=reply_message,
                        type="messages",
                        title="Ask me about the volleyball events!",
                    )

        process_button.click(
            process_video,
            inputs=video_input,
            outputs=[
                predictions_output,
                video_state,
                results_state,
                render_button,
                rendered_video,
                chat_container,
            ],
        )
        render_button.click(
            render_results, inputs=[video_state, results_state], outputs=rendered_video
        )

    demo.launch(server_name="0.0.0.0", share=True, pwa=True)
