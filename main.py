
import os
import uuid
import numpy as np
import requests
import json
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import subprocess

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("yolov8n-pose.pt")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def get_keypoint(keypoints, index):
    kp = keypoints.data[0][index]
    return [kp[0].item(), kp[1].item()]

def score_metric(value, benchmark_ideal, benchmark_range):
    deviation = abs(value - benchmark_ideal)
    return round(max(0, 100 - (deviation / benchmark_range) * 100))

def calculate_phase_scores(torso_avg, knee_avg, hip_avg, elbow_left, elbow_right):
    torso_score = score_metric(torso_avg, 45, 45)
    knee_score = score_metric(knee_avg, 90, 90)
    hip_score = score_metric(hip_avg, 165, 45)
    elbow_left_score = score_metric(elbow_left, 95, 45)
    elbow_right_score = score_metric(elbow_right, 95, 45)
    arm_symmetry_score = score_metric(abs(elbow_left - elbow_right), 0, 15)
    block_start_score = torso_score
    acceleration_score = round((knee_score + torso_score) / 2)
    max_velocity_score = round((hip_score + knee_score) / 2)
    arm_score = round((elbow_left_score + elbow_right_score + arm_symmetry_score) / 3)
    overall = round((block_start_score + acceleration_score + max_velocity_score + arm_score) / 4)
    return {
        "overall": overall,
        "block_start": block_start_score,
        "acceleration": acceleration_score,
        "max_velocity": max_velocity_score,
        "arm_mechanics": arm_score,
        "breakdown": {
            "torso_lean": torso_score,
            "knee_drive": knee_score,
            "hip_extension": hip_score,
            "elbow_left": elbow_left_score,
            "elbow_right": elbow_right_score,
            "arm_symmetry": arm_symmetry_score
        }
    }

@app.post("/analyze")
async def analyze_sprint(file: UploadFile = File(...)):
    # Save uploaded video
    ext = file.filename.split(".")[-1]
    video_path = f"/tmp/{uuid.uuid4()}.{ext}"
    mp4_path = video_path.replace(f".{ext}", ".mp4")
    
    with open(video_path, "wb") as f:
        f.write(await file.read())
    
    # Convert to mp4 if needed
    if ext.lower() != "mp4":
        subprocess.run(["ffmpeg", "-i", video_path, "-vcodec", "libx264", "-acodec", "aac", mp4_path, "-y"])
    else:
        mp4_path = video_path

    # Run pose detection
    results = model(mp4_path, save=False, task="pose", stream = True)
    results = list(results)

    # Detect sprint start
    hip_x_positions = []
    for frame in results:
        if frame.keypoints is not None and len(frame.keypoints.data) > 0:
            kps = frame.keypoints.data[0]
            hip_x = (kps[11][0].item() + kps[12][0].item()) / 2
            hip_x_positions.append(hip_x)

    sprint_start = 0
    max_movement = 0
    for i in range(len(hip_x_positions) - 10):
        movement = abs(hip_x_positions[i + 10] - hip_x_positions[i])
        if movement > max_movement:
            max_movement = movement
            sprint_start = i
    sprint_start = max(0, sprint_start - 5)

    # Calculate angles
    torso_angles, knee_angles, hip_angles = [], [], []
    elbow_left_angles, elbow_right_angles = [], []
    CONF = 0.5

    for frame in results[sprint_start:]:
        if frame.keypoints is None or len(frame.keypoints.data) == 0:
            continue
        kps = frame.keypoints.data[0]

        if all(kps[i][2].item() > CONF for i in [5, 6, 11, 12]):
            ls = get_keypoint(frame.keypoints, 5)
            rs = get_keypoint(frame.keypoints, 6)
            lh = get_keypoint(frame.keypoints, 11)
            rh = get_keypoint(frame.keypoints, 12)
            sm = [(ls[0]+rs[0])/2, (ls[1]+rs[1])/2]
            hm = [(lh[0]+rh[0])/2, (lh[1]+rh[1])/2]
            vr = [hm[0], hm[1]-100]
            torso_angles.append(calculate_angle(sm, hm, vr))

        if all(kps[i][2].item() > CONF for i in [11, 13, 15]):
            knee_angles.append(calculate_angle(
                get_keypoint(frame.keypoints, 11),
                get_keypoint(frame.keypoints, 13),
                get_keypoint(frame.keypoints, 15)))

        if all(kps[i][2].item() > CONF for i in [5, 11, 13]):
            hip_angles.append(calculate_angle(
                get_keypoint(frame.keypoints, 5),
                get_keypoint(frame.keypoints, 11),
                get_keypoint(frame.keypoints, 13)))

        if all(kps[i][2].item() > CONF for i in [5, 7, 9]):
            elbow_left_angles.append(calculate_angle(
                get_keypoint(frame.keypoints, 5),
                get_keypoint(frame.keypoints, 7),
                get_keypoint(frame.keypoints, 9)))

        if all(kps[i][2].item() > CONF for i in [6, 8, 10]):
            elbow_right_angles.append(calculate_angle(
                get_keypoint(frame.keypoints, 6),
                get_keypoint(frame.keypoints, 8),
                get_keypoint(frame.keypoints, 10)))

    torso_avg = round(float(np.mean(torso_angles)), 1) if torso_angles else 0
    knee_avg = round(float(np.mean(knee_angles)), 1) if knee_angles else 0
    hip_avg = round(float(np.mean(hip_angles)), 1) if hip_angles else 0
    elbow_left = round(float(np.mean(elbow_left_angles)), 1) if elbow_left_angles else 0
    elbow_right = round(float(np.mean(elbow_right_angles)), 1) if elbow_right_angles else 0

    scores = calculate_phase_scores(torso_avg, knee_avg, hip_avg, elbow_left, elbow_right)

    # Get AI feedback
    sprint_metrics = {
        "torso_lean": {"average": torso_avg, "benchmark": "45° block start → 5° max velocity"},
        "knee_drive": {"average": knee_avg, "benchmark": "Under 90° at peak drive"},
        "hip_extension": {"average": hip_avg, "benchmark": "160-170° at toe-off"},
        "elbow_angle": {"left": elbow_left, "right": elbow_right, "benchmark": "80-110°"}
    }

    prompt = f"""
    You are an expert sprint coach analyzing biomechanical data from a sprinter video.
    The overall score has been calculated as {scores["overall"]}/100 — use this exact score.
    Give a detailed but encouraging phase by phase coaching report.
    For each metric explain what it means in plain English and give one specific drill to fix it.
    Keep the tone like a supportive coach, not a robot.

    Sprint Metrics:
    {json.dumps(sprint_metrics, indent=2)}

    Phase Scores:
    - Block Start: {scores["block_start"]}/100
    - Acceleration: {scores["acceleration"]}/100
    - Max Velocity: {scores["max_velocity"]}/100
    - Arm Mechanics: {scores["arm_mechanics"]}/100
    - Overall: {scores["overall"]}/100

    Structure your response as:
    1. Overall Score: {scores["overall"]}/100
    2. Block Start Analysis
    3. Acceleration Phase Analysis
    4. Max Velocity Analysis
    5. Arm Mechanics
    6. Top 3 Priority Improvements
    """

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "max_tokens": 1000}
    )

    feedback = response.json()["choices"][0]["message"]["content"]

    # Cleanup
    os.remove(video_path)
    if mp4_path != video_path:
        os.remove(mp4_path)

    return {
        "scores": scores,
        "metrics": {
            "torso_lean": torso_avg,
            "knee_drive": knee_avg,
            "hip_extension": hip_avg,
            "elbow_left": elbow_left,
            "elbow_right": elbow_right
        },
        "feedback": feedback
    }

@app.get("/")
def root():
    return {"status": "SprintIQ API is running"}
