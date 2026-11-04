import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
import tempfile
import os
import urllib.request
from pathlib import Path

st.set_page_config(page_title="筋トレフォームチェック", page_icon="🏋️", layout="wide")
st.title("🏋️ 筋トレフォームチェック")
st.caption("MediaPipe Pose Landmarkerによる動画解析・試作版（学習済み分類モデルは未接続）")

TASK_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
TASK_PATH = Path("pose_landmarker_lite.task")

@st.cache_resource
def load_landmarker():
    if not TASK_PATH.exists():
        urllib.request.urlretrieve(TASK_URL, TASK_PATH)
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(TASK_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return options

def angle(a,b,c):
    a,b,c = (np.array([q.x,q.y], dtype=float) for q in (a,b,c))
    v1,v2=a-b,c-b
    return float(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-8),-1,1))))

# MediaPipe landmark indices: shoulder=11/12, elbow=13/14, wrist=15/16,
# hip=23/24, knee=25/26, ankle=27/28.
def measure(lm, exercise):
    if exercise=="アームカール":
        return angle(lm[12],lm[14],lm[16]), "肘角度"
    if exercise=="ベンチプレス":
        return angle(lm[12],lm[14],lm[16]), "肘角度"
    return angle(lm[24],lm[26],lm[28]), "膝角度"

def process(video_bytes, exercise):
    from mediapipe.tasks.python import vision
    source=tempfile.NamedTemporaryFile(suffix=".mp4",delete=False)
    source.write(video_bytes);source.close()
    cap=cv2.VideoCapture(source.name)
    fps=cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps<=0: fps=30
    frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frames>0 and frames/fps>120:
        cap.release();os.unlink(source.name)
        raise ValueError("動画は120秒以内にしてください。")
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width<=0 or height<=0:
        cap.release();os.unlink(source.name)
        raise ValueError("動画を読み取れませんでした。MP4(H.264)をお試しください。")
    scale=min(1,640/width)
    w,h=int(width*scale),int(height*scale)
    output=tempfile.NamedTemporaryFile(suffix=".mp4",delete=False);output.close()
    writer=cv2.VideoWriter(output.name,cv2.VideoWriter_fourcc(*"mp4v"),min(fps,30), (w,h))
    records=[];reps=0;phase="up";last_ms=-1
    options=load_landmarker()
    progress=st.progress(0)
    try:
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            i=0
            while True:
                ok,frame=cap.read()
                if not ok or i>=3600:break
                frame=cv2.resize(frame,(w,h))
                rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                timestamp=max(last_ms+1,int(i*1000/fps));last_ms=timestamp
                image=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
                result=landmarker.detect_for_video(image,timestamp)
                if result.pose_landmarks:
                    lm=result.pose_landmarks[0]
                    value,label=measure(lm,exercise)
                    records.append((round(i/fps,2),round(value,1)))
                    # Simple heuristic only, not trained model inference.
                    down = value<95 if exercise!="スクワット" else value<105
                    up = value>150 if exercise!="スクワット" else value>160
                    if phase=="up" and down:phase="down"
                    elif phase=="down" and up:reps+=1;phase="up"
                    for a,b in ((11,13),(13,15),(12,14),(14,16),(11,12),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28)):
                        x1,y1=int(lm[a].x*w),int(lm[a].y*h)
                        x2,y2=int(lm[b].x*w),int(lm[b].y*h)
                        cv2.line(frame,(x1,y1),(x2,y2),(0,255,120),2)
                    cv2.putText(frame,f"Angle: {value:.0f}",(15,35),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
                cv2.putText(frame,f"Reps: {reps}",(15,70),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
                writer.write(frame)
                i+=1
                if i%15==0:progress.progress(min(1,i/max(frames,1)))
    finally:
        cap.release();writer.release();os.unlink(source.name);progress.empty()
    with open(output.name,"rb") as f: processed=f.read()
    os.unlink(output.name)
    return reps,records,processed

exercise=st.selectbox("種目",["ベンチプレス","アームカール","スクワット"])
video=st.file_uploader("動画をアップロード（MP4推奨、120秒以内）",type=["mp4","mov","avi"])
if video:
    st.video(video)
    if st.button("解析開始",type="primary"):
        try:
            with st.spinner("姿勢を解析しています…"):
                reps,records,processed=process(video.getvalue(),exercise)
            st.metric("推定回数",reps)
            if records:
                import pandas as pd
                df=pd.DataFrame(records,columns=["時間（秒）","角度（度）"])
                st.line_chart(df.set_index("時間（秒）"))
                st.download_button("角度データCSV",df.to_csv(index=False).encode("utf-8-sig"),"angles.csv","text/csv")
            else:st.warning("人物を検出できませんでした。全身が映る動画で再試行してください。")
            st.download_button("骨格描画動画を保存",processed,"pose_result.mp4","video/mp4")
            st.info("回数は角度しきい値による参考値です。フォームの正誤・100点採点は学習済みモデル接続後に追加します。")
        except Exception as e:
            st.error(f"解析に失敗しました: {e}")
            st.info("ログを確認してください。動画はMP4(H.264)形式が推奨です。")
