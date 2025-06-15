
FROM python:3.9-slim

# Install required dependencies
RUN apt update && \
    apt install -y ffmpeg libportaudio2 libasound2 tk && \
    pip install numpy scipy soundfile pandas sounddevice matplotlib

WORKDIR /app

# Copy scripts into the container
COPY lfn_gui_batch_analyzer.py /app/
COPY lfn_realtime_monitor.py /app/
COPY LFN_Monitoring_Log_Template.csv /app/

CMD ["python", "lfn_gui_batch_analyzer.py"]
