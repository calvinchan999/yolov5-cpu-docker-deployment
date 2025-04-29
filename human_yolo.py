import sys
from pathlib import Path
import os
import ssl

# Get the absolute path to your project directory
FILE = Path(__file__).resolve()
ROOT = FILE.parent
YOLOV5_ROOT = ROOT / 'yolov5'
if str(YOLOV5_ROOT) not in sys.path:
    sys.path.append(str(YOLOV5_ROOT))

# Import required packages
import random
import time
import numpy as np
import cv2
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import LoadStreams
from yolov5.utils.general import (check_img_size, non_max_suppression, scale_boxes)
from yolov5.utils.plots import Annotator, colors
from yolov5.utils.torch_utils import select_device, smart_inference_mode
import logging
import subprocess
import queue
import threading
import paho.mqtt.client as mqtt
import json
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get configuration from environment variables with defaults
MQTT_ENABLED = os.getenv('MQTT_ENABLED', 'true').lower() == 'true'
MQTT_BROKER = os.getenv('MQTT_BROKER', '127.0.0.1')
MQTT_PORT = int(os.getenv('MQTT_PORT', '8883'))
MQTT_TOPIC = os.getenv('MQTT_TOPIC', 'object/detection/inference')
MQTT_USERNAME = os.getenv('MQTT_USERNAME', 'test')
MQTT_PASSWORD = os.getenv('MQTT_PASSWORD', 'test')
MQTT_PUBLISH_INTERVAL = int(os.getenv('MQTT_PUBLISH_INTERVAL', '1'))
MQTT_USE_SSL = os.getenv('MQTT_USE_SSL', 'true').lower() == 'true'

ROBOT_ID = os.getenv('ROBOT_ID', '418.1')

# YOLO Configuration
YOLO_WEIGHTS = os.getenv('YOLO_WEIGHTS', 'yolov5n.pt')
YOLO_CONF_THRES = float(os.getenv('YOLO_CONF_THRES', '0.30'))
YOLO_IOU_THRES = float(os.getenv('YOLO_IOU_THRES', '0.35'))
YOLO_MAX_DET = int(os.getenv('YOLO_MAX_DET', '20'))
YOLO_CLASSES = int(os.getenv('YOLO_CLASSES', '0'))
YOLO_AGNOSTIC_NMS= os.getenv('YOLO_AGNOSTIC_NMS', 'true').lower == 'true'

# RTMP Configuration
RTMP_URL = os.getenv('RTMP_URL', 'rtmp://127.0.0.1:1935/hihi')
# RTSP Configuration
RTSP_URL = os.getenv('RTSP_URL', 'rtsp://admin:test@169.254.89.235:554/Streaming/Channels/101')
FPS = int(os.getenv('FPS', '10'))
WIDTH = int(os.getenv('WIDTH', '640'))
HEIGHT = int(os.getenv('HEIGHT', '360'))
KEYFRAME_INTERVAL = int(os.getenv('KEYFRAME_INTERVAL', '30'))

# Performance Configuration
PROCESS_WIDTH = int(os.getenv('PROCESS_WIDTH', '256'))
PROCESS_HEIGHT = int(os.getenv('PROCESS_HEIGHT', '256'))
INFERENCE_INTERVAL = float(os.getenv('INFERENCE_INTERVAL', '0.1'))
MAX_BATCH_SIZE = int(os.getenv('MAX_BATCH_SIZE', '1'))
NUM_THREADS = int(os.getenv('NUM_THREADS', '4'))
ENABLE_OPTIMIZATION = os.getenv('ENABLE_OPTIMIZATION', 'true').lower() == 'true'

# Add after the configuration variables
DETECTION_PEOPLE = 0  # Global variable to track detection state

# Log all configuration values
logging.info("Configuration:")
logging.info(f"MQTT_BROKER: {MQTT_BROKER}")
logging.info(f"MQTT_PORT: {MQTT_PORT}")
logging.info(f"MQTT_TOPIC: {MQTT_TOPIC}")
logging.info(f"ROBOT_ID: {ROBOT_ID}")
logging.info(f"RTMP_URL: {RTMP_URL}")
logging.info(f"RTSP_URL: {RTSP_URL}")
logging.info(f"FPS: {FPS}")
logging.info(f"PROCESS_WIDTH x PROCESS_HEIGHT: {PROCESS_WIDTH}x{PROCESS_HEIGHT}")
logging.info(f"YOLO_WEIGHTS: {YOLO_WEIGHTS}")
logging.info(f"YOLO_CONF_THRES: {YOLO_CONF_THRES}")
logging.info(f"YOLO_IOU_THRES: {YOLO_IOU_THRES}")
logging.info(f"YOLO_MAX_DET: {YOLO_MAX_DET}")
logging.info(f"YOLO_CLASSES: {YOLO_CLASSES}")
logging.info(f"YOLO_AGNOSTIC_NMS: {YOLO_AGNOSTIC_NMS}")

class MQTTClient:
    def __init__(self, broker, port, username, password):
        # Use MQTT v3.1.1
        self.client = mqtt.Client(protocol=mqtt.MQTTv311)
        self.broker = broker
        self.port = port
        self.username = username
        self.password = password
        self.connected = False

        # Set up callbacks
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_publish = self.on_publish
        self.client.on_log = self.on_log

        # Enable automatic reconnection
        self.client.reconnect_delay_set(min_delay=1, max_delay=30)

    def on_connect(self, client, userdata, flags, rc):
        rc_codes = {
            0: "Connection successful",
            1: "Connection refused - incorrect protocol version",
            2: "Connection refused - invalid client identifier",
            3: "Connection refused - server unavailable",
            4: "Connection refused - bad username or password",
            5: "Connection refused - not authorized"
        }
        if rc == 0:
            self.connected = True
            logging.info("Successfully connected to MQTT broker")
        else:
            self.connected = False
            logging.error(f"Failed to connect to MQTT broker: {rc_codes.get(rc, f'Unknown error {rc}')}")

    def on_disconnect(self, client, userdata, rc):
        self.connected = False
        if rc != 0:
            logging.warning(f"Unexpected MQTT disconnection with code: {rc}. Will automatically reconnect...")
        else:
            logging.info("MQTT client disconnected")

    def on_publish(self, client, userdata, mid):
        logging.debug(f"Message {mid} has been published")

    def on_log(self, client, userdata, level, buf):
        logging.debug(f"MQTT Log: {buf}")

    def connect(self):
        try:
            # Set username and password if provided
            if self.username and self.password:
                self.client.username_pw_set(self.username, self.password)
                logging.info("MQTT credentials configured")

            # Configure SSL/TLS if enabled
            if MQTT_USE_SSL:
                logging.info("Configuring MQTT with SSL/TLS")
                try:
                    self.client.tls_set(
                        cert_reqs=ssl.CERT_REQUIRED,
                        tls_version=ssl.PROTOCOL_TLSv1_2,
                        ciphers=None
                    )
                    self.client.tls_insecure_set(False)
                except Exception as ssl_error:
                    logging.error(f"SSL configuration failed: {str(ssl_error)}")
                    raise
            else:
                logging.info("Using MQTT without SSL/TLS")

            # Connect to broker
            logging.info(f"Attempting to connect to MQTT broker at {self.broker}:{self.port}")
            connect_result = self.client.connect(self.broker, self.port, keepalive=60)
            if connect_result != 0:
                logging.error(f"Initial MQTT connection failed with result code: {connect_result}")
                return False

            # Start the loop in a background thread
            self.client.loop_start()

            # Wait for connection to be established
            timeout = 10
            start_time = time.time()
            while not self.connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)

            if not self.connected:
                logging.error("Failed to connect to MQTT broker within timeout period")
                self.client.loop_stop()
                return False

            logging.info("MQTT client successfully connected and ready")
            return True

        except Exception as e:
            logging.error(f"Failed to connect to MQTT broker: {str(e)}")
            self.connected = False
            return False

    def publish_detection(self, count):
        if not self.connected:
            logging.warning("Not connected to MQTT broker, cannot publish message")
            return

        try:
            # Prepare message
            message = {
                "robotId": ROBOT_ID,
                "pose": None,
                "detectionType": None,
                "base64Image": None,
                "metadata": None,
                "count": count
            }

            # Publish message with QoS 1 and retain=False
            result = self.client.publish(
                topic=MQTT_TOPIC,
                payload=json.dumps(message),
                qos=1,
                retain=False
            )

            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                logging.error(f"Failed to publish message: {mqtt.error_string(result.rc)}")
            else:
                result.wait_for_publish()  # Wait for message to be published
                logging.info(f"Published MQTT message: {json.dumps(message)}")

        except Exception as e:
            logging.error(f"Failed to publish MQTT message: {str(e)}")

    def disconnect(self):
        if self.connected:
            self.client.loop_stop()
            self.client.disconnect()
            self.connected = False
            logging.info("Disconnected from MQTT broker")

class RTMPStreamer:
    def __init__(self, rtmp_url, width, height, fps):
        self.rtmp_url = rtmp_url
        self.width = width
        self.height = height
        self.fps = fps
        self.process = None
        self.frame_queue = queue.Queue(maxsize=180)  # Reduced buffer size for lower latency
        self.is_running = False
        self._shutdown = threading.Event()

    def start(self):
        command = [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{self.width}x{self.height}',
            '-r', str(self.fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-g', str(KEYFRAME_INTERVAL),
            '-keyint_min', str(KEYFRAME_INTERVAL),
            '-x264opts', 'no-scenecut',
            '-b:v', '1M',  # Adjust bitrate as needed
            '-maxrate', '1M',
            '-bufsize', '2M',
            '-f', 'flv',
            self.rtmp_url
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)
        self.is_running = True
        self.stream_thread = threading.Thread(target=self._stream_frames, daemon=True)
        self.stream_thread.start()

    def write(self, frame):
        if not self.frame_queue.full():
            self.frame_queue.put(frame)

    def _stream_frames(self):
        while not self._shutdown.is_set():
            try:
                frame = self.frame_queue.get(timeout=1)
                if frame is None:
                    break
                self.process.stdin.write(frame.tobytes())
            except queue.Empty:
                continue
            except BrokenPipeError:
                logging.error("Broken pipe. Restarting stream...")
                self.restart()
            except Exception as e:
                logging.error(f"Streaming error: {e}")
                self.restart()

    def restart(self):
        self.release()
        time.sleep(5)  # Wait before restarting
        self.start()

    def release(self):
        self._shutdown.set()
        self.frame_queue.put(None)  # Send sentinel
        if self.stream_thread:
            self.stream_thread.join(timeout=5)
        if self.process:
            try:
                self.process.stdin.close()
                self.process.terminate()
                self.process.wait(timeout=5)
            except:
                self.process.kill()
        logging.info("RTMP stream closed")

class HeadlessVideoStream:
    def __init__(self, source, img_size=(480, 480), stride=32, auto=True, transforms=None):
        self.source = source
        self.img_size = img_size
        self.stride = stride
        self.transforms = transforms

        # Initialize video capture with optimized buffer size
        self.cap = cv2.VideoCapture(self.source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size
        assert self.cap.isOpened(), f'Failed to open {self.source}'

        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize frame reading thread
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.frame_queue = queue.Queue(maxsize=2)  # Reduced queue size for lower latency
        self.stopped = False
        self.thread.start()

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stop()
                break

            # Clear queue and put new frame
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            self.frame_queue.put(frame)

    def read(self):
        try:
            frame = self.frame_queue.get(timeout=1.0)
            # Prepare image for YOLO
            img = cv2.resize(frame, self.img_size)
            img = img.transpose((2, 0, 1))  # HWC to CHW
            img = np.ascontiguousarray(img)
            return True, img, [frame], None, ''
        except queue.Empty:
            return False, None, None, None, 'No frames available'

    def stop(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join()
        self.cap.release()

@smart_inference_mode()
def run(
    weights=YOLO_WEIGHTS,
    source=RTSP_URL,
    imgsz=(PROCESS_WIDTH, PROCESS_HEIGHT),
    conf_thres=YOLO_CONF_THRES,
    iou_thres=YOLO_IOU_THRES,
    max_det=YOLO_MAX_DET,
    device='cpu',
    classes=YOLO_CLASSES,
    agnostic_nms=YOLO_AGNOSTIC_NMS,
    mqtt_client=None
):
    import torch
    import torch.multiprocessing as mp

    # Initialize
    device = select_device(device)

    global DETECTION_PEOPLE  # Add global declaration

    # Load model
    model = DetectMultiBackend(weights, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # Initialize video stream
    stream = HeadlessVideoStream(source, img_size=imgsz)
    time.sleep(0.5)

    if not stream.cap.isOpened():
        raise Exception("Failed to open RTSP stream")

    # Initialize RTMP streamer
    streamer = None
    retry_count = 0
    max_rtmp_retries = 3

    logging.info(f"Source FPS: {stream.fps}, Target FPS: {FPS}")

    # Initialize RTMP streamer with retries
    while retry_count < max_rtmp_retries:
        try:
            streamer = RTMPStreamer(RTMP_URL, WIDTH, HEIGHT, FPS)
            streamer.start()
            logging.info(f"Successfully connected to RTMP server at {RTMP_URL}")
            break
        except Exception as e:
            retry_count += 1
            logging.error(f"RTMP connection attempt {retry_count} failed: {e}")
            if retry_count < max_rtmp_retries:
                time.sleep(5)
            else:
                logging.error("Failed to establish RTMP connection. Continuing without streaming...")
                break

    # Initialize timing variables
    last_inference_time = time.time()
    last_publish_time = time.time()
    frame_count = 0

    try:
        while True:
            # Read frame
            ret, im, im0s, _, _ = stream.read()
            if not ret:
                logging.error("Failed to read frame")
                raise Exception("Failed to read frame from RTSP stream")

            current_time = time.time()

            # Process every frame (removed frame skipping)
            frame_count += 1

            # Control inference rate
            if current_time - last_inference_time < INFERENCE_INTERVAL:
                # Still stream the frame even if we don't do inference
                if streamer and streamer.is_running:
                    try:
                        resized_frame = cv2.resize(im0s[0], (WIDTH, HEIGHT))
                        streamer.write(resized_frame)
                    except Exception as e:
                        logging.error(f"Streaming error: {e}")
                        if streamer:
                            streamer.restart()
                continue

            last_inference_time = current_time

            # Convert image to tensor
            try:
                im = torch.from_numpy(im)
                im = im.to(device)
                im = im.float()
                im /= 255.0
                if len(im.shape) == 3:
                    im = im[None]
            except Exception as e:
                logging.error(f"Error converting image to tensor: {e}")
                continue

            # Inference
            try:
                pred = model(im, augment=False, visualize=False)
            except Exception as e:
                logging.error(f"Inference error: {e}")
                continue

            # NMS
            try:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            except Exception as e:
                logging.error(f"NMS error: {e}")
                continue

            # Process predictions
            for i, det in enumerate(pred):
                detection_count = len(det)
                
                if DETECTION_PEOPLE != detection_count:
                    mqtt_client.publish_detection(detection_count)
                    DETECTION_PEOPLE = detection_count
                    logging.info(f"State change: {DETECTION_PEOPLE} people detected")


                im0 = im0s[i].copy()
                annotator = Annotator(im0, line_width=2)

                # Display count on top left
                cv2.putText(im0, f'Count: {len(det)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                           1, (0, 255, 0), 2, cv2.LINE_AA)

                if len(det):
                    # Convert to float64 for higher precision
                    det = det.to(torch.float64)

                    # Calculate scaling factors with higher precision
                    gain = torch.tensor([im0.shape[1] / im.shape[2],
                                      im0.shape[0] / im.shape[3],
                                      im0.shape[1] / im.shape[2],
                                      im0.shape[0] / im.shape[3]],
                                      dtype=torch.float64, device=device)

                    # Scale boxes with higher precision
                    det[:, :4] = det[:, :4] * gain

                    # Convert back to float32 for drawing
                    det = det.to(torch.float32)

                    # Draw boxes
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)
                        annotator.box_label(xyxy, "", color=colors(c, True))

                # Stream if RTMP is available
                if streamer and streamer.is_running:
                    try:
                        resized_frame = cv2.resize(im0, (WIDTH, HEIGHT))
                        streamer.write(resized_frame)
                    except Exception as e:
                        logging.error(f"Streaming error: {e}")
                        if streamer:
                            streamer.restart()

    except KeyboardInterrupt:
        logging.info("Stopping detection...")
    except Exception as e:
        logging.error(f"Detection error: {e}")
        raise
    finally:
        stream.stop()
        if streamer:
            streamer.release()

def main():
    max_retries = 3
    retry_delay = 5
    retry_count = 0
    mqtt_retry_count = 0
    max_mqtt_retries = 3

    # Initialize MQTT client only if enabled
    mqtt_client = None
    if MQTT_ENABLED:
        while mqtt_retry_count < max_mqtt_retries:
            try:
                mqtt_client = MQTTClient(MQTT_BROKER, MQTT_PORT, MQTT_USERNAME, MQTT_PASSWORD)
                if mqtt_client.connect():
                    logging.info("MQTT client successfully initialized")
                    break
                mqtt_retry_count += 1
                if mqtt_retry_count < max_mqtt_retries:
                    logging.info(f"Retrying MQTT connection in {retry_delay} seconds... (Attempt {mqtt_retry_count}/{max_mqtt_retries})")
                    time.sleep(retry_delay)
            except Exception as e:
                logging.error(f"MQTT initialization error: {e}")
                mqtt_retry_count += 1
                if mqtt_retry_count < max_mqtt_retries:
                    time.sleep(retry_delay)

        if not mqtt_client or not mqtt_client.connected:
            logging.error("Failed to initialize MQTT client after maximum retries")
            if MQTT_ENABLED:  # Only return if MQTT was supposed to be enabled
                return
    else:
        logging.info("MQTT is disabled by configuration")

    try:
        while True:
            try:
                retry_count = 0
                # Pass the MQTT client (which might be None if disabled)
                run(mqtt_client=mqtt_client)
            except Exception as e:
                retry_count += 1
                logging.error(f"Error: {e}")

                if retry_count >= max_retries:
                    logging.error("Max retries reached. Waiting longer before next attempt...")
                    time.sleep(retry_delay * 2)
                    retry_count = 0
                else:
                    logging.info(f"Retrying in {retry_delay} seconds... (Attempt {retry_count}/{max_retries})")
                    time.sleep(retry_delay)
    finally:
        # Only disconnect MQTT if it was enabled and connected
        if mqtt_client and mqtt_client.connected:
            mqtt_client.disconnect()

if __name__ == '__main__':
    main()
