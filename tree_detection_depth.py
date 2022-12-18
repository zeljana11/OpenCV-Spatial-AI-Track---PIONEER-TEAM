"""
The code is edited from docs (https://docs.luxonis.com/projects/api/en/latest/samples/Yolo/tiny_yolo/)
"""

from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import time
import argparse
import json
from depthai_sdk.utils import frameNorm, cropToAspectRatio

# configPath = Path(r"C:\Users\Vlade\BioSense\Projects\depthai3\best.json")
# model = r"C:\Users\Vlade\BioSense\Projects\depthai3\best_openvino_2021.4_6shave.blob"

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Provide model name or model path for inference",
                    default='yolo_v5_openvino_2021.4_6shave.blob', type=str)
parser.add_argument("-c", "--config", help="Provide config path for inference",
                    default='yolo_v5.json', type=str)
args = parser.parse_args()

# parse config
configPath = Path(args.config)
if not configPath.exists():
    raise ValueError("Path {} does not exist!".format(configPath))

with configPath.open() as f:
    config = json.load(f)
nnConfig = config.get("nn_config", {})

# parse input shape
if "input_size" in nnConfig:
    W, H = tuple(map(int, nnConfig.get("input_size").split('x')))

# extract metadata
metadata = nnConfig.get("NN_specific_metadata", {})
classes = metadata.get("classes", {})
coordinates = metadata.get("coordinates", {})
anchors = metadata.get("anchors", {})
anchorMasks = metadata.get("anchor_masks", {})
iouThreshold = metadata.get("iou_threshold", {})
confidenceThreshold = metadata.get("confidence_threshold", {})

print(metadata)

# parse labels
nnMappings = config.get("mappings", {})
labels = nnMappings.get("labels", {})

# get model path
model = args.model
nnPath = Path(model)

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
fps = 20

# NETWORK CONFIG:
# node
detectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
# config
detectionNetwork.setConfidenceThreshold(confidenceThreshold)
detectionNetwork.setNumClasses(classes)
detectionNetwork.setCoordinateSize(coordinates)
detectionNetwork.setAnchors(anchors)
detectionNetwork.setAnchorMasks(anchorMasks)
detectionNetwork.setIouThreshold(iouThreshold)
detectionNetwork.setBlobPath(nnPath)
detectionNetwork.setNumInferenceThreads(2)
detectionNetwork.input.setBlocking(False)

# COLOR CONFIG
# node
node_camera_rgb = pipeline.create(dai.node.ColorCamera)
# config
node_camera_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
node_camera_rgb.setFps(fps)
node_camera_rgb.setInterleaved(False)
node_camera_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
node_camera_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
node_camera_rgb.setPreviewSize(416, 416)



# MONO & STEREO CONFIG
# nodes
node_camera_mono_left = pipeline.create(dai.node.MonoCamera)
node_camera_mono_right = pipeline.create(dai.node.MonoCamera)
node_stereo_depth = pipeline.createStereoDepth()
# config
# mono_camera_resolution = resolution_depth
# node_camera_mono_left.setResolution(mono_camera_resolution)
node_camera_mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
node_camera_mono_left.setFps(fps)
# node_camera_mono_right.setResolution(mono_camera_resolution)
node_camera_mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
node_camera_mono_right.setFps(fps)
node_stereo_depth.setDepthAlign(dai.CameraBoardSocket.RGB)
node_stereo_depth.setExtendedDisparity(False)
# link
node_camera_mono_left.out.link(node_stereo_depth.left)
node_camera_mono_right.out.link(node_stereo_depth.right)


# OUTPUTS AND LINKING
# outputs
xout_depth = pipeline.createXLinkOut()
xout_depth.setStreamName("depth")
xout_color = pipeline.createXLinkOut()
xout_color.setStreamName("color")
xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("nn")

# links
# # regular
# node_camera_rgb.video.link(xout_color.input)
# # node_camera_rgb.preview.link(xout_color.input)
# node_stereo_depth.depth.link(xout_depth.input)
# # node_stereo_depth.disparity.link(xout_depth.input)

# passthrough
node_stereo_depth.depth.link(detectionNetwork.inputDepth)
# manip.out.link(detectionNetwork.input)
node_camera_rgb.preview.link(detectionNetwork.input)
# node_camera_rgb.video.link(detectionNetwork.input)
detectionNetwork.passthrough.link(xout_color.input)
detectionNetwork.passthroughDepth.link(xout_depth.input)
detectionNetwork.out.link(xout_nn.input)


# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queues
    qRgb = device.getOutputQueue(name="color", maxSize=4, blocking=False)
    qDep = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    frame = None
    detections = []
    startTime = time.monotonic()
    counter = 0
    color2 = (0, 120, 255)
    disparityMultiplier = 255 / node_stereo_depth.initialConfig.getMaxDisparity()

    def prepareFrame2(frame, f2, detections):
        color = (0, 120, 255)
        for detection in detections:

            # Denormalize bounding box
            bbox = frameNorm(frame, [detection.xmin, detection.ymin, detection.xmax, detection.ymax])
            crop = f2[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            th = 1500
            mask = (crop < th) & (crop != 0)
            dist = np.nanmedian(crop[mask])
            if np.isnan(dist):
                dist = 0

            label = labels[detection.label]
            cv2.putText(frame, str(label), (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(frame, "{:.2f}".format(detection.confidence * 100), (bbox[0] + 10, bbox[1] + 35),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x)} mm", (bbox[0] + 10, bbox[1] + 50),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y)} mm", (bbox[0] + 10, bbox[1] + 65),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z)} mm", (bbox[0] + 10, bbox[1] + 80),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(frame, f"Z2: {int(dist)} mm", (bbox[0] + 10, bbox[1] + 95),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, cv2.FONT_HERSHEY_SIMPLEX)

        return frame

    while True:
        inRgb = qRgb.get()
        inDet = qDet.get()
        inDep = qDep.get()

        if inRgb is not None:
            frame_rgb = inRgb.getCvFrame()

        if inDet is not None:
            detections = inDet.detections
            counter += 1

        if inDet is not None:
            frame_dep = inDep.getFrame()
            frame_dep = cropToAspectRatio(frame_dep, (416, 416))
            frame_dep = cv2.resize(frame_dep, (416, 416), interpolation=cv2.INTER_NEAREST)
            frame_dep_color = (((np.clip(frame_dep, a_min=300, a_max=3000) - 300) / 2700)*255).astype('uint8')
            frame_dep_color = cv2.applyColorMap(frame_dep_color, cv2.COLORMAP_JET)

        frame = frame_rgb
        if frame is not None:
            frame = prepareFrame2(frame, frame_dep, detections)
            cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                        (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color2)
            cv2.imshow('rgb', frame)#cv2.resize(frame, dsize=(750, 750)))
            cv2.imshow('depth', frame_dep_color)#cv2.resize(frame_dep_color, dsize=(750, 750)))

        if cv2.waitKey(1) == ord('q'):
            break