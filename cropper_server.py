"""
RetinaFace crop server — reimplements detect_pipe.py pipeline inline.
Model loads once at startup; every POST /crop runs inference in-memory.

This is intended to run in a Docker container, and is not 
optimized/secured for production use outside of that context.

POST /crop  body: {"image_base64": "...", "filename": "photo.jpg",
                   "crop_size": 100, "fill_color": 0.5, ...}
            200: {"status": "ok", "filename": "photo_crop_square.jpg", "image_base64": "..."}
            404: {"status": "error", "message": "No face detected"}
            500: {"status": "error", "message": "..."}
GET  /health  200: {"status": "ok"}

Env vars: CROPPER_PORT (5000), CROPPER_HOST (0.0.0.0),
          CROPPER_MODEL (./weights/Resnet50_Final.pth),
          CROPPER_NETWORK (resnet50), CROPPER_CPU (1)
"""

import base64
import http.server
import json
import os
import socketserver

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF

from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CROPPER_PORT    = int(os.environ.get("CROPPER_PORT") or 5000)
CROPPER_HOST    = os.environ.get("CROPPER_HOST") or "0.0.0.0"
CROPPER_MODEL   = os.environ.get("CROPPER_MODEL") or "./weights/Resnet50_Final.pth"
CROPPER_NETWORK = os.environ.get("CROPPER_NETWORK") or "resnet50"
CROPPER_CPU     = (os.environ.get("CROPPER_CPU") or "1") != "0"

# ---------------------------------------------------------------------------
# Model helpers — copied verbatim from detect_pipe.py lines 64-97
# ---------------------------------------------------------------------------
def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

# copied verbatim from detect_pipe.py lines 163-172
def resize_square_aspect_cv2(img, desired_size=100):
    old_size = img.shape[0:2]
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size][::-1])
    return cv2.resize(img, new_size)

# ---------------------------------------------------------------------------
# rotate_image — same logic as detect_pipe.py lines 119-144, but
# fill_color is now a parameter instead of reading from args (line 138)
# ---------------------------------------------------------------------------
def rotate_image(image, landmarks, fill_color=0.5):
    origin        = landmarks[[5, 6]]
    middle_finger = landmarks[[7, 8]]
    nose          = landmarks[[9, 10]]
    orientation_vector = middle_finger - origin
    destination_vector = np.array([1., 0.])
    dir_unit_vector = orientation_vector / np.linalg.norm(orientation_vector)
    angle_rad = np.arccos(np.clip(np.dot(destination_vector, dir_unit_vector), -1.0, 1.0))
    angle_degrees = -180 / np.pi * angle_rad
    angle_degrees = angle_degrees if (orientation_vector[1] < 0) else -angle_degrees
    rot_mat = cv2.getRotationMatrix2D((float(nose[0]), float(nose[1])), angle_degrees, 1.0)
    fill = int(fill_color * 256)
    return cv2.warpAffine(image, rot_mat, image.shape[1::-1],
                          flags=cv2.INTER_LINEAR, borderValue=(fill, fill, fill))

# ---------------------------------------------------------------------------
# Detection helper — extracted from detect_pipe.py lines 289-350
# ---------------------------------------------------------------------------
def _detect_best_face(img_raw, net, cfg, device, confidence_threshold,
                      top_k, nms_threshold, keep_top_k):
    """One forward pass. Returns highest-confidence detection row [x1,y1,x2,y2,score,lx1,ly1,...] or None."""
    resize = 1
    img = np.float32(img_raw)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([im_width, im_height, im_width, im_height]).to(device)
    img -= (104, 117, 123)
    img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.no_grad():
        loc, conf, landms = net(img)

    priorbox   = PriorBox(cfg, image_size=(im_height, im_width))
    prior_data = priorbox.forward().to(device).data

    boxes  = (decode(loc.data.squeeze(0), prior_data, cfg['variance']) * scale / resize).cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    scale1  = torch.Tensor([img.shape[3], img.shape[2]] * 5).to(device)
    landms_ = (decode_landm(landms.data.squeeze(0), prior_data, cfg['variance']) * scale1 / resize).cpu().numpy()

    inds = np.where(scores > confidence_threshold)[0]
    if len(inds) == 0:
        return None
    boxes, landms_, scores = boxes[inds], landms_[inds], scores[inds]

    order = scores.argsort()[::-1][:top_k]
    boxes, landms_, scores = boxes[order], landms_[order], scores[order]

    dets    = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep    = py_cpu_nms(dets, nms_threshold)
    dets    = dets[keep][:keep_top_k]
    landms_ = landms_[keep][:keep_top_k]

    dets = np.concatenate((dets, landms_), axis=1)
    return dets[dets[:, 4].argmax()] if len(dets) > 0 else None

# ---------------------------------------------------------------------------
# Two-pass crop pipeline — mirrors detect_pipe.py __main__ loop (lines 252-441)
# ---------------------------------------------------------------------------
def crop_face(image_bytes, params):
    """
    Pass 1: detect face → compute rotation angle → rotate image
    Pass 2: detect face on rotated image → crop → resize → return JPEG bytes
    Returns None if no face detected.
    """
    confidence_threshold = float(params.get('confidence_threshold', 0.02))
    top_k                = int(params.get('top_k', 5000))
    nms_threshold        = float(params.get('nms_threshold', 0.4))
    keep_top_k           = int(params.get('keep_top_k', 750))
    crop_size            = int(params.get('crop_size', 100))
    fill_color           = float(params.get('fill_color', 0.5))

    nparr = np.frombuffer(image_bytes, np.uint8)
    img_original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_original is None:
        raise ValueError("Failed to decode image")
    img_original = resize_square_aspect_cv2(img_original, 400)

    det1 = _detect_best_face(img_original, NET, CFG, DEVICE,
                              confidence_threshold, top_k, nms_threshold, keep_top_k)
    if det1 is None:
        return None

    img_rotated = rotate_image(img_original, det1, fill_color)

    det2 = _detect_best_face(img_rotated, NET, CFG, DEVICE,
                              confidence_threshold, top_k, nms_threshold, keep_top_k)
    if det2 is None:
        return None

    b               = det2
    img_rotated_pil = TF.to_pil_image(img_rotated)
    img_crop        = img_rotated_pil.crop((b[0], b[1], b[2], b[3]))
    img_square_crop = np.ascontiguousarray(img_crop.resize((crop_size, crop_size)))
    ok, buf = cv2.imencode('.jpg', img_square_crop)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()

# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class CropperHandler(http.server.BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"status": "error", "message": "Not found"})

    def do_POST(self):
        if self.path == "/crop":
            self._handle_crop()
        else:
            self._send_json(404, {"status": "error", "message": "Not found"})

    def _handle_crop(self):
        length    = int(self.headers.get("Content-Length", 0))
        body      = json.loads(self.rfile.read(length))
        image_b64 = body.get("image_base64")
        filename  = body.get("filename", "image.jpg")

        if not image_b64:
            self._send_json(400, {"status": "error", "message": "Missing image_base64"})
            return

        try:
            crop_bytes = crop_face(base64.b64decode(image_b64), body)
        except Exception as exc:
            self._send_json(500, {"status": "error", "message": str(exc)})
            return

        if crop_bytes is None:
            self._send_json(404, {"status": "error", "message": "No face detected"})
            return

        stem = os.path.splitext(filename)[0]
        self._send_json(200, {
            "status":       "ok",
            "filename":     f"{stem}_crop_square.jpg",
            "image_base64": base64.b64encode(crop_bytes).decode(),
        })

    def _send_json(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True

# ---------------------------------------------------------------------------
# Startup: load model once into module-level globals
# ---------------------------------------------------------------------------
torch.set_grad_enabled(False)
CFG    = {**(cfg_re50 if CROPPER_NETWORK == "resnet50" else cfg_mnet), 'pretrain': False}
DEVICE = torch.device("cpu" if CROPPER_CPU else "cuda")
NET    = RetinaFace(cfg=CFG, phase='test')
NET    = load_model(NET, CROPPER_MODEL, CROPPER_CPU)
NET.eval()
NET    = NET.to(DEVICE)
cudnn.benchmark = True
print(f"Model loaded. Listening on {CROPPER_HOST}:{CROPPER_PORT}", flush=True)

if __name__ == "__main__":
    server = ThreadedHTTPServer((CROPPER_HOST, CROPPER_PORT), CropperHandler)
    server.serve_forever()
