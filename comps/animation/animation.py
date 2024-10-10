# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2013--2023, librosa development team.
# Copyright 1999-2003 The OpenLDAP Foundation, Redwood City, California, USA.  All Rights Reserved.
# Copyright (c) 2012, Anaconda, Inc. All rights reserved.

import os
import pathlib
import platform
import subprocess
import time

import cv2
import ffmpeg
import numpy as np

# Wav2Lip-GFPGAN
import requests
import Wav2Lip.audio as audio
import Wav2Lip.face_detection as face_detection
from basicsr.utils import imwrite
from gfpgan import GFPGANer
from Wav2Lip.models import Wav2Lip

cur_path = pathlib.Path(__file__).parent.resolve()
comps_path = os.path.join(cur_path, "../")

# Habana
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
from utils import *

# GenAIComps
from comps import (
    AnimationDoc,
    Base64ByteStrDoc,
    ServiceType,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)

args = get_args()
print("args: ", args)

# Specify device
if args.device == "hpu" and hthpu.is_available():
    device = "hpu"
elif args.device == "cuda":
    device = "cuda"
elif args.device == "cpu":
    device = "cpu"
else:
    device = "cpu"
    print("Invalid device argument, fall back to cpu")
print("Using {} for inference.".format(device))


# Register the microservice
@register_microservice(
    name="opea_service@animation",
    service_type=ServiceType.ANIMATION,
    endpoint="/v1/animation",
    host="0.0.0.0",
    port=args.port,
    input_datatype=Base64ByteStrDoc,
)
@register_statistics(names=["opea_service@animation"])
def animate(input: Base64ByteStrDoc):
    start = time.time()

    if not os.path.exists("inputs"):
        os.makedirs("inputs")
    if not os.path.exists("temp"):
        os.makedirs("temp")
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    print(args.face, args.audio)
    if not os.path.isfile(args.face):
        raise ValueError("--face argument must be a valid path to video/image file")
    elif args.face.split(".")[-1] in ["jpg", "jpeg", "png"]:
        full_frames = [cv2.imread(args.face)]
        fps = args.fps
    else:
        video_stream = cv2.VideoCapture(args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        print("Reading video frames...")
        full_frames = []
        while True:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            if args.resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1] // args.resize_factor, frame.shape[0] // args.resize_factor))
            if args.rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE())

            y1, y2, x1, x2 = args.crop
            if x2 == -1:
                x2 = frame.shape[1]
            if y2 == -1:
                y2 = frame.shape[0]
            frame = frame[y1:y2, x1:x2]
            full_frames.append(frame)

    print("Number of frames available for inference: " + str(len(full_frames)))

    if args.audio != "None":
        if not args.audio.endswith(".wav"):
            os.makedirs("temp", exist_ok=True)
            print("Extracting raw audio...")
            # command = f"ffmpeg -y -i {args.audio} -strict -2 temp/temp.wav"
            # subprocess.call(command, shell=True)

            ffmpeg.input(args.audio).output("temp/temp.wav", strict="-2").run(overwrite_output=True)
            args.audio = "temp/temp.wav"
    else:
        sr, y = base64_to_int16_to_wav(input.byte_str, "temp/temp.wav")
        args.audio = "temp/temp.wav"

    wav = audio.load_wav(args.audio, 16000)
    mel = audio.melspectrogram(wav)
    print("Mel spectrogram shape: " + str(mel.shape))
    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError("Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again")

    # one single video frame corresponds to 80/25*0.01 = 0.032 seconds (or 32 milliseconds) of audio
    # 30 fps video will match closer to audio, than 25 fps
    mel_chunks = []
    mel_idx_multiplier = 80.0 / fps
    i = 0
    mel_step_size = 16
    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size :])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1
    print(f"Length of mel chunks: {len(mel_chunks)}")

    full_frames = full_frames[: len(mel_chunks)]

    batch_size = args.wav2lip_batch_size
    gen = datagen(args, full_frames.copy(), mel_chunks)

    # iterate over the generator
    for i, (img_batch, mel_batch, frames, coords) in enumerate(
        tqdm(gen, total=int(np.ceil(float(len(mel_chunks)) / batch_size)))
    ):
        if i == 0:
            # load Wav2Lip model
            model = load_model(args)
            print("Wav2Lip Model loaded")

            # load BG sampler if needed
            if args.inference_mode == "wav2clip+gfpgan" and args.bg_upsampler == "realesrgan":
                model_bg_upsampler = load_bg_upsampler(args)
                print("Model BG Sampler loaded")
            # model_bg_upsampler = torch.compile(model_bg_upsampler, backend="hpu_backend")
            # print("Model BG Sampler compiled")
            else:
                model_bg_upsampler = None
                print("Model BG Sampler not loaded")

            # load GFPGAN model if needed
            if args.inference_mode == "wav2clip+gfpgan":
                model_restorer = load_gfpgan(args, model_bg_upsampler)
                print("Model GFPGAN and face helper loaded")
            else:
                model_restorer = None
                print("Model GFPGAN not loaded")

            frame_h, frame_w = full_frames[0].shape[:-1]
            if args.inference_mode == "wav2clip_only":
                out = cv2.VideoWriter("temp/result.avi", cv2.VideoWriter_fourcc(*"DIVX"), fps, (frame_w, frame_h))
            else:
                out = cv2.VideoWriter(
                    "temp/result.avi",
                    cv2.VideoWriter_fourcc(*"DIVX"),
                    fps,
                    (frame_w * args.upscale, frame_h * args.upscale),
                )

        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0

        for p, f, c in tqdm(zip(pred, frames, coords), total=pred.shape[0]):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            f[y1:y2, x1:x2] = p  # patching

            # restore faces and background if necessary
            if args.inference_mode == "wav2clip+gfpgan":
                cropped_faces, restored_faces, f = model_restorer.enhance(
                    f, has_aligned=args.aligned, only_center_face=args.only_center_face, paste_back=True
                )
            out.write(f)
    out.release()

    # command = "ffmpeg -y -i {} -i {} -strict -2 -c:v libx264 -crf 23 -preset medium -c:a aac {}".format(
    #     args.audio, "temp/result.avi", args.outfile
    # )
    # subprocess.call(command, shell=platform.system() != "Windows")

    ffmpeg.output(
        ffmpeg.input(args.audio),
        ffmpeg.input("temp/result.avi"),
        args.outfile,
        strict="-2",
        crf=23,
        vcodec="libx264",
        preset="medium",
        acodec="aac",
    ).run(overwrite_output=True)

    statistics_dict["opea_service@animation"].append_latency(time.time() - start, None)
    # return_str = f"Video generated successfully, check {args.outfile} for the result."
    return AnimationDoc(video_save_path=args.outfile)


if __name__ == "__main__":
    print("Animation initialized.")
    opea_microservices["opea_service@animation"].start()
