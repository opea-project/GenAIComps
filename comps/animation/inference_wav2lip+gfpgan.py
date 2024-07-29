#%% Imports
# wav2lip
from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse
import Wav2Lip.audio as audio 
import Wav2Lip.face_detection as face_detection
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch
from Wav2Lip.models import Wav2Lip
import platform

# gfpgan
from basicsr.utils import imwrite
from GFPGAN.gfpgan import GFPGANer

# ctao 7/11
import habana_frameworks.torch.hpu as hthpu
import habana_frameworks.torch.core as htcore
device = "hpu" if hthpu.is_available() else "cpu"
print('Using {} for inference.'.format(device))
import pdb
import time


#%% Argument parsing
parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')
# General config
parser.add_argument('--inference_mode', type=str, choices=['wav2clip_only', 'wav2clip+gfpgan'],
					default='wav2clip+gfpgan', help='whether to use just wav2clip or include gfpgan')
# Wav2Lip config
parser.add_argument('--checkpoint_path', type=str, 
					help='Name of saved checkpoint to load weights from', required=True)
parser.add_argument('--face', type=str, 
					help='Filepath of video/image that contains faces to use', required=True)
parser.add_argument('--audio', type=str, 
					help='Filepath of video/audio file to use as raw audio source', required=True)
parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.', 
								default='results/result_voice.mp4')
parser.add_argument('--static', type=bool, 
					help='If True, then use only first video frame for inference', default=False)
parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)', 
					default=25., required=False)
parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], 
					help='Padding (top, bottom, left, right). Please adjust to include chin at least')
parser.add_argument('--face_det_batch_size', type=int, 
					help='Batch size for face detection', default=16)
parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=128)
parser.add_argument('--resize_factor', default=1, type=int, 
			help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')
parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], 
					help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. ' 
					'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')
parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], 
					help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
					'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')
parser.add_argument('--rotate', default=False, action='store_true',
					help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
					'Use if you get a flipped result, despite feeding a normal looking video')
parser.add_argument('--nosmooth', default=False, action='store_true',
					help='Prevent smoothing face detections over a short temporal window')
# GFPGAN
parser.add_argument('-i', '--input', type=str, default='inputs/whole_imgs', help='Input image or folder. Default: inputs/whole_imgs')
parser.add_argument('-o', '--output', type=str, default='results', help='Output folder. Default: results')
# we use version to select models, which is more user-friendly
parser.add_argument('--img_size', type=int, default=96, help='size to reshape the detected face')
parser.add_argument('-v', '--version', type=str, default='1.3', help='GFPGAN model version. Option: 1 | 1.2 | 1.3. Default: 1.3')
parser.add_argument('-s', '--upscale', type=int, default=2, help='The final upsampling scale of the image. Default: 2')
parser.add_argument('--bg_upsampler', type=str, default='realesrgan', help='background upsampler. Default: realesrgan')
parser.add_argument( '--bg_tile', type=int, default=400, help='Tile size for background sampler, 0 for no tile during testing. Default: 400')
parser.add_argument('--suffix', type=str, default=None, help='Suffix of the restored faces')
parser.add_argument('--only_center_face', action='store_true', help='Only restore the center face')
parser.add_argument('--aligned', action='store_true', help='Input are aligned faces')
parser.add_argument('--save_faces', default=False, help='Save the restored faces')
parser.add_argument('--ext', type=str, default='auto', help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs. Default: auto')
args = parser.parse_args()

if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
	args.static = True

os.makedirs(args.output, exist_ok=True)


#%% Custom functions
def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes


def face_detect(images):
	detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
											flip_input=False, device=device)

	batch_size = args.face_det_batch_size
	
	while 1:
		predictions = []
		try:
			for i in tqdm(range(0, len(images), batch_size)):
				predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
		except RuntimeError:
			if batch_size == 1: 
				raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
			batch_size //= 2
			print('Recovering from OOM error; New batch size: {}'.format(batch_size))
			continue
		break

	results = []
	pady1, pady2, padx1, padx2 = args.pads
	for rect, image in zip(predictions, images):
		if rect is None:
			cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
			raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

		y1 = max(0, rect[1] - pady1)
		y2 = min(image.shape[0], rect[3] + pady2)
		x1 = max(0, rect[0] - padx1)
		x2 = min(image.shape[1], rect[2] + padx2)
		
		results.append([x1, y1, x2, y2])

	boxes = np.array(results)
	if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
	results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

	del detector
	return results 


def datagen(frames, mels):
	img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if args.box[0] == -1:
		if not args.static:
			face_det_results = face_detect(frames) # BGR2RGB for CNN face detection
		else:
			face_det_results = face_detect([frames[0]])
	else:
		print('Using the specified bounding box instead of face detection...')
		y1, y2, x1, x2 = args.box
		face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

	for i, m in enumerate(mels):
		idx = 0 if args.static else i%len(frames)
		frame_to_save = frames[idx].copy()
		face, coords = face_det_results[idx].copy()

		face = cv2.resize(face, (args.img_size, args.img_size))
			
		img_batch.append(face)
		mel_batch.append(m)
		frame_batch.append(frame_to_save)
		coords_batch.append(coords)

		if len(img_batch) >= args.wav2lip_batch_size:
			img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

			img_masked = img_batch.copy()
			img_masked[:, args.img_size//2:] = 0

			img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
			mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

			yield img_batch, mel_batch, frame_batch, coords_batch
			img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if len(img_batch) > 0:
		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

		img_masked = img_batch.copy()
		img_masked[:, args.img_size//2:] = 0

		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
		mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

		yield img_batch, mel_batch, frame_batch, coords_batch


def _load(checkpoint_path):
	if device == 'cuda':
		checkpoint = torch.load(checkpoint_path)
	else:
		checkpoint = torch.load(checkpoint_path,
								map_location=lambda storage, loc: storage)
	return checkpoint


def load_model(path):
	model = Wav2Lip()
	print("Load checkpoint from: {}".format(path))
	checkpoint = _load(path)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)
	return model.eval().to(device)


def load_bg_upsampler(args):
	if (not torch.cuda.is_available()) or (not hthpu.is_available()):  # CPU
		import warnings
		warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
						'If you really want to use it, please modify the corresponding codes.')
		bg_upsampler = None
	else:
		from basicsr.archs.rrdbnet_arch import RRDBNet
		from realesrgan import RealESRGANer
		model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2) 
		bg_upsampler = RealESRGANer(
			scale=2,
			model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
			model=model,
			tile=args.bg_tile,
			tile_pad=10,
			pre_pad=0,
			half=True)  # need to set False in CPU mode
	return bg_upsampler.eval().to(device)


def load_gfpgan(args, bg_upsampler):
	if args.version == '1':
		arch = 'original'
		channel_multiplier = 1
		model_name = 'GFPGANv1'
	elif args.version == '1.2':
		arch = 'clean'
		channel_multiplier = 2
		model_name = 'GFPGANCleanv1-NoCE-C2'
	elif args.version == '1.3':
		arch = 'clean'
		channel_multiplier = 2
		model_name = 'GFPGANv1.3'
	else:
		raise ValueError(f'Wrong model version {args.version}.')
	
	# determine model path
	model_path = path.join('GFPGAN/experiments/pretrained_models', model_name + '.pth')
	if not path.isfile(model_path):
		model_path = path.join('GFPGAN/realesrgan/weights', model_name + '.pth')
	if not path.isfile(model_path):
		raise ValueError(f'Model {model_name} does not exist')
	
	restorer = GFPGANer(model_path=model_path,
						upscale=args.upscale,
						arch=arch,
						channel_multiplier=channel_multiplier,
						bg_upsampler=bg_upsampler,
						device=device)
	
	# torch.compile
	# restorer.face_helper.face_det = torch.compile(restorer.face_helper.face_det, backend="hpu_backend")
	# restorer.face_helper.face_parse = torch.compile(restorer.face_helper.face_parse, backend="hpu_backend")
	# restorer.gfpgan = torch.compile(restorer.gfpgan, backend="hpu_backend") # some compilation issue
	# print("Model GFPGAN and face helper compiled")

	return restorer


def restore_images(args):
	return


#%% Main function
def main():
	print(args.face, args.audio)
	if not os.path.isfile(args.face):
        #try:
        #    args.face.save('temp/face.png', 'PNG')
        #    print('Face saved')
        #    args.face = 'temp/face.png')
        # except:
	    raise ValueError('--face argument must be a valid path to video/image file')

	elif args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
		full_frames = [cv2.imread(args.face)]
		fps = args.fps

	else:
		video_stream = cv2.VideoCapture(args.face)
		fps = video_stream.get(cv2.CAP_PROP_FPS)

		print('Reading video frames...')

		full_frames = []
		while 1:
			still_reading, frame = video_stream.read()
			if not still_reading:
				video_stream.release()
				break
			if args.resize_factor > 1:
				frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))

			if args.rotate:
				frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

			y1, y2, x1, x2 = args.crop
			if x2 == -1: x2 = frame.shape[1]
			if y2 == -1: y2 = frame.shape[0]

			frame = frame[y1:y2, x1:x2]

			full_frames.append(frame)

	print ("Number of frames available for inference: "+str(len(full_frames)))

	if not args.audio.endswith('.wav'):
		os.makedirs('temp', exist_ok=True)
		print('Extracting raw audio...')
		command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, 'temp/temp.wav')

		subprocess.call(command, shell=True)
		args.audio = 'temp/temp.wav'

	wav = audio.load_wav(args.audio, 16000)
	mel = audio.melspectrogram(wav)
	print(mel.shape)

	if np.isnan(mel.reshape(-1)).sum() > 0:
		raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

	mel_chunks = []
	mel_idx_multiplier = 80./fps 
	i = 0
	mel_step_size = 16
	while 1:
		start_idx = int(i * mel_idx_multiplier)
		if start_idx + mel_step_size > len(mel[0]):
			mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
			break
		mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
		i += 1

	print("Length of mel chunks: {}".format(len(mel_chunks)))

	full_frames = full_frames[:len(mel_chunks)]

	batch_size = args.wav2lip_batch_size
	gen = datagen(full_frames.copy(), mel_chunks)

	for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
											total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
		if i == 0:
			model = load_model(args.checkpoint_path)
			print ("Model loaded")

			# ctao 7/12
			# model = torch.compile(model, backend="hpu_backend")
			# print("Model compiled")

			# ctao 7/12
			if args.inference_mode == 'wav2clip+gfpgan' and args.bg_upsampler == 'realesrgan':
				model_bg_upsampler = load_bg_upsampler(args)
				print("Model BG Sampler loaded")
				# model_bg_upsampler = torch.compile(model_bg_upsampler, backend="hpu_backend")
				# print("Model BG Sampler compiled")
			else:
				model_bg_upsampler = None
				print("Model BG Sampler not loaded")

			# ctao 7/12
			if args.inference_mode == 'wav2clip+gfpgan':
				restorer = load_gfpgan(args, model_bg_upsampler)
				print("Model GFPGAN and face helper loaded")

			frame_h, frame_w = full_frames[0].shape[:-1]
			if args.inference_mode == 'wav2clip_only':
				out = cv2.VideoWriter('temp/result.avi', 
									  cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))
			else:
				out = cv2.VideoWriter('temp/result.avi', 
									  cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w * args.upscale,
												  							 frame_h * args.upscale))

		# pdb.set_trace() # passed

		img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
		mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

		with torch.no_grad():
			pred = model(mel_batch, img_batch)

		pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
		
		for p, f, c in tqdm(zip(pred, frames, coords), total=pred.shape[0]):
			y1, y2, x1, x2 = c
			p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
			f[y1:y2, x1:x2] = p # patching

			# ctao 7/12 - before writing to out, apply gfpgan
			# pdb.set_trace() # passed
			
			# restore faces and background if necessary
			if args.inference_mode == "wav2clip+gfpgan":
				cropped_faces, restored_faces, f = restorer.enhance(f, 
																	has_aligned=args.aligned, 
																	only_center_face=args.only_center_face, 
																	paste_back=True)
				# pdb.set_trace() # passed
			out.write(f)

	out.release()

	# pdb.set_trace() # passed

	# Add audio
	# command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 -c:v h264_v4l2m2m -c:a aac {}'.format(args.audio, 'temp/result.avi', args.outfile)
	command = 'ffmpeg -y -i {} -i {} -strict -2 -c:v libx264 -crf 23 -preset medium -c:a aac {}'.format(args.audio, 'temp/result.avi', args.outfile)

	subprocess.call(command, shell=platform.system() != 'Windows')


if __name__ == '__main__':
	start_time = time.time()
	main()
	end_time = time.time()
	print(f"Face animation took {(end_time - start_time):.2f} seconds")
