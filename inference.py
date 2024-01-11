import logging
from os import listdir
import time
import numpy as np
import cv2, os, audio
import subprocess
from tqdm import tqdm
from glob import glob
import torch, face_detection
from models import Wav2Lip
import platform

logger = logging.getLogger(__name__)


class InferenceRequest:
    def __init__(self, checkpoint_path, face, audio, outfile='results/result_voice.mp4', push_url='', static=False,
                 fps=25.0, pads=[0, 10, 0, 0], face_det_batch_size=16, wav2lip_batch_size=128,
                 resize_factor=1, crop=[0, -1, 0, -1], box=[-1, -1, -1, -1], rotate=False, nosmooth=False):
        self.checkpoint_path = checkpoint_path
        self.face = face
        self.audio = audio
        self.outfile = outfile
        self.push_url = push_url
        self.static = static
        self.fps = fps
        self.pads = pads
        self.face_det_batch_size = face_det_batch_size
        self.wav2lip_batch_size = wav2lip_batch_size
        self.resize_factor = resize_factor
        self.crop = crop
        self.box = box
        self.rotate = rotate
        self.nosmooth = nosmooth
        self.img_size = 96
        # Determine if the face input is static based on the file extension
        if os.path.isfile(face) and face.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']:
            self.static = True

mel_step_size = 16

class Wav2LipSyncer:
    def __init__(self, model_path: str):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.face_detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device=self.device)
        self.model = self.load_model(model_path)
        self.frames = []
        self.mel_chunks = []
        self.cache = {}
        
    def load_model(self, checkpoint_path):
        model = Wav2Lip()
        start_time = time.time()
        print("Loading checkpoint from: {}".format(checkpoint_path))
        if self.device == 'cuda':
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        s = checkpoint["state_dict"]
        new_s = {k.replace('module.', ''): v for k, v in s.items()}
        model.load_state_dict(new_s)
        model = model.to(self.device)
        logger.info("Loaded checkpoint: {} in {}s".format(checkpoint_path, time.time() - start_time))
        return model.eval()

    def get_smoothened_boxes(self, boxes, T):
        for i in range(len(boxes)):
            if i + T > len(boxes):
               window = boxes[len(boxes) - T:]
            else:
                window = boxes[i : i + T]
            boxes[i] = np.mean(window, axis=0)
        return boxes

    def face_detect(self, images, request: InferenceRequest):
        batch_size = request.face_det_batch_size
        
        while 1:
            predictions = []
            try:
                for i in tqdm(range(0, len(images), batch_size)):
                    predictions.extend(self.face_detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
            except RuntimeError:
                if batch_size == 1: 
                    raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
                batch_size //= 2
                print('Recovering from OOM error; New batch size: {}'.format(batch_size))
                continue
            break

        results = []
        pady1, pady2, padx1, padx2 = request.pads
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
        if not request.nosmooth: boxes = self.get_smoothened_boxes(boxes, T=5)
        results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

        return results 

    def datagen(self, frames, mels, request: InferenceRequest):
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if request.box[0] == -1:
            if not request.static:
                face_det_results = self.face_detect(frames, request) # BGR2RGB for CNN face detection
            else:
                face_det_results = self.face_detect([frames[0]], request)
        else:
            print('Using the specified bounding box instead of face detection...')
            y1, y2, x1, x2 = request.box
            face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

        for i, m in enumerate(mels):
            idx = 0 if request.static else i%len(frames)
            frame_to_save = frames[idx].copy()
            face, coords = face_det_results[idx].copy()

            face = cv2.resize(face, (request.img_size, request.img_size))
                
            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)

            if len(img_batch) >= request.wav2lip_batch_size:
                img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

                img_masked = img_batch.copy()
                img_masked[:, request.img_size//2:] = 0

                img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
                mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

                yield img_batch, mel_batch, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if len(img_batch) > 0:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, request.img_size//2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch

    def process_video(self, request: InferenceRequest):
        start_time = time.time()
        if not os.path.isfile(request.face):
            raise ValueError('--face argument must be a valid path to video/image file')

        elif request.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
            full_frames = [cv2.imread(request.face)]
            fps = request.fps

        else:
            video_stream = cv2.VideoCapture(request.face)
            fps = video_stream.get(cv2.CAP_PROP_FPS)

            logger.info('Reading video frames...')

            full_frames = self.get_full_frames(request, video_stream)
        logger.info('Number of frames available for inference: {}'.format(len(full_frames)) + " time: " + str(time.time() - start_time))

        if not request.audio.endswith('.wav'):
            print('Extracting raw audio...')
            command = 'ffmpeg -y -i {} -strict -2 {}'.format(request.audio, 'temp/temp.wav')

            subprocess.call(command, shell=True)
            request.audio = 'temp/temp.wav'

        wav = audio.load_wav(request.audio, 16000)
        mel = audio.melspectrogram(wav)

        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

        mel_chunks = []
        mel_idx_multiplier = 80./fps 
        i = 0
        while 1:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
            i += 1

        logger.info('Length of mel chunks: {}'.format(len(mel_chunks)) + " time: " + str(time.time() - start_time))

        full_frames = full_frames[:len(mel_chunks)]

        batch_size = request.wav2lip_batch_size
        gen = self.datagen(full_frames.copy(), mel_chunks, request)

        logger.info('Starting to generate the frames' + " time: " + str(time.time() - start_time))

        self.infer(request, full_frames, fps, mel_chunks, batch_size, gen)

        logger.info('request.outfile: {}'.format(request.outfile)  + " time: " + str(time.time() - start_time))

        # 假设 request 是一个对象，包含了 audio, outfile 和 push_url 属性
        audio_input = request.audio
        video_input = request.outfile
        push_url = request.push_url

       # 定义临时中间文件的路径
        temp_file = '/personal/cache/temp_output' + str(time.time()) + '.flv'

        # 第一个 ffmpeg 命令：处理音频和视频文件
        command1 = [
            'ffmpeg', '-y',
            '-i', audio_input,
            '-i', video_input,
            '-strict', 'experimental',
            '-q:v', '1',
            '-c:v', 'libx264',
            '-preset', 'veryfast',
            '-maxrate', '3000k',
            '-bufsize', '6000k',
            '-pix_fmt', 'yuv420p',
            '-g', '50',
            '-c:a', 'aac',
            '-b:a', '160k',
            '-ac', '2',
            '-ar', '44100',
            '-f', 'flv',
            '-loglevel', 'error',
            temp_file  # 输出到临时文件
        ]

        # 执行第一个 ffmpeg 命令并等待其完成
        subprocess.run(command1, check=True)

        # 确保临时文件存在
        if not os.path.isfile(temp_file):
            raise FileNotFoundError(f"The temporary file {temp_file} was not created by ffmpeg.")

        # 第二个 ffmpeg 命令：读取临时文件并推送到流媒体服务器
        command2 = [
            'ffmpeg',
            '-re',
            '-i', temp_file,  # 从临时文件读取输入
            '-f', 'flv',
            '-c', 'copy',
            '-f', 'flv',
            '-loglevel', 'error',
            push_url
        ]

        # 执行第二个 ffmpeg 命令并等待其完成
        subprocess.run(command2, check=True)

        # 清理临时文件
        os.remove(temp_file)

        logger.info('Push video to push_url: {}'.format(request.push_url) + " time: " + str(time.time() - start_time))

    def infer(self, request: InferenceRequest, full_frames, fps, mel_chunks, batch_size, gen):
        for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
                                                total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
            if i == 0:
                frame_h, frame_w = full_frames[0].shape[:-1]
                out = cv2.VideoWriter(request.outfile,
                                        cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.device)

            with torch.no_grad():
                pred = self.model(mel_batch, img_batch)

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
            
            for p, f, c in zip(pred, frames, coords):
                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

                f[y1:y2, x1:x2] = p
                out.write(f)

        out.release()

    

    def get_full_frames(self, request: InferenceRequest, video_stream: cv2.VideoCapture):
         # Check if we already have the frames in cache
        if request.face in self.cache:
            return self.cache[request.face]
        
        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            if request.resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1]//request.resize_factor, frame.shape[0]//request.resize_factor))

            if request.rotate:
                frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

            y1, y2, x1, x2 = request.crop
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]

            full_frames.append(frame)

        # Cache the frames
        self.cache[request.face] = full_frames
        return full_frames

def main(request: InferenceRequest):
    syncer = Wav2LipSyncer(request)
    syncer.process_video()

if __name__ == '__main__':
    request = InferenceRequest(
    checkpoint_path='../model/wav2lip_gan.pth',
    face='../data/ranran2.mp4',
    audio='../data/ranran-jp.wav'
    )
    # ... other parameters as needed

    main(request)
