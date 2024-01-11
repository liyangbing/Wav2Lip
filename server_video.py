# -*- coding: utf-8 -*-
import json
import logging
import subprocess
import time

from flask import Flask, request

import sys
from inference import InferenceRequest, Wav2LipSyncer
from util import ResponseCode, create_response_custom, response_success


# Set up logger for this module
logger = logging.getLogger(__name__)

model_path = '/personal/model/wav2lip.pth'

cache_path = '/personal/data/'

class Server:
    """
    Server class to provide video streaming services.
    """
    def __init__(self, flask_app: Flask):
        """
        Initializes the Server with a given Flask application.

        :param flask_app: Instance of Flask
        """
        self.flask_app = flask_app

    def register(self, syncer: Wav2LipSyncer):
        """
        Registers the routes for the video streaming services.
        """
        @self.flask_app.route('/video/inference', methods=['GET'])
        def refer_video():
            """
            API endpoint to refer to a video using file_name and push_url.
            """
            audio = request.args.get('audio')
            if not audio:
                return create_response_custom(ResponseCode.BAD_REQUEST.value, 'audio is required')
            
            face = request.args.get('face')
            if not face:
                return create_response_custom(ResponseCode.BAD_REQUEST.value, 'face is required')
            
            push_url = request.args.get('push_url')
            if not push_url:
                return create_response_custom(ResponseCode.BAD_REQUEST.value, 'push_url is required')
            
            # 把audio后缀改为mp4
            outfile = audio.split('.')[0] + '.avi'

            inferenceRequest = InferenceRequest(
                checkpoint_path=model_path,
                face=cache_path + face,
                audio=audio,
                outfile=outfile,
                push_url=push_url
                )
            
            print('inferenceRequest: checkpoint_path: {}, face: {}, audio: {}, outfile: {}'.format(
                inferenceRequest.checkpoint_path, inferenceRequest.face, inferenceRequest.audio, inferenceRequest.outfile
            ))
            start_time = time.time()
            syncer.process_video(inferenceRequest)
            print('process_video cost: {}s'.format(time.time() - start_time))

            result = {'file_name': outfile, 'push_url': push_url}
            return response_success(result)
        
        
        @self.flask_app.route('/video/prepare', methods=['GET'])
        def prepare_video():
            """
            API endpoint to prepare the video file for streaming.
            """
            file = request.args.get('file')
            if not file:
                return create_response_custom(ResponseCode.BAD_REQUEST.value, 'file is required')
            
            result = {'file': file}
            return response_success(result)

# Entry point for the application
if __name__ == '__main__':


    # 设置日志格式
    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(filename)s:%(lineno)d')


    # Instantiate the Flask application
    flask_app = Flask(__name__)

    # Store Flask app context in the config
    app_context = flask_app.app_context()

    # Create Server instance and register routes
    syncer = Wav2LipSyncer(model_path)

    server = Server(flask_app)
    server.register(syncer)

    # Display the URL map for debugging purposes
    with flask_app.test_request_context():
        print(flask_app.url_map)

    # Print the system path for debugging
    print(sys.path)

    # Run the Flask application
    flask_app.run(host='0.0.0.0', port=50004)