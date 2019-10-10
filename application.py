from flask import Flask, request, jsonify, send_file
from flask_restplus import Api, Resource, fields
from PIL import Image
from werkzeug.datastructures import FileStorage

from classifier import api_controller
import cv2

application = Flask(__name__)
api = Api(
    application,
    version='1.0',
    title='Facial Recognition API',
    description='''
    Send an image to an endpoint and it will return the same image with
    rectangles around the faces detected in the image.
    '''
)
namespace = api.namespace('/', description='Methods')

single_parser = api.parser()
single_parser.add_argument('file', location='files', type=FileStorage, required=True)

@namespace.route('/recognize-faces')
class FacialRecognizer(Resource):
    """Uses the image provided in POST route to feed into open-cv and detect faces"""
    @api.doc(parser=single_parser, description='Upload an image with people\'s faces.')
    def post(self):
        args = single_parser.parse_args()
        file_path = api_controller(args)

        return send_file(file_path)


if __name__ == '__main__':
    application.run()