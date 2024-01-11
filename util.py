from enum import Enum
from flask import jsonify

# 定义reponse状态码枚举值

class ResponseCode(Enum):
    SUCCESS = 200
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    NOT_FOUND = 404
    INTERNAL_SERVER_ERROR = 500
    IS_EXISTS = 1001
    IS_NOT_EXISTS = 1002
    OVER_LIMIT = 1002

    # 可以根据需要添加其他状态码
    LIVE_TYPE_HAS_SCRIPT = 1110 # live_type下有script
    LIVE_TYPE_NOT_EXIST = 1111 # live_type不存在

    KB_IS_IN_USE = 1120 # kb正在使用

    CHAT_IS_NOT_ACTIVATED = 1130 # 应用未有会话激活

    FILE_TYPE_NOT_SUPPORT = 1140 # 文件类型不支持

    USER_USERNAME_OR_PASSWORD_ERROR = 1150 # 用户名或密码错误
    

def create_response_custom(code=ResponseCode.SUCCESS.value, message=ResponseCode.SUCCESS):
    response = {
        'code': code,
        'message': message
    }

    return response

def create_response(code : ResponseCode):
    response = {
        'code': code.value,
        'message': code.name
    }

    return response



def response_success(data=None):
    response = {
        'code': ResponseCode.SUCCESS.value,
        'message': ResponseCode.SUCCESS.name,
        # data转换为json
        'data': data
    }

    return response