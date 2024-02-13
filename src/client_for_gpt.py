#!/usr/bin/env python

import rospy
import io
import base64  
import socket
from sensor_msgs.msg import Image

from cv_bridge import CvBridge
from PIL import Image as PILImage

SERVER_HOST = '192.168.130.202'
SERVER_PORT = 12346
bridge = CvBridge()

def convert(cv2_img):
    pil_im = PILImage.fromarray(cv2_img)
    output = io.BytesIO()
    pil_im.save(output, format="png")

    image_as_string = base64.b64encode(output.getvalue())

    return image_as_string

       
def ScannedImageCallback(img_msg):
    try: 
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((SERVER_HOST, SERVER_PORT))

        cv2_img = bridge.imgmsg_to_cv2(img_msg, "bgr8")
        image_as_string = convert(cv2_img)
        size = len(image_as_string)
        
        client_socket.sendall("SIZE "+str(size))
        answer = client_socket.recv(4096)
        print("ACK: " + answer.decode('utf-8'))

        if answer == 'GOT SIZE':
            print("ciao")
            client_socket.sendall(image_as_string.encode('utf-8'))
            print("miao")
            answer = client_socket.recv(4096)
            print("ACK: " + answer.decode('utf-8'))

            if answer == 'GOT IMAGE' :
                client_socket.sendall("BYE BYE ")
                print('Image successfully send to server')

        client_socket.send(image_as_string.encode('utf-8'))

        ack_msg_back = client_socket.recv(1024)
        

    except Exception as e:
        print("Error sending Images: ", e)

    finally:
        client_socket.close()

    return 


# def send_joint_angles(host, port, joint_angles):
#     try:
#         client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         client_socket.connect((host, port))

#         joint_angles_str = ','.join(map(str, joint_angles))

#         client_socket.send(joint_angles_str.encode('utf-8'))
        
#         ack_msg_back = client_socket.recv(1024)
        
#         print("ACK: " + ack_msg_back.decode('utf-8'))

#     except Exception as e:
#         print("Error sending joint angles:", str(e))

#     finally:
#         client_socket.close()


def listener():

    rospy.Subscriber("/client_image_topic", Image, ScannedImageCallback)
    rospy.spin()

if __name__ == '__main__':

    rospy.init_node('client_for_gpt', anonymous=True)
    listener()
