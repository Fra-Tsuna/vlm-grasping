import loader 
import detection
import socket
import importlib
 
loader_instance = loader.Loader()

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_address = ('localhost', 10000)
server_socket.bind(server_address)

server_socket.listen(1)


while True:
    print('In attesa di una connessione...')
    connection, client_address = server_socket.accept()

    try:        
        data = connection.recv(1024)
        message = data.decode()
        if message != "":
            print(message)
        
        if "detection" in message:
            print('reload the module and run')
            importlib.reload(detection)
            detection_instance = detection.Detection()
            detection_instance.set_loader(loader_instance)
            print("Detection completed")
        else:
            pass
    except:
        pass
