import socket                   # Import socket module

port = 60000                    # Reserve a port for your service.
s = socket.socket()             # Create a socket object
host = '192.168.1.12'     # Get local machine name
s.bind((host, port))            # Bind to the port
s.listen(5)                     # Now wait for client connection.

fnames = ["sign_1.jpg", "sign_2.jpg", "sign_3.jpg", "sign_4.jpg", "sign_5.jpg"]

print ("Server listening....")

def server():
	while True:
		for fname in fnames:
			conn, addr = s.accept()
			with open('signs/'+fname, 'wb') as f:
				print("file opend")
				while True:
					print("receiving data...")
					data = conn.recv(1024)
					#print("data = %s", (data))
					if not data:
						#f.close()
						break
					f.write(data)
		
			#f.close()
		conn.close()
		send_result()
	s.close()

def send_result():
	conn, addr = s.accept()
	data = "result"
	conn.send(data.encode("UTF-8"))
	conn.close()

server()
#send_result()

