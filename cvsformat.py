with open("test.txt") as f:
	chars = f.read().split()
with open("formatted.csv", 'w') as f:
	f.write("ID,Category\n")
	for i,c in enumerate(chars):
		print c
		f.write(str(i+1)+","+str(c)+"\n")