'''
Author = Ranjan Satapathy
'''

from collections import OrderedDict as od
from collections import Counter


def input(current_session_name, face_name, speech_name):
	try:
		'''
		Input from face recognition
		'''
		name = open(face_name,"r").read()
		if name not in current_session_name:
			current_session_name[name] += 1  #Addds the new name with a counter
			print"Your name is ", name		
		else:
			current_session_name[name] += 1
			print"Your name is ", name, 'and I have met you before'
	except:
		'''
		open the speech input

		'''
#Do we need number of times robot interacts with a person?
def main():
	if __name__== "__main__" :
		current_session_name = Counter()
		face_name = ''
		speech_name = ''
		read_file(current_session_name, face_name, speech_name) 
