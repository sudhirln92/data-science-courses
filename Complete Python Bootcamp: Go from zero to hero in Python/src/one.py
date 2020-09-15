# one.py

def myfunc():
	print('Hello')
	print('FUNC() in one')

print('Top level in oen.py')


 # built in variable name
if __name__ == '__main__':
 	print('One.py is being run directly')
else:
 	print('One.py has been imported')