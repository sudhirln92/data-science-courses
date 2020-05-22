# two.py

import one

print('Top level in Two.py')

one.myfunc()

if __name__ == '__main__':
	print('Tow.py is being run directly')
else:
	print('Two.py has been imported')