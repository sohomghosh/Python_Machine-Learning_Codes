import re
def isValidEmail(email):
	if re.match(r"[^@]+@[^@]+\.[^@]+", email):
		return True
	else:
		return False
