# Import

from flask import Flask
import pyrebase
import re
import io
import base64
import matplotlib.pyplot as plt, mpld3

#Global

app = Flask(__name__)

@app.route('/')
def hello_world():
	return 'Hello, World!'

@app.route('/plot')
def build_plot():
	plt.plot([3,1,4,1,5], 'ks-', mec='w', mew=5, ms=20)
	mpld3.show()
	return

@app.route('/new_account/<username>/<password>/<email>')
def new_account(username, password, email):
	users = db.child('user').get()
	for user in users.each():
		each_user = user.val()
		print (each_user['username'])
		if each_user['username'] == username:
			return 'Name already use'
	if len(password) < 8:
		return 'Passwords to short'
	pattern = r"\"?([-a-zA-Z0-9.`?{}]+@\w+\.\w+)\"?"
	pattern = re.compile(pattern)
	if not re.match(pattern, email):
		print ("You failed to match %s" % (email))
		return 'wrong mail'
	new_user = {"username": username, "password": password, "email": email}
	print (new_user)
	db.child("user").push(new_user, fire_user['idToken'])
	return 'ok'

@app.route('/login/<username>/<password>')
def login(usernamne, password):
	return 'ok'


@app.route('/test')
def test():
	archer = {'username': 'FlorianBord', 'password': 'Stage4223423', 'email': 'mail@mail.fr'}
	db.child("user").push(archer, fire_user['idToken'])
	return 'Agent Added'