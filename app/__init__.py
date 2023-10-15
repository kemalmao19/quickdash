from flask import Flask

App = Flask("__name__", static_folder='static', static_url_path='/static')

from app import routes
