# app.wsgi
from main import app as application

if __name__ == "__app__":
    application.run(debug=True)
