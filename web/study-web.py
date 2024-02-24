import sqlite3

# Sample dictionary
data = {
    "id": [1, 2, 3],
    "name": ["Alice", "Bob", "Charlie"],
    "age": [30, 25, 35]
}

# Connect to SQLite database
conn = sqlite3.connect('data.db')
c = conn.cursor()

# Create table
c.execute('''CREATE TABLE IF NOT EXISTS users
             (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)''')

# Clear existing data
c.execute('''DELETE FROM users''')

# Insert data from dictionary into the table
for i in range(len(data['id'])):
    #c.execute('''INSERT INTO users (id, name, age) VALUES (?, ?, ?)''',
    #          (data['id'][i], data['name'][i], data['age'][i]))            # this command does not work
    c.execute('''CREATE TABLE IF NOT EXISTS users
                (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, age INTEGER)''')


# Commit changes and close connection
conn.commit()
conn.close()


###################### Create Flask Web App ###################################################################################################
from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
db = SQLAlchemy(app)

class User(db.Model):
    #__tablename__ = '"User"'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50))
    age = db.Column(db.Integer)

@app.route('/')
def index():
    users = User.query.all()
    return render_template('index.html', users=users)


###################### Dump the tables to debug ###################################################################################################
from sqlalchemy import create_engine, inspect

# Create an SQLAlchemy engine
engine = create_engine('sqlite:///data.db')

# Create an inspector object
inspector = inspect(engine)

# Get all table names
table_names = inspector.get_table_names()

# Print the table names
print("Tables in the database:")
for table_name in table_names:
    print(table_name)
print(f'\n')


if __name__ == '__main__':
    app.run(debug=True)


###################### Create a Jinja2 Template to display the page ##########################################################################################################
# NOTE: This is stored at the default location of templates/ in our project
# to change this we would use the Flask Constructur as follows:
# app = Flask(__index__, app = Flask(__name__, template_folder='my_custom_templates'))
#
################################################################################################################################
# NOTE:
#WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
#That warning indicates we should use a more secure app to run our website in production; so we perform the following in prod:
#pip install gunicorn
#gunicorn your_app_name:app
