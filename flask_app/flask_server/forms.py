from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from flask_login import current_user
from wtforms import StringField, PasswordField, SubmitField, BooleanField, TextAreaField, SelectField, IntegerField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError
from flask_server.models import User


class RegistrationForm(FlaskForm):
    username = StringField('Username',
                           validators=[DataRequired(), Length(min=2, max=20)])
    email = StringField('Email',
                        validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password',
                                     validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Sign Up')

    def validate_username(self, username):
        user=User.query.filter_by(username= username.data).first()
        if user:
            raise ValidationError('That username is taken. Please choose a different one')

    def validate_email(self, email):
        user=User.query.filter_by(email= email.data).first()
        if user:
            raise ValidationError('That email is taken. Please choose a different one')


class LoginForm(FlaskForm):
    email = StringField('Email',
                        validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember = BooleanField('Remember Me')
    submit = SubmitField('Login')

class UpdateAccountForm(FlaskForm):
    username = StringField('Username',
                           validators=[DataRequired(), Length(min=2, max=20)])
    email = StringField('Email',
                        validators=[DataRequired(), Email()])
    picture = FileField('Update Profile Picture', validators=[FileAllowed(['jpg', 'png'])])
    submit = SubmitField('update')

    def validate_username(self, username):
        if username.data!= current_user.username:
            user=User.query.filter_by(username= username.data).first()
            if user:
                raise ValidationError('That username is taken. Please choose a different one')

    def validate_email(self, email):
        if email.data!= current_user.email:
            user=User.query.filter_by(email= email.data).first()
            if user:
                raise ValidationError('That email is taken. Please choose a different one')

class PostForm(FlaskForm):
    title = StringField('Title of prediction', validators=[DataRequired()])
    content = TextAreaField('Description of prediction', validators=[DataRequired()])
    picture = FileField('File (.tif) containing area information to be predicted', validators=[FileAllowed(['tif'])])
    country = SelectField('Country of prediction', choices = [('Guinea', 'Guinea'), ('Congo', 'Congo')], validators=[DataRequired()])
    color1 = TextAreaField('Color of virgin forest pixels', default='#064518', validators=[DataRequired()])
    color2 = TextAreaField('Color of deforested forest pixels', default='#DEDC93', validators=[DataRequired()])
    color3 = TextAreaField('Color of no forest pixels', default='#A3A39B', validators=[DataRequired()])
    submit= SubmitField('Post')
