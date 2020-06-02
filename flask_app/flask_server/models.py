from datetime import datetime
from flask_server import db, login_manager
from flask_login import UserMixin

@login_manager.user_loader
def Load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id=db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    image_file = db.Column(db.String(20),nullable=False, default='default.jpg')
    password=  db.Column(db.String(60),nullable=False)
    posts= db.relationship('Post', backref= 'author', lazy=True)


class Post(db.Model):
    id=db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), unique=True, nullable=False)
    date_posted = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    tiff = db.Column(db.String(20), nullable=False)
    msi = db.Column(db.String(20), nullable=False)
    rgb = db.Column(db.String(20), nullable=False)
    mask = db.Column(db.String(20), nullable=False)
    infra = db.Column(db.String(20), nullable=False)
    mask_msi = db.Column(db.String(20), nullable=False)
    mask_rgb = db.Column(db.String(20), nullable=False)
    msi_rgb = db.Column(db.String(20), nullable=False)
    mask_infra = db.Column(db.String(20), nullable=False)
    rgb_infra = db.Column(db.String(20), nullable=False)
    msi_infra = db.Column(db.String(20), nullable=False)
    mask_msi_infra = db.Column(db.String(20), nullable=False)
    mask_rgb_infra = db.Column(db.String(20), nullable=False)
    msi_rgb_infra = db.Column(db.String(20), nullable=False)
    msi_rgb_mask = db.Column(db.String(20), nullable=False)
    all_imgs = db.Column(db.String(20), nullable=False)
    kpis = db.Column(db.String(20), nullable=False)
    content = db.Column(db.Text, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'),nullable=False)
