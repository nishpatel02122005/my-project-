from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(15), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=True)
    password_hash = db.Column(db.String(150), nullable=False)
    
    # Admin fields
    is_admin = db.Column(db.Boolean, default=False)
    admin_role = db.Column(db.String(20), default='user') # 'user', 'superadmin', 'viewer'
    
    # User status and tracking
    join_date = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(20), default='active') # 'active', 'suspended'
    
    predictions = db.relationship('Prediction', backref='user', lazy=True)
    audit_logs = db.relationship('AuditLog', backref='admin', lazy=True)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Input features
    pregnancies = db.Column(db.Float, nullable=False)
    glucose = db.Column(db.Float, nullable=False)
    blood_pressure = db.Column(db.Float, nullable=False)
    skin_thickness = db.Column(db.Float, nullable=False)
    insulin = db.Column(db.Float, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    diabetes_pedigree = db.Column(db.Float, nullable=False)
    age = db.Column(db.Float, nullable=False)
    
    # Prediction results
    probability_positive = db.Column(db.Float, nullable=False)
    result_class = db.Column(db.Integer, nullable=False)

class AuditLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    admin_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True) # ID of admin who performed action
    action = db.Column(db.String(255), nullable=False) # e.g., 'Deleted Prediction #12', 'Suspended User john'
    target_type = db.Column(db.String(50), nullable=True) # 'User', 'Prediction', 'Settings'
    target_id = db.Column(db.Integer, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
