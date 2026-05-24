import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from models import db, User, Prediction, AuditLog

app = Flask(__name__)
app.config['SECRET_KEY'] = 'super-secret-key-change-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///diabetes_app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash('Unauthorized access. Admin privileges required.', 'error')
            return redirect(url_for('predict'))
        return f(*args, **kwargs)
    return decorated_function

def superadmin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin or current_user.admin_role != 'superadmin':
            flash('Unauthorized access. Superadmin privileges required.', 'error')
            return redirect(url_for('admin_dashboard'))
        return f(*args, **kwargs)
    return decorated_function

def log_audit(action, target_type=None, target_id=None):
    if current_user.is_authenticated and current_user.is_admin:
        log = AuditLog(
            admin_id=current_user.id,
            action=action,
            target_type=target_type,
            target_id=target_id
        )
        db.session.add(log)
        db.session.commit()

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'model.pkl')
ml_model = None

def get_model():
    global ml_model
    if ml_model is None:
        try:
            ml_model = joblib.load(MODEL_PATH)
        except Exception as e:
            print(f"Error loading model: {e}")
    return ml_model

@app.cli.command("init-db")
def init_db_command():
    """Clear the existing data and create new tables."""
    db.drop_all()
    db.create_all()
    if not User.query.filter_by(username='admin').first():
        admin = User(
            username='admin',
            password_hash=generate_password_hash('admin123', method='pbkdf2:sha256'),
            is_admin=True,
            admin_role='superadmin',
            email='admin@example.com'
        )
        db.session.add(admin)
        db.session.commit()
        print("Initialized the database and created default admin (admin/admin123).")
    else:
        print("Database already initialized.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('predict'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            flash('Logged in successfully.', 'success')
            if user.is_admin:
                return redirect(url_for('admin_dashboard'))
            return redirect(url_for('predict'))
            
        flash('Invalid username or password.', 'error')
    return render_template('auth/login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('predict'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        if user:
            flash('Username already exists.', 'error')
            return redirect(url_for('register'))
            
        new_user = User(
            username=username,
            password_hash=generate_password_hash(password, method='pbkdf2:sha256')
        )
        db.session.add(new_user)
        db.session.commit()
        
        login_user(new_user)
        flash('Registration successful! Welcome.', 'success')
        return redirect(url_for('predict'))
        
    return render_template('auth/register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        try:
            features = {
                'Pregnancies': float(request.form['pregnancies']),
                'Glucose': float(request.form['glucose']),
                'BloodPressure': float(request.form['blood_pressure']),
                'SkinThickness': float(request.form['skin_thickness']),
                'Insulin': float(request.form['insulin']),
                'BMI': float(request.form['bmi']),
                'DiabetesPedigreeFunction': float(request.form['diabetes_pedigree']),
                'Age': float(request.form['age'])
            }
            
            model = get_model()
            if not model:
                flash('Model not loaded. Contact administrator.', 'error')
                return redirect(url_for('predict'))
                
            df = pd.DataFrame([features])
            prediction_class = int(model.predict(df)[0])
            probabilities = model.predict_proba(df)[0]
            prob_positive = float(probabilities[1])
            
            new_prediction = Prediction(
                user_id=current_user.id,
                pregnancies=features['Pregnancies'],
                glucose=features['Glucose'],
                blood_pressure=features['BloodPressure'],
                skin_thickness=features['SkinThickness'],
                insulin=features['Insulin'],
                bmi=features['BMI'],
                diabetes_pedigree=features['DiabetesPedigreeFunction'],
                age=features['Age'],
                probability_positive=prob_positive,
                result_class=prediction_class
            )
            db.session.add(new_prediction)
            db.session.commit()
            
            prediction_data = {
                'probability_positive': prob_positive * 100,
                'probability_negative': float(probabilities[0]) * 100,
                'result_class': prediction_class,
                'features': features
            }
            
            if prediction_class == 1:
                ref = {}
                age = features['Age']
                bmi = features['BMI']
                glucose = features['Glucose']
                insulin = features['Insulin']
                
                if age < 18:
                    ref['age'] = ("Insulin therapy primarily, Metformin sometimes considered", "amber")
                elif age <= 40:
                    ref['age'] = ("Metformin first line, Ozempic/Jardiance if needed", "green")
                elif age <= 60:
                    ref['age'] = ("Metformin + combination therapy, monitor kidney/heart", "amber")
                else:
                    ref['age'] = ("Gentler options like Januvia preferred, lower doses", "amber")
                    
                if bmi < 25:
                    ref['bmi'] = ("Insulin or Januvia recommended", "amber")
                elif bmi <= 30:
                    ref['bmi'] = ("Metformin first choice", "green")
                else:
                    ref['bmi'] = ("Ozempic or Semaglutide preferred — reduces weight and glucose", "amber")
                    
                if glucose < 140:
                    ref['glucose'] = ("Lifestyle + Metformin alone may be enough", "green")
                elif glucose <= 199:
                    ref['glucose'] = ("Metformin + one add-on drug", "amber")
                elif glucose <= 249:
                    ref['glucose'] = ("Dual or triple drug combination likely needed", "red")
                else:
                    ref['glucose'] = ("Insulin usually required immediately", "red")
                    
                if insulin <= 50:
                    ref['kidney'] = ("Most medications safe", "green")
                elif insulin <= 150:
                    ref['kidney'] = ("Reduce Metformin dose, monitor carefully", "amber")
                else:
                    ref['kidney'] = ("Consult doctor before any medication", "red")
                    
                prediction_data['medicine_reference'] = ref
            
            return render_template('result.html', data=prediction_data)
            
        except Exception as e:
            flash(f'Error processing prediction: {str(e)}', 'error')
            return redirect(url_for('predict'))
            
    return render_template('predict.html')

from datetime import datetime, date

@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    total_users = User.query.count()
    total_predictions = Prediction.query.count()
    high_risk_count = Prediction.query.filter_by(result_class=1).count()
    
    today = date.today()
    todays_activity = Prediction.query.filter(Prediction.timestamp >= today).count()
    
    # Risk Distribution (Using probability_positive)
    # Low: < 0.33, Medium: 0.33 - 0.66, High: > 0.66
    all_preds = Prediction.query.all()
    risk_dist = [0, 0, 0] # Low, Medium, High
    for p in all_preds:
        if p.probability_positive < 0.33:
            risk_dist[0] += 1
        elif p.probability_positive < 0.66:
            risk_dist[1] += 1
        else:
            risk_dist[2] += 1
            
    recent_predictions = Prediction.query.order_by(Prediction.timestamp.desc()).limit(10).all()
    
    return render_template('admin/dashboard.html', 
                           total_users=total_users, 
                           total_predictions=total_predictions,
                           high_risk_count=high_risk_count,
                           todays_activity=todays_activity,
                           risk_dist=risk_dist,
                           predictions=recent_predictions)

@app.route('/admin/users')
@admin_required
def admin_users():
    users = User.query.all()
    return render_template('admin/users.html', users=users)

@app.route('/admin/users/<int:user_id>/toggle_status', methods=['POST'])
@superadmin_required
def toggle_user_status(user_id):
    user = User.query.get_or_404(user_id)
    if user.is_admin:
        flash("Cannot suspend another admin.", "error")
        return redirect(url_for('admin_users'))
    user.status = 'suspended' if user.status == 'active' else 'active'
    db.session.commit()
    log_audit(f"{'Suspended' if user.status == 'suspended' else 'Activated'} user {user.username}", "User", user.id)
    flash(f"User {user.username} status changed to {user.status}.", "success")
    return redirect(url_for('admin_users'))

@app.route('/admin/predictions')
@admin_required
def admin_predictions():
    predictions = Prediction.query.order_by(Prediction.timestamp.desc()).all()
    return render_template('admin/predictions.html', predictions=predictions)

@app.route('/admin/predictions/<int:pred_id>/delete', methods=['POST'])
@superadmin_required
def delete_prediction(pred_id):
    pred = Prediction.query.get_or_404(pred_id)
    db.session.delete(pred)
    db.session.commit()
    log_audit(f"Deleted prediction record #{pred_id}", "Prediction", pred_id)
    flash(f"Prediction {pred_id} deleted successfully.", "success")
    return redirect(url_for('admin_predictions'))

@app.route('/admin/predictions/export')
@admin_required
def export_predictions():
    import csv
    from io import StringIO
    from flask import Response
    
    si = StringIO()
    cw = csv.writer(si)
    cw.writerow(['ID', 'Username', 'Age', 'BMI', 'Glucose', 'BloodPressure', 'Pregnancies', 'SkinThickness', 'Insulin', 'DiabetesPedigree', 'Result', 'Pos_Prob', 'Timestamp'])
    
    predictions = Prediction.query.all()
    for p in predictions:
        cw.writerow([p.id, p.user.username, p.age, p.bmi, p.glucose, p.blood_pressure, 
                     p.pregnancies, p.skin_thickness, p.insulin, p.diabetes_pedigree,
                     'Diabetic' if p.result_class == 1 else 'Non-Diabetic', 
                     f"{p.probability_positive*100:.1f}%", p.timestamp])
                     
    output = Response(si.getvalue(), mimetype='text/csv')
    output.headers["Content-Disposition"] = "attachment; filename=predictions_export.csv"
    log_audit("Exported predictions to CSV", "Prediction")
    return output

@app.route('/admin/reports')
@admin_required
def admin_reports():
    preds = Prediction.query.all()
    diabetic_count = sum(1 for p in preds if p.result_class == 1)
    non_diabetic_count = len(preds) - diabetic_count
    
    avg_glucose = sum(p.glucose for p in preds) / len(preds) if preds else 0
    avg_bmi = sum(p.bmi for p in preds) / len(preds) if preds else 0
    avg_age = sum(p.age for p in preds) / len(preds) if preds else 0
    
    return render_template('admin/reports.html', 
                          diabetic=diabetic_count,
                          non_diabetic=non_diabetic_count,
                          avg_glucose=avg_glucose,
                          avg_bmi=avg_bmi,
                          avg_age=avg_age)

@app.route('/admin/audit_logs')
@admin_required
def admin_audit_logs():
    logs = AuditLog.query.order_by(AuditLog.timestamp.desc()).all()
    return render_template('admin/audit_logs.html', logs=logs)

@app.route('/admin/settings', methods=['GET', 'POST'])
@admin_required
def admin_settings():
    if request.method == 'POST':
        current_pw = request.form.get('current_password')
        new_pw = request.form.get('new_password')
        
        if not check_password_hash(current_user.password_hash, current_pw):
            flash('Current password is incorrect.', 'error')
        else:
            current_user.password_hash = generate_password_hash(new_pw, method='pbkdf2:sha256')
            db.session.commit()
            log_audit("Changed their password", "Settings")
            flash('Password updated successfully.', 'success')
            
        return redirect(url_for('admin_settings'))
    return render_template('admin/settings.html')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
