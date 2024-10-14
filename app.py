import joblib
from flask import Flask, render_template, redirect, url_for, request, flash
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField
from wtforms.validators import InputRequired, Email, Length
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sqlalchemy.exc import IntegrityError
from wtforms import StringField, SubmitField, TextAreaField, BooleanField
from wtforms.validators import InputRequired, Length, Email
from flask_wtf.file import FileField, FileRequired
from flask_wtf import FlaskForm

# Loading models
filename = 'diabetes-prediction-rfc-model.pkl'
classifier = pickle.load(open(filename, 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
model1 = pickle.load(open('model1.pkl', 'rb'))

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
bootstrap = Bootstrap(app)
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Database model for user
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(15), unique=True)
    email = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(80))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Flask-WTForms for Login and Registration
class LoginForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=8, max=80)])
    remember = BooleanField('remember me')

class RegisterForm(FlaskForm):
    email = StringField('Email', validators=[InputRequired(), Email(message='Invalid email'), Length(max=50)])
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=8, max=80)])

# Form for admin to create or edit users
class AdminUserForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=15)])
    email = StringField('Email', validators=[InputRequired(), Email(message='Invalid email'), Length(max=50)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=8, max=80)])


'''
class PatientRegistrationForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired(), Length(max=50)])
    email = StringField('Email', validators=[InputRequired(), Email(), Length(max=50)])
    password = StringField('Password', validators=[InputRequired(), Length(min=6, max=50)])
    patient_name = StringField('Patient Name', validators=[InputRequired(), Length(max=100)])
    patient_id = StringField('Patient ID', validators=[InputRequired(), Length(max=20)])
    age = StringField('Age', validators=[InputRequired(), Length(max=3)])
    gender = StringField('Gender', validators=[InputRequired(), Length(max=6)])
    birthdate = StringField('Birthdate', validators=[InputRequired()])
    phone_number = StringField('Phone Number', validators=[InputRequired(), Length(max=15)])
    address = StringField('Address', validators=[InputRequired(), Length(max=200)])
    city = StringField('City', validators=[InputRequired(), Length(max=100)])
    state = StringField('State', validators=[InputRequired(), Length(max=100)])
    zip_code = StringField('Zip Code', validators=[InputRequired(), Length(max=10)])
    emergency_contact_name = StringField('Emergency Contact Name', validators=[InputRequired()])
    emergency_contact_phone = StringField('Emergency Contact Phone', validators=[InputRequired()])
    relationship_to_patient = StringField('Relationship to Patient', validators=[InputRequired()])
    medical_history = TextAreaField('Medical History', validators=[Length(max=500)])
    current_medications = StringField('Current Medications', validators=[Length(max=200)])
    allergies = StringField('Allergies', validators=[Length(max=200)])
    blood_type = StringField('Blood Type', validators=[Length(max=3)])
    insurance_provider = StringField('Insurance Provider', validators=[Length(max=100)])
    insurance_policy_number = StringField('Insurance Policy Number', validators=[Length(max=50)])
    insurance_group_number = StringField('Insurance Group Number', validators=[Length(max=50)])
    consent_to_treatment = BooleanField('Consent to Treatment')
    consent_to_share_info = BooleanField('Consent to Share Information')
    submit = SubmitField('Register Now')
'''
class DiabetesData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    pregnancies = db.Column(db.Integer, nullable=False)
    glucose = db.Column(db.Float, nullable=False)
    bloodpressure = db.Column(db.Float, nullable=False)
    skinthickness = db.Column(db.Float, nullable=False)
    insulin = db.Column(db.Float, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    dpf = db.Column(db.Float, nullable=False)
    age = db.Column(db.Integer, nullable=False)
    prediction = db.Column(db.String(10), nullable=False)
   

class BreastCancerData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    clump_thickness = db.Column(db.Integer, nullable=False)
    uniform_cell_size = db.Column(db.Integer, nullable=False)
    uniform_cell_shape = db.Column(db.Integer, nullable=False)
    marginal_adhesion = db.Column(db.Integer, nullable=False)
    single_epithelial_size = db.Column(db.Integer, nullable=False)
    bare_nuclei = db.Column(db.Integer, nullable=False)
    bland_chromatin = db.Column(db.Integer, nullable=False)
    normal_nucleoli = db.Column(db.Integer, nullable=False)
    mitoses = db.Column(db.Integer, nullable=False)
    prediction = db.Column(db.String(10), nullable=False) 
     # Stores either 'Malignant' or 'Benign'

class KidneyDiseasePrediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    blood_pressure = db.Column(db.Float, nullable=False)
    specific_gravity = db.Column(db.Float, nullable=False)
    albumin = db.Column(db.Float, nullable=False)
    blood_sugar_level = db.Column(db.Float, nullable=False)
    red_blood_cells_count = db.Column(db.Float, nullable=False)
    pus_cell_count = db.Column(db.Float, nullable=False)
    pus_cell_clumps = db.Column(db.Float, nullable=False)
    prediction = db.Column(db.String(50), nullable=False)
    

class LiverData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    total_bilirubin = db.Column(db.Float, nullable=False)
    direct_bilirubin = db.Column(db.Float, nullable=False)
    alkaline_phosphotase = db.Column(db.Integer, nullable=False)
    alamine_aminotransferase = db.Column(db.Integer, nullable=False)
    total_proteins = db.Column(db.Float, nullable=False)
    albumin = db.Column(db.Float, nullable=False)
    albumin_globulin_ratio = db.Column(db.Float, nullable=False)
    prediction = db.Column(db.String(50), nullable=False)
    

    def __init__(self, total_bilirubin, direct_bilirubin, alkaline_phosphotase,
                 alamine_aminotransferase, total_proteins, albumin, albumin_globulin_ratio):
        self.total_bilirubin = total_bilirubin
        self.direct_bilirubin = direct_bilirubin
        self.alkaline_phosphotase = alkaline_phosphotase
        self.alamine_aminotransferase = alamine_aminotransferase
        self.total_proteins = total_proteins
        self.albumin = albumin
        self.albumin_globulin_ratio = albumin_globulin_ratio
        

    
class HeartDiseasePrediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    age = db.Column(db.Float, nullable=False)
    trestbps = db.Column(db.Float, nullable=False)
    chol = db.Column(db.Float, nullable=False)
    thalach = db.Column(db.Float, nullable=False)
    oldpeak = db.Column(db.Float, nullable=False)
    sex_0 = db.Column(db.Float, nullable=False)
    sex_1 = db.Column(db.Float, nullable=False)
    cp_0 = db.Column(db.Float, nullable=False)
    cp_1 = db.Column(db.Float, nullable=False)
    cp_2 = db.Column(db.Float, nullable=False)
    cp_3 = db.Column(db.Float, nullable=False)
    fbs_0 = db.Column(db.Float, nullable=False)
    restecg_0 = db.Column(db.Float, nullable=False)
    restecg_1 = db.Column(db.Float, nullable=False)
    restecg_2 = db.Column(db.Float, nullable=False)
    exang_0 = db.Column(db.Float, nullable=False)
    exang_1 = db.Column(db.Float, nullable=False)
    slope_0 = db.Column(db.Float, nullable=False)
    slope_1 = db.Column(db.Float, nullable=False)
    slope_2 = db.Column(db.Float, nullable=False)
    ca_0 = db.Column(db.Float, nullable=False)
    ca_1 = db.Column(db.Float, nullable=False)
    ca_2 = db.Column(db.Float, nullable=False)
    thal_1 = db.Column(db.Float, nullable=False)
    thal_2 = db.Column(db.Float, nullable=False)
    thal_3 = db.Column(db.Float, nullable=False)
    result = db.Column(db.String(50), nullable=False)
    
    
# Routes for rendering templates and login/signup functionality
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/help')
def help():
    return render_template("help.html")

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/tc')
def terms():
    return render_template("tc.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            
            # Check if the user is the admin
            if user.username == 'admin':
                return redirect('http://127.0.0.1:5000/admin')  # Redirect to admin page
            else:
                return redirect(url_for('dashboard'))  # Redirect to normal user dashboard
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template("login.html", form=form)
'''
@app.route('/register_patient', methods=['GET', 'POST'])
def register_patient():
    form = PatientRegistrationForm()  # Initialize your form
    if form.validate_on_submit():
        # Save patient information in the database
        # (You'll need to implement the Patient model and save logic here)
        flash('Patient registered successfully!', 'success')
        return redirect(url_for('register_patient'))  # Redirect or render a success page

    return render_template('register_patient.html', form=form)
'''

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = RegisterForm()

    if form.validate_on_submit():
        # Check if the username or email already exists
        existing_user = User.query.filter((User.username == form.username.data) | (User.email == form.email.data)).first()
        if existing_user:
            flash('Username or email already exists. Please choose another.', 'error')
            return redirect(url_for('signup'))

        # Hash password and add the new user
        hashed_password = generate_password_hash(form.password.data, method='sha256')
        new_user = User(username=form.username.data, email=form.email.data, password=hashed_password)

        db.session.add(new_user)

        try:
            db.session.commit()
            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('login'))
        except IntegrityError:
            db.session.rollback()
            flash('An error occurred during signup. Please try again.', 'error')

    # Render the signup form on GET request or if validation fails
    return render_template('signup.html', form=form)

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

# Disease prediction routes
@app.route("/disindex")
def disindex():
    return render_template("disindex.html")

@app.route("/cancer")
@login_required
def cancer():
    return render_template("cancer.html")

@app.route("/diabetes")
@login_required
def diabetes():
    return render_template("diabetes.html")

@app.route("/heart")
@login_required
def heart():
    return render_template("heart.html")

@app.route("/kidney")
@login_required
def kidney():
    return render_template("kidney.html")

@app.route("/liver")
@login_required
def liver():
    return render_template("liver.html")


# Value prediction functions for Kidney and Liver
def ValuePredictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1, size)
    if size == 7:
        loaded_model = joblib.load('kidney_model.pkl')
        result = loaded_model.predict(to_predict)
    return result[0]

@app.route("/predictkidney", methods=['POST'])
def predictkidney():
    if request.method == "POST":
        to_predict_list = list(map(float, request.form.values()))
        result = ValuePredictor(to_predict_list, 7)
        prediction = "high risk of Kidney Disease" if int(result) == 1 else "low risk of Kidney Disease"
        
        # Save to database
        new_prediction = KidneyDiseasePrediction(
            blood_pressure=to_predict_list[0],
            specific_gravity=to_predict_list[1],
            albumin=to_predict_list[2],
            blood_sugar_level=to_predict_list[3],
            red_blood_cells_count=to_predict_list[4],
            pus_cell_count=to_predict_list[5],
            pus_cell_clumps=to_predict_list[6],
            prediction=prediction
        )
        db.session.add(new_prediction)
        db.session.commit()
        
    return render_template("kidney_result.html", prediction_text=prediction)

#liver model
def ValuePred(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1, size)
    if size == 7:
        loaded_model = joblib.load('liver_model.pkl')
        result = loaded_model.predict(to_predict)
    return result[0]

@app.route('/predictliver', methods=['POST'])
def predictliver():
    if request.method == "POST":
        # Convert form values to a list of float values
        to_predict_list = list(map(float, request.form.values()))
        
        # Assuming the function ValuePred is already defined and 7 is the feature length
        result = ValuePred(to_predict_list, 7)
        
        # Check the prediction result and set the appropriate message
        prediction = "high risk of Liver Disease" if int(result) == 1 else "low risk of Liver Disease"
    
    # Render the prediction result template
    return render_template("liver_result.html", prediction_text=prediction)

@app.route('/liver', methods=['GET'])
def liver_report():
    patient_name = request.args.get('patient_name', '')
    patient_id = request.args.get('patient_id', '')
    age = request.args.get('age', '')
    gender = request.args.get('gender', '')
    
    # Default diagnostic values to None
    total_bilirubin = request.args.get('Total_Bilirubin', '')
    direct_bilirubin = request.args.get('Direct_Bilirubin', '')
    alkaline_phosphotase = request.args.get('Alkaline_Phosphotase', '')
    alamine_aminotransferase = request.args.get('Alamine_Aminotransferase', '')
    total_proteins = request.args.get('Total_Protiens', '')
    albumin = request.args.get('Albumin', '')
    albumin_globulin_ratio = request.args.get('Albumin_and_Globulin_Ratio', '')

    # Check if all form values are provided for prediction
    if total_bilirubin and direct_bilirubin and alkaline_phosphotase and alamine_aminotransferase and total_proteins and albumin and albumin_globulin_ratio:
        to_predict_list = [
            float(total_bilirubin),
            float(direct_bilirubin),
            float(alkaline_phosphotase),
            float(alamine_aminotransferase),
            float(total_proteins),
            float(albumin),
            float(albumin_globulin_ratio)
        ]
        
        # Assume ValuePred is defined and takes the input list and number of features
        result = ValuePred(to_predict_list, 7)
        prediction_text = "high risk of Liver Disease" if int(result) == 1 else "low risk of Liver Disease"
    else:
        prediction_text = "Please provide all the required values."

    
    
    # Render the template with dynamic values
    return render_template('liver_report.html',
                           patient_name=patient_name,
                           patient_id=patient_id,
                           age=age,
                           gender=gender,
                        
                           total_bilirubin=total_bilirubin,
                           direct_bilirubin=direct_bilirubin,
                           alkaline_phosphotase=alkaline_phosphotase,
                           alamine_aminotransferase=alamine_aminotransferase,
                           total_proteins=total_proteins,
                           albumin=albumin,
                           albumin_globulin_ratio=albumin_globulin_ratio,
                           prediction_text=prediction_text)

#####################################################################################################
# cancer model
@app.route('/predict', methods=['POST'])
def predict():
    # Get input features from form
    input_features = [int(x) for x in request.form.values()]
    
    # Feature names
    features_name = ['clump_thickness', 'uniform_cell_size', 'uniform_cell_shape', 'marginal_adhesion',
                     'single_epithelial_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses']
    
    # Create a DataFrame for prediction
    features_value = [np.array(input_features)]
    df = pd.DataFrame(features_value, columns=features_name)
    
    # Predict using the trained model
    output = model.predict(df)[0]  # Get the single output (assume model gives array)
    
    # Interpret the prediction result
    res_val = "a high risk of Breast Cancer" if output == 4 else "a low risk of Breast Cancer"
    
    # Save the input and result to the database
    new_data = BreastCancerData(
        clump_thickness=input_features[0],
        uniform_cell_size=input_features[1],
        uniform_cell_shape=input_features[2],
        marginal_adhesion=input_features[3],
        single_epithelial_size=input_features[4],
        bare_nuclei=input_features[5],
        bland_chromatin=input_features[6],
        normal_nucleoli=input_features[7],
        mitoses=input_features[8],
        prediction=res_val
    )
    
    # Commit the data to the database
    db.session.add(new_data)
    db.session.commit()
    
    # Return the result page
    return render_template('cancer_result.html', prediction_text=f'Patient has {res_val}')

################################################################################################
# diabates
df1 = pd.read_csv('diabetes.csv')

# Renaming DiabetesPedigreeFunction as DPF
df1 = df1.rename(columns={'DiabetesPedigreeFunction': 'DPF'})

# Replacing the 0 values from ['Glucose','BloodPressure','SkinThickness','Insulin','BMI'] by NaN
df_copy = df1.copy(deep=True)
df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df_copy[['Glucose', 'BloodPressure',
                                                                                    'SkinThickness', 'Insulin',
                                                                                    'BMI']].replace(0, np.NaN)

# Replacing NaN value by mean, median depending upon distribution
df_copy['Glucose'].fillna(df_copy['Glucose'].mean(), inplace=True)
df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean(), inplace=True)
df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median(), inplace=True)
df_copy['Insulin'].fillna(df_copy['Insulin'].median(), inplace=True)
df_copy['BMI'].fillna(df_copy['BMI'].median(), inplace=True)

# Model Building

X = df1.drop(columns='Outcome')
y = df1['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
# Creating Random Forest Model

classifier = RandomForestClassifier(n_estimators=20)
classifier.fit(X_train, y_train)

# Creating a pickle file for the classifier
filename = 'diabetes-prediction-rfc-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))

@app.route('/predictt', methods=['POST'])
def predictt():
    if request.method == 'POST':
        # Retrieve data from the form
        pregnancies = int(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        bloodpressure = float(request.form['bloodpressure'])
        skinthickness = float(request.form['skinthickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])

        # Prepare data for prediction
        data = np.array([[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, dpf, age]])
        my_prediction = classifier.predict(data)
        prediction = 'Positive' if my_prediction == 1 else 'Negative'

        # Save the form data and prediction result to the database
        new_data = DiabetesData(pregnancies=pregnancies, glucose=glucose, bloodpressure=bloodpressure, 
                                skinthickness=skinthickness, insulin=insulin, bmi=bmi, dpf=dpf, age=age, 
                                prediction=prediction)
        db.session.add(new_data)
        db.session.commit()

        # Render the prediction result
        return render_template('diab_result.html', prediction=prediction)

    
@app.route('/get_user_data/<int:user_id>', methods=['GET'])
def get_user_data(user_id):
    user = User.query.get(user_id)
    if user:
        # Send the user data as JSON
        user_data = {
            'pregnancies': user.pregnancies,
            'glucose': user.glucose,
            'bloodpressure': user.bloodpressure,
            'skinthickness': user.skinthickness,
            'insulin': user.insulin,
            'bmi': user.bmi,
            'dpf': user.dpf,
            'age': user.age,
            'prediction': user.prediction
        }
        
#########################################################################################################
    
@app.route('/predictheart', methods=['POST'])
def predictheart():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]

    features_name = ["age", "trestbps", "chol", "thalach", "oldpeak", "sex_0",
                     "sex_1", "cp_0", "cp_1", "cp_2", "cp_3", "fbs_0",
                     "restecg_0", "restecg_1", "restecg_2", "exang_0", "exang_1",
                     "slope_0", "slope_1", "slope_2", "ca_0", "ca_1", "ca_2", "thal_1",
                     "thal_2", "thal_3"]

    df = pd.DataFrame(features_value, columns=features_name)
    output = model1.predict(df)

    if output == 1:
        res_val = "a high risk of Heart Disease"
    else:
        res_val = "a low risk of Heart Disease"

    # Save the result in the database
    new_prediction = HeartDiseasePrediction(
        age=input_features[0],
        trestbps=input_features[1],
        chol=input_features[2],
        thalach=input_features[3],
        oldpeak=input_features[4],
        sex_0=input_features[5],
        sex_1=input_features[6],
        cp_0=input_features[7],
        cp_1=input_features[8],
        cp_2=input_features[9],
        cp_3=input_features[10],
        fbs_0=input_features[11],
        restecg_0=input_features[12],
        restecg_1=input_features[13],
        restecg_2=input_features[14],
        exang_0=input_features[15],
        exang_1=input_features[16],
        slope_0=input_features[17],
        slope_1=input_features[18],
        slope_2=input_features[19],
        ca_0=input_features[20],
        ca_1=input_features[21],
        ca_2=input_features[22],
        thal_1=input_features[23],
        thal_2=input_features[24],
        thal_3=input_features[25],
        result=res_val
    )
    
    db.session.add(new_prediction)
    db.session.commit()

    return render_template('heart_result.html', prediction_text='Patient has {}'.format(res_val))
######################################################################################################

# Admin panel for user management (CRUD)
@app.route('/admin')
@login_required
def admin_panel():
    if current_user.username != 'admin':
        flash("You don't have access to this page.")
        return redirect(url_for('index'))
    users = User.query.all()
    return render_template('admin.html', users=users)

@app.route('/admin/user/create', methods=['GET', 'POST'])
@login_required
def admin_create_user():
    if current_user.username != 'admin'or 'Admin':
        flash("You don't have access to this page.")
        return redirect(url_for('index'))
    
    form = AdminUserForm()
    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data, method='sha256')
        new_user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('User created successfully!')
        return redirect(url_for('admin_panel'))
    
    return render_template('.html', form=form)

@app.route('/admin/user/edit/<int:id>', methods=['GET', 'POST'])
@login_required
def admin_edit_user(id):
    if current_user.username != 'admin':
        flash("You don't have access to this page.")
        return redirect(url_for('index'))
    
    user = User.query.get_or_404(id)
    form = AdminUserForm(obj=user)
    
    if form.validate_on_submit():
        user.username = form.username.data
        user.email = form.email.data
        if form.password.data:
            user.password = generate_password_hash(form.password.data, method='sha256')
        db.session.commit()
        flash('User updated successfully!')
        return redirect(url_for('admin_panel'))
    
    return render_template('edit_user.html', form=form)

@app.route('/admin/user/delete/<int:id>', methods=['POST'])
@login_required
def admin_delete_user(id):
    if current_user.username != 'admin':
        flash("You don't have access to this page.")
        return redirect(url_for('index'))
    
    user = User.query.get_or_404(id)
    db.session.delete(user)
    db.session.commit()
    flash('User deleted successfully!')
    return redirect(url_for('admin_panel'))

# Main execution
if __name__ == "__main__":
    db.create_all()
    app.run(debug=True)