import os
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Get the absolute path of this script
base_dir = os.path.abspath(os.path.dirname(__file__))

app = Flask(
    __name__,
    template_folder='templates',
    static_folder='static',
    static_url_path=''
)

# Load your trained models
model_depression = joblib.load(os.path.join(base_dir, 'Model', 'best_model_depression.pkl'))
model_anxiety    = joblib.load(os.path.join(base_dir, 'Model', 'best_model_anxiety.pkl'))
model_stress     = joblib.load(os.path.join(base_dir, 'Model', 'best_model_stress.pkl'))

# One-hot for Age
AGE_ONEHOT = {
    "18-22":    [1,0,0,0,0],
    "23-26":    [0,1,0,0,0],
    "27-30":    [0,0,1,0,0],
    "Above 30": [0,0,0,1,0],
    "Below 18": [0,0,0,0,1],
}

# One-hot for Departments
DEPT_ONEHOT = {
    'Biological Sciences':                 [1,0,0,0,0,0,0,0,0,0],
    'Business & Entrepreneurship':         [0,1,0,0,0,0,0,0,0,0],
    'Civil Engineering':                   [0,0,1,0,0,0,0,0,0,0],
    'Computer Engineering / CS':           [0,0,0,1,0,0,0,0,0,0],
    'Electrical / Electronics Engineering':[0,0,0,0,1,0,0,0,0,0],
    'Env & Life Sciences':                 [0,0,0,0,0,1,0,0,0,0],
    'Mechanical Engineering':              [0,0,0,0,0,0,1,0,0,0],
    'Other':                               [0,0,0,0,0,0,0,1,0,0],
    'Other Engineering':                   [0,0,0,0,0,0,0,0,1,0],
    'Pharmacy / Public Health':            [0,0,0,0,0,0,0,0,0,1],
}

# Map CGPA string to int
CGPA_MAP = {
    'Below 2.50': 0,
    '2.50 - 2.99': 1,
    '3.00 - 3.39': 2,
    '3.40 - 3.79': 3,
    '3.80 - 4.00': 4,
    'Other': -1
}

# Recommendation functions
def recommend_anxiety_action(pred_label):
    if pred_label == 0:
        return (
            "Minimal anxiety: No immediate intervention needed. "
            "Encourage maintaining healthy habits like adequate sleep, "
            "balanced nutrition, physical exercise, and mindfulness practices "
            "to continue supporting emotional well-being."
        )
    elif pred_label == 1:
        return (
            "Mild anxiety: Recommend participation in peer support groups, "
            "stress-relief workshops, or online mental wellness resources. "
            "Encourage practicing breathing exercises, journaling, and "
            "developing time management skills to reduce academic stress."
        )
    elif pred_label == 2:
        return (
            "Moderate anxiety: Suggest scheduling individual counseling sessions "
            "with a mental health professional. Explore structured anxiety "
            "management programs such as cognitive-behavioral techniques, and "
            "recommend building a supportive social network for sharing concerns."
        )
    elif pred_label == 3:
        return (
            "Severe anxiety: Advise immediate referral to professional mental "
            "health services. A comprehensive clinical evaluation may be needed "
            "to assess possible anxiety disorders. Encourage crisis resources if "
            "the student feels overwhelmed or unable to function day-to-day."
        )
    else:
        return (
            "Consult a mental health counselor for further assessment if symptoms "
            "do not match any standard category."
        )

def recommend_stress_action(pred_label):
    if pred_label == 0:
        return (
            "Low stress: No immediate intervention required. Encourage students "
            "to maintain a healthy study-life balance by taking regular breaks, "
            "engaging in hobbies, maintaining social connections, and using basic "
            "stress-reduction techniques such as mindfulness or light exercise."
        )
    elif pred_label == 1:
        return (
            "Moderate stress: Suggest participating in structured stress-management "
            "workshops, group support programs, or guided mindfulness and relaxation sessions. "
            "Encourage identifying stress triggers, improving time-management skills, "
            "and exploring healthy coping strategies to manage workload demands."
        )
    elif pred_label == 2:
        return (
            "High perceived stress: Recommend more focused intervention through "
            "individual counseling with a mental health professional. Discuss therapeutic "
            "options, explore stress-reduction programs, and coordinate academic support "
            "resources such as tutoring or workload adjustments if necessary. "
            "Ongoing monitoring of stress levels is highly encouraged."
        )
    else:
        return (
            "Follow up with a qualified mental health counselor for more in-depth assessment "
            "if the stress level does not clearly fit these categories."
        )

def recommend_depression_action(pred_label):
    if pred_label == 0:
        return (
            "No depression: No formal intervention needed. Encourage the student to "
            "continue engaging in positive activities such as regular exercise, "
            "maintaining social connections, and pursuing hobbies or academic interests "
            "to support emotional resilience."
        )
    elif pred_label == 1:
        return (
            "Minimal depression: Provide mental health awareness materials, including "
            "self-help resources and information on early warning signs. Encourage the "
            "student to monitor their mood and reach out to support networks if symptoms "
            "worsen or persist."
        )
    elif pred_label == 2:
        return (
            "Mild depression: Recommend participating in peer support groups or "
            "community-based wellness programs. Suggest regular check-ins with a "
            "counselor or academic advisor, and encourage healthy coping skills such as "
            "journaling, mindfulness, and time management."
        )
    elif pred_label == 3:
        return (
            "Moderate depression: Recommend professional counseling with a licensed "
            "mental health provider. Encourage the student to explore therapeutic options "
            "such as cognitive-behavioral therapy, and monitor symptoms regularly to "
            "prevent escalation."
        )
    elif pred_label == 4:
        return (
            "Moderately severe depression: Strongly encourage referral for a full clinical "
            "psychological evaluation. Recommend combining professional therapy with "
            "psychiatric consultation if needed, and provide information on crisis resources "
            "in case of acute distress."
        )
    elif pred_label == 5:
        return (
            "Severe depression: Advise urgent referral to psychiatric services for a "
            "comprehensive evaluation and potential medication. Ensure the student is aware "
            "of emergency hotlines and 24/7 crisis support, and coordinate immediate follow-up "
            "to safeguard their well-being."
        )
    else:
        return (
            "Follow up with a licensed mental health professional for a thorough assessment "
            "if the symptoms do not clearly fit a standard category."
        )

# Gauge mapping for frontend
GAUGE_DATA = {
    'depression': {
        'levels': ['No Depression', 'Minimal', 'Mild', 'Moderate', 'Mod. Severe', 'Severe'],
        'icons': ['😊','🙂','😐','🙁','😦','😢'],
        'colors': ['#43c463','#92d36e','#ffe066','#ffb366','#ff8666','#e04848']
    },
    'anxiety': {
        'levels': ['Minimal', 'Mild', 'Moderate', 'Severe'],
        'icons': ['😊','🙂','😐','😢'],
        'colors': ['#43c463','#92d36e','#ffe066','#e04848']
    },
    'stress': {
        'levels': ['Low', 'Moderate', 'High'],
        'icons': ['😊','😐','😢'],
        'colors': ['#43c463','#ffe066','#e04848']
    }
}

# --- Main Page Routes ---
@app.route('/')
def main():
    return render_template('page-profile.html')

@app.route('/index.html')
def index_page():
    return render_template('index.html')

@app.route('/settings.html')
def settings_page():
    return render_template('settings.html')

@app.route('/page-profile.html')
def profile_page():
    return render_template('page-profile.html')

# --- Predict Page Routes ---

@app.route('/page-predict.html')
def predict_page():
    return render_template('page-predict.html')


@app.route('/predict-stress.html')
def show_predict_stress():
    return render_template('predict-stress.html')

@app.route('/predict-stress', methods=['POST'])
def predict_stress():
    data = request.get_json()
    features = []
    features.extend(AGE_ONEHOT.get(data['Age'], [0,0,0,0,0]))
    features.extend(DEPT_ONEHOT.get(data['Departments'], [0]*10))
    features.append(int(data['Gender']))
    features.append(int(data['Year']))
    features.append(CGPA_MAP.get(data['CGPA'], -1))
    features.append(int(data['Scholarship']))
    features.extend([
        int(data['Stress_UpsetAcademic']),
        int(data['Stress_LossOfControl']),
        int(data['Stress_NervousStressed']),
        int(data['Stress_OverwhelmedByTasks']),
        int(data['Stress_ConfidentHandling']),
        int(data['Stress_OnTrack']),
        int(data['Stress_ControlIrritation']),
        int(data['Stress_Performance']),
        int(data['Stress_AngeredByGrades']),
        int(data['Stress_DifficultiesPileUp'])
    ])
    X = np.array([features])
    pred = int(model_stress.predict(X)[0])
    STRESS_LABELS = {
        0: 'Low Stress',
        1: 'Moderate Stress',
        2: 'High Perceived Stress'
    }
    recommendation = recommend_stress_action(pred)
    result_label = STRESS_LABELS.get(pred, str(pred))
    return jsonify({
        'result': result_label,
        'recommendation': recommendation,
        'gauge_type': 'stress',
        'gauge_value': pred
    })

@app.route('/predict-Depression.html')
def show_depression_form():
    return render_template('predict-Depression.html')

@app.route('/predict-Depression', methods=['POST'])
def predict_depression():
    data = request.get_json()
    features = []
    features.extend(AGE_ONEHOT.get(data['Age'], [0,0,0,0,0]))
    features.extend(DEPT_ONEHOT.get(data['Departments'], [0]*10))
    features.append(int(data['Gender']))
    features.append(int(data['Year']))
    features.append(CGPA_MAP.get(data['CGPA'], -1))
    features.append(int(data['Scholarship']))
    features.extend([
        int(data['Depression_LowInterest']),
        int(data['Depression_Hopeless']),
        int(data['Depression_SleepIssues']),
        int(data['Depression_LowEnergy']),
        int(data['Depression_AppetiteChange']),
        int(data['Depression_SelfWorth']),
        int(data['Depression_Concentration']),
        int(data['Depression_MovementChanges']),
        int(data['Depression_SuicidalThoughts'])
    ])
    X = np.array([features])
    pred = int(model_depression.predict(X)[0])
    DEPRESSION_LABELS = {
        0: 'No Depression',
        1: 'Minimal Depression',
        2: 'Mild Depression',
        3: 'Moderate Depression',
        4: 'Moderately Severe Depression',
        5: 'Severe Depression'
    }
    recommendation = recommend_depression_action(pred)
    result_label = DEPRESSION_LABELS.get(pred, str(pred))
    return jsonify({
        'result': result_label,
        'recommendation': recommendation,
        'gauge_type': 'depression',
        'gauge_value': pred
    })

@app.route('/predict-Anxiety.html')
def show_predict_anxiety():
    return render_template('predict-Anxiety.html')

@app.route('/predict-Anxiety', methods=['POST'])
def predict_anxiety():
    data = request.get_json()
    features = []
    features.extend(AGE_ONEHOT.get(data['Age'], [0,0,0,0,0]))
    features.extend(DEPT_ONEHOT.get(data['Departments'], [0]*10))
    features.append(int(data['Gender']))
    features.append(int(data['Year']))
    features.append(CGPA_MAP.get(data['CGPA'], -1))
    features.append(int(data['Scholarship']))
    features.extend([
        int(data['Anxiety_NervousOnEdge']),
        int(data['Anxiety_UnstoppableWorry']),
        int(data['Anxiety_TroubleRelaxing']),
        int(data['Anxiety_Irritated']),
        int(data['Anxiety_Overthinking']),
        int(data['Anxiety_Restless']),
        int(data['Anxiety_Fearful'])
    ])
    X = np.array([features])
    pred = int(model_anxiety.predict(X)[0])
    ANXIETY_LABELS = {
        0: 'Minimal Anxiety',
        1: 'Mild Anxiety',
        2: 'Moderate Anxiety',
        3: 'Severe Anxiety'
    }
    recommendation = recommend_anxiety_action(pred)
    result_label = ANXIETY_LABELS.get(pred, str(pred))
    return jsonify({
        'result': result_label,
        'recommendation': recommendation,
        'gauge_type': 'anxiety',
        'gauge_value': pred
    })

# --- Result Page Routes ---
@app.route('/predict-result-anxiety.html')
def predict_result_anxiety():
    result = request.args.get('result', '')
    recommendation = request.args.get('recommendation', '')
    gauge_type = request.args.get('gauge_type', 'anxiety')
    gauge_value = request.args.get('gauge_value', 0)
    return render_template(
        'predict-result-anxiety.html',
        result=result,
        recommendation=recommendation,
        gauge_type=gauge_type,
        gauge_value=int(gauge_value),
        gauge_data=GAUGE_DATA
    )

@app.route('/predict-result-depression.html')
def predict_result_depression():
    result = request.args.get('result', '')
    recommendation = request.args.get('recommendation', '')
    gauge_type = request.args.get('gauge_type', 'depression')
    gauge_value = request.args.get('gauge_value', 0)
    return render_template(
        'predict-result-depression.html',
        result=result,
        recommendation=recommendation,
        gauge_type=gauge_type,
        gauge_value=int(gauge_value),
        gauge_data=GAUGE_DATA
    )

@app.route('/predict-result-stress.html')
def predict_result_stress():
    result = request.args.get('result', '')
    recommendation = request.args.get('recommendation', '')
    gauge_type = request.args.get('gauge_type', 'stress')
    gauge_value = request.args.get('gauge_value', 0)
    return render_template(
        'predict-result-stress.html',
        result=result,
        recommendation=recommendation,
        gauge_type=gauge_type,
        gauge_value=int(gauge_value),
        gauge_data=GAUGE_DATA
    )

# Additional dashboard routes...

@app.route('/dashboard2.html')
def index_page2():
    return render_template('dashboard2.html')

@app.route('/dashboard3.html')
def index_page3():
    return render_template('dashboard3.html')

@app.route('/dashboard4.html')
def index_page4():
    return render_template('dashboard4.html')

@app.route('/index5.html')
def index_page5():
    return render_template('index5.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
